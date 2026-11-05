"""Focused reset contract for completed restricted-v2 request sessions."""

from __future__ import annotations

import json
import pathlib
import subprocess
import unittest

import jax
import jax.numpy as jnp

from simulator import (
	ACTION_TACKLE,
	MAX_REQUEST_EPISODE,
	SESSION_NOT_TERMINAL,
	SESSION_OVERFLOW,
	SESSION_RESET,
	SESSION_STALE_EPISODE,
	SESSION_UNSUPPORTED,
	SessionReset,
	make_request_session,
	project_request_view,
	reset_request_session,
	submit_choice,
)
from test.test_request_session import _as_list, _leaf_equal, _state, _view_projection

ROOT = pathlib.Path(__file__).resolve().parents[1]
ORACLE = ROOT / "reference" / "neutral-turn-oracle.cjs"
_COMPILED_RESET = jax.jit(jax.vmap(reset_request_session))


def _stack(trees):
	return jax.tree_util.tree_map(lambda *values: jnp.stack(values), *trees)


def _lane(tree, index):
	return jax.tree_util.tree_map(lambda value: value[index], tree)


class RequestSessionResetTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		completed = subprocess.run(
			["node", str(ORACLE), "--session-duels"],
			cwd=ROOT,
			check=True,
			capture_output=True,
			text=True,
			timeout=25,
		)
		cls.fixture = json.loads(completed.stdout)
		cls.scenarios = {scenario["id"]: scenario for scenario in cls.fixture["scenarios"]}

	def _terminal_session(self, scenario_id, episode=None):
		scenario = self.scenarios[scenario_id]
		terminal = make_request_session(
			_state(scenario["outcome"]["state"]),
			episode=len(scenario["turns"]) if episode is None else episode,
		)
		return terminal, scenario

	def test_reset_uses_native_fresh_request_and_rejects_prior_episode_submission(self):
		self.assertEqual(
			self.fixture["kind"],
			"gen5-restricted-neutral-tackle-staged-session-full-duel-oracle",
		)
		self.assertIn("each player commits separately through Battle.choose", self.fixture["initialBoundary"])
		terminal, _ = self._terminal_session("tied-first-p1")
		fresh_scenario = self.scenarios["unequal-first-p2"]
		fresh_snapshot = fresh_scenario["initial"]["state"]
		fresh_battle = _state(fresh_snapshot)
		old_episode = int(terminal.episode)

		reset = reset_request_session(terminal, fresh_battle, old_episode)

		self.assertIsInstance(reset, SessionReset)
		self.assertEqual(int(reset.status), SESSION_RESET)
		self.assertEqual(int(reset.session.episode), old_episode + 1)
		self.assertEqual(_as_list(reset.session.pending_choices), [0, 0])
		self.assertTrue(_leaf_equal(reset.session.battle, fresh_battle))
		self.assertEqual(_as_list(reset.session.battle.rng), fresh_snapshot["rng"])
		for audience, side in ((0, "p1"), (1, "p2")):
			view = project_request_view(reset.session, audience)
			self.assertEqual(int(view.episode), old_episode + 1)
			self.assertFalse(bool(view.own_choice_submitted))
			self.assertEqual(
				_view_projection(view.player_view),
				{"request": fresh_scenario["initial"]["requests"][side], "events": []},
			)
			self.assertNotIn("rng", view.player_view._fields)
			self.assertNotIn("pending_choices", view.player_view._fields)

		stale = submit_choice(
			reset.session,
			jnp.asarray(0, jnp.int32),
			jnp.asarray(ACTION_TACKLE, jnp.int32),
			old_episode,
		)
		self.assertEqual(int(stale.status), SESSION_STALE_EPISODE)
		self.assertTrue(_leaf_equal(stale.session, reset.session))
		self.assertEqual(int(stale.transition.draws.count), 0)

	def test_compiled_heterogeneous_reset_lanes_are_independent_and_preserve_rejections(self):
		terminal, _ = self._terminal_session("tied-first-p1")
		live_source = make_request_session(_state(self.scenarios["unequal-first-p2"]["initial"]["state"]))
		live_pending = live_source._replace(
			pending_choices=jnp.asarray([ACTION_TACKLE, 0], dtype=jnp.int32)
		)
		stale_source = terminal._replace(pending_choices=jnp.asarray([2, 0], dtype=jnp.int32))
		unsupported_source = live_pending
		overflow_source, _ = self._terminal_session("unequal-first-p1", MAX_REQUEST_EPISODE)
		invalid_metadata = terminal._replace(pending_choices=jnp.asarray([2, 0], dtype=jnp.int32))
		terminal_pending = terminal._replace(
			pending_choices=jnp.asarray([ACTION_TACKLE, 0], dtype=jnp.int32)
		)

		fresh = _state(self.scenarios["unequal-first-p2"]["initial"]["state"])
		unsupported_replacement = fresh._replace(variant_id=jnp.asarray(999, jnp.int32))
		terminal_replacement = _state(self.scenarios["unequal-first-p2"]["outcome"]["state"])
		sessions = [
			terminal,
			live_pending,
			stale_source,
			unsupported_source,
			overflow_source,
			invalid_metadata,
			terminal_pending,
			terminal,
		]
		battles = [
			fresh,
			fresh,
			unsupported_replacement,
			unsupported_replacement,
			fresh,
			fresh,
			fresh,
			terminal_replacement,
		]
		expected_episodes = jnp.asarray(
			[
				int(terminal.episode),
				int(live_pending.episode),
				int(stale_source.episode) - 1,
				int(unsupported_source.episode),
				MAX_REQUEST_EPISODE,
				int(invalid_metadata.episode),
				int(terminal_pending.episode),
				int(terminal.episode),
			],
			dtype=jnp.int32,
		)

		result = _COMPILED_RESET(_stack(sessions), _stack(battles), expected_episodes)
		self.assertEqual(
			_as_list(result.status),
			[
				SESSION_RESET,
				SESSION_NOT_TERMINAL,
				SESSION_STALE_EPISODE,
				SESSION_UNSUPPORTED,
				SESSION_OVERFLOW,
				SESSION_UNSUPPORTED,
				SESSION_UNSUPPORTED,
				SESSION_UNSUPPORTED,
			],
		)

		for index, (source, replacement, status) in enumerate(zip(sessions, battles, _as_list(result.status))):
			actual = _lane(result.session, index)
			if status == SESSION_RESET:
				expected = source._replace(
					battle=replacement,
					pending_choices=jnp.zeros((2,), jnp.int32),
					episode=source.episode + jnp.asarray(1, jnp.int32),
				)
				self.assertTrue(_leaf_equal(actual, expected), (index, "reset lane uses only its supplied battle"))
			else:
				self.assertTrue(_leaf_equal(actual, source), (index, "rejected lane preserves entire session"))

	def test_scalar_boundaries_and_structural_validation(self):
		terminal, _ = self._terminal_session("unequal-first-p1", MAX_REQUEST_EPISODE - 1)
		fresh = _state(self.scenarios["tied-first-p1"]["initial"]["state"])
		last_safe = reset_request_session(terminal, fresh, MAX_REQUEST_EPISODE - 1)
		self.assertEqual(int(last_safe.status), SESSION_RESET)
		self.assertEqual(int(last_safe.session.episode), MAX_REQUEST_EPISODE)
		self.assertTrue(_leaf_equal(last_safe.session.battle, fresh))

		bad_pending_shape = terminal._replace(pending_choices=jnp.zeros((3,), jnp.int32))
		with self.assertRaisesRegex(ValueError, r"pending_choices must have shape \(2,\)"):
			reset_request_session(bad_pending_shape, fresh, MAX_REQUEST_EPISODE - 1)

		bad_replacement_dtype = fresh._replace(rng=fresh.rng.astype(jnp.int32))
		with self.assertRaisesRegex(TypeError, "initial_state.rng must have dtype"):
			reset_request_session(terminal, bad_replacement_dtype, MAX_REQUEST_EPISODE - 1)

		with self.assertRaisesRegex(TypeError, "expected_episode must be an integer fitting signed int32"):
			reset_request_session(terminal, fresh, jnp.asarray(MAX_REQUEST_EPISODE - 1, jnp.float32))


if __name__ == "__main__":
	unittest.main()
