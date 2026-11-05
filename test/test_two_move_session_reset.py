"""Focused native checks for trusted v4 request-session reset."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from typing import get_type_hints

import jax
import jax.numpy as jnp
import numpy as np

import test.test_neutral_two_move as prior
import test.test_two_move_session as session_tests
from simulator import (
	MAX_REQUEST_EPISODE,
	SESSION_NOT_TERMINAL,
	SESSION_OVERFLOW,
	SESSION_RESET,
	SESSION_RESOLVED,
	SESSION_STALE_EPISODE,
	SESSION_UNSUPPORTED,
	SESSION_WAITING,
	TwoMoveSessionReset,
	make_two_move_request_session,
	reset_two_move_request_session,
)
from simulator.neutral_two_move import ACTION_POUND, ACTION_TACKLE


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "reference" / "neutral-two-move-oracle.cjs"
COMPILED_RESET_BATCH = jax.jit(jax.vmap(reset_two_move_request_session))


def _stack_sessions(sessions):
	return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *sessions)


def _lane(tree, index):
	return jax.tree_util.tree_map(lambda value: value[index], tree)


class TwoMoveSessionResetTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		completed = subprocess.run(
			["node", str(ORACLE), "--sessions"],
			cwd=ROOT,
			check=True,
			capture_output=True,
			text=True,
			timeout=10,
		)
		cls.fixture = json.loads(completed.stdout)
		cls.scenarios = {scenario["id"]: scenario for scenario in cls.fixture["scenarios"]}

	def _assert_request(self, view, expected, label):
		projection = prior._jax_view_projection(view.player_view)
		self.assertEqual(projection["request"], expected, (label, "native request"))
		self.assertEqual(projection["events"], [], (label, "fresh view has no events"))
		self.assertEqual(
			prior._as_list(view.player_view.legal_move_mask),
			[not move["disabled"] for move in expected["own"]["moves"]],
			(label, "native legal mask"),
		)

	def _assert_draws(self, result, expected, label):
		trace = result.transition.draws
		self.assertEqual(int(trace.count), len(expected), (label, "draw count"))
		for index, draw in enumerate(expected):
			phase = draw.get("phase", draw["context"])
			bound = 2 if draw["context"] == "speed-sort" else draw["bound"]
			self.assertEqual(int(trace.bounds[index]), bound, (label, index, "bound"))
			self.assertEqual(int(trace.values[index]), draw["value"], (label, index, "value"))
			self.assertEqual(int(trace.contexts[index]), prior.DRAW_CONTEXTS[phase], (label, index, "context"))
			self.assertEqual(prior._as_list(trace.before[index]), draw["before"], (label, index, "before RNG"))
			self.assertEqual(prior._as_list(trace.after[index]), draw["after"], (label, index, "after RNG"))
			if index:
				self.assertEqual(draw["before"], expected[index - 1]["after"], (label, index, "draw chain"))

	def _assert_empty_submission(self, result, session, label):
		self.assertTrue(prior._leaf_equal(result.session, session), (label, "complete session preserved"))
		self.assertTrue(prior._leaf_equal(result.transition.state, session.battle), (label, "battle preserved"))
		self.assertEqual(int(result.transition.draws.count), 0, (label, "no RNG draws"))
		self.assertFalse(any(prior._as_list(result.transition.events.executed)), (label, "no events"))
		self.assertEqual(prior._as_list(result.transition.event_move_ids), [-1, -1], (label, "no move IDs"))

	def test_native_terminal_reset_and_fresh_local_token(self):
		self.assertEqual(self.fixture["kind"], "gen5-restricted-neutral-two-move-staged-session-oracle")
		self.assertEqual(self.fixture["variant"], prior.VARIANT_NAME)
		self.assertEqual(self.fixture["upstream"]["commit"], "1242c1e2d9763ba5b85d33f13aecb89bae19ae3d")
		self.assertEqual(len(self.scenarios), 10)
		self.assertTrue(get_type_hints(TwoMoveSessionReset))
		self.assertTrue(get_type_hints(reset_two_move_request_session))

		terminal_state = prior._state(self.scenarios["first-action-ko"]["turn"]["after"])
		self.assertTrue(bool(terminal_state.terminal), "fixture is a real native first-action knockout")
		old_episode = 37
		old_session = make_two_move_request_session(terminal_state, episode=old_episode)
		self.assertEqual(prior._as_list(old_session.pending_choices), [0, 0])

		scenario = self.scenarios["tackle-pound-unequal-first-p1"]
		fresh = prior._state(scenario["initial"]["state"])
		result = reset_two_move_request_session(old_session, fresh, old_episode)
		self.assertEqual(int(result.status), SESSION_RESET)
		self.assertTrue(prior._leaf_equal(result.session.battle, fresh), "copies complete native initial snapshot including RNG")
		self.assertEqual(prior._as_list(result.session.pending_choices), [0, 0])
		new_episode = old_episode + 1
		self.assertEqual(int(result.session.episode), new_episode, "episode advances from local old value")
		self.assertNotEqual(int(result.session.episode), int(fresh.turn), "native turn does not set local episode")

		for audience, side in ((0, "p1"), (1, "p2")):
			view = session_tests.COMPILED_REQUEST_VIEWS[audience](result.session, audience)
			self.assertEqual(int(view.episode), new_episode, (side, "local episode"))
			self.assertFalse(bool(view.own_choice_submitted), (side, "cleared own bit"))
			self._assert_request(view, scenario["initial"]["requests"][side], side)

		stale = session_tests.COMPILED_SUBMIT(result.session, 0, ACTION_TACKLE, old_episode)
		self.assertEqual(int(stale.status), SESSION_STALE_EPISODE, "pre-reset token is stale")
		self._assert_empty_submission(stale, result.session, "stale pre-reset submit")

		turn = scenario["turn"]
		first = session_tests.COMPILED_SUBMIT(result.session, 0, turn["choices"][0], new_episode)
		self.assertEqual(int(first.status), SESSION_WAITING, "new token accepts first choice")
		self.assertTrue(prior._leaf_equal(first.session.battle, fresh), "first choice preserves replacement battle/RNG")
		self.assertEqual(int(first.session.episode), new_episode)
		self.assertEqual(int(first.transition.draws.count), 0)
		second = session_tests.COMPILED_SUBMIT(first.session, 1, turn["choices"][1], new_episode)
		self.assertEqual(int(second.status), SESSION_RESOLVED, "new token resolves native next turn")
		self.assertEqual(int(second.session.episode), new_episode + 1)
		self.assertEqual(prior._as_list(second.session.pending_choices), [0, 0])
		self.assertTrue(prior._leaf_equal(second.session.battle, prior._state(turn["after"])), "full native next state, PP and RNG")
		self.assertEqual(int(second.transition.status), prior.STATUS_OK)
		self._assert_draws(second, turn["secondCommit"]["draws"], "fresh-token next native turn")
		for audience, side in ((0, "p1"), (1, "p2")):
			view = session_tests.COMPILED_SUBMISSION_VIEWS[audience](second, audience)
			self.assertEqual(
				prior._jax_view_projection(view.resolved_events)["events"],
				turn["events"][side],
				(side, "recipient events"),
			)
			self.assertEqual(
				prior._jax_view_projection(view.player_view)["request"],
				turn["nextRequests"][side],
				(side, "next native request"),
			)

	def test_compiled_heterogeneous_reset_precedence_and_boundaries(self):
		terminal = prior._state(self.scenarios["first-action-ko"]["turn"]["after"])
		normal_scenario = self.scenarios["tackle-pound-unequal-first-p1"]
		depleted_scenario = self.scenarios["depleted-tackle-legal-pound"]
		other_scenario = self.scenarios["pound-tackle-tied-first-p2"]
		fresh_normal = prior._state(normal_scenario["initial"]["state"])
		fresh_depleted = prior._state(depleted_scenario["initial"]["state"])
		fresh_other = prior._state(other_scenario["initial"]["state"])
		self.assertEqual(int(fresh_depleted.pp[0, 0]), 0, "native prepared replacement has one exhausted slot")
		self.assertGreater(int(fresh_depleted.pp[0, 1]), 0, "native request remains supported")

		def old(episode, pending=(0, 0), battle=terminal):
			return make_two_move_request_session(battle, episode=episode)._replace(
				pending_choices=jnp.asarray(pending, dtype=jnp.int32)
			)

		bad_variant = fresh_other._replace(variant_id=jnp.asarray(99, dtype=jnp.int32))
		sessions = (
			old(10),                                      # accepted, ordinary replacement
			old(20),                                      # accepted, partial PP replacement
			old(5, pending=(ACTION_TACKLE, 0), battle=fresh_normal),  # live valid commitment
			old(21, pending=(ACTION_TACKLE, 0), battle=fresh_depleted),  # live held move has no PP
			old(6),                                       # stale beats bad replacement
			old(7),                                       # unsupported replacement variant
			old(8),                                       # terminal replacement unsupported
			old(MAX_REQUEST_EPISODE),                    # overflow
			old(9, pending=(ACTION_TACKLE, 0)),           # terminal held Tackle
			old(10, pending=(ACTION_POUND, 0)),           # terminal held Pound
			old(11, pending=(3, 0)),                      # corrupt pending metadata
			old(0)._replace(episode=jnp.asarray(-1, dtype=jnp.int32)),  # negative local episode
			old(MAX_REQUEST_EPISODE - 1),                 # last safe increment
		)
		replacements = (
			fresh_normal,
			fresh_depleted,
			fresh_normal,
			fresh_normal,
			bad_variant,
			bad_variant,
			terminal,
			fresh_normal,
			fresh_normal,
			fresh_normal,
			fresh_normal,
			fresh_other,
			fresh_other,
		)
		expected_episodes = jnp.asarray(
			[
				10, 20, 5, 21, 5, 7, 8, MAX_REQUEST_EPISODE, 9, 10, 11, -1,
				MAX_REQUEST_EPISODE - 1,
			],
			dtype=jnp.int32,
		)
		batch = COMPILED_RESET_BATCH(_stack_sessions(sessions), jax.tree_util.tree_map(
			lambda *leaves: jnp.stack(leaves), *replacements
		), expected_episodes)
		self.assertEqual(
			prior._as_list(batch.status),
			[
				SESSION_RESET,
				SESSION_RESET,
				SESSION_NOT_TERMINAL,
				SESSION_UNSUPPORTED,
				SESSION_STALE_EPISODE,
				SESSION_UNSUPPORTED,
				SESSION_UNSUPPORTED,
				SESSION_OVERFLOW,
				SESSION_UNSUPPORTED,
				SESSION_UNSUPPORTED,
				SESSION_UNSUPPORTED,
				SESSION_UNSUPPORTED,
				SESSION_RESET,
			],
			"compiled reset status precedence",
		)

		for lane, expected_battle, expected_episode in (
			(0, fresh_normal, 11),
			(1, fresh_depleted, 21),
			(12, fresh_other, MAX_REQUEST_EPISODE),
		):
			result = _lane(batch, lane)
			self.assertTrue(prior._leaf_equal(result.session.battle, expected_battle), (lane, "accepted replacement exact"))
			self.assertEqual(prior._as_list(result.session.pending_choices), [0, 0], (lane, "accepted clears choices"))
			self.assertEqual(int(result.session.episode), expected_episode, (lane, "accepted episode value"))
		for lane in range(2, 12):
			result = _lane(batch, lane)
			self.assertTrue(prior._leaf_equal(result.session, sessions[lane]), (lane, "rejection preserves full session"))

		for lane, scenario in ((0, normal_scenario), (1, depleted_scenario)):
			accepted = _lane(batch, lane).session
			for audience, side in ((0, "p1"), (1, "p2")):
				view = session_tests.COMPILED_REQUEST_VIEWS[audience](accepted, audience)
				self._assert_request(view, scenario["initial"]["requests"][side], (lane, side))

		with self.assertRaises(ValueError):
			reset_two_move_request_session(sessions[0], fresh_normal, np.asarray([10], dtype=np.int32))
		with self.assertRaises(TypeError):
			reset_two_move_request_session(sessions[0], fresh_normal, np.asarray(10.0, dtype=np.float32))
		with self.assertRaises(ValueError):
			reset_two_move_request_session(
				sessions[0]._replace(pending_choices=jnp.asarray([0], dtype=jnp.int32)), fresh_normal, 10
			)
		with self.assertRaises(TypeError):
			reset_two_move_request_session(
				sessions[0]._replace(episode=np.asarray(10, dtype=np.int64)), fresh_normal, 10
			)
		with self.assertRaises(TypeError):
			reset_two_move_request_session(
				sessions[0], fresh_normal._replace(move_ids=np.asarray([33, 1], dtype=np.int64)), 10
			)


if __name__ == "__main__":
	unittest.main()
