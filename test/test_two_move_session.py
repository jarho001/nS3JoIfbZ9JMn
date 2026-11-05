"""Focused native checks for private staged v4 Tackle/Pound requests."""

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
from simulator import (
	MAX_REQUEST_EPISODE,
	SESSION_DUPLICATE,
	SESSION_INVALID_ACTION,
	SESSION_INVALID_PLAYER,
	SESSION_OVERFLOW,
	SESSION_RESOLVED,
	SESSION_STALE_EPISODE,
	SESSION_TERMINAL,
	SESSION_UNSUPPORTED,
	SESSION_WAITING,
	STATUS_OK,
	STATUS_TERMINAL,
	TwoMoveRequestSession,
	TwoMoveRequestView,
	TwoMoveSessionSubmission,
	TwoMoveSubmissionView,
	make_two_move_request_session,
	project_two_move_request_view,
	project_two_move_submission_view,
	submit_two_move_choice,
)
from simulator.neutral_tackle import PHASE_MOVE_REQUEST, PHASE_UNSUPPORTED


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "reference" / "neutral-two-move-oracle.cjs"
MOVE_ID_BY_NAME = {"tackle": prior.MOVE_TACKLE, "pound": prior.MOVE_POUND}
COMPILED_SUBMIT = jax.jit(submit_two_move_choice)
COMPILED_REQUEST_VIEWS = tuple(
	jax.jit(project_two_move_request_view, static_argnums=1) for _ in (0, 1)
)
COMPILED_SUBMISSION_VIEWS = tuple(
	jax.jit(project_two_move_submission_view, static_argnums=1) for _ in (0, 1)
)


def _leaf(value):
	return np.asarray(value).tolist()


def _session_stack(sessions):
	return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *sessions)


class TwoMoveSessionTests(unittest.TestCase):
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

	def _empty_trace(self, result, label):
		self.assertEqual(int(result.transition.draws.count), 0, (label, "draw count"))
		self.assertFalse(any(prior._as_list(result.transition.events.executed)), (label, "executed events"))
		self.assertEqual(prior._as_list(result.transition.event_move_ids), [-1, -1], (label, "event move IDs"))

	def _assert_draws(self, result, expected, label):
		trace = result.transition.draws
		self.assertEqual(int(trace.count), len(expected), (label, "draw count"))
		for index, draw in enumerate(expected):
			phase = draw.get("phase", draw["context"])
			bound = 2 if draw["context"] == "speed-sort" else draw["bound"]
			self.assertEqual(int(trace.bounds[index]), bound, (label, index, "bound"))
			self.assertEqual(int(trace.values[index]), draw["value"], (label, index, "value"))
			self.assertEqual(int(trace.contexts[index]), prior.DRAW_CONTEXTS[phase], (label, index, "phase"))
			self.assertEqual(prior._as_list(trace.before[index]), draw["before"], (label, index, "before RNG"))
			self.assertEqual(prior._as_list(trace.after[index]), draw["after"], (label, index, "after RNG"))
			if index:
				self.assertEqual(draw["before"], expected[index - 1]["after"], (label, index, "draw chain"))

	def _assert_rejected(self, session, player, action, episode, expected_status, label):
		result = COMPILED_SUBMIT(session, jnp.asarray(player, jnp.int32), jnp.asarray(action, jnp.int32), jnp.asarray(episode, jnp.int32))
		self.assertEqual(int(result.status), expected_status, (label, "status"))
		self.assertTrue(prior._leaf_equal(result.session, session), (label, "input session preserved"))
		self.assertTrue(prior._leaf_equal(result.before, session.battle), (label, "before state"))
		self.assertTrue(prior._leaf_equal(result.transition.state, session.battle), (label, "transition state"))
		self._empty_trace(result, label)
		return result

	def test_staged_native_turns_privacy_rejections_and_batch(self):
		self.assertEqual(self.fixture["kind"], "gen5-restricted-neutral-two-move-staged-session-oracle")
		self.assertEqual(self.fixture["variant"], prior.VARIANT_NAME)
		self.assertEqual(self.fixture["upstream"]["commit"], "1242c1e2d9763ba5b85d33f13aecb89bae19ae3d")
		self.assertEqual(len(self.scenarios), 10)
		for typename in (TwoMoveRequestSession, TwoMoveSessionSubmission, TwoMoveRequestView, TwoMoveSubmissionView):
			self.assertTrue(get_type_hints(typename), typename.__name__)
		for function in (
			make_two_move_request_session,
			submit_two_move_choice,
			project_two_move_request_view,
			project_two_move_submission_view,
		):
			self.assertTrue(get_type_hints(function), function.__name__)

		for scenario_id, scenario in self.scenarios.items():
			turn = scenario["turn"]
			state = prior._state(scenario["initial"]["state"])
			session = make_two_move_request_session(state)
			for audience, side in ((0, "p1"), (1, "p2")):
				view = COMPILED_REQUEST_VIEWS[audience](session, audience)
				self.assertEqual(int(view.episode), 0, (scenario_id, side, "initial episode"))
				self.assertFalse(bool(view.own_choice_submitted), (scenario_id, side, "initial own commit bit"))
				self.assertEqual(
					prior._jax_view_projection(view.player_view)["request"],
					scenario["initial"]["requests"][side],
					(scenario_id, side, "fresh request parity"),
				)
				self.assertEqual(set(view._fields), {"player_view", "episode", "own_choice_submitted"})
				for forbidden in ("pending_choices", "opponent_choice", "battle", "rng"):
					self.assertNotIn(forbidden, view._fields)

			first_player = 0 if turn["firstPlayer"] == "p1" else 1
			second_player = 1 - first_player
			first_choice = turn["choices"][first_player]
			first = COMPILED_SUBMIT(session, first_player, first_choice, 0)
			self.assertEqual(int(first.status), SESSION_WAITING, (scenario_id, "first commit"))
			self.assertTrue(prior._leaf_equal(first.session.battle, state), (scenario_id, "first commit battle freeze"))
			self.assertEqual(prior._as_list(first.session.pending_choices), [first_choice if first_player == 0 else 0, first_choice if first_player == 1 else 0])
			self.assertEqual(int(first.session.episode), 0, (scenario_id, "first commit episode"))
			self._empty_trace(first, (scenario_id, "first commit"))
			self.assertEqual(prior._as_list(first.transition.state.pp), scenario["initial"]["state"]["pp"])
			first_commit = turn["firstCommit"]
			self.assertTrue(first_commit["accepted"])
			self.assertEqual(first_commit["draws"], [])
			self.assertEqual(first_commit["before"], first_commit["after"])
			self.assertEqual(first_commit["requests"], first_commit["requestsAfter"])
			self.assertEqual(first_commit["events"], {"p1": [], "p2": []})

			for audience in (0, 1):
				before_view = COMPILED_REQUEST_VIEWS[audience](session, audience)
				after_view = COMPILED_REQUEST_VIEWS[audience](first.session, audience)
				self.assertTrue(bool(after_view.own_choice_submitted) == (audience == first_player), (scenario_id, audience, "private own bit"))
				if audience != first_player:
					self.assertTrue(prior._leaf_equal(before_view, after_view), (scenario_id, audience, "opponent view unchanged"))

			second_choice = turn["choices"][second_player]
			result = COMPILED_SUBMIT(first.session, second_player, second_choice, 0)
			self.assertEqual(int(result.status), SESSION_RESOLVED, (scenario_id, "second commit"))
			self.assertEqual(int(result.transition.status), STATUS_OK, (scenario_id, "v4 transition"))
			self.assertEqual(prior._as_list(result.session.pending_choices), [0, 0], (scenario_id, "clear both choices"))
			self.assertEqual(int(result.session.episode), 1, (scenario_id, "advance episode once"))
			prior._assert_state(self, result.session.battle, turn["after"], (scenario_id, "resolved battle"))
			self._assert_draws(result, turn["secondCommit"]["draws"], scenario_id)
			expected_actors = [0 if actor == "p1" else 1 for actor in turn["actionOrder"]]
			actual_actors = [int(actor) for actor in prior._as_list(result.transition.events.actors) if actor >= 0]
			self.assertEqual(actual_actors, expected_actors, (scenario_id, "action order"))
			move_events = [event for event in turn["events"]["p1"] if event["type"] == "move"]
			expected_move_ids = [MOVE_ID_BY_NAME[event["move"]] for event in move_events]
			self.assertEqual(prior._as_list(result.transition.event_move_ids), expected_move_ids + [-1] * (2 - len(expected_move_ids)))

			for audience, side in ((0, "p1"), (1, "p2")):
				view = COMPILED_SUBMISSION_VIEWS[audience](result, audience)
				self.assertEqual(int(view.episode), 1, (scenario_id, side, "submission view episode"))
				self.assertFalse(bool(view.own_choice_submitted), (scenario_id, side, "cleared own bit"))
				self.assertEqual(
					prior._jax_view_projection(view.resolved_events)["events"],
					turn["events"][side],
					(scenario_id, side, "prior resolved recipient events"),
				)
				if turn["nextRequests"][side] is not None:
					self.assertEqual(
						prior._jax_view_projection(view.player_view)["request"],
						turn["nextRequests"][side],
						(scenario_id, side, "new current request"),
					)
				else:
					self.assertFalse(any(prior._as_list(view.player_view.legal_move_mask)), (scenario_id, side, "terminal request mask"))
			self.assertTrue(prior._leaf_equal(result.before, state), (scenario_id, "submission before state"))

		# The opponent's whole view is invariant to either legal p1 choice.
		privacy_state = prior._state(self.scenarios["tackle-tackle-unequal-first-p1"]["initial"]["state"])
		privacy_session = make_two_move_request_session(privacy_state)
		p1_tackle = COMPILED_SUBMIT(privacy_session, 0, 1, 0)
		p1_pound = COMPILED_SUBMIT(privacy_session, 0, 2, 0)
		p2_initial = COMPILED_REQUEST_VIEWS[1](privacy_session, 1)
		p2_after_tackle = COMPILED_REQUEST_VIEWS[1](p1_tackle.session, 1)
		p2_after_pound = COMPILED_REQUEST_VIEWS[1](p1_pound.session, 1)
		self.assertTrue(prior._leaf_equal(p2_initial, p2_after_tackle))
		self.assertTrue(prior._leaf_equal(p2_after_tackle, p2_after_pound))
		self.assertTrue(bool(COMPILED_REQUEST_VIEWS[0](p1_tackle.session, 0).own_choice_submitted))
		self.assertTrue(bool(COMPILED_REQUEST_VIEWS[0](p1_pound.session, 0).own_choice_submitted))

		base_state = privacy_state
		base = make_two_move_request_session(base_state)
		self._assert_rejected(base, 8, 0, 0, SESSION_INVALID_PLAYER, "invalid player before action")
		self._assert_rejected(base, 0, 0, 0, SESSION_INVALID_ACTION, "zero action")
		self._assert_rejected(base, 0, -1, 0, SESSION_INVALID_ACTION, "negative action")
		self._assert_rejected(base, 0, 3, 0, SESSION_INVALID_ACTION, "out-of-range action")
		self._assert_rejected(base, 8, 0, 1, SESSION_STALE_EPISODE, "stale episode before invalid player and action")
		self._assert_rejected(base, 8, 0, 0, SESSION_INVALID_PLAYER, "invalid player")

		bad_pending = base._replace(pending_choices=jnp.asarray([3, 0], dtype=jnp.int32))
		self._assert_rejected(bad_pending, 0, 1, 0, SESSION_UNSUPPORTED, "unsupported pending value")
		bad_phase = base._replace(battle=base_state._replace(request_phase=jnp.asarray(PHASE_UNSUPPORTED, jnp.int32)))
		self._assert_rejected(bad_phase, 0, 1, 0, SESSION_UNSUPPORTED, "unsupported battle")
		negative_episode = base._replace(episode=jnp.asarray(-1, dtype=jnp.int32))
		self._assert_rejected(negative_episode, 0, 1, -1, SESSION_UNSUPPORTED, "negative episode metadata")

		held_zero_pp_state = base_state._replace(pp=base_state.pp.at[0, 0].set(0))
		held_zero_pp = make_two_move_request_session(held_zero_pp_state)._replace(
			pending_choices=jnp.asarray([1, 0], dtype=jnp.int32)
		)
		self._assert_rejected(held_zero_pp, 0, 2, 0, SESSION_UNSUPPORTED, "held action lost PP")

		overflow = base._replace(
			episode=jnp.asarray(MAX_REQUEST_EPISODE, dtype=jnp.int32),
			pending_choices=jnp.asarray([1, 0], dtype=jnp.int32),
		)
		self._assert_rejected(overflow, 0, 1, MAX_REQUEST_EPISODE, SESSION_OVERFLOW, "overflow before duplicate")
		self._assert_rejected(
			base._replace(pending_choices=jnp.asarray([1, 0], dtype=jnp.int32)),
			0,
			2,
			0,
			SESSION_DUPLICATE,
			"duplicate before selected PP",
		)
		unavailable_state = base_state._replace(pp=base_state.pp.at[0, 0].set(0))
		unavailable = make_two_move_request_session(unavailable_state)
		self._assert_rejected(unavailable, 0, 1, 0, SESSION_INVALID_ACTION, "unavailable selected Tackle")

		terminal_state = prior._state(self.scenarios["first-action-ko"]["turn"]["after"])
		terminal_state = terminal_state._replace(pp=jnp.zeros((2, 2), dtype=jnp.int32))
		terminal = make_two_move_request_session(terminal_state)._replace(
			pending_choices=jnp.asarray([3, 0], dtype=jnp.int32)
		)
		terminal_result = self._assert_rejected(
			terminal, 0, 1, 0, SESSION_TERMINAL, "terminal before unsupported metadata and PP"
		)
		self.assertEqual(int(terminal_result.transition.status), STATUS_TERMINAL, "valid terminal request status")
		self._assert_rejected(terminal, 0, 0, 0, SESSION_INVALID_ACTION, "invalid action before terminal")

		with self.assertRaises((TypeError, ValueError)):
			make_two_move_request_session(base_state, episode=True)
		with self.assertRaises(ValueError):
			make_two_move_request_session(base_state, episode=MAX_REQUEST_EPISODE + 1)
		with self.assertRaises((TypeError, ValueError)):
			submit_two_move_choice(base, np.asarray([0, 1], dtype=np.int32), 1, 0)
		with self.assertRaises(TypeError):
			submit_two_move_choice(base, 0, 1, np.asarray(0.0, dtype=np.float32))
		with self.assertRaises(TypeError):
			project_two_move_request_view(
				base._replace(pending_choices=np.asarray([0, 0], dtype=np.int64)), 0
			)

		batch_resolve_scenario = self.scenarios["tackle-pound-unequal-first-p1"]
		batch_resolve_state = prior._state(batch_resolve_scenario["initial"]["state"])
		batch_resolve_session = make_two_move_request_session(batch_resolve_state)._replace(
			pending_choices=jnp.asarray([1, 0], dtype=jnp.int32)
		)
		batch_sessions = (
			base,
			batch_resolve_session,
			base,
			base._replace(pending_choices=jnp.asarray([1, 0], jnp.int32)),
			unavailable,
			make_two_move_request_session(terminal_state),
		)
		batch_players = jnp.asarray([0, 1, 8, 0, 0, 0], dtype=jnp.int32)
		batch_actions = jnp.asarray([1, 2, 0, 2, 1, 1], dtype=jnp.int32)
		batch_episodes = jnp.asarray([0, 0, 1, 0, 0, 0], dtype=jnp.int32)
		batch = jax.jit(jax.vmap(submit_two_move_choice))(
			_session_stack(batch_sessions), batch_players, batch_actions, batch_episodes
		)
		self.assertEqual(
			prior._as_list(batch.status),
			[SESSION_WAITING, SESSION_RESOLVED, SESSION_STALE_EPISODE, SESSION_DUPLICATE, SESSION_INVALID_ACTION, SESSION_TERMINAL],
			"heterogeneous jit(vmap(submit)) statuses",
		)
		self.assertEqual(prior._as_list(batch.session.episode), [0, 1, 0, 0, 0, 0])
		self.assertEqual(prior._as_list(batch.session.pending_choices[1]), [0, 0])
		waiting = jax.tree_util.tree_map(lambda value: value[0], batch)
		self.assertTrue(prior._leaf_equal(waiting.session.battle, base.battle), "vmap waiting lane full battle freeze")
		self.assertTrue(np.array_equal(np.asarray(waiting.session.battle.rng), np.asarray(base.battle.rng)), "vmap waiting lane RNG freeze")
		self.assertEqual(prior._as_list(waiting.session.pending_choices), [1, 0])
		self.assertEqual(int(waiting.session.episode), 0)
		self._empty_trace(waiting, "vmap waiting lane")

		resolved = jax.tree_util.tree_map(lambda value: value[1], batch)
		self.assertTrue(prior._leaf_equal(resolved.before, batch_resolve_state), "vmap resolved lane before state")
		prior._assert_state(
			self,
			resolved.session.battle,
			batch_resolve_scenario["turn"]["after"],
			"vmap resolved lane native battle",
		)
		self.assertEqual(int(resolved.transition.status), STATUS_OK, "vmap resolved lane transition status")
		self._assert_draws(resolved, batch_resolve_scenario["turn"]["secondCommit"]["draws"], "vmap resolved lane")
		self.assertEqual(prior._as_list(resolved.session.pending_choices), [0, 0], "vmap resolved lane clear choices")
		self.assertEqual(int(resolved.session.episode), 1, "vmap resolved lane episode")
		for lane in (2, 3, 4, 5):
			lane_result = jax.tree_util.tree_map(lambda value: value[lane], batch)
			self.assertTrue(prior._leaf_equal(lane_result.session, batch_sessions[lane]), ("vmap rejected lane", lane))
			self.assertTrue(prior._leaf_equal(lane_result.before, batch_sessions[lane].battle), ("vmap rejected before", lane))
			self.assertTrue(prior._leaf_equal(lane_result.transition.state, batch_sessions[lane].battle), ("vmap rejected transition state", lane))
			self._empty_trace(lane_result, ("vmap rejected lane", lane))
