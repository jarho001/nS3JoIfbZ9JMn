"""Full-battle differential for private request-session submissions."""

from __future__ import annotations

import json
import pathlib
import subprocess
import unittest

import jax.numpy as jnp

from simulator import (
	ACTION_TACKLE,
	PHASE_MOVE_REQUEST,
	PHASE_TERMINAL,
	SESSION_RESOLVED,
	SESSION_TERMINAL,
	SESSION_WAITING,
	STATUS_PENDING,
	STATUS_TERMINAL,
	make_request_session,
	project_request_view,
)
from test.test_request_session import (
	_COMPILED_SUBMIT,
	_COMPILED_VIEW,
	DRAW_CONTEXTS,
	_as_list,
	_leaf_equal,
	_state,
	_view_projection,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
ORACLE = ROOT / "reference" / "neutral-turn-oracle.cjs"


class RequestSessionDuelTests(unittest.TestCase):
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

	def _assert_snapshot(self, state, snapshot, label):
		for name, field in (
			("hp", "hp"),
			("max_hp", "maxHp"),
			("pp", "pp"),
			("level", "level"),
			("attack", "attack"),
			("defense", "defense"),
			("speed", "speed"),
			("rng", "rng"),
		):
			self.assertEqual(_as_list(getattr(state, name)), snapshot[field], (*label, field))
		self.assertEqual(int(state.turn), snapshot["turn"], (*label, "turn"))
		self.assertEqual(
			int(state.request_phase),
			PHASE_TERMINAL if snapshot["requestPhase"] == "terminal" else PHASE_MOVE_REQUEST,
			(*label, "request phase"),
		)
		self.assertEqual(bool(state.terminal), snapshot["terminal"], (*label, "terminal"))
		self.assertEqual(int(state.winner), snapshot["winner"], (*label, "winner"))

	def _assert_draws(self, transition, expected, label):
		self.assertEqual(int(transition.draws.count), len(expected), (*label, "draw count"))
		for index, draw in enumerate(expected):
			trace = transition.draws
			self.assertEqual(
				int(trace.bounds[index]), 2 if draw["context"] == "speed-sort" else draw["bound"],
				(*label, index, "bound"),
			)
			self.assertEqual(int(trace.values[index]), draw["value"], (*label, index, "value"))
			self.assertEqual(int(trace.contexts[index]), DRAW_CONTEXTS[draw.get("phase", draw["context"])], (*label, index, "context"))
			self.assertEqual(_as_list(trace.before[index]), draw["before"], (*label, index, "before RNG"))
			self.assertEqual(_as_list(trace.after[index]), draw["after"], (*label, index, "after RNG"))
			if index:
				self.assertEqual(draw["before"], expected[index - 1]["after"], (*label, index, "draw chain"))

	def _assert_no_private_trace(self, view, label):
		self.assertNotIn("pending_choices", view._fields, label)
		self.assertNotIn("opponent_choice_submitted", view._fields, label)
		self.assertNotIn("rng", view.player_view._fields, label)
		self.assertNotIn("draws", view.player_view._fields, label)
		self.assertNotIn("pending_choices", view.resolved_events._fields, label)
		self.assertNotIn("rng", view.resolved_events._fields, label)
		self.assertNotIn("draws", view.resolved_events._fields, label)

	def test_compiled_private_submissions_match_native_duels_through_terminal(self):
		self.assertEqual(self.fixture["kind"], "gen5-restricted-neutral-tackle-staged-session-full-duel-oracle")
		self.assertIn("each player commits separately through Battle.choose", self.fixture["initialBoundary"])
		self.assertEqual(
			set(self.scenarios),
			{"tied-first-p1", "tied-first-p2", "unequal-first-p1", "unequal-first-p2"},
		)

		for speed_control in ("tied", "unequal"):
			first_p1 = self.scenarios[f"{speed_control}-first-p1"]
			first_p2 = self.scenarios[f"{speed_control}-first-p2"]
			self.assertEqual(first_p1["initial"], first_p2["initial"], (speed_control, "same initialized native battle"))
			self.assertEqual(first_p1["firstPlayer"], "p1")
			self.assertEqual(first_p2["firstPlayer"], "p2")

		paired_results = {}
		for scenario in self.fixture["scenarios"]:
			label = scenario["id"]
			initial = scenario["initial"]
			session = make_request_session(_state(initial["state"]))
			self._assert_snapshot(session.battle, initial["state"], (label, "initial"))
			self.assertEqual(_as_list(session.pending_choices), [0, 0], (label, "initial commitments"))
			self.assertEqual(int(session.episode), 0, (label, "initial episode"))

			current_actor_views = {}
			for audience, side in ((0, "p1"), (1, "p2")):
				view = project_request_view(session, audience)
				self.assertEqual(_view_projection(view.player_view), {"request": initial["requests"][side], "events": []}, (label, side, "initial request"))
				self.assertEqual(int(view.episode), 0)
				self.assertFalse(bool(view.own_choice_submitted))
				self.assertNotIn("rng", view.player_view._fields)
				self.assertNotIn("pending_choices", view.player_view._fields)
				current_actor_views[side] = view.player_view

			turn_views = []
			self.assertLessEqual(len(scenario["turns"]), self.fixture["turnLimit"], (label, "bounded native duel"))
			for turn_index, turn in enumerate(scenario["turns"]):
				turn_label = (label, turn_index)
				self.assertFalse(bool(session.battle.terminal), (*turn_label, "active before request"))
				self._assert_snapshot(session.battle, turn["before"], (*turn_label, "before"))
				self.assertEqual(_as_list(session.pending_choices), [0, 0], (*turn_label, "no stale commitments"))
				self.assertEqual(int(session.episode), turn_index, (*turn_label, "local episode"))

				first_player = 0 if turn["firstPlayer"] == "p1" else 1
				second_player = 1 - first_player
				first = _COMPILED_SUBMIT(
					session,
					jnp.asarray(first_player, jnp.int32),
					jnp.asarray(ACTION_TACKLE, jnp.int32),
					session.episode,
				)
				self.assertEqual(int(first.status), SESSION_WAITING, (*turn_label, "first submission status"))
				self.assertEqual(int(first.transition.status), STATUS_PENDING, (*turn_label, "pending transition status"))
				self.assertEqual(_as_list(first.session.pending_choices), [int(index == first_player) for index in range(2)], (*turn_label, "private commit pair"))
				self.assertTrue(_leaf_equal(first.session.battle, session.battle), (*turn_label, "partial battle freeze"))
				self.assertEqual(int(first.transition.draws.count), 0, (*turn_label, "partial RNG freeze"))
				self.assertEqual(_as_list(first.transition.events.executed), [False, False], (*turn_label, "partial events"))
				self.assertTrue(turn["firstCommit"]["accepted"], (*turn_label, "native partial accepted"))
				self.assertEqual(turn["firstCommit"]["draws"], [], (*turn_label, "native partial draws"))
				self.assertEqual(turn["firstCommit"]["requests"], turn["requests"], (*turn_label, "native requests frozen"))
				self.assertEqual(turn["firstCommit"]["privateChoicesHeld"], 1, (*turn_label, "native private choice"))
				self._assert_snapshot(first.session.battle, turn["firstCommit"]["state"], (*turn_label, "native partial state"))

				partial_views = {}
				for audience, side in ((0, "p1"), (1, "p2")):
					view = _COMPILED_VIEW(first, audience)
					self._assert_no_private_trace(view, (*turn_label, side, "partial view"))
					self.assertEqual(_view_projection(view.player_view), {"request": turn["requests"][side], "events": []}, (*turn_label, side, "request while one commit held"))
					self.assertEqual(_view_projection(view.resolved_events)["events"], [], (*turn_label, side, "no partial resolved events"))
					self.assertEqual(int(view.episode), turn_index, (*turn_label, side, "partial episode"))
					self.assertEqual(bool(view.own_choice_submitted), audience == first_player, (*turn_label, side, "own commit bit"))
					self.assertTrue(_leaf_equal(view.player_view, current_actor_views[side]), (*turn_label, side, "opponent commit does not change actor view"))
					partial_views[side] = view.player_view

				resolved = _COMPILED_SUBMIT(
					first.session,
					jnp.asarray(second_player, jnp.int32),
					jnp.asarray(ACTION_TACKLE, jnp.int32),
					first.session.episode,
				)
				self.assertEqual(int(resolved.status), SESSION_RESOLVED, (*turn_label, "second submission status"))
				self.assertEqual(int(resolved.transition.status), 0, (*turn_label, "resolved transition status"))
				self.assertEqual(_as_list(resolved.session.pending_choices), [0, 0], (*turn_label, "cleared commitments"))
				self.assertEqual(int(resolved.session.episode), turn_index + 1, (*turn_label, "next episode"))
				self.assertTrue(_leaf_equal(resolved.session.battle, resolved.transition.state), (*turn_label, "committed battle"))
				self.assertEqual(turn["totalDrawCount"], len(turn["draws"]), (*turn_label, "native total draws"))
				self.assertEqual(turn["pendingChoiceCount"], 0, (*turn_label, "native cleared choices"))
				self.assertEqual(int(resolved.transition.draws.count), turn["totalDrawCount"], (*turn_label, "native draw count"))
				self._assert_snapshot(resolved.transition.state, turn["after"], (*turn_label, "after"))
				self._assert_draws(resolved.transition, turn["draws"], turn_label)
				actors = [int(actor) for actor in _as_list(resolved.transition.events.actors) if actor >= 0]
				self.assertEqual(["p1" if actor == 0 else "p2" for actor in actors], turn["actionOrder"], (*turn_label, "native action order"))
				executed = _as_list(resolved.transition.events.executed)
				self.assertEqual(executed, [index < len(turn["actionOrder"]) for index in range(2)], (*turn_label, "executed action mask"))

				resolved_views = {}
				for audience, side in ((0, "p1"), (1, "p2")):
					view = _COMPILED_VIEW(resolved, audience)
					self._assert_no_private_trace(view, (*turn_label, side, "resolved view"))
					self.assertEqual(_view_projection(view.resolved_events), turn["views"][side], (*turn_label, side, "recipient events"))
					self.assertFalse(bool(view.own_choice_submitted), (*turn_label, side, "commit cleared from actor view"))
					resolved_views[side] = view.resolved_events
					if turn["nextRequests"][side] is None:
						self.assertTrue(bool(view.player_view.terminal), (*turn_label, side, "terminal request"))
						self.assertEqual(int(view.player_view.request_phase), PHASE_TERMINAL, (*turn_label, side, "terminal phase"))
						self.assertEqual(_as_list(view.player_view.legal_move_mask), [False], (*turn_label, side, "terminal no-action mask"))
						self.assertEqual(int(view.player_view.winner), int(turn["after"]["winner"]), (*turn_label, side, "terminal winner projection"))
					else:
						self.assertEqual(_view_projection(view.player_view), {"request": turn["nextRequests"][side], "events": []}, (*turn_label, side, "next native request"))
						self.assertEqual(int(view.player_view.request_phase), PHASE_MOVE_REQUEST, (*turn_label, side, "next request phase"))
						current_actor_views[side] = view.player_view

				turn_views.append({
					"partial": partial_views,
					"resolved": resolved_views,
					"native": {key: turn[key] for key in ("before", "requests", "firstCommit", "draws", "actionOrder", "events", "after", "nextRequests", "views")},
				})
				session = resolved.session

			outcome = scenario["outcome"]
			self.assertTrue(outcome["terminal"], (label, "native terminal"))
			self.assertIn(outcome["winner"], ("p1", "p2"), (label, "native winner"))
			self.assertEqual(outcome["pendingChoiceCount"], 0, (label, "terminal choices cleared"))
			self._assert_snapshot(session.battle, outcome["state"], (label, "terminal outcome"))
			self.assertTrue(bool(session.battle.terminal), (label, "JAX terminal"))
			self.assertEqual(int(session.battle.winner), outcome["sideWinner"], (label, "JAX terminal winner"))
			self.assertEqual(_as_list(session.pending_choices), [0, 0], (label, "terminal commitments cleared"))
			self.assertEqual(int(session.episode), len(scenario["turns"]), (label, "terminal episode"))

			terminal_rejection = _COMPILED_SUBMIT(
				session,
				jnp.asarray(0, jnp.int32),
				jnp.asarray(ACTION_TACKLE, jnp.int32),
				session.episode,
			)
			self.assertEqual(int(terminal_rejection.status), SESSION_TERMINAL, (label, "terminal submit status"))
			self.assertEqual(int(terminal_rejection.transition.status), STATUS_TERMINAL, (label, "terminal transition status"))
			self.assertTrue(_leaf_equal(terminal_rejection.session, session), (label, "terminal submission preserves session"))
			self.assertEqual(int(terminal_rejection.transition.draws.count), 0, (label, "terminal submit has no RNG draws"))
			self.assertEqual(_as_list(terminal_rejection.transition.events.executed), [False, False], (label, "terminal submit has no actions"))
			self._assert_snapshot(terminal_rejection.session.battle, outcome["state"], (label, "terminal submission state"))

			paired_results[label] = {
				"speed_control": "tied" if label.startswith("tied") else "unequal",
				"first_player": scenario["firstPlayer"],
				"initial": initial,
				"turns": turn_views,
				"outcome": outcome,
				"session": session,
			}

		for speed_control in ("tied", "unequal"):
			p1 = paired_results[f"{speed_control}-first-p1"]
			p2 = paired_results[f"{speed_control}-first-p2"]
			self.assertEqual(p1["initial"], p2["initial"], (speed_control, "paired initial"))
			self.assertEqual(p1["outcome"], p2["outcome"], (speed_control, "paired native terminal"))
			self.assertTrue(_leaf_equal(p1["session"], p2["session"]), (speed_control, "submission order invariant final session"))
			self.assertEqual(len(p1["turns"]), len(p2["turns"]), (speed_control, "paired turn count"))
			for index, (first_turn, second_turn) in enumerate(zip(p1["turns"], p2["turns"])):
				self.assertTrue(_leaf_equal(first_turn["partial"]["p1"], second_turn["partial"]["p1"]), (speed_control, index, "p1 view invariant to first submitter"))
				self.assertTrue(_leaf_equal(first_turn["partial"]["p2"], second_turn["partial"]["p2"]), (speed_control, index, "p2 view invariant to first submitter"))
				self.assertTrue(_leaf_equal(first_turn["resolved"]["p1"], second_turn["resolved"]["p1"]), (speed_control, index, "p1 events invariant to first submitter"))
				self.assertTrue(_leaf_equal(first_turn["resolved"]["p2"], second_turn["resolved"]["p2"]), (speed_control, index, "p2 events invariant to first submitter"))
				self.assertEqual(first_turn["native"]["before"], second_turn["native"]["before"], (speed_control, index, "paired before"))
				self.assertEqual(first_turn["native"]["draws"], second_turn["native"]["draws"], (speed_control, index, "paired draws"))
				self.assertEqual(first_turn["native"]["events"], second_turn["native"]["events"], (speed_control, index, "paired events"))
				self.assertEqual(first_turn["native"]["after"], second_turn["native"]["after"], (speed_control, index, "paired after"))


if __name__ == "__main__":
	unittest.main()
