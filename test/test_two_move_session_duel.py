"""Complete native battle parity through the staged v4 request-session API."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp

import test.test_neutral_two_move as prior
from simulator import (
	SESSION_RESOLVED,
	SESSION_TERMINAL,
	SESSION_WAITING,
	STATUS_OK,
	TwoMoveRequestSession,
	make_two_move_request_session,
	project_two_move_request_view,
	project_two_move_submission_view,
	submit_two_move_choice,
)
from simulator.neutral_tackle import PHASE_TERMINAL
from simulator.neutral_two_move import INVALID_MOVE_ID


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "reference" / "neutral-two-move-oracle.cjs"
MOVE_ID_BY_NAME = {"tackle": prior.MOVE_TACKLE, "pound": prior.MOVE_POUND}

COMPILED_SUBMIT = jax.jit(submit_two_move_choice)
COMPILED_REQUEST_VIEW = jax.jit(project_two_move_request_view, static_argnums=1)
COMPILED_SUBMISSION_VIEW = jax.jit(project_two_move_submission_view, static_argnums=1)


class TwoMoveSessionDuelTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		completed = subprocess.run(
			["node", str(ORACLE), "--session-duels"],
			cwd=ROOT,
			check=True,
			capture_output=True,
			text=True,
			timeout=20,
		)
		cls.fixture = json.loads(completed.stdout)
		cls.scenarios = {scenario["id"]: scenario for scenario in cls.fixture["scenarios"]}

	def _empty_transition(self, result, label):
		self.assertEqual(int(result.transition.draws.count), 0, (label, "draw count"))
		self.assertFalse(any(prior._as_list(result.transition.events.executed)), (label, "events"))
		self.assertEqual(prior._as_list(result.transition.event_move_ids), [INVALID_MOVE_ID, INVALID_MOVE_ID],
			(label, "move identities"))

	def _assert_draws(self, result, expected, label):
		trace = result.transition.draws
		self.assertEqual(int(trace.count), len(expected), (label, "draw count"))
		self.assertLessEqual(len(expected), 16, (label, "draw capacity"))
		move_count = sum(event["type"] == "move" for event in label["turn"]["events"]["p1"])
		self.assertEqual(
			len([draw for draw in expected if draw["context"] != "speed-sort"]),
			move_count * 3,
			(label["scenario"], label["index"], "ordinary draw accounting"),
		)
		for index, draw in enumerate(expected):
			phase = draw.get("phase", draw["context"])
			bound = 2 if draw["context"] == "speed-sort" else draw["bound"]
			self.assertEqual(int(trace.bounds[index]), bound, (label["scenario"], label["index"], index, "bound"))
			self.assertEqual(int(trace.values[index]), draw["value"], (label["scenario"], label["index"], index, "value"))
			self.assertEqual(int(trace.contexts[index]), prior.DRAW_CONTEXTS[phase],
				(label["scenario"], label["index"], index, "context"))
			self.assertEqual(prior._as_list(trace.before[index]), draw["before"],
				(label["scenario"], label["index"], index, "pre-draw RNG limbs"))
			self.assertEqual(prior._as_list(trace.after[index]), draw["after"],
				(label["scenario"], label["index"], index, "post-draw RNG limbs"))
			if index:
				self.assertEqual(draw["before"], expected[index - 1]["after"],
					(label["scenario"], label["index"], index, "native draw chain"))
		if expected:
			self.assertEqual(expected[0]["before"], label["turn"]["before"]["rng"],
				(label["scenario"], label["index"], "first draw starts at prior RNG"))
			self.assertEqual(expected[-1]["after"], label["turn"]["after"]["rng"],
				(label["scenario"], label["index"], "last draw ends at resolved RNG"))

	def _assert_request(self, view, expected_request, scenario_id, turn_index, side):
		projection = prior._jax_view_projection(view.player_view)
		self.assertEqual(projection["request"], expected_request,
			(scenario_id, turn_index, side, "current request"))
		self.assertEqual(projection["events"], [], (scenario_id, turn_index, side, "current request has no events"))
		legal = [not move["disabled"] for move in expected_request["own"]["moves"]]
		self.assertEqual(prior._as_list(view.player_view.legal_move_mask), legal,
			(scenario_id, turn_index, side, "current legal mask"))

	def _assert_turn(self, session, turn, scenario, resolved_count):
		scenario_id = scenario["id"]
		index = turn["index"]
		self.assertEqual(int(session.episode), resolved_count, (scenario_id, index, "incoming local episode"))
		prior._assert_state(self, session.battle, turn["before"], (scenario_id, index, "incoming native state"))
		self.assertEqual(prior._as_list(session.pending_choices), [0, 0],
			(scenario_id, index, "pending choices cleared before turn"))

		current_views = [COMPILED_REQUEST_VIEW(session, audience) for audience in (0, 1)]
		for audience, side in ((0, "p1"), (1, "p2")):
			self.assertEqual(int(current_views[audience].episode), resolved_count,
				(scenario_id, index, side, "current episode"))
			self.assertFalse(bool(current_views[audience].own_choice_submitted),
				(scenario_id, index, side, "no pending own choice"))
			self._assert_request(current_views[audience], turn["requests"][side], scenario_id, index, side)

		first_player = 0 if scenario["firstPlayer"] == "p1" else 1
		second_player = 1 - first_player
		first_choice = turn["choices"][first_player]
		first = COMPILED_SUBMIT(session, first_player, first_choice, resolved_count)
		label = {"scenario": scenario_id, "index": index, "turn": turn}
		self.assertEqual(int(first.status), SESSION_WAITING, (scenario_id, index, "first commitment"))
		self.assertTrue(prior._leaf_equal(first.session.battle, session.battle),
			(scenario_id, index, "first commitment preserves battle"))
		self.assertEqual(prior._as_list(first.session.pending_choices),
			[turn["choices"][0] if first_player == 0 else 0,
			 turn["choices"][1] if first_player == 1 else 0],
			(scenario_id, index, "one private pending choice"))
		self.assertEqual(int(first.session.episode), resolved_count, (scenario_id, index, "first commit keeps episode"))
		self.assertTrue(prior._leaf_equal(first.before, session.battle), (scenario_id, index, "first before state"))
		self.assertTrue(prior._leaf_equal(first.transition.state, session.battle),
			(scenario_id, index, "first transition state unchanged"))
		self._empty_transition(first, (scenario_id, index, "first commitment"))

		native_first = turn["firstCommit"]
		self.assertTrue(native_first["accepted"], (scenario_id, index, "native first accepted"))
		self.assertEqual(native_first["draws"], [], (scenario_id, index, "native first draws"))
		self.assertEqual(native_first["before"], native_first["after"],
			(scenario_id, index, "native first preserves full battle/PP/RNG"))
		self.assertEqual(native_first["requests"], native_first["requestsAfter"],
			(scenario_id, index, "native first preserves both requests"))
		self.assertEqual(native_first["events"], {"p1": [], "p2": []},
			(scenario_id, index, "native first events"))
		for audience in (0, 1):
			after_first = COMPILED_REQUEST_VIEW(first.session, audience)
			self.assertEqual(bool(after_first.own_choice_submitted), audience == first_player,
				(scenario_id, index, audience, "private own submitted bit"))
			self._assert_request(after_first, turn["requests"]["p1" if audience == 0 else "p2"],
				scenario_id, index, f"after first p{audience + 1}")
			if audience != first_player:
				self.assertTrue(prior._leaf_equal(current_views[audience], after_first),
					(scenario_id, index, audience, "full opponent view invariance"))

		second_choice = turn["choices"][second_player]
		result = COMPILED_SUBMIT(first.session, second_player, second_choice, resolved_count)
		self.assertEqual(int(result.status), SESSION_RESOLVED, (scenario_id, index, "second commitment"))
		self.assertEqual(int(result.transition.status), STATUS_OK, (scenario_id, index, "v4 transition"))
		self.assertTrue(prior._leaf_equal(result.before, session.battle), (scenario_id, index, "resolved before state"))
		prior._assert_state(self, result.session.battle, turn["after"], (scenario_id, index, "resolved state"))
		self.assertEqual(prior._as_list(result.session.pending_choices), [0, 0],
			(scenario_id, index, "clear both pending choices"))
		self.assertEqual(int(result.session.episode), resolved_count + 1,
			(scenario_id, index, "advance local episode exactly once"))
		self._assert_draws(result, turn["secondCommit"]["draws"], label)

		expected_actors = [0 if actor == "p1" else 1 for actor in turn["actionOrder"]]
		actual_actors = [int(actor) for actor in prior._as_list(result.transition.events.actors) if actor >= 0]
		self.assertEqual(actual_actors, expected_actors, (scenario_id, index, "native public move order"))
		self.assertEqual(prior._as_list(result.transition.events.executed),
			[index < len(expected_actors) for index in range(2)], (scenario_id, index, "executed rows"))
		move_events = [event for event in turn["events"]["p1"] if event["type"] == "move"]
		for event in move_events:
			actor = 0 if event["actor"] == "p1" else 1
			self.assertEqual(event["move"], "tackle" if turn["choices"][actor] == 1 else "pound",
				(scenario_id, index, event["actor"], "native selected move identity"))
		expected_ids = [MOVE_ID_BY_NAME[event["move"]] for event in move_events]
		self.assertEqual(prior._as_list(result.transition.event_move_ids),
			expected_ids + [INVALID_MOVE_ID] * (2 - len(expected_ids)),
			(scenario_id, index, "public move IDs"))

		for audience, side in ((0, "p1"), (1, "p2")):
			view = COMPILED_SUBMISSION_VIEW(result, audience)
			self.assertEqual(int(view.episode), resolved_count + 1, (scenario_id, index, side, "submission episode"))
			self.assertFalse(bool(view.own_choice_submitted), (scenario_id, index, side, "pending cleared in view"))
			self.assertEqual(prior._jax_view_projection(view.resolved_events)["events"], turn["events"][side],
				(scenario_id, index, side, "prior resolved recipient events"))
			next_request = turn["nextRequests"][side]
			if next_request is None:
				self.assertTrue(bool(view.player_view.terminal), (scenario_id, index, side, "terminal request"))
				self.assertFalse(any(prior._as_list(view.player_view.legal_move_mask)),
					(scenario_id, index, side, "terminal legal mask"))
				winner_side = "p1" if turn["after"]["winner"] == 1 else "p2"
				self.assertEqual(prior._jax_view_projection(view.player_view)["events"],
					[{"type": "winner", "side": winner_side}],
					(scenario_id, index, side, "current terminal outcome"))
			else:
				self.assertEqual(prior._jax_view_projection(view.player_view)["events"], [],
					(scenario_id, index, side, "current request excludes prior events"))
				self._assert_request(view, next_request, scenario_id, index, side)
				fresh = COMPILED_REQUEST_VIEW(result.session, audience)
				self.assertTrue(prior._leaf_equal(fresh.player_view, view.player_view),
					(scenario_id, index, side, "current request equals fresh projection"))
			self.assertTrue(prior._leaf_equal(result.before, session.battle),
				(scenario_id, index, side, "submission preserves its before-state evidence"))
		return result.session, result

	def test_complete_native_battles_match_every_staged_session_turn(self):
		self.assertEqual(self.fixture["kind"], "gen5-restricted-neutral-two-move-complete-session-duel-oracle")
		self.assertEqual(self.fixture["variant"], prior.VARIANT_NAME)
		self.assertEqual(self.fixture["upstream"]["commit"], "1242c1e2d9763ba5b85d33f13aecb89bae19ae3d")
		self.assertEqual(set(self.scenarios), {
			f"{base}-first-{first}"
			for base in ("complete-p1-faster", "complete-p2-faster", "complete-tied-speed", "complete-last-tackle-pp")
			for first in ("p1", "p2")
		})
		self.assertEqual(len(self.scenarios), 8)
		asymmetric_choice_seen = False

		for scenario_id, scenario in self.scenarios.items():
			self.assertTrue(scenario["terminal"]["ended"], (scenario_id, "native terminal"))
			self.assertIn(scenario["terminal"]["winner"], (1, 2), (scenario_id, "native winner"))
			self.assertLessEqual(len(scenario["turns"]), 56, scenario_id)
			self.assertEqual(len(scenario["turns"]), scenario["terminal"]["state"]["turn"],
				(scenario_id, "native turn progression"))
			self.assertTrue(all(any(pp > 0 for pp in slots) for slots in scenario["terminal"]["state"]["pp"]),
				(scenario_id, "both actors retain a legal move slot"))
			if scenario["baseId"] == "complete-p1-faster":
				self.assertGreater(scenario["initial"]["state"]["speed"][0], scenario["initial"]["state"]["speed"][1])
			elif scenario["baseId"] == "complete-p2-faster":
				self.assertGreater(scenario["initial"]["state"]["speed"][1], scenario["initial"]["state"]["speed"][0])
			elif scenario["baseId"] == "complete-tied-speed":
				self.assertEqual(scenario["initial"]["state"]["speed"][0], scenario["initial"]["state"]["speed"][1])
				self.assertTrue(any(draw["context"] == "speed-sort"
					for turn in scenario["turns"] for draw in turn["secondCommit"]["draws"]))

			state = prior._state(scenario["initial"]["state"])
			session = make_two_move_request_session(state)
			self.assertIsInstance(session, TwoMoveRequestSession)
			for audience, side in ((0, "p1"), (1, "p2")):
				view = COMPILED_REQUEST_VIEW(session, audience)
				self.assertEqual(int(view.episode), 0, (scenario_id, side, "fresh episode"))
				self.assertFalse(bool(view.own_choice_submitted), (scenario_id, side, "fresh own bit"))
				self._assert_request(view, scenario["initial"]["requests"][side], scenario_id, 0, side)

			for resolved_count, turn in enumerate(scenario["turns"]):
				self.assertEqual(turn["index"], resolved_count, (scenario_id, "native turn index"))
				self.assertEqual(turn["plannedChoices"], [1 + (resolved_count % 2)] * 2,
					(scenario_id, resolved_count, "alternating planned choices"))
				asymmetric_choice_seen |= turn["choices"][0] != turn["choices"][1]
				session, _ = self._assert_turn(session, turn, scenario, resolved_count)

			self.assertEqual(int(session.episode), len(scenario["turns"]), (scenario_id, "terminal episode"))
			self.assertEqual(int(session.battle.turn), len(scenario["turns"]), (scenario_id, "terminal native turn"))
			prior._assert_state(self, session.battle, scenario["terminal"]["state"], (scenario_id, "terminal session state"))
			self.assertTrue(bool(session.battle.terminal), (scenario_id, "genuine terminal state"))
			self.assertEqual(int(session.battle.request_phase), PHASE_TERMINAL, (scenario_id, "terminal phase"))
			self.assertEqual(int(session.battle.winner), scenario["terminal"]["winner"], (scenario_id, "terminal winner"))
			self.assertEqual(prior._as_list(session.pending_choices), [0, 0], (scenario_id, "terminal pending slots"))
			for audience in (0, 1):
				terminal_view = COMPILED_REQUEST_VIEW(session, audience)
				self.assertFalse(any(prior._as_list(terminal_view.player_view.legal_move_mask)),
					(scenario_id, audience, "terminal all-false legal mask"))
				self.assertFalse(bool(terminal_view.own_choice_submitted),
					(scenario_id, audience, "terminal own bit"))

			terminal_submit = COMPILED_SUBMIT(session, 0, 1, len(scenario["turns"]))
			self.assertEqual(int(terminal_submit.status), SESSION_TERMINAL, (scenario_id, "terminal absorption status"))
			self.assertTrue(prior._leaf_equal(terminal_submit.session, session), (scenario_id, "terminal session unchanged"))
			self.assertTrue(prior._leaf_equal(terminal_submit.before, session.battle), (scenario_id, "terminal before unchanged"))
			self.assertTrue(prior._leaf_equal(terminal_submit.transition.state, session.battle),
				(scenario_id, "terminal transition state unchanged"))
			self._empty_transition(terminal_submit, (scenario_id, "terminal absorbing submission"))

			move_sequences = {"p1": [], "p2": []}
			for turn in scenario["turns"]:
				for event in turn["events"]["p1"]:
					if event["type"] == "move":
						move_sequences[event["actor"]].append(event["move"])
			for player, sequence in move_sequences.items():
				self.assertIn("tackle", sequence, (scenario_id, player, "Tackle executed"))
				self.assertGreater(sequence.index("pound"), sequence.index("tackle"),
					(scenario_id, player, "Pound executed after Tackle"))

			if scenario["baseId"] == "complete-last-tackle-pp":
				self.assertEqual(scenario["setupEvidence"]["engineApi"], "battle.makeRequest('move')")
				self.assertEqual(scenario["setupEvidence"]["rngBefore"], scenario["setupEvidence"]["rngAfter"])
				self.assertEqual(scenario["initial"]["state"]["pp"][0][0], 1)
				self.assertEqual(scenario["turns"][0]["after"]["pp"][0][0], 0)
				self.assertTrue(scenario["turns"][0]["nextRequests"]["p1"]["own"]["moves"][0]["disabled"])
				self.assertFalse(scenario["turns"][0]["nextRequests"]["p1"]["own"]["moves"][1]["disabled"])
				self.assertEqual(scenario["turns"][1]["choices"][0], 2)
				self.assertEqual(scenario["turns"][1]["events"]["p1"][0]["actor"], "p1")
				self.assertIn("pound", [event["move"] for event in scenario["turns"][1]["events"]["p1"]
					if event["type"] == "move" and event["actor"] == "p1"])

		for base in ("complete-p1-faster", "complete-p2-faster", "complete-tied-speed", "complete-last-tackle-pp"):
			p1_first = self.scenarios[f"{base}-first-p1"]
			p2_first = self.scenarios[f"{base}-first-p2"]
			self.assertEqual(p1_first["initial"], p2_first["initial"], (base, "paired initial state"))
			self.assertEqual(p1_first["terminal"], p2_first["terminal"], (base, "paired terminal state"))
			self.assertEqual(len(p1_first["turns"]), len(p2_first["turns"]), (base, "paired turn count"))
			for index, (p1_turn, p2_turn) in enumerate(zip(p1_first["turns"], p2_first["turns"])):
				for key in ("before", "requests", "plannedChoices", "choices", "commands", "firstCommit",
						"secondCommit", "actionOrder", "events", "after", "nextRequests"):
					self.assertEqual(p1_turn[key], p2_turn[key], (base, index, key, "paired semantic parity"))
		self.assertTrue(asymmetric_choice_seen, "last-Tackle-PP fallback must exercise asymmetric choices")


if __name__ == "__main__":
	unittest.main()
