"""Complete-battle native differential checks for the v4 two-move Mew API."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import test.test_neutral_two_move as prior
from simulator.neutral_tackle import (
	PHASE_TERMINAL,
	STATUS_OK,
	STATUS_TERMINAL,
)
from simulator.neutral_two_move import (
	INVALID_MOVE_ID,
	MOVE_POUND,
	MOVE_TACKLE,
	project_two_move_view,
	transition_two_move,
)


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "reference" / "neutral-two-move-oracle.cjs"
MOVE_ID_BY_NAME = {"tackle": MOVE_TACKLE, "pound": MOVE_POUND}


def _step_with_views(state, choices):
	result = transition_two_move(state, choices)
	idle_choices = jnp.zeros((2,), dtype=jnp.int32)
	before_idle = transition_two_move(state, idle_choices)
	after_idle = transition_two_move(result.state, idle_choices)
	return (
		result,
		project_two_move_view(state, before_idle, 0),
		project_two_move_view(state, before_idle, 1),
		project_two_move_view(state, result, 0),
		project_two_move_view(state, result, 1),
		project_two_move_view(result.state, after_idle, 0),
		project_two_move_view(result.state, after_idle, 1),
	)


class NeutralTwoMoveDuelTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		completed = subprocess.run(
			["node", str(ORACLE), "--duels"],
			cwd=ROOT,
			check=True,
			capture_output=True,
			text=True,
		)
		cls.fixture = json.loads(completed.stdout)
		cls.scenarios = {scenario["id"]: scenario for scenario in cls.fixture["scenarios"]}

	def _assert_draws(self, result, turn, scenario_id):
		expected_draws = turn["draws"]
		self.assertEqual(int(result.draws.count), len(expected_draws), (scenario_id, turn["index"], "draw count"))
		self.assertLessEqual(len(expected_draws), 16, (scenario_id, turn["index"], "draw capacity"))
		move_count = sum(event["type"] == "move" for event in turn["events"]["p1"])
		self.assertEqual(
			len([draw for draw in expected_draws if draw["context"] != "speed-sort"]),
			move_count * 3,
			(scenario_id, turn["index"], "ordinary draw accounting"),
		)
		for index, expected in enumerate(expected_draws):
			phase = expected.get("phase", expected["context"])
			bound = 2 if expected["context"] == "speed-sort" else expected["bound"]
			self.assertEqual(int(result.draws.bounds[index]), bound, (scenario_id, turn["index"], index, "bound"))
			self.assertEqual(int(result.draws.values[index]), expected["value"], (scenario_id, turn["index"], index, "value"))
			self.assertEqual(
				int(result.draws.contexts[index]), prior.DRAW_CONTEXTS[phase],
				(scenario_id, turn["index"], index, "context"),
			)
			self.assertEqual(prior._as_list(result.draws.before[index]), expected["before"],
				(scenario_id, turn["index"], index, "pre-draw limbs"))
			self.assertEqual(prior._as_list(result.draws.after[index]), expected["after"],
				(scenario_id, turn["index"], index, "post-draw limbs"))
			if index:
				self.assertEqual(expected["before"], expected_draws[index - 1]["after"],
					(scenario_id, turn["index"], index, "native draw chain"))
		if expected_draws:
			self.assertEqual(expected_draws[0]["before"], turn["before"]["rng"],
				(scenario_id, turn["index"], "first draw starts at state RNG"))
			self.assertEqual(expected_draws[-1]["after"], turn["after"]["rng"],
				(scenario_id, turn["index"], "last draw ends at state RNG"))

	def _assert_views(self, result, views, turn, scenario_id):
		before_views = views[:2]
		turn_views = views[2:4]
		next_views = views[4:6]
		for audience, side in ((0, "p1"), (1, "p2")):
			before_projection = prior._jax_view_projection(before_views[audience])
			self.assertEqual(before_projection["request"], turn["requests"][side],
				(scenario_id, turn["index"], side, "current native request"))
			self.assertEqual(before_projection["events"], [],
				(scenario_id, turn["index"], side, "current request has no new events"))
			legal = [not move["disabled"] for move in turn["requests"][side]["own"]["moves"]]
			self.assertEqual(prior._as_list(before_views[audience].legal_move_mask), legal,
				(scenario_id, turn["index"], side, "current legal mask"))
			self.assertEqual(
				prior._jax_view_projection(turn_views[audience])["events"], turn["events"][side],
				(scenario_id, turn["index"], side, "recipient native events"),
			)
			next_request = turn["nextRequests"][side]
			if next_request is None:
				self.assertFalse(any(prior._as_list(next_views[audience].legal_move_mask)),
					(scenario_id, turn["index"], side, "terminal next mask"))
			else:
				next_projection = prior._jax_view_projection(next_views[audience])
				self.assertEqual(next_projection["request"], next_request,
					(scenario_id, turn["index"], side, "next native request"))
				next_legal = [not move["disabled"] for move in next_request["own"]["moves"]]
				self.assertEqual(prior._as_list(next_views[audience].legal_move_mask), next_legal,
					(scenario_id, turn["index"], side, "next legal mask"))

	def _assert_turn(self, result, views, turn, scenario_id):
		self.assertEqual(int(result.status), STATUS_OK, (scenario_id, turn["index"], "status"))
		prior._assert_state(self, result.state, turn["after"], (scenario_id, turn["index"], "after"))
		self.assertEqual(result.draws.bounds.shape, (16,), (scenario_id, turn["index"], "draw bounds shape"))
		self.assertEqual(result.draws.values.shape, (16,), (scenario_id, turn["index"], "draw values shape"))
		self.assertEqual(result.draws.before.shape, (16, 4), (scenario_id, turn["index"], "draw limbs shape"))
		self._assert_draws(result, turn, scenario_id)

		expected_actors = [0 if actor == "p1" else 1 for actor in turn["actionOrder"]]
		actual_actors = [int(actor) for actor in prior._as_list(result.events.actors) if actor >= 0]
		self.assertEqual(actual_actors, expected_actors, (scenario_id, turn["index"], "action order"))
		self.assertEqual(prior._as_list(result.events.executed),
			[index < len(expected_actors) for index in range(2)],
			(scenario_id, turn["index"], "executed action rows"))
		move_events = [event for event in turn["events"]["p1"] if event["type"] == "move"]
		expected_ids = [MOVE_ID_BY_NAME[event["move"]] for event in move_events]
		self.assertEqual(prior._as_list(result.event_move_ids), expected_ids + [INVALID_MOVE_ID] * (2 - len(expected_ids)),
			(scenario_id, turn["index"], "executed move identities"))
		self.assertEqual(result.event_move_ids.dtype, jnp.int32)

		before_pp = np.asarray(turn["before"]["pp"], dtype=np.int32)
		expected_pp = before_pp.copy()
		for event in move_events:
			actor = 0 if event["actor"] == "p1" else 1
			choice = turn["choices"][actor]
			self.assertEqual(event["move"], "tackle" if choice == 1 else "pound",
				(scenario_id, turn["index"], event["actor"], "submitted move identity"))
			expected_pp[actor, choice - 1] -= 1
		self.assertEqual(prior._as_list(result.state.pp), expected_pp.tolist(),
			(scenario_id, turn["index"], "only executed selected PP is spent"))

		for actor, player in enumerate(("p1", "p2")):
			request = turn["requests"][player]
			planned = turn["plannedChoices"][actor]
			expected_choice = planned
			if request["own"]["moves"][planned - 1]["disabled"]:
				expected_choice = 3 - planned
			self.assertEqual(turn["choices"][actor], expected_choice,
				(scenario_id, turn["index"], player, "planned choice/fallback"))
			self.assertIn(turn["commands"][actor], request["legalActions"],
				(scenario_id, turn["index"], player, "submitted command is native-legal"))
		self._assert_views(result, views, turn, scenario_id)

	def test_complete_native_battles_match_every_jax_turn_and_terminal(self):
		self.assertEqual(self.fixture["kind"], "gen5-restricted-neutral-two-move-complete-duel-oracle")
		self.assertEqual(self.fixture["variant"], prior.VARIANT_NAME)
		self.assertEqual(self.fixture["upstream"]["commit"], "1242c1e2d9763ba5b85d33f13aecb89bae19ae3d")
		self.assertEqual(set(self.scenarios), {
			"complete-p1-faster", "complete-p2-faster", "complete-tied-speed", "complete-last-tackle-pp",
		})
		compiled_step = jax.jit(_step_with_views)
		idle_choices = jnp.zeros((2,), dtype=jnp.int32)

		for scenario_id, scenario in self.scenarios.items():
			self.assertTrue(scenario["terminal"]["ended"], scenario_id)
			self.assertIn(scenario["terminal"]["winner"], (1, 2), scenario_id)
			self.assertLessEqual(len(scenario["turns"]), 56, scenario_id)
			self.assertEqual(len(scenario["turns"]), scenario["terminal"]["state"]["turn"],
				(scenario_id, "native turn count consumed"))
			self.assertTrue(all(any(pp > 0 for pp in slots) for slots in scenario["terminal"]["state"]["pp"]),
				(scenario_id, "both sides retain a move slot"))
			if scenario_id == "complete-p1-faster":
				self.assertGreater(scenario["initial"]["state"]["speed"][0], scenario["initial"]["state"]["speed"][1])
			elif scenario_id == "complete-p2-faster":
				self.assertGreater(scenario["initial"]["state"]["speed"][1], scenario["initial"]["state"]["speed"][0])
			elif scenario_id == "complete-tied-speed":
				self.assertEqual(scenario["initial"]["state"]["speed"][0], scenario["initial"]["state"]["speed"][1])
				self.assertTrue(any(draw["context"] == "speed-sort" for turn in scenario["turns"] for draw in turn["draws"]))

			state = prior._state(scenario["initial"]["state"])
			prior._assert_state(self, state, scenario["initial"]["state"], (scenario_id, "native initial state"))
			initial_views = compiled_step(state, idle_choices)[1:3]
			for audience, side in ((0, "p1"), (1, "p2")):
				self.assertEqual(prior._jax_view_projection(initial_views[audience])["request"],
					scenario["initial"]["requests"][side], (scenario_id, side, "initial native request"))

			move_sequences = {"p1": [], "p2": []}
			for turn in scenario["turns"]:
				self.assertEqual(int(state.turn), turn["before"]["turn"], (scenario_id, turn["index"], "threaded turn"))
				prior._assert_state(self, state, turn["before"], (scenario_id, turn["index"], "threaded before state"))
				choices = jnp.asarray(turn["choices"], dtype=jnp.int32)
				result, *views = compiled_step(state, choices)
				self._assert_turn(result, views, turn, scenario_id)
				self.assertEqual(turn["winner"], int(result.state.winner),
					(scenario_id, turn["index"], "native winner"))
				for event in turn["events"]["p1"]:
					if event["type"] == "move":
						move_sequences[event["actor"]].append(event["move"])
				state = result.state
			self.assertEqual(len(scenario["turns"]), scenario["turns"][-1]["index"] + 1, (scenario_id, "all scripted turns consumed"))
			for player, sequence in move_sequences.items():
				self.assertIn("tackle", sequence, (scenario_id, player, "Tackle executed"))
				self.assertGreater(sequence.index("pound"), sequence.index("tackle"),
					(scenario_id, player, "Pound executed after Tackle"))
			prior._assert_state(self, state, scenario["terminal"]["state"], (scenario_id, "actual terminal state"))
			self.assertTrue(bool(state.terminal), (scenario_id, "genuine native terminal"))
			self.assertEqual(int(state.winner), scenario["terminal"]["winner"], (scenario_id, "native winner"))
			self.assertEqual(int(state.request_phase), PHASE_TERMINAL, (scenario_id, "terminal request phase"))

			terminal_noop, *terminal_views = compiled_step(state, jnp.asarray([1, 1], dtype=jnp.int32))
			self.assertEqual(int(terminal_noop.status), STATUS_TERMINAL, (scenario_id, "absorbing status"))
			self.assertTrue(prior._leaf_equal(terminal_noop.state, state), (scenario_id, "absorbing full state"))
			self.assertTrue(np.array_equal(np.asarray(terminal_noop.state.rng), np.asarray(state.rng)),
				(scenario_id, "absorbing RNG"))
			self.assertEqual(int(terminal_noop.draws.count), 0, (scenario_id, "absorbing draws"))
			self.assertFalse(any(prior._as_list(terminal_noop.events.executed)), (scenario_id, "absorbing events"))
			self.assertEqual(prior._as_list(terminal_noop.event_move_ids), [INVALID_MOVE_ID, INVALID_MOVE_ID],
				(scenario_id, "absorbing move IDs"))
			for audience in (0, 1):
				self.assertFalse(any(prior._as_list(terminal_views[audience].legal_move_mask)),
					(scenario_id, audience, "terminal legal mask"))

		boundary = self.scenarios["complete-last-tackle-pp"]
		self.assertEqual(boundary["setupEvidence"]["engineApi"], "battle.makeRequest('move')")
		self.assertEqual(boundary["setupEvidence"]["rngBefore"], boundary["setupEvidence"]["rngAfter"])
		self.assertEqual(boundary["initial"]["state"]["pp"][0][0], 1)
		self.assertEqual(boundary["turns"][0]["after"]["pp"][0][0], 0)
		self.assertTrue(boundary["turns"][0]["nextRequests"]["p1"]["own"]["moves"][0]["disabled"])
		self.assertFalse(boundary["turns"][0]["nextRequests"]["p1"]["own"]["moves"][1]["disabled"])
		self.assertEqual(boundary["turns"][1]["choices"][0], 2)
