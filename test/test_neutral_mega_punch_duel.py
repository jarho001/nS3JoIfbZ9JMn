from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from test import test_neutral_mega_punch as mega_helpers
from simulator.neutral_mega_punch import project_mega_punch_view, transition_mega_punch
from simulator.neutral_tackle import STATUS_OK, STATUS_TERMINAL


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "reference" / "neutral-mega-punch-oracle.cjs"
COMPILED_TRANSITION = jax.jit(transition_mega_punch)


class NeutralMegaPunchDuelTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		completed = subprocess.run(
			["node", str(ORACLE), "--duels"],
			cwd=ROOT,
			check=True,
			capture_output=True,
			text=True,
			timeout=20,
		)
		cls.fixture = json.loads(completed.stdout)

	def test_complete_native_battles_thread_full_state_to_terminal(self):
		fixture = self.fixture
		self.assertEqual(fixture["kind"], "gen5-restricted-neutral-mega-punch-complete-duel-oracle")
		self.assertEqual(fixture["upstream"]["commit"], "1242c1e2d9763ba5b85d33f13aecb89bae19ae3d")
		self.assertLessEqual(fixture["turnCap"], 32)
		self.assertGreaterEqual(len(fixture["duels"]), 4)
		self.assertLessEqual(len(fixture["duels"]), 6)
		coverage = fixture["coverage"]
		self.assertGreater(coverage["initiativeFirst"]["p1"], 0)
		self.assertGreater(coverage["initiativeFirst"]["p2"], 0)
		self.assertGreater(coverage["missByActor"]["p1"], 0)
		self.assertGreater(coverage["missByActor"]["p2"], 0)
		self.assertGreater(coverage["missBySecondActingSlot"], 0)
		self.assertGreater(coverage["normalHits"], 0)
		self.assertGreater(coverage["critical"], 0)
		self.assertGreater(coverage["firstActionKoSkips"], 0)
		self.assertGreater(coverage["secondActionTerminal"], 0)

		observed = {
			"battles": len(fixture["duels"]),
			"turns": 0,
			"attempts": 0,
			"hits": 0,
			"misses": 0,
			"critical": 0,
			"firstActionKoSkips": 0,
			"secondActionTerminal": 0,
			"missByActor": {"p1": 0, "p2": 0},
			"missBySecondActingSlot": 0,
			"normalHits": 0,
			"battlesWithMisses": 0,
			"battlesWithRepeatedMisses": 0,
			"doubleMissTurns": 0,
			"initiativeFirst": {"p1": 0, "p2": 0},
		}
		for duel in fixture["duels"]:
			state = mega_helpers._make_state(duel["initial"])
			self.assertEqual(duel["initial"]["pp"], [32, 32], duel["id"])
			self.assertNotEqual(*duel["initial"]["speed"], duel["id"])
			self.assertTrue(duel["terminal"], duel["id"])
			self.assertIn(duel["winner"], ("p1", "p2"), duel["id"])
			self.assertLessEqual(len(duel["turns"]), fixture["turnCap"], duel["id"])
			previous_rng = duel["initial"]["rng"]
			misses_in_battle = 0
			first_action = None
			for turn_index, turn in enumerate(duel["turns"]):
				label = f"{duel['id']} turn {turn_index + 1}"
				observed["turns"] += 1
				self.assertEqual(turn["before"]["rng"], previous_rng, label)
				self.assertNotEqual(turn["after"]["rng"], turn["before"]["rng"], label)
				self.assertGreater(len(turn["draws"]), 0, label)
				self.assertEqual(turn["choices"], {"p1": "move 1", "p2": "move 1"}, label)
				self.assertEqual(turn["before"]["requestPhase"], "move", label)
				self.assertEqual(turn["requests"]["p1"]["boundary"], "mega-punch", label)
				self.assertEqual(turn["requests"]["p2"]["boundary"], "mega-punch", label)
				self.assertEqual(turn["requests"]["p1"]["legalActions"], ["move 1"], label)
				self.assertEqual(turn["requests"]["p2"]["legalActions"], ["move 1"], label)
				mega_helpers._assert_tree_equal(
					self,
					state,
					mega_helpers._make_state(turn["before"]),
					f"{label} threaded native current state",
				)
				actual = COMPILED_TRANSITION(state, jnp.asarray([1, 1], dtype=jnp.int32))
				expected = mega_helpers._native_expected({"turn": turn})
				mega_helpers._assert_tree_equal(self, actual, expected, f"{label} complete transition")

				moves = turn["events"]["p1"]
				move_events = [event for event in moves if event["type"] == "move"]
				miss_events = [event for event in moves if event["type"] == "miss"]
				critical_events = [event for event in moves if event["type"] == "critical"]
				observed["attempts"] += len(move_events)
				observed["misses"] += len(miss_events)
				observed["hits"] += len(move_events) - len(miss_events)
				observed["critical"] += len(critical_events)
				observed["normalHits"] += len(move_events) - len(miss_events) - len(critical_events)
				misses_in_battle += len(miss_events)
				if len(miss_events) == 2:
					observed["doubleMissTurns"] += 1
				for event in miss_events:
					observed["missByActor"][event["actor"]] += 1
					if turn["actionOrder"].index(event["actor"]) == 1:
						observed["missBySecondActingSlot"] += 1
				if turn_index == 0:
					first_action = turn["actionOrder"][0]
					observed["initiativeFirst"][first_action] += 1
				if turn["after"]["terminal"]:
					if len(turn["actionOrder"]) == 1:
						observed["firstActionKoSkips"] += 1
					if len(turn["actionOrder"]) == 2:
						observed["secondActionTerminal"] += 1
				if miss_events:
					missed_slots = {turn["actionOrder"].index(event["actor"]) for event in miss_events}
					for audience in (0, 1):
						wrapped = project_mega_punch_view(state, actual, audience)
						view = wrapped.player_view
						for slot in missed_slots:
							self.assertTrue(bool(np.asarray(wrapped.missed[slot])), label)
							self.assertFalse(bool(np.asarray(view.critical[slot])), label)
							self.assertEqual(int(view.damage_value[slot]), -1, label)
							self.assertFalse(bool(np.asarray(view.damage_visible[slot])), label)
							self.assertEqual(int(view.target_hp_value[slot]), -1, label)
							self.assertFalse(bool(np.asarray(view.target_hp_is_percent[slot])), label)
							self.assertFalse(bool(np.asarray(view.fainted[slot])), label)

				for audience, side in ((0, "p1"), (1, "p2")):
					projected = mega_helpers._projection_to_native_json(state, actual, audience)
					self.assertEqual(projected["request"], turn["requests"][side], f"{label} {side} current request")
					self.assertEqual(projected, turn["views"][side], f"{label} {side} selected native events")

				previous_rng = turn["after"]["rng"]
				if turn["after"]["terminal"]:
					self.assertEqual(turn["after"], duel["final"], f"{label} native final snapshot")
					self.assertEqual(turn["after"]["requestPhase"], "terminal", label)
					self.assertEqual(turn["after"]["requestsPresent"], [False, False], label)
					self.assertEqual(turn["nextRequests"], {"p1": None, "p2": None}, label)
					final_state = mega_helpers._make_state(duel["final"], phase=mega_helpers._phase_for_native_after(duel["final"]))
					mega_helpers._assert_tree_equal(self, actual.state, final_state, f"{duel['id']} terminal native snapshot")
					for audience in (0, 1):
						terminal_view = project_mega_punch_view(
							actual.state,
							mega_helpers._empty_expected(actual.state, STATUS_TERMINAL),
							audience,
						).player_view
						self.assertFalse(np.asarray(terminal_view.legal_move_mask).any(), f"{duel['id']} terminal legal actions")
					post_terminal = COMPILED_TRANSITION(actual.state, jnp.asarray([-1, 2], dtype=jnp.int32))
					mega_helpers._assert_tree_equal(
						self,
						post_terminal,
						mega_helpers._empty_expected(actual.state, STATUS_TERMINAL),
						f"{duel['id']} rejected post-terminal submission",
					)
					state = actual.state
					break

				self.assertFalse(turn["after"]["terminal"], label)
				self.assertTrue(all(pp > 0 for pp in turn["after"]["pp"]), label)
				self.assertEqual(turn["after"]["requestPhase"], "move", label)
				self.assertEqual(turn["after"]["requestsPresent"], [True, True], label)
				for audience, side in ((0, "p1"), (1, "p2")):
					next_request = mega_helpers._projection_to_native_json(
						actual.state,
						mega_helpers._empty_expected(actual.state, STATUS_OK),
						audience,
					)["request"]
					self.assertEqual(next_request, turn["nextRequests"][side], f"{label} {side} actual next request")
				state = actual.state

			self.assertIsNotNone(first_action, duel["id"])
			self.assertTrue(duel["final"]["terminal"], duel["id"])
			self.assertEqual(duel["final"]["winner"], 1 if duel["winner"] == "p1" else 2, duel["id"])
			if misses_in_battle > 0:
				observed["battlesWithMisses"] += 1
			if misses_in_battle > 1:
				observed["battlesWithRepeatedMisses"] += 1

		self.assertEqual(coverage, observed)


if __name__ == "__main__":
	unittest.main()
