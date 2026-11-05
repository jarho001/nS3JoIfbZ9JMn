from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import test.test_accuracy_choice as helpers
from simulator.accuracy_choice import transition_accuracy_choice
from simulator.neutral_tackle import STATUS_OK, STATUS_TERMINAL


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "reference" / "accuracy-choice-oracle.cjs"
COMPILED_TRANSITION = jax.jit(transition_accuracy_choice)


class AccuracyChoiceDuelTest(unittest.TestCase):
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

    def test_threaded_native_duels_reach_terminal_with_live_pp(self):
        fixture = self.fixture
        self.assertEqual(fixture["kind"], "gen5-restricted-accuracy-choice-duels-oracle")
        self.assertEqual(fixture["upstream"]["commit"], "1242c1e2d9763ba5b85d33f13aecb89bae19ae3d")
        self.assertEqual(fixture["variant"], helpers.VARIANT_NAME)
        self.assertEqual(fixture["turnCap"], 32)
        self.assertEqual(len(fixture["battles"]), 6)

        coverage = {
            "battles": 0,
            "turns": 0,
            "genuineTerminalOutcomes": 0,
            "terminalAfterSecondAction": 0,
            "firstActionKoSkips": 0,
            "submittedChoicePairs": {"1,1": 0, "1,2": 0, "2,1": 0, "2,2": 0},
            "selectedMoveAttempts": {"tackle": 0, "megaPunch": 0},
            "selectedMoveHits": {"tackle": 0, "megaPunch": 0},
            "hits": 0,
            "missesByActorAndSelectedMove": {
                "p1": {"tackle": 0, "megaPunch": 0},
                "p2": {"tackle": 0, "megaPunch": 0},
            },
            "criticalHits": 0,
            "criticalHitsBySelectedMove": {"tackle": 0, "megaPunch": 0},
            "doubleMissTurns": 0,
            "repeatedSameActorMissesAcrossConsecutiveTurns": 0,
            "depletionContinuations": {"megaPunchThenTackle": 0, "tackleThenMegaPunch": 0},
        }

        for battle in fixture["battles"]:
            with self.subTest(battle=battle["id"]):
                coverage["battles"] += 1
                self.assertFalse(battle["initial"]["terminal"])
                self.assertNotEqual(battle["initial"]["speed"][0], battle["initial"]["speed"][1])
                if battle["depletion"] is None:
                    self.assertEqual(battle["initial"]["pp"], [[56, 32], [56, 32]])
                else:
                    self.assertEqual(battle["initial"]["pp"][1], [56, 32])
                    if battle["depletion"] == "megaPunch":
                        self.assertEqual(battle["initial"]["pp"][0], [56, 1])
                    else:
                        self.assertEqual(battle["initial"]["pp"][0], [1, 32])
                    self.assertEqual(battle["initial"]["rng"], battle["turns"][0]["before"]["rng"])

                self.assertEqual(battle["initial"], battle["turns"][0]["before"])
                state = helpers._state(battle["initial"])
                actual_turns = []
                previous_turn = None
                previous_miss_actors = set()

                for turn_index, turn in enumerate(battle["turns"]):
                    label = f"{battle['id']} turn {turn_index + 1}"
                    if previous_turn is not None:
                        self.assertEqual(previous_turn["after"], turn["before"], f"{label} native snapshot chain")
                        self.assertEqual(previous_turn["nextRequests"], turn["requests"], f"{label} live request chain")
                        self.assertEqual(previous_turn["after"]["rng"], turn["before"]["rng"], f"{label} original PRNG chain")
                    self.assertFalse(turn["before"]["terminal"], label)
                    for side in ("p1", "p2"):
                        self.assertEqual(turn["requests"][side]["boundary"], "accuracy-choice", label)
                        self.assertIn(f"move {turn['choices'][side]}", turn["requests"][side]["legalActions"], label)

                    native_before = helpers._state(turn["before"])
                    helpers._assert_tree_equal(self, state, native_before, f"{label} threaded input equals native before")
                    choices = jnp.asarray([turn["choices"]["p1"], turn["choices"]["p2"]], dtype=jnp.int32)
                    _, expected = helpers._native_expected({"turn": turn, "choices": turn["choices"]})
                    actual = COMPILED_TRANSITION(state, choices)
                    helpers._assert_tree_equal(self, actual, expected, f"{label} complete native transition")
                    self.assertEqual(int(actual.status), STATUS_OK, label)
                    helpers.AccuracyChoiceTest._assert_recipients(
                        self, {"turn": turn, "choices": turn["choices"]}, state, actual
                    )
                    actual_turns.append(actual)

                    for side in ("p1", "p2"):
                        native_move_events = [event for event in turn["events"][side] if event["type"] == "move"]
                        native_move_ids = [move_id for event, move_id in zip(turn["events"][side], turn["eventMoveIds"][side]) if event["type"] == "move"]
                        self.assertEqual(
                            native_move_ids,
                            [33 if turn["choices"][event["actor"]] == 1 else 5 for event in native_move_events],
                            f"{label} {side} split recipient move IDs",
                        )

                    coverage["turns"] += 1
                    pair_key = f"{turn['choices']['p1']},{turn['choices']['p2']}"
                    coverage["submittedChoicePairs"][pair_key] += 1
                    before_pp = np.asarray(state.pp).tolist()
                    after_pp = np.asarray(actual.state.pp).tolist()
                    executed = np.asarray(actual.events.executed).tolist()
                    actors = np.asarray(actual.events.actors).tolist()
                    move_ids = np.asarray(actual.event_move_ids).tolist()
                    missed = np.asarray(actual.events.missed).tolist()
                    hit = np.asarray(actual.events.hit).tolist()
                    critical = np.asarray(actual.events.critical).tolist()

                    for row, did_execute in enumerate(executed):
                        if not did_execute:
                            self.assertEqual(move_ids[row], -1, f"{label} unused event slot")
                            continue
                        actor = actors[row]
                        actor_side = "p1" if actor == 0 else "p2"
                        action = turn["choices"][actor_side]
                        move_key = "tackle" if action == 1 else "megaPunch"
                        self.assertEqual(move_ids[row], 33 if action == 1 else 5, label)
                        coverage["selectedMoveAttempts"][move_key] += 1
                        self.assertEqual(
                            after_pp[actor][action - 1], before_pp[actor][action - 1] - 1,
                            f"{label} selected PP spent",
                        )
                        other_slot = 1 - (action - 1)
                        self.assertEqual(after_pp[actor][other_slot], before_pp[actor][other_slot], f"{label} unselected PP preserved")
                        coverage["hits"] += int(hit[row])
                        coverage["criticalHits"] += int(critical[row])
                        coverage["selectedMoveHits"][move_key] += int(hit[row])
                        coverage["criticalHitsBySelectedMove"][move_key] += int(critical[row])
                        if missed[row]:
                            coverage["missesByActorAndSelectedMove"][actor_side][move_key] += 1
                            self.assertEqual(
                                after_pp[actor][action - 1], before_pp[actor][action - 1] - 1,
                                f"{label} miss spends selected PP",
                            )

                    miss_actors = {
                        "p1" if actors[row] == 0 else "p2"
                        for row, did_execute in enumerate(executed)
                        if did_execute and missed[row]
                    }
                    if len(miss_actors) == 2:
                        coverage["doubleMissTurns"] += 1
                    coverage["repeatedSameActorMissesAcrossConsecutiveTurns"] += len(miss_actors & previous_miss_actors)
                    previous_miss_actors = miss_actors

                    if sum(executed) == 1:
                        coverage["firstActionKoSkips"] += 1
                        self.assertTrue(bool(actual.state.terminal), f"{label} single executed action is a native KO terminal")
                        self.assertEqual(executed, [True, False], label)
                        skipped_actor = 1 - actors[0]
                        self.assertEqual(after_pp[skipped_actor], before_pp[skipped_actor], f"{label} skipped opponent PP preserved")
                    if bool(actual.state.terminal) and sum(executed) == 2:
                        coverage["terminalAfterSecondAction"] += 1

                    state = actual.state
                    previous_turn = turn

                self.assertGreater(len(battle["turns"]), 0)
                self.assertLessEqual(len(battle["turns"]), 32)
                self.assertEqual(battle["finalState"], battle["turns"][-1]["after"])
                self.assertTrue(battle["finalState"]["terminal"])
                self.assertEqual(battle["finalState"]["requestPhase"], "terminal")
                self.assertEqual(battle["finalState"]["requestsPresent"], [False, False])
                self.assertEqual(battle["turns"][-1]["nextRequests"], {"p1": None, "p2": None})
                self.assertIn(battle["finalState"]["winner"], (1, 2))
                coverage["genuineTerminalOutcomes"] += 1

                terminal_rejection = COMPILED_TRANSITION(state, jnp.asarray([1, 1], dtype=jnp.int32))
                expected_terminal = helpers._empty_result(state, STATUS_TERMINAL)
                helpers._assert_tree_equal(self, terminal_rejection, expected_terminal, f"{battle['id']} post-terminal rejection")
                self.assertEqual(int(terminal_rejection.draws.count), 0, battle["id"])
                self.assertEqual(np.asarray(terminal_rejection.event_move_ids).tolist(), [-1, -1], battle["id"])
                for audience in (0, 1):
                    terminal_view = helpers.project_accuracy_choice_view(state, terminal_rejection, audience).player_view
                    self.assertEqual(np.asarray(terminal_view.legal_move_mask).tolist(), [False, False], battle["id"])
                    self.assertTrue(bool(terminal_view.terminal), battle["id"])
                    self.assertEqual(int(terminal_view.winner), battle["finalState"]["winner"], battle["id"])

                if battle["depletion"] == "megaPunch":
                    first, follow = battle["turns"][:2]
                    first_actual, follow_actual = actual_turns[:2]
                    self.assertEqual(first["choices"]["p1"], 2, battle["id"])
                    self.assertEqual(first["draws"][0]["value"], 93, battle["id"])
                    self.assertEqual(first["after"]["pp"][0][1], 0, battle["id"])
                    self.assertTrue(first["nextRequests"]["p1"]["moves"][1]["disabled"])
                    self.assertEqual(follow["requests"], first["nextRequests"], battle["id"])
                    self.assertEqual(follow["choices"]["p1"], 1, battle["id"])
                    p1_follow_ids = [
                        int(move_id)
                        for move_id, actor, did_execute in zip(
                            np.asarray(follow_actual.event_move_ids),
                            np.asarray(follow_actual.events.actors),
                            np.asarray(follow_actual.events.executed),
                        )
                        if int(actor) == 0 and bool(did_execute)
                    ]
                    self.assertIn(33, p1_follow_ids, battle["id"])
                    self.assertEqual(int(first_actual.state.pp[0, 1]), 0, battle["id"])
                    coverage["depletionContinuations"]["megaPunchThenTackle"] += 1
                elif battle["depletion"] == "tackle":
                    first, follow = battle["turns"][:2]
                    follow_actual = actual_turns[1]
                    self.assertEqual(first["choices"]["p1"], 1, battle["id"])
                    self.assertEqual(first["after"]["pp"][0][0], 0, battle["id"])
                    self.assertTrue(first["nextRequests"]["p1"]["moves"][0]["disabled"])
                    self.assertEqual(follow["requests"], first["nextRequests"], battle["id"])
                    self.assertEqual(follow["choices"]["p1"], 2, battle["id"])
                    p1_follow_ids = [
                        int(move_id)
                        for move_id, actor, did_execute in zip(
                            np.asarray(follow_actual.event_move_ids),
                            np.asarray(follow_actual.events.actors),
                            np.asarray(follow_actual.events.executed),
                        )
                        if int(actor) == 0 and bool(did_execute)
                    ]
                    self.assertIn(5, p1_follow_ids, battle["id"])
                    coverage["depletionContinuations"]["tackleThenMegaPunch"] += 1

        self.assertEqual(coverage["battles"], 6)
        self.assertGreaterEqual(coverage["genuineTerminalOutcomes"], 4)
        self.assertEqual(
            {key for key, count in coverage["submittedChoicePairs"].items() if count},
            {"1,1", "1,2", "2,1", "2,2"},
        )
        self.assertGreater(coverage["selectedMoveAttempts"]["tackle"], 0)
        self.assertGreater(coverage["selectedMoveAttempts"]["megaPunch"], 0)
        self.assertGreater(coverage["selectedMoveHits"]["megaPunch"], 0)
        self.assertGreater(coverage["criticalHitsBySelectedMove"]["megaPunch"], 0)
        self.assertGreater(coverage["missesByActorAndSelectedMove"]["p1"]["megaPunch"], 0)
        self.assertGreater(coverage["missesByActorAndSelectedMove"]["p2"]["megaPunch"], 0)
        self.assertGreater(coverage["criticalHits"], 0)
        self.assertGreater(coverage["firstActionKoSkips"], 0)
        self.assertGreater(coverage["terminalAfterSecondAction"], 0)
        self.assertEqual(coverage["depletionContinuations"], {"megaPunchThenTackle": 1, "tackleThenMegaPunch": 1})
        self.assertEqual(coverage, fixture["coverage"])


if __name__ == "__main__":
    unittest.main()
