"""Differential tests for the explicitly named speed-tie Tackle variant."""

from __future__ import annotations

import json
import pathlib
import subprocess
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from simulator import (
	DIAGNOSTIC_INVALID_BUDGET,
	DISPOSITION_BUDGET_EXHAUSTED,
	DISPOSITION_COMPLETED,
	DISPOSITION_INVALID_BUDGET,
	DISPOSITION_STOPPED,
	DRAW_ACTION_UPDATE,
	DRAW_BEFORE_TURN,
	DRAW_MOVE_HIT_UPDATE,
	DRAW_QUEUE_SORT,
	DRAW_RESIDUAL_UPDATE,
	PHASE_MOVE_REQUEST,
	PHASE_PENDING,
	PHASE_TERMINAL,
	STATUS_INVALID_CHOICE,
	STATUS_OK,
	STATUS_PENDING,
	STATUS_TERMINAL,
	STATUS_UNSUPPORTED,
	make_tie_state,
	project_tie_view,
	run_tie_duel,
	transition_with_speed_ties,
)
from simulator.neutral_tackle import DRAW_ACCURACY, DRAW_CRITICAL, DRAW_DAMAGE, PHASE_UNSUPPORTED


ROOT = pathlib.Path(__file__).resolve().parents[1]
ORACLE = ROOT / "reference" / "neutral-turn-oracle.cjs"
DRAW_CONTEXTS = {
	"accuracy": DRAW_ACCURACY,
	"critical": DRAW_CRITICAL,
	"damage": DRAW_DAMAGE,
	"queue-sort": DRAW_QUEUE_SORT,
	"before-turn": DRAW_BEFORE_TURN,
	"turn-start-update": DRAW_ACTION_UPDATE,
	"move-hit-update": DRAW_MOVE_HIT_UPDATE,
	"action-update": DRAW_ACTION_UPDATE,
	"residual-update": DRAW_RESIDUAL_UPDATE,
}
_COMPILED_TRANSITION = jax.jit(transition_with_speed_ties)
_COMPILED_DUEL = jax.jit(run_tie_duel, static_argnames=("max_turns",))


def _state(snapshot):
	return make_tie_state(
		hp=snapshot["hp"],
		max_hp=snapshot["maxHp"],
		pp=snapshot["pp"],
		level=snapshot["level"],
		attack=snapshot["attack"],
		defense=snapshot["defense"],
		speed=snapshot["speed"],
		rng=snapshot["rng"],
		turn=snapshot["turn"],
		request_phase=PHASE_TERMINAL if snapshot["terminal"] else PHASE_MOVE_REQUEST,
		terminal=snapshot["terminal"],
		winner=snapshot["winner"],
	)


def _as_list(value):
	return np.asarray(value).tolist()


def _leaf_equal(left, right):
	left_leaves = jax.tree_util.tree_leaves(left)
	right_leaves = jax.tree_util.tree_leaves(right)
	return len(left_leaves) == len(right_leaves) and all(
		np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(left_leaves, right_leaves)
	)


def _view_projection(view):
	request = {
		"phase": "move",
		"legalActions": ["move 1"] if bool(np.asarray(view.legal_move_mask)[0]) else [],
		"own": {
			"species": "Mew",
			"level": int(view.own_level),
			"hpCondition": f"{int(view.own_hp)}/{int(view.own_max_hp)}",
			"stats": {"atk": int(view.own_attack), "def": int(view.own_defense), "spe": int(view.own_speed)},
			"move": {"name": "Tackle", "id": "tackle", "pp": int(view.own_pp), "disabled": False},
		},
	}
	events = []
	for index, actor in enumerate(_as_list(view.action_order)):
		if actor < 0:
			continue
		target = int(view.target_player[index])
		actor_id = "p1" if actor == 0 else "p2"
		target_id = "p1" if target == 0 else "p2"
		events.append({"type": "move", "actor": actor_id, "move": "tackle", "target": target_id})
		if bool(view.critical[index]):
			events.append({"type": "critical", "target": target_id})
		hp = int(view.target_hp_value[index])
		if hp == 0:
			condition = "0 fnt"
		elif bool(view.target_hp_is_percent[index]):
			condition = f"{hp}/100"
		else:
			condition = f"{hp}/{int(view.own_max_hp)}"
		events.append({"type": "damage", "target": target_id, "condition": condition})
		if bool(view.fainted[index]):
			events.append({"type": "faint", "target": target_id})
	if bool(view.terminal):
		winner = int(view.winner)
		events.append({"type": "winner", "side": "p1" if winner == 1 else "p2" if winner == 2 else "tie"})
	elif bool(view.turn_event):
		events.append({"type": "turn", "number": int(view.turn_number)})
	return {"request": request, "events": events}


def _assert_state(test, actual, expected, label):
	for field, actual_value in (
		("hp", actual.hp),
		("maxHp", actual.max_hp),
		("pp", actual.pp),
		("level", actual.level),
		("attack", actual.attack),
		("defense", actual.defense),
		("speed", actual.speed),
		("rng", actual.rng),
	):
		test.assertEqual(_as_list(actual_value), expected[field], (label, field))
	for field, value in (("turn", actual.turn), ("terminal", actual.terminal), ("winner", actual.winner)):
		test.assertEqual(bool(value) if field == "terminal" else int(value), expected[field], (label, field))
	phase = PHASE_TERMINAL if expected["requestPhase"] == "terminal" else PHASE_MOVE_REQUEST
	test.assertEqual(int(actual.request_phase), phase, (label, "requestPhase"))


def _hand_damage(level, attack, defense, critical, roll):
	level_term = (2 * level) // 5 + 2
	core = (level_term * 50 * attack // defense) // 50
	pre_roll = core + 2
	if critical:
		pre_roll *= 2
	return max(1, pre_roll * (100 - roll) // 100)


class NeutralTackleSpeedTieTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		completed = subprocess.run(
			["node", str(ORACLE), "--ties"],
			cwd=ROOT,
			check=True,
			capture_output=True,
			text=True,
		)
		cls.fixture = json.loads(completed.stdout)
		cls.scenarios = {row["id"]: row for row in cls.fixture["scenarios"]}
		legacy = subprocess.run(
			["node", str(ORACLE), "--duels"],
			cwd=ROOT,
			check=True,
			capture_output=True,
			text=True,
		)
		cls.legacy_fixture = json.loads(legacy.stdout)

	def _assert_turn(self, row, turn, scenario_id, turn_index):
		label = (scenario_id, turn_index)
		self.assertTrue(bool(row.valid), label)
		self.assertEqual(int(row.transition.status), STATUS_OK, label)
		_assert_state(self, row.before, turn["before"], (*label, "before"))
		_assert_state(self, row.transition.state, turn["after"], (*label, "after"))
		actors = [int(actor) for actor in _as_list(row.transition.events.actors) if actor >= 0]
		self.assertEqual(["p1" if actor == 0 else "p2" for actor in actors], turn["actionOrder"], (*label, "order"))
		self.assertEqual(int(row.transition.draws.count), len(turn["draws"]), (*label, "draw count"))
		self.assertLessEqual(len(turn["draws"]), 16, label)
		self.assertEqual(len(turn["draws"]), len(actors) * 3 + sum(draw["context"] == "speed-sort" for draw in turn["draws"]), (*label, "draw accounting"))
		for draw_index, expected in enumerate(turn["draws"]):
			actual = row.transition.draws
			self.assertEqual(int(actual.bounds[draw_index]), 2 if expected["context"] == "speed-sort" else expected["bound"], (*label, draw_index, "bound"))
			self.assertEqual(int(actual.values[draw_index]), expected["value"], (*label, draw_index, "value"))
			self.assertEqual(int(actual.contexts[draw_index]), DRAW_CONTEXTS[expected["phase"]], (*label, draw_index, "context"))
			self.assertEqual(_as_list(actual.before[draw_index]), expected["before"], (*label, draw_index, "before limbs"))
			self.assertEqual(_as_list(actual.after[draw_index]), expected["after"], (*label, draw_index, "after limbs"))
			if draw_index:
				self.assertEqual(expected["before"], turn["draws"][draw_index - 1]["after"], (*label, draw_index, "native chain"))
		self.assertEqual(_as_list(row.transition.events.executed), [index < len(actors) for index in range(2)], (*label, "event mask"))
		for audience, field in ((0, "p1_view"), (1, "p2_view")):
			view = getattr(row, field)
			self.assertEqual(_view_projection(view), turn["views"]["p1" if audience == 0 else "p2"], (*label, "view", audience))
			self.assertNotIn("rng", view._fields)
			self.assertNotIn("draws", view._fields)
			for action_index, actor in enumerate(_as_list(view.action_order)):
				if actor < 0:
					continue
				target = int(view.target_player[action_index])
				if target == audience:
					observed_loss = max(0, turn["before"]["hp"][target] - turn["after"]["hp"][target])
					self.assertTrue(bool(view.damage_visible[action_index]), (*label, "visible", audience, action_index))
					self.assertEqual(int(view.damage_value[action_index]), observed_loss, (*label, "loss", audience, action_index))
				else:
					self.assertFalse(bool(view.damage_visible[action_index]), (*label, "hidden", audience, action_index))
					self.assertEqual(int(view.damage_value[action_index]), -1, (*label, "hidden damage", audience, action_index))
		for action_index, actor in enumerate(actors):
			action_draws = [draw for draw in turn["draws"] if draw["context"] != "speed-sort"]
			critical = action_draws[action_index * 3 + 1]["value"] == 0
			self.assertEqual(bool(row.transition.events.critical[action_index]), critical, (*label, action_index, "critical"))
			self.assertEqual(
				int(row.transition.events.damage[action_index]),
				_hand_damage(
					turn["before"]["level"][actor],
					turn["before"]["attack"][actor],
					turn["before"]["defense"][1 - actor],
					critical,
					action_draws[action_index * 3 + 2]["value"],
				),
				(*label, action_index, "independent damage"),
			)

	def test_complete_native_tied_duels_match_every_state_draw_and_view(self):
		self.assertEqual(self.fixture["kind"], "gen5-restricted-neutral-tackle-speed-tie-full-duel-oracle")
		self.assertEqual(len({tuple(row["seed"]) for row in self.fixture["scenarios"]}), 3)
		self.assertTrue(any(row["turns"][0]["actionOrder"][0] == "p1" for row in self.fixture["scenarios"]))
		self.assertTrue(any(row["turns"][0]["actionOrder"][0] == "p2" for row in self.fixture["scenarios"]))
		first_ko = self.scenarios["equal-speed-p1-initial-order-and-first-action-ko"]
		self.assertEqual(first_ko["turns"][-1]["actionOrder"], ["p2"])
		self.assertEqual(len(first_ko["turns"][-1]["draws"]), 7)
		self.assertTrue(any(draw["phase"] == "critical" and draw["value"] == 0 for turn in self.scenarios["equal-speed-critical"]["turns"] for draw in turn["draws"]))

		for scenario in self.fixture["scenarios"]:
			self.assertTrue(scenario["outcome"]["terminal"], scenario["id"])
			self.assertLessEqual(len(scenario["turns"]), self.fixture["turnLimit"], scenario["id"])
			state = _state(scenario["initial"]["state"])
			copy = jax.tree_util.tree_map(lambda leaf: np.array(leaf, copy=True), state)
			eager = run_tie_duel(state, max_turns=56)
			compiled = _COMPILED_DUEL(state, max_turns=56)
			self.assertTrue(_leaf_equal(eager, compiled), (scenario["id"], "eager/JIT"))
			self.assertEqual(int(eager.disposition), DISPOSITION_COMPLETED, scenario["id"])
			self.assertEqual(int(eager.turn_count), len(scenario["turns"]), scenario["id"])
			self.assertEqual(int(eager.diagnostic_status), STATUS_OK, scenario["id"])
			self.assertEqual(_as_list(eager.trace.valid), [True] * len(scenario["turns"]) + [False] * (56 - len(scenario["turns"])))
			_assert_state(self, eager.state, scenario["outcome"]["state"], (scenario["id"], "outcome"))
			self.assertTrue(_leaf_equal(state, copy), (scenario["id"], "input immutable"))

			for index, turn in enumerate(scenario["turns"]):
				self.assertEqual(turn["actionOrder"][0], "p1" if turn["draws"][0]["value"] == 0 else "p2", (scenario["id"], index, "queue tie selection"))
				row = jax.tree_util.tree_map(lambda leaf: leaf[index], eager.trace)
				self._assert_turn(row, turn, scenario["id"], index)
				if index:
					self.assertEqual(turn["before"]["hp"], scenario["turns"][index - 1]["after"]["hp"])
					self.assertEqual(turn["before"]["pp"], scenario["turns"][index - 1]["after"]["pp"])
					self.assertEqual(turn["before"]["rng"], scenario["turns"][index - 1]["after"]["rng"])
			for index in range(len(scenario["turns"]), 56):
				self.assertEqual(int(eager.trace.transition.draws.count[index]), 0, (scenario["id"], index, "padding draws"))
				self.assertFalse(any(_as_list(eager.trace.transition.events.executed[index])), (scenario["id"], index, "padding events"))
				self.assertFalse(bool(eager.trace.p1_view.legal_move_mask[index, 0]), (scenario["id"], index, "padding request"))

	def test_unequal_speed_control_resume_and_mixed_vmap_lanes(self):
		scenarios = list(self.fixture["scenarios"])
		states = [_state(row["initial"]["state"]) for row in scenarios]
		control_scenario = next(row for row in self.legacy_fixture["scenarios"] if row["id"] == "p2-first-full-duel")
		control_state = _state(control_scenario["initial"]["state"])
		self.assertNotEqual(int(control_state.speed[0]), int(control_state.speed[1]))
		control_transition = _COMPILED_TRANSITION(control_state, jnp.asarray([1, 1], dtype=jnp.int32))
		self.assertEqual(int(control_transition.status), STATUS_OK)
		self.assertEqual(_as_list(control_transition.events.actors), [1, 0])
		self.assertEqual(int(control_transition.draws.count), 6)
		self.assertEqual(_as_list(control_transition.draws.contexts[:6]), [DRAW_ACCURACY, DRAW_CRITICAL, DRAW_DAMAGE] * 2)
		self.assertFalse(any(int(value) >= DRAW_QUEUE_SORT for value in _as_list(control_transition.draws.contexts[:6])))
		control_duel = run_tie_duel(control_state, max_turns=56)
		self.assertEqual(int(control_duel.turn_count), len(control_scenario["turns"]))
		for index, expected in enumerate(control_scenario["turns"]):
			row = jax.tree_util.tree_map(lambda leaf: leaf[index], control_duel.trace)
			normalized = dict(expected)
			normalized["draws"] = [{**draw, "phase": draw["context"]} for draw in expected["draws"]]
			self._assert_turn(row, normalized, control_scenario["id"], index)
		_assert_state(self, control_duel.state, control_scenario["outcome"]["state"], (control_scenario["id"], "outcome"))

		full = run_tie_duel(states[0], max_turns=8)
		limited = run_tie_duel(states[0], max_turns=8, turn_budget=3)
		self.assertEqual(int(limited.disposition), DISPOSITION_BUDGET_EXHAUSTED)
		resumed = run_tie_duel(limited.state, max_turns=8, turn_budget=8)
		self.assertEqual(int(resumed.disposition), DISPOSITION_COMPLETED)
		self.assertTrue(_leaf_equal(full.state, resumed.state))
		for index in range(int(full.turn_count)):
			full_row = jax.tree_util.tree_map(lambda leaf: leaf[index], full.trace)
			resumed_row = jax.tree_util.tree_map(lambda leaf: leaf[index if index < 3 else index - 3], limited.trace if index < 3 else resumed.trace)
			self.assertTrue(_leaf_equal(full_row, resumed_row), (index, "resume row"))
		exact_capacity = run_tie_duel(states[0], max_turns=len(scenarios[0]["turns"]), turn_budget=len(scenarios[0]["turns"]))
		self.assertEqual(int(exact_capacity.disposition), DISPOSITION_COMPLETED)
		self.assertEqual(int(exact_capacity.turn_count), len(scenarios[0]["turns"]))

		pp_one = states[0]._replace(pp=jnp.asarray([1, 56], dtype=jnp.int32))
		pp_one_turn = run_tie_duel(pp_one, max_turns=1, turn_budget=1)
		self.assertEqual(int(pp_one_turn.disposition), DISPOSITION_BUDGET_EXHAUSTED)
		self.assertEqual(int(pp_one_turn.turn_count), 1)
		self.assertEqual(int(pp_one_turn.state.pp[0]), 0)
		self.assertEqual(int(pp_one_turn.state.request_phase), PHASE_UNSUPPORTED)
		pp_attempt = run_tie_duel(pp_one_turn.state, max_turns=2, turn_budget=2)
		self.assertEqual(int(pp_attempt.disposition), DISPOSITION_STOPPED)
		self.assertEqual(int(pp_attempt.diagnostic_status), STATUS_UNSUPPORTED)
		self.assertEqual(int(pp_attempt.turn_count), 0)
		self.assertEqual(int(pp_attempt.trace.transition.draws.count[0]), 0)
		self.assertTrue(_leaf_equal(pp_one_turn.state, pp_attempt.state))

		pending = states[2]._replace(request_phase=jnp.asarray(PHASE_PENDING, dtype=jnp.int32))
		terminal_snapshot = scenarios[0]["outcome"]["state"]
		terminal = _state(terminal_snapshot)
		unsupported = states[1]._replace(pp=states[1].pp.at[0].set(0))
		invalid_budget_lane = states[2]
		batch_states = [states[0], states[1], states[2], control_state, pending, terminal, unsupported, invalid_budget_lane]
		batch = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *batch_states)
		budgets = jnp.asarray([3, 8, 0, 1, 8, 0, 8, 9], dtype=jnp.int32)
		run_batch = jax.jit(jax.vmap(lambda lane, budget: run_tie_duel(lane, max_turns=8, turn_budget=budget)))
		batched = run_batch(batch, budgets)
		self.assertEqual(_as_list(batched.turn_count), [3, len(scenarios[1]["turns"]), 0, 1, 0, 0, 0, 0])
		self.assertEqual(_as_list(batched.disposition), [DISPOSITION_BUDGET_EXHAUSTED, DISPOSITION_COMPLETED, DISPOSITION_BUDGET_EXHAUSTED, DISPOSITION_BUDGET_EXHAUSTED, DISPOSITION_STOPPED, DISPOSITION_COMPLETED, DISPOSITION_STOPPED, DISPOSITION_INVALID_BUDGET])
		self.assertEqual(int(batched.trace.transition.draws.count[3, 0]), 6, "unequal-speed lane has no speed-sort draws")
		self.assertEqual(int(batched.diagnostic_status[4]), STATUS_PENDING)
		self.assertEqual(int(batched.diagnostic_status[5]), STATUS_TERMINAL)
		self.assertEqual(int(batched.diagnostic_status[6]), STATUS_UNSUPPORTED)
		self.assertEqual(int(batched.diagnostic_status[7]), DIAGNOSTIC_INVALID_BUDGET)
		for lane_index, (candidate, budget) in enumerate(zip(batch_states, _as_list(budgets))):
			individual = run_tie_duel(candidate, max_turns=8, turn_budget=budget)
			lane = jax.tree_util.tree_map(lambda leaf: leaf[lane_index], batched)
			self.assertTrue(_leaf_equal(individual, lane), (lane_index, "vmapped lane equals individual"))
			count = int(lane.turn_count)
			self.assertEqual(_as_list(lane.trace.valid), [True] * count + [False] * (8 - count), (lane_index, "valid prefix"))
			for slot in range(count, 8):
				self.assertEqual(int(lane.trace.transition.draws.count[slot]), 0, (lane_index, slot, "padding draw"))
				self.assertFalse(any(_as_list(lane.trace.transition.events.executed[slot])), (lane_index, slot, "padding event"))
			if budget == 0 or lane_index >= 4:
				self.assertTrue(_leaf_equal(candidate, lane.state), (lane_index, "inactive state/RNG freeze"))
		permutation = np.asarray([7, 5, 2, 0, 6, 4, 3, 1])
		reversed_batch = jax.tree_util.tree_map(lambda value: value[permutation], batch)
		reversed_result = run_batch(reversed_batch, budgets[permutation])
		self.assertTrue(_leaf_equal(reversed_result, jax.tree_util.tree_map(lambda value: value[permutation], batched)))

	def test_tie_variant_diagnostics_and_invalid_runtime_budget_do_not_advance(self):
		state = _state(self.scenarios["equal-speed-p1-initial-order-and-first-action-ko"]["initial"]["state"])
		pending = state._replace(request_phase=jnp.asarray(PHASE_PENDING, dtype=jnp.int32))
		terminal = state._replace(terminal=jnp.asarray(True), request_phase=jnp.asarray(PHASE_TERMINAL, dtype=jnp.int32), winner=jnp.asarray(1, dtype=jnp.int32))
		unsupported = state._replace(pp=state.pp.at[0].set(0))
		for candidate, choices, status in (
			(pending, [1, 1], STATUS_PENDING),
			(unsupported, [1, 1], STATUS_UNSUPPORTED),
			(state, [-1, 1], STATUS_INVALID_CHOICE),
			(state, [0, 1], STATUS_PENDING),
			(terminal, [1, 1], STATUS_TERMINAL),
		):
			before = jax.tree_util.tree_map(lambda leaf: np.array(leaf, copy=True), candidate)
			result = _COMPILED_TRANSITION(candidate, jnp.asarray(choices, dtype=jnp.int32))
			self.assertEqual(int(result.status), status)
			self.assertEqual(int(result.draws.count), 0)
			self.assertTrue(_leaf_equal(candidate, result.state))
			self.assertFalse(any(_as_list(result.events.executed)))
			self.assertTrue(_leaf_equal(candidate, before))
		invalid_budget = run_tie_duel(state, max_turns=8, turn_budget=jnp.asarray(9, dtype=jnp.int32))
		self.assertEqual(int(invalid_budget.disposition), DISPOSITION_INVALID_BUDGET)
		self.assertEqual(int(invalid_budget.turn_count), 0)
		self.assertEqual(int(invalid_budget.trace.transition.draws.count[0]), 0)
		self.assertTrue(_leaf_equal(state, invalid_budget.state))
		self.assertFalse(bool(invalid_budget.trace.valid[0]))


if __name__ == "__main__":
	unittest.main()
