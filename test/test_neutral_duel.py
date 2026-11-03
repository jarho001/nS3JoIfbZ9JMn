"""Full native-duel differential tests for the bounded JAX executor."""

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
	DIAGNOSTIC_NONE,
	DISPOSITION_BUDGET_EXHAUSTED,
	DISPOSITION_COMPLETED,
	DISPOSITION_INVALID_BUDGET,
	DISPOSITION_STOPPED,
	MAX_DUEL_TURNS,
	PHASE_MOVE_REQUEST,
	PHASE_PENDING,
	PHASE_TERMINAL,
	STATUS_OK,
	STATUS_PENDING,
	STATUS_TERMINAL,
	STATUS_UNSUPPORTED,
	BattleState,
	make_state,
	run_duel,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
ORACLE = ROOT / "reference" / "neutral-turn-oracle.cjs"
DRAW_CONTEXTS = {"accuracy": 1, "critical": 2, "damage": 3}
_COMPILED_RUN_DUEL = jax.jit(run_duel, static_argnames=("max_turns",))


def _state(snapshot):
	return make_state(
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


def _jax_view_projection(view):
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


def _assert_state_matches(test, actual, expected, label):
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
	test.assertEqual(expected["requestsPresent"], [not expected["terminal"], not expected["terminal"]], label)


def _hand_damage(level, attack, defense, critical, roll):
	level_term = (2 * level) // 5 + 2
	core = (level_term * 50 * attack // defense) // 50
	pre_roll = core + 2
	if critical:
		pre_roll *= 2
	return max(1, pre_roll * (100 - roll) // 100)


class NeutralDuelTests(unittest.TestCase):
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
		cls.scenarios = {row["id"]: row for row in cls.fixture["scenarios"]}

	def _assert_turn_matches(self, row, turn, scenario_id, turn_index):
		label = (scenario_id, turn_index)
		self.assertTrue(bool(row.valid), label)
		self.assertEqual(int(row.transition.status), STATUS_OK, label)
		_assert_state_matches(self, row.before, turn["before"], (*label, "before"))
		_assert_state_matches(self, row.transition.state, turn["after"], (*label, "after"))

		actors = [int(actor) for actor in _as_list(row.transition.events.actors) if actor >= 0]
		self.assertEqual(["p1" if actor == 0 else "p2" for actor in actors], turn["actionOrder"], (*label, "order"))
		self.assertEqual(int(row.transition.draws.count), len(turn["draws"]), (*label, "draw count"))
		self.assertEqual(len(turn["draws"]), len(actors) * 3, (*label, "three draws per action"))
		for draw_index, expected in enumerate(turn["draws"]):
			actual = row.transition.draws
			self.assertEqual(int(actual.bounds[draw_index]), expected["bound"], (*label, draw_index, "bound"))
			self.assertEqual(int(actual.values[draw_index]), expected["value"], (*label, draw_index, "value"))
			self.assertEqual(int(actual.contexts[draw_index]), DRAW_CONTEXTS[expected["context"]], (*label, draw_index, "context"))
			self.assertEqual(_as_list(actual.before[draw_index]), expected["before"], (*label, draw_index, "before limbs"))
			self.assertEqual(_as_list(actual.after[draw_index]), expected["after"], (*label, draw_index, "after limbs"))
		for audience, field in ((0, "p1_view"), (1, "p2_view")):
			view = getattr(row, field)
			self.assertEqual(
				_jax_view_projection(view),
				turn["views"]["p1" if audience == 0 else "p2"],
				(*label, "view", audience),
			)
			self.assertNotIn("rng", view._fields)
			self.assertNotIn("opponent_hp", view._fields)
			self.assertNotIn("unprojected_packets", view._fields)
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
			critical_roll = turn["draws"][action_index * 3 + 1]["value"]
			damage_roll = turn["draws"][action_index * 3 + 2]["value"]
			critical = critical_roll == 0
			self.assertEqual(bool(row.transition.events.critical[action_index]), critical, (*label, action_index, "critical"))
			self.assertEqual(
				int(row.transition.events.damage[action_index]),
				_hand_damage(
					turn["before"]["level"][actor],
					turn["before"]["attack"][actor],
					turn["before"]["defense"][1 - actor],
					critical,
					damage_roll,
				),
				(*label, action_index, "independent damage"),
			)

	def test_actual_full_native_duels_match_every_compiled_state_draw_and_view(self):
		self.assertEqual(self.fixture["kind"], "gen5-restricted-neutral-tackle-full-duel-oracle")
		self.assertEqual(len({tuple(row["seed"]) for row in self.fixture["scenarios"]}), 4)
		self.assertTrue(any(row["turns"][0]["actionOrder"][0] == "p1" for row in self.fixture["scenarios"]))
		self.assertTrue(any(row["turns"][0]["actionOrder"][0] == "p2" for row in self.fixture["scenarios"]))
		critical_case = self.scenarios["critical-full-duel"]
		self.assertTrue(any(draw["context"] == "critical" and draw["value"] == 0 for turn in critical_case["turns"] for draw in turn["draws"]))
		self.assertTrue(any(event["type"] == "critical" for turn in critical_case["turns"] for event in turn["views"]["p1"]["events"]))
		ko_case = self.scenarios["one-action-ko-full-duel"]
		self.assertEqual(ko_case["turns"][0]["actionOrder"], ["p2"])
		self.assertTrue(ko_case["outcome"]["terminal"])
		self.assertIn(ko_case["outcome"]["winner"], ("p1", "p2"))
		self.assertTrue(any(len(row["turns"]) > 1 for row in self.fixture["scenarios"]))

		for scenario in self.fixture["scenarios"]:
			self.assertTrue(scenario["outcome"]["terminal"], scenario["id"])
			self.assertIn(scenario["outcome"]["winner"], ("p1", "p2"), scenario["id"])
			self.assertLessEqual(len(scenario["turns"]), self.fixture["turnLimit"], scenario["id"])
			state = _state(scenario["initial"]["state"])
			initial_state_copy = jax.tree_util.tree_map(lambda leaf: np.array(leaf, copy=True), state)
			eager = run_duel(state, max_turns=MAX_DUEL_TURNS)
			compiled = _COMPILED_RUN_DUEL(state, max_turns=MAX_DUEL_TURNS)
			self.assertTrue(_leaf_equal(eager, compiled), (scenario["id"], "eager/JIT"))
			self.assertEqual(int(eager.disposition), DISPOSITION_COMPLETED, scenario["id"])
			self.assertEqual(int(eager.turn_count), len(scenario["turns"]), scenario["id"])
			self.assertEqual(int(eager.diagnostic_status), STATUS_OK, scenario["id"])
			self.assertEqual(_as_list(eager.trace.valid), [True] * len(scenario["turns"]) + [False] * (MAX_DUEL_TURNS - len(scenario["turns"])))
			_assert_state_matches(self, eager.state, scenario["outcome"]["state"], (scenario["id"], "outcome"))
			self.assertTrue(_leaf_equal(state, initial_state_copy), (scenario["id"], "input immutable"))

			for index, turn in enumerate(scenario["turns"]):
				row = jax.tree_util.tree_map(lambda leaf: leaf[index], eager.trace)
				self._assert_turn_matches(row, turn, scenario["id"], index)
				if index:
					self.assertEqual(turn["before"]["hp"], scenario["turns"][index - 1]["after"]["hp"])
					self.assertEqual(turn["before"]["pp"], scenario["turns"][index - 1]["after"]["pp"])
					self.assertEqual(turn["before"]["rng"], scenario["turns"][index - 1]["after"]["rng"])

			for index in range(len(scenario["turns"]), MAX_DUEL_TURNS):
				self.assertEqual(int(eager.trace.transition.draws.count[index]), 0, (scenario["id"], index, "padding draws"))
				self.assertFalse(any(_as_list(eager.trace.transition.events.executed[index])), (scenario["id"], index, "padding events"))
				self.assertFalse(bool(eager.trace.p1_view.legal_move_mask[index, 0]), (scenario["id"], index, "padding request"))
				self.assertFalse(bool(eager.trace.p1_view.terminal[index]), (scenario["id"], index, "padding outcome"))

	def test_budget_resume_exact_capacity_and_pp_one_unsupported_boundary(self):
		scenario = self.scenarios["p2-first-full-duel"]
		initial = _state(scenario["initial"]["state"])
		limited = run_duel(initial, max_turns=3)
		self.assertEqual(int(limited.disposition), DISPOSITION_BUDGET_EXHAUSTED)
		self.assertEqual(int(limited.turn_count), 3)
		self.assertEqual(int(limited.diagnostic_status), STATUS_OK)
		_assert_state_matches(self, limited.state, scenario["turns"][2]["after"], "budget state")

		resumed = run_duel(limited.state, max_turns=8, turn_budget=8)
		exact = run_duel(initial, max_turns=8, turn_budget=8)
		self.assertEqual(int(resumed.disposition), DISPOSITION_COMPLETED)
		self.assertEqual(int(resumed.turn_count), 5)
		self.assertEqual(int(exact.disposition), DISPOSITION_COMPLETED, "terminal on final capacity slot beats budget exhaustion")
		self.assertEqual(int(exact.turn_count), 8)
		self.assertTrue(_leaf_equal(resumed.state, exact.state))
		self.assertTrue(_leaf_equal(
			jax.tree_util.tree_map(lambda leaf: leaf[:3], limited.trace),
			jax.tree_util.tree_map(lambda leaf: leaf[:3], exact.trace),
		))
		self.assertTrue(_leaf_equal(
			jax.tree_util.tree_map(lambda leaf: leaf[:5], resumed.trace),
			jax.tree_util.tree_map(lambda leaf: leaf[3:8], exact.trace),
		))
		combined = np.concatenate((np.asarray(limited.trace.valid), np.asarray(resumed.trace.valid)))
		self.assertEqual(combined.tolist(), [True] * 8 + [False] * 3)

		one_pp = initial._replace(pp=jnp.asarray([1, 1], dtype=jnp.int32))
		one_slot = run_duel(one_pp, max_turns=1)
		self.assertEqual(int(one_slot.disposition), DISPOSITION_BUDGET_EXHAUSTED)
		self.assertEqual(int(one_slot.turn_count), 1)
		self.assertEqual(int(one_slot.diagnostic_status), STATUS_OK)
		self.assertFalse(bool(one_slot.state.terminal))
		self.assertEqual(_as_list(one_slot.state.pp), [0, 0])
		self.assertEqual(_as_list(one_slot.trace.valid), [True])

		unsupported = run_duel(one_slot.state, max_turns=2)
		self.assertEqual(int(unsupported.disposition), DISPOSITION_STOPPED)
		self.assertEqual(int(unsupported.diagnostic_status), STATUS_UNSUPPORTED)
		self.assertEqual(int(unsupported.turn_count), 0)
		self.assertEqual(_as_list(unsupported.trace.valid), [False, False])
		self.assertTrue(_leaf_equal(unsupported.state, one_slot.state))
		boundary_in_budget = run_duel(one_pp, max_turns=2, turn_budget=2)
		self.assertEqual(int(boundary_in_budget.disposition), DISPOSITION_STOPPED)
		self.assertEqual(int(boundary_in_budget.diagnostic_status), STATUS_UNSUPPORTED)
		self.assertEqual(int(boundary_in_budget.turn_count), 1)
		self.assertEqual(_as_list(boundary_in_budget.trace.valid), [True, False])
		self.assertTrue(_leaf_equal(boundary_in_budget.state, one_slot.state))

	def test_terminal_pending_unsupported_and_structural_boundaries(self):
		ko = self.scenarios["one-action-ko-full-duel"]["outcome"]["state"]
		terminal = _state(ko)
		completed = run_duel(terminal, max_turns=2)
		self.assertEqual(int(completed.disposition), DISPOSITION_COMPLETED)
		self.assertEqual(int(completed.turn_count), 0)
		self.assertEqual(int(completed.diagnostic_status), STATUS_TERMINAL)
		self.assertEqual(_as_list(completed.trace.valid), [False, False])
		self.assertTrue(_leaf_equal(completed.state, terminal))
		zero_budget_terminal = run_duel(terminal, max_turns=2, turn_budget=0)
		self.assertEqual(int(zero_budget_terminal.disposition), DISPOSITION_COMPLETED)
		self.assertEqual(int(zero_budget_terminal.diagnostic_status), STATUS_TERMINAL)
		self.assertEqual(int(zero_budget_terminal.turn_count), 0)

		active = _state(self.scenarios["p2-first-full-duel"]["initial"]["state"])
		pending = active._replace(request_phase=jnp.asarray(PHASE_PENDING, dtype=jnp.int32))
		pending_result = run_duel(pending, max_turns=2)
		self.assertEqual(int(pending_result.disposition), DISPOSITION_STOPPED)
		self.assertEqual(int(pending_result.diagnostic_status), STATUS_PENDING)
		self.assertEqual(int(pending_result.turn_count), 0)
		self.assertTrue(_leaf_equal(pending_result.state, pending))
		unsupported_state = active._replace(speed=jnp.asarray([120, 120], dtype=jnp.int32))
		unsupported_result = run_duel(unsupported_state, max_turns=2)
		self.assertEqual(int(unsupported_result.disposition), DISPOSITION_STOPPED)
		self.assertEqual(int(unsupported_result.diagnostic_status), STATUS_UNSUPPORTED)
		self.assertTrue(_leaf_equal(unsupported_result.state, unsupported_state))
		for bad_capacity in (0, MAX_DUEL_TURNS + 1):
			with self.assertRaises(ValueError):
				run_duel(active, max_turns=bad_capacity)
		for bad_capacity in (True, 1.0, jnp.asarray(1, dtype=jnp.int32)):
			with self.assertRaises(TypeError):
				run_duel(active, max_turns=bad_capacity)
		with self.assertRaises(TypeError):
			run_duel(tuple(active), max_turns=1)
		with self.assertRaises(ValueError):
			run_duel(active._replace(hp=jnp.zeros((3,), dtype=jnp.int32)), max_turns=1)
		with self.assertRaises(TypeError):
			run_duel(active._replace(hp=jnp.asarray([1, 1], dtype=jnp.int16)), max_turns=1)
		with self.assertRaises(TypeError):
			run_duel(active, max_turns=2, turn_budget=1.0)
		invalid_jit = jax.jit(lambda state, budget: run_duel(state, max_turns=2, turn_budget=budget))
		for bad_budget in (-1, 3):
			invalid_eager = run_duel(active, max_turns=2, turn_budget=bad_budget)
			invalid_dynamic = invalid_jit(active, jnp.asarray(bad_budget, dtype=jnp.int32))
			for invalid_result in (invalid_eager, invalid_dynamic):
				self.assertEqual(int(invalid_result.disposition), DISPOSITION_INVALID_BUDGET)
				self.assertEqual(int(invalid_result.diagnostic_status), DIAGNOSTIC_INVALID_BUDGET)
				self.assertTrue(_leaf_equal(invalid_result.state, active))
				self.assertEqual(_as_list(invalid_result.trace.valid), [False, False])
		zero_budget = run_duel(active, max_turns=2, turn_budget=0)
		self.assertEqual(int(zero_budget.disposition), DISPOSITION_BUDGET_EXHAUSTED)
		self.assertEqual(int(zero_budget.diagnostic_status), DIAGNOSTIC_NONE)
		self.assertTrue(_leaf_equal(zero_budget.state, active))

	def test_jit_vmap_mixed_lengths_diagnostics_budgets_and_lane_order(self):
		first = self.scenarios["p2-first-full-duel"]
		second = self.scenarios["p1-first-full-duel"]
		ko = self.scenarios["one-action-ko-full-duel"]
		base_states = [
			_state(first["initial"]["state"]),
			_state(second["initial"]["state"]),
			_state(first["initial"]["state"]),
			_state(ko["initial"]["state"]),
			_state(first["initial"]["state"])._replace(speed=jnp.asarray([120, 120], dtype=jnp.int32)),
			_state(ko["outcome"]["state"]),
			_state(first["initial"]["state"])._replace(request_phase=jnp.asarray(PHASE_PENDING, dtype=jnp.int32)),
			_state(first["initial"]["state"]),
		]
		budgets = jnp.asarray([0, 1, 8, 8, 8, 0, 8, 9], dtype=jnp.int32)
		batch = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *base_states)
		batched_run = jax.jit(jax.vmap(lambda state, budget: run_duel(state, max_turns=8, turn_budget=budget)))
		batched = batched_run(batch, budgets)
		self.assertEqual(_as_list(batched.disposition), [
			DISPOSITION_BUDGET_EXHAUSTED,
			DISPOSITION_BUDGET_EXHAUSTED,
			DISPOSITION_COMPLETED,
			DISPOSITION_COMPLETED,
			DISPOSITION_STOPPED,
			DISPOSITION_COMPLETED,
			DISPOSITION_STOPPED,
			DISPOSITION_INVALID_BUDGET,
		])
		self.assertEqual(_as_list(batched.turn_count), [0, 1, 8, 1, 0, 0, 0, 0])
		self.assertEqual(_as_list(batched.diagnostic_status), [
			DIAGNOSTIC_NONE,
			STATUS_OK,
			STATUS_OK,
			STATUS_OK,
			STATUS_UNSUPPORTED,
			STATUS_TERMINAL,
			STATUS_PENDING,
			DIAGNOSTIC_INVALID_BUDGET,
		])
		for lane, state in enumerate(base_states):
			separate = run_duel(state, max_turns=8, turn_budget=int(budgets[lane]))
			self.assertTrue(_leaf_equal(jax.tree_util.tree_map(lambda leaf: leaf[lane], batched), separate), lane)
			if lane >= 4:
				self.assertTrue(_leaf_equal(separate.state, state), (lane, "diagnostic state unchanged"))
				self.assertEqual(_as_list(separate.trace.valid), [False] * 8, (lane, "diagnostic empty trace"))

		permutation = jnp.asarray([7, 4, 2, 0, 6, 3, 1, 5], dtype=jnp.int32)
		reordered = batched_run(
			jax.tree_util.tree_map(lambda leaf: leaf[permutation], batch),
			budgets[permutation],
		)
		for new_lane, old_lane in enumerate(np.asarray(permutation).tolist()):
			self.assertTrue(
				_leaf_equal(
					jax.tree_util.tree_map(lambda leaf: leaf[new_lane], reordered),
					jax.tree_util.tree_map(lambda leaf: leaf[old_lane], batched),
				),
				(old_lane, new_lane),
			)


if __name__ == "__main__":
	unittest.main()
