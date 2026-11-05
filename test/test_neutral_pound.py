"""Differential tests for the restricted pinned Gen 5 Pound turn variant."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from simulator import (
	ACTION_POUND,
	DRAW_ACTION_UPDATE,
	DRAW_BEFORE_TURN,
	DRAW_MOVE_HIT_UPDATE,
	DRAW_QUEUE_SORT,
	DRAW_RESIDUAL_UPDATE,
	MOVE_POUND,
	PHASE_MOVE_REQUEST,
	PHASE_PENDING,
	PHASE_TERMINAL,
	POUND_ACCURACY,
	POUND_BASE_PP,
	POUND_BASE_POWER,
	POUND_CRITICAL_RATIO,
	POUND_MAX_PP,
	POUND_VARIANT_ID,
	POUND_VARIANT_NAME,
	STATUS_INVALID_CHOICE,
	STATUS_OK,
	STATUS_PENDING,
	STATUS_TERMINAL,
	STATUS_UNSUPPORTED,
	make_pound_state,
	project_pound_view,
	transition_pound,
	transition_with_speed_ties,
)
from simulator.neutral_tackle import (
	DRAW_ACCURACY,
	DRAW_CRITICAL,
	DRAW_DAMAGE,
	MOVE_TACKLE,
	VARIANT_ID as TACKLE_VARIANT_ID,
	transition as transition_tackle,
)
from simulator.neutral_tackle_ties import _empty_tie_transition


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "reference" / "neutral-pound-oracle.cjs"
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


def _state(snapshot):
	phase = PHASE_TERMINAL if snapshot["terminal"] else PHASE_MOVE_REQUEST
	return make_pound_state(
		hp=snapshot["hp"],
		max_hp=snapshot["maxHp"],
		pp=snapshot["pp"],
		level=snapshot["level"],
		attack=snapshot["attack"],
		defense=snapshot["defense"],
		speed=snapshot["speed"],
		rng=snapshot["rng"],
		turn=snapshot["turn"],
		request_phase=phase,
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


def _hand_damage(level, attack, defense, critical, roll, base_power=POUND_BASE_POWER):
	"""Independent ordinary Gen 5 damage arithmetic for Pound versus Mew."""
	level_term = (2 * level) // 5 + 2
	core = (level_term * base_power * attack // defense) // 50
	pre_roll = core + 2
	if critical:
		pre_roll *= 2
	return max(1, pre_roll * (100 - roll) // 100)


def _jax_view_projection(view):
	request = {
		"phase": "move",
		"legalActions": ["move 1"] if bool(np.asarray(view.legal_move_mask)[0]) else [],
		"own": {
			"species": "Mew",
			"level": int(view.own_level),
			"hpCondition": f"{int(view.own_hp)}/{int(view.own_max_hp)}",
			"stats": {
				"atk": int(view.own_attack),
				"def": int(view.own_defense),
				"spe": int(view.own_speed),
			},
			"move": {"name": "Pound", "id": "pound", "pp": int(view.own_pp), "disabled": False},
		},
	}
	events = []
	actors = _as_list(view.action_order)
	for index, actor in enumerate(actors):
		if actor < 0:
			continue
		target = int(view.target_player[index])
		target_id = "p1" if target == 0 else "p2"
		actor_id = "p1" if actor == 0 else "p2"
		events.append({"type": "move", "actor": actor_id, "move": "pound", "target": target_id})
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
	test.assertEqual(int(actual.variant_id), POUND_VARIANT_ID, (label, "variant_id"))
	test.assertEqual(int(actual.move_id), MOVE_POUND, (label, "move_id"))


class NeutralPoundTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		completed = subprocess.run(
			["node", str(ORACLE)],
			cwd=ROOT,
			check=True,
			capture_output=True,
			text=True,
		)
		cls.fixture = json.loads(completed.stdout)
		cls.scenarios = {scenario["id"]: scenario for scenario in cls.fixture["scenarios"]}

	def _assert_turn(self, result, turn, scenario_id):
		self.assertEqual(int(result.status), STATUS_OK, scenario_id)
		_assert_state(self, result.state, turn["after"], (scenario_id, "after"))
		self.assertEqual(result.draws.bounds.shape, (16,), (scenario_id, "tie trace bounds shape"))
		self.assertEqual(result.draws.values.shape, (16,), (scenario_id, "tie trace values shape"))
		self.assertEqual(result.draws.before.shape, (16, 4), (scenario_id, "tie trace limb shape"))
		actors = [int(actor) for actor in _as_list(result.events.actors) if actor >= 0]
		self.assertEqual(["p1" if actor == 0 else "p2" for actor in actors], turn["actionOrder"], (scenario_id, "order"))
		self.assertEqual(int(result.draws.count), len(turn["draws"]), (scenario_id, "draw count"))
		self.assertLessEqual(len(turn["draws"]), 16, scenario_id)
		action_draws = [draw for draw in turn["draws"] if draw["context"] != "speed-sort"]
		self.assertEqual(len(action_draws), len(actors) * 3, (scenario_id, "action draw accounting"))
		self.assertEqual(len(turn["draws"]), len(action_draws) + sum(draw["context"] == "speed-sort" for draw in turn["draws"]))
		for draw_index, expected in enumerate(turn["draws"]):
			actual = result.draws
			self.assertEqual(int(actual.bounds[draw_index]), 2 if expected["context"] == "speed-sort" else expected["bound"], (scenario_id, draw_index, "bound"))
			self.assertEqual(int(actual.values[draw_index]), expected["value"], (scenario_id, draw_index, "value"))
			self.assertEqual(int(actual.contexts[draw_index]), DRAW_CONTEXTS[expected["phase"]], (scenario_id, draw_index, "context"))
			self.assertEqual(_as_list(actual.before[draw_index]), expected["before"], (scenario_id, draw_index, "before limbs"))
			self.assertEqual(_as_list(actual.after[draw_index]), expected["after"], (scenario_id, draw_index, "after limbs"))
			if draw_index:
				self.assertEqual(expected["before"], turn["draws"][draw_index - 1]["after"], (scenario_id, draw_index, "native chain"))
		self.assertEqual(_as_list(result.events.executed), [index < len(actors) for index in range(2)], (scenario_id, "event mask"))
		for audience, side in ((0, "p1"), (1, "p2")):
			view = project_pound_view(_state(turn["before"]), result, audience)
			self.assertEqual(
				{"request": turn["views"][side]["request"], "events": turn["views"][side]["events"]},
				_jax_view_projection(view),
				(scenario_id, "view", side),
			)
			self.assertNotIn("rng", view._fields)
			self.assertNotIn("draws", view._fields)
			for action_index, actor in enumerate(_as_list(view.action_order)):
				if actor < 0:
					continue
				target = int(view.target_player[action_index])
				if target == audience:
					observed_loss = max(0, turn["before"]["hp"][target] - turn["after"]["hp"][target])
					self.assertTrue(bool(view.damage_visible[action_index]), (scenario_id, side, action_index, "visible"))
					self.assertEqual(int(view.damage_value[action_index]), observed_loss, (scenario_id, side, action_index, "loss"))
				else:
					self.assertFalse(bool(view.damage_visible[action_index]), (scenario_id, side, action_index, "hidden"))
					self.assertEqual(int(view.damage_value[action_index]), -1, (scenario_id, side, action_index, "hidden damage"))
		for action_index, actor in enumerate(actors):
			critical_draw = action_draws[action_index * 3 + 1]
			damage_draw = action_draws[action_index * 3 + 2]
			critical = critical_draw["value"] == 0
			self.assertEqual(bool(result.events.critical[action_index]), critical, (scenario_id, action_index, "critical"))
			pound_damage = _hand_damage(
				turn["before"]["level"][actor],
				turn["before"]["attack"][actor],
				turn["before"]["defense"][1 - actor],
				critical,
				damage_draw["value"],
			)
			self.assertEqual(int(result.events.damage[action_index]), pound_damage, (scenario_id, action_index, "independent Pound damage"))
			if not critical:
				tackle_damage = _hand_damage(
					turn["before"]["level"][actor],
					turn["before"]["attack"][actor],
					turn["before"]["defense"][1 - actor],
					False,
					damage_draw["value"],
					base_power=50,
				)
				if scenario_id == "p2-first-surviving" and action_index == 0:
					self.assertNotEqual(pound_damage, tackle_damage, (scenario_id, action_index, "Pound power must differ from Tackle"))
		after_state = _state(turn["after"])
		if turn["nextRequests"]["p1"] is None:
			self.assertTrue(bool(after_state.terminal), (scenario_id, "terminal has no next requests"))
			self.assertIsNone(turn["nextRequests"]["p2"])
			for audience in (0, 1):
				idle = _empty_tie_transition(after_state, STATUS_TERMINAL)
				view = project_pound_view(after_state, idle, audience)
				self.assertFalse(bool(view.legal_move_mask[0]), (scenario_id, audience, "terminal request"))
		else:
			idle = _empty_tie_transition(after_state, STATUS_PENDING)
			for audience, side in ((0, "p1"), (1, "p2")):
				view = project_pound_view(after_state, idle, audience)
				self.assertEqual(turn["nextRequests"][side], _jax_view_projection(view)["request"], (scenario_id, side, "next request"))

	def test_pinned_pound_turns_match_native_state_rng_events_and_both_views(self):
		self.assertEqual(self.fixture["kind"], "gen5-restricted-neutral-pound-turn-oracle")
		self.assertEqual(self.fixture["variant"], POUND_VARIANT_NAME)
		self.assertEqual(self.fixture["generation"], 5)
		self.assertEqual(self.fixture["upstream"]["commit"], "1242c1e2d9763ba5b85d33f13aecb89bae19ae3d")
		self.assertEqual((POUND_BASE_POWER, POUND_ACCURACY, POUND_BASE_PP, POUND_MAX_PP, POUND_CRITICAL_RATIO), (40, 100, 35, 56, 1))
		self.assertEqual(ACTION_POUND, 1)
		self.assertEqual(len(self.scenarios), 5)
		self.assertEqual(self.scenarios["p2-first-surviving"]["turns"][0]["actionOrder"][0], "p2")
		self.assertEqual(self.scenarios["p1-first-surviving"]["turns"][0]["actionOrder"][0], "p1")
		self.assertEqual(self.scenarios["equal-speed-tied-turn"]["initial"]["state"]["speed"][0], self.scenarios["equal-speed-tied-turn"]["initial"]["state"]["speed"][1])
		critical_turn = self.scenarios["native-critical-hit"]["turns"][0]
		self.assertTrue(any(draw["context"] == "critical" and draw["value"] == 0 for draw in critical_turn["draws"]))
		self.assertTrue(any(event["type"] == "critical" for event in critical_turn["views"]["p1"]["events"]))
		ko_turn = self.scenarios["first-action-ko"]["turns"][0]
		self.assertEqual(ko_turn["actionOrder"], ["p2"])
		self.assertEqual(ko_turn["after"]["hp"][0], 0)
		ko_condition = next(event["condition"] for event in ko_turn["events"]["p1"] if event["type"] == "damage")
		self.assertEqual(ko_condition, "0 fnt")
		self.assertTrue(ko_turn["after"]["terminal"])

		compiled = jax.jit(transition_pound)
		compiled_view = jax.jit(project_pound_view, static_argnums=2)
		for scenario in self.fixture["scenarios"]:
			initial = _state(scenario["initial"]["state"])
			self.assertEqual(_as_list(initial.pp), scenario["initial"]["state"]["pp"])
			self.assertTrue(all(int(value) == POUND_MAX_PP for value in initial.pp), "pinned Gen 5 request supplies maximum Pound PP")
			self.assertEqual(int(initial.variant_id), POUND_VARIANT_ID)
			self.assertEqual(int(initial.move_id), MOVE_POUND)
			turn = scenario["turns"][0]
			_assert_state(self, initial, turn["before"], (scenario["id"], "before"))
			choices = jnp.asarray([ACTION_POUND, ACTION_POUND], dtype=jnp.int32)
			eager = transition_pound(initial, choices)
			jit_result = compiled(initial, choices)
			self.assertTrue(_leaf_equal(eager, jit_result), (scenario["id"], "eager/JIT"))
			if scenario["id"] == "first-action-ko":
				self.assertGreater(int(eager.events.damage[0]), turn["before"]["hp"][0], "trusted Pound damage retains KO overkill")
			self._assert_turn(eager, turn, scenario["id"])
			for audience in (0, 1):
				self.assertTrue(
					_leaf_equal(
						compiled_view(initial, eager, audience),
						project_pound_view(initial, eager, audience),
					),
					(scenario["id"], audience, "compiled view"),
				)
			self.assertTrue(_leaf_equal(initial, _state(turn["before"])), (scenario["id"], "input unchanged"))

	def test_variant_boundaries_no_advance_and_structure_validation(self):
		turn = self.scenarios["p2-first-surviving"]["turns"][0]
		pound = _state(turn["before"])
		choices = jnp.asarray([1, 1], dtype=jnp.int32)
		for candidate, action, status in (
			(pound, [-1, 1], STATUS_INVALID_CHOICE),
			(pound, [0, 1], STATUS_PENDING),
			(pound, [2, 1], STATUS_UNSUPPORTED),
			(pound._replace(request_phase=jnp.asarray(PHASE_PENDING, dtype=jnp.int32)), [1, 1], STATUS_PENDING),
		):
			result = transition_pound(candidate, np.asarray(action, dtype=np.int32))
			self.assertEqual(int(result.status), status)
			self.assertTrue(_leaf_equal(result.state, candidate))
			self.assertEqual(int(result.draws.count), 0)
			self.assertFalse(any(_as_list(result.events.executed)))

		terminal_snapshot = self.scenarios["first-action-ko"]["turns"][0]["after"]
		terminal = _state(terminal_snapshot)
		terminal_result = transition_pound(terminal, choices)
		self.assertEqual(int(terminal_result.status), STATUS_TERMINAL)
		self.assertTrue(_leaf_equal(terminal_result.state, terminal))
		self.assertEqual(int(terminal_result.draws.count), 0)

		tackle = pound._replace(
			variant_id=jnp.asarray(TACKLE_VARIANT_ID, dtype=jnp.int32),
			move_id=jnp.asarray(MOVE_TACKLE, dtype=jnp.int32),
		)
		for result in (transition_pound(tackle, choices), transition_tackle(pound, choices), transition_with_speed_ties(pound, choices)):
			self.assertEqual(int(result.status), STATUS_UNSUPPORTED)
			self.assertTrue(_leaf_equal(result.state, tackle if int(result.state.variant_id) == TACKLE_VARIANT_ID else pound))
			self.assertEqual(int(result.draws.count), 0)
			self.assertFalse(any(_as_list(result.events.executed)))

		for malformed, exception, message in (
			(pound._replace(rng=pound.rng[:3]), ValueError, "rng must have shape"),
			(pound._replace(rng=pound.rng.astype(jnp.int32)), TypeError, "rng must have dtype"),
		):
			with self.assertRaisesRegex(exception, message):
				transition_pound(malformed, choices)
		with self.assertRaises(ValueError):
			transition_pound(pound, [1, 1, 1])

		valid = transition_pound(pound, choices)
		for audience in (0, 1):
			view = project_pound_view(pound, valid, audience)
			self.assertTrue(bool(view.legal_move_mask[0]))
			cross_variant = project_pound_view(tackle, transition_pound(tackle, choices), audience)
			self.assertFalse(bool(cross_variant.legal_move_mask[0]))
		with self.assertRaisesRegex(ValueError, "rng must have shape"):
			project_pound_view(pound._replace(rng=pound.rng[:3]), valid, 0)

	def test_compiled_heterogeneous_lanes_are_independent(self):
		scenarios = self.scenarios
		p2_first = _state(scenarios["p2-first-surviving"]["turns"][0]["before"])
		p1_first = _state(scenarios["p1-first-surviving"]["turns"][0]["before"])
		tied = _state(scenarios["equal-speed-tied-turn"]["turns"][0]["before"])
		ko = _state(scenarios["first-action-ko"]["turns"][0]["before"])
		terminal = _state(scenarios["first-action-ko"]["turns"][0]["after"])
		unsupported = p2_first._replace(
			variant_id=jnp.asarray(TACKLE_VARIANT_ID, dtype=jnp.int32),
			move_id=jnp.asarray(MOVE_TACKLE, dtype=jnp.int32),
		)
		pending = p2_first._replace(request_phase=jnp.asarray(PHASE_PENDING, dtype=jnp.int32))
		states = [p2_first, p1_first, tied, ko, unsupported, pending, terminal]
		choices = jnp.asarray([[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [0, 1], [1, 1]], dtype=jnp.int32)
		batch = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *states)
		compiled_batch = jax.jit(jax.vmap(transition_pound))(batch, choices)
		self.assertEqual(
			_as_list(compiled_batch.status),
			[STATUS_OK, STATUS_OK, STATUS_OK, STATUS_OK, STATUS_UNSUPPORTED, STATUS_PENDING, STATUS_TERMINAL],
		)
		for lane, state in enumerate(states):
			separate = transition_pound(state, choices[lane])
			batched_lane = jax.tree_util.tree_map(lambda value: value[lane], compiled_batch)
			self.assertTrue(_leaf_equal(batched_lane, separate), (lane, "batch equals scalar"))
			if lane >= 4:
				self.assertTrue(_leaf_equal(batched_lane.state, state), (lane, "rejected state frozen"))
				self.assertEqual(int(batched_lane.draws.count), 0, (lane, "rejected draws empty"))
				self.assertFalse(any(_as_list(batched_lane.events.executed)), (lane, "rejected events empty"))
		for lane in (4, 5, 6):
			self.assertTrue(_leaf_equal(jax.tree_util.tree_map(lambda value: value[lane], compiled_batch.state), states[lane]))
			self.assertEqual(int(compiled_batch.draws.count[lane]), 0)
			self.assertEqual(int(compiled_batch.state.rng[lane, 0]), int(states[lane].rng[0]))
		permutation = np.asarray([6, 2, 4, 0, 5, 1, 3])
		reordered = jax.jit(jax.vmap(transition_pound))(
			jax.tree_util.tree_map(lambda value: value[permutation], batch), choices[permutation]
		)
		self.assertTrue(
			_leaf_equal(reordered, jax.tree_util.tree_map(lambda value: value[permutation], compiled_batch)),
			"batch lane reorder preserves results",
		)


if __name__ == "__main__":
	unittest.main()
