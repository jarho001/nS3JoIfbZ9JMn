from __future__ import annotations

import json
import subprocess
import typing
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import simulator
from simulator.neutral_mega_punch import (
	DRAW_ACCURACY,
	DRAW_CRITICAL,
	DRAW_DAMAGE,
	MEGA_PUNCH_ACCURACY,
	MEGA_PUNCH_BASE_POWER,
	MEGA_PUNCH_BASE_PP,
	MEGA_PUNCH_CRITICAL_RATIO,
	MEGA_PUNCH_MAX_PP,
	MOVE_MEGA_PUNCH,
	MegaPunchDrawTrace,
	MegaPunchEvents,
	MegaPunchPlayerView,
	MegaPunchTransition,
	VARIANT_ID,
	VARIANT_NAME,
	make_mega_punch_state,
	project_mega_punch_view,
	transition_mega_punch,
)
from simulator.neutral_tackle import (
	PHASE_MOVE_REQUEST,
	PHASE_PENDING,
	PHASE_TERMINAL,
	PHASE_UNSUPPORTED,
	STATUS_INVALID_CHOICE,
	STATUS_OK,
	STATUS_PENDING,
	STATUS_TERMINAL,
	STATUS_UNSUPPORTED,
)


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "reference" / "neutral-mega-punch-oracle.cjs"


def _side_index(side: str) -> int:
	return 0 if side == "p1" else 1


def _phase_for_native_after(after):
	if after["terminal"]:
		return PHASE_TERMINAL
	if any(pp == 0 for pp in after["pp"]):
		return PHASE_UNSUPPORTED
	return PHASE_MOVE_REQUEST


def _make_state(native, *, phase=None):
	return make_mega_punch_state(
		hp=native["hp"],
		max_hp=native["maxHp"],
		pp=native["pp"],
		level=native["level"],
		attack=native["attack"],
		defense=native["defense"],
		speed=native["speed"],
		rng=native["rng"],
		turn=native["turn"],
		request_phase=phase if phase is not None else PHASE_MOVE_REQUEST,
		terminal=native["terminal"],
		winner=native["winner"],
	)


def _manual_damage(before, actor, critical, roll):
	target = 1 - actor
	level_term = (2 * before["level"][actor]) // 5 + 2
	core = ((level_term * MEGA_PUNCH_BASE_POWER * before["attack"][actor]) // before["defense"][target]) // 50
	pre_roll = core + 2
	if critical:
		pre_roll *= 2
	return max(1, ((pre_roll * (100 - roll)) // 100) & 0xFFFF)


def _jnp_array(values, dtype):
	return jnp.asarray(values, dtype=dtype)


def _native_expected(scenario):
	turn = scenario["turn"]
	before = turn["before"]
	after = turn["after"]
	state = _make_state(after, phase=_phase_for_native_after(after))
	missed_actors = {
		event["actor"] for event in turn["events"]["p1"] if event["type"] == "miss"
	}
	critical_targets = {
		event["target"] for event in turn["events"]["p1"] if event["type"] == "critical"
	}
	executed = [False, False]
	actors = [-1, -1]
	hit = [False, False]
	missed = [False, False]
	critical = [False, False]
	damage = [0, 0]
	target_hp = [0, 0]
	target_max_hp = [0, 0]
	for slot, actor_side in enumerate(turn["actionOrder"]):
		actor = _side_index(actor_side)
		target = 1 - actor
		target_side = "p1" if target == 0 else "p2"
		is_miss = actor_side in missed_actors
		executed[slot] = True
		actors[slot] = actor
		hit[slot] = not is_miss
		missed[slot] = is_miss
		critical[slot] = (target_side in critical_targets) and not is_miss
		if not is_miss:
			# The two-turn fixture has at most two action attempts. Associate this
			# actor's hit roll by walking the ordered native draw prefix below.
			position = 0
			for current_side in turn["actionOrder"]:
				current_actor = _side_index(current_side)
				current_missed = current_side in missed_actors
				if current_actor == actor:
					critical_roll = turn["draws"][position + 1]["value"]
					roll = turn["draws"][position + 2]["value"]
					damage[slot] = _manual_damage(before, actor, critical_roll == 0, roll)
					break
				position += 1 if current_missed else 3
			else:
				raise AssertionError(f"No native draws found for actor {actor_side}.")
		target_hp[slot] = after["hp"][target]
		target_max_hp[slot] = before["maxHp"][target]
	events = MegaPunchEvents(
		_jnp_array(executed, jnp.bool_),
		_jnp_array(actors, jnp.int32),
		_jnp_array(hit, jnp.bool_),
		_jnp_array(missed, jnp.bool_),
		_jnp_array(critical, jnp.bool_),
		_jnp_array(damage, jnp.int32),
		_jnp_array(target_hp, jnp.int32),
		_jnp_array(target_max_hp, jnp.int32),
	)
	context_ids = {"accuracy": DRAW_ACCURACY, "critical": DRAW_CRITICAL, "damage": DRAW_DAMAGE}
	draws = turn["draws"]
	count = len(draws)
	bounds = [0] * 6
	values = [-1] * 6
	contexts = [0] * 6
	seed_before = [[0] * 4 for _ in range(6)]
	seed_after = [[0] * 4 for _ in range(6)]
	for index, draw in enumerate(draws):
		bounds[index] = draw["bound"]
		values[index] = draw["value"]
		contexts[index] = context_ids[draw["context"]]
		seed_before[index] = draw["before"]
		seed_after[index] = draw["after"]
	draw_trace = MegaPunchDrawTrace(
		jnp.asarray(count, dtype=jnp.int32),
		_jnp_array(bounds, jnp.int32),
		_jnp_array(values, jnp.int32),
		_jnp_array(contexts, jnp.int32),
		_jnp_array(seed_before, jnp.uint16),
		_jnp_array(seed_after, jnp.uint16),
	)
	return MegaPunchTransition(state, jnp.asarray(STATUS_OK, dtype=jnp.int32), events, draw_trace)


def _projection_to_native_json(before, result, audience):
	projected = project_mega_punch_view(before, result, audience)
	view = projected.player_view
	own = int(view.audience)
	request = {
		"phase": "move",
		"legalActions": ["move 1"] if bool(np.asarray(view.legal_move_mask)[0]) else [],
		"boundary": "mega-punch",
		"own": {
			"species": "Mew",
			"level": int(view.own_level),
			"hpCondition": f"{int(view.own_hp)}/{int(view.own_max_hp)}",
			"stats": {"atk": int(view.own_attack), "def": int(view.own_defense), "spe": int(view.own_speed)},
			"move": {"name": "Mega Punch", "id": "megapunch", "pp": int(view.own_pp), "disabled": False},
		},
	}
	events = []
	actors = np.asarray(view.action_order).tolist()
	misses = np.asarray(projected.missed).tolist()
	critical = np.asarray(view.critical).tolist()
	targets = np.asarray(view.target_player).tolist()
	hp_values = np.asarray(view.target_hp_value).tolist()
	hp_percent = np.asarray(view.target_hp_is_percent).tolist()
	fainted = np.asarray(view.fainted).tolist()
	for slot, actor in enumerate(actors):
		if actor < 0:
			continue
		actor_id = "p1" if actor == 0 else "p2"
		target_id = "p1" if targets[slot] == 0 else "p2"
		events.append({"type": "move", "actor": actor_id, "move": "mega punch", "target": target_id})
		if misses[slot]:
			events.append({"type": "miss", "actor": actor_id, "target": target_id})
			continue
		if critical[slot]:
			events.append({"type": "critical", "target": target_id})
		if hp_values[slot] == 0:
			condition = "0 fnt"
		elif hp_percent[slot]:
			condition = f"{hp_values[slot]}/100"
		else:
			condition = f"{hp_values[slot]}/{int(view.own_max_hp)}"
		events.append({"type": "damage", "target": target_id, "condition": condition})
		if fainted[slot]:
			events.append({"type": "faint", "target": target_id})
	if bool(view.terminal):
		winner = int(view.winner)
		events.append({"type": "winner", "side": "p1" if winner == 1 else "p2" if winner == 2 else "tie"})
	elif bool(view.turn_event):
		events.append({"type": "turn", "number": int(view.turn_number)})
	return {"request": request, "events": events}


def _empty_expected(state, status):
	return MegaPunchTransition(
		state,
		jnp.asarray(status, dtype=jnp.int32),
		MegaPunchEvents(
			jnp.zeros((2,), dtype=jnp.bool_),
			jnp.full((2,), -1, dtype=jnp.int32),
			jnp.zeros((2,), dtype=jnp.bool_),
			jnp.zeros((2,), dtype=jnp.bool_),
			jnp.zeros((2,), dtype=jnp.bool_),
			jnp.zeros((2,), dtype=jnp.int32),
			jnp.zeros((2,), dtype=jnp.int32),
			jnp.zeros((2,), dtype=jnp.int32),
		),
		MegaPunchDrawTrace(
			jnp.asarray(0, dtype=jnp.int32),
			jnp.zeros((6,), dtype=jnp.int32),
			jnp.full((6,), -1, dtype=jnp.int32),
			jnp.zeros((6,), dtype=jnp.int32),
			jnp.zeros((6, 4), dtype=jnp.uint16),
			jnp.zeros((6, 4), dtype=jnp.uint16),
		),
	)


def _stack_pytrees(trees):
	return jax.tree_util.tree_map(lambda *values: jnp.stack(values), *trees)


def _assert_tree_equal(test, actual, expected, label):
	actual_leaves, actual_def = jax.tree_util.tree_flatten(actual)
	expected_leaves, expected_def = jax.tree_util.tree_flatten(expected)
	test.assertEqual(actual_def, expected_def, label)
	test.assertEqual(len(actual_leaves), len(expected_leaves), label)
	for index, (actual_leaf, expected_leaf) in enumerate(zip(actual_leaves, expected_leaves)):
		np.testing.assert_array_equal(np.asarray(actual_leaf), np.asarray(expected_leaf), f"{label} leaf {index}")


class NeutralMegaPunchTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		completed = subprocess.run(
			["node", str(ORACLE)],
			cwd=ROOT,
			check=True,
			capture_output=True,
			text=True,
			timeout=20,
		)
		cls.fixture = json.loads(completed.stdout)

	def test_native_turns_and_heterogeneous_compiled_batch(self):
		fixture = self.fixture
		self.assertEqual(fixture["variant"], VARIANT_NAME)
		self.assertEqual(fixture["upstream"]["commit"], "1242c1e2d9763ba5b85d33f13aecb89bae19ae3d")
		self.assertEqual(
			fixture["moveFacts"],
			{
				"name": "Mega Punch",
				"id": "megapunch",
				"type": "Normal",
				"category": "Physical",
				"basePower": MEGA_PUNCH_BASE_POWER,
				"accuracy": MEGA_PUNCH_ACCURACY,
				"basePp": MEGA_PUNCH_BASE_PP,
				"maxPp": MEGA_PUNCH_MAX_PP,
				"criticalRatio": MEGA_PUNCH_CRITICAL_RATIO,
			},
		)
		self.assertEqual(len(fixture["scenarios"]), 6)
		for row in fixture["scenarios"]:
			speeds = row["turn"]["before"]["speed"]
			self.assertTrue(all(speed > 0 for speed in speeds), row["id"])
			self.assertNotEqual(speeds[0], speeds[1], row["id"])

		states = []
		expected = []
		for scenario in fixture["scenarios"]:
			turn = scenario["turn"]
			before = _make_state(turn["before"])
			states.append(before)
			expected.append(_native_expected(scenario))
			self.assertEqual(turn["choices"], {"p1": "move 1", "p2": "move 1"})
			self.assertEqual(turn["requests"]["p1"]["own"]["move"]["id"], "megapunch")
			self.assertEqual(turn["requests"]["p2"]["own"]["move"]["id"], "megapunch")

		by_id = {row["id"]: row["turn"] for row in fixture["scenarios"]}
		self.assertEqual(by_id["p1-first-normal-hit"]["actionOrder"][0], "p1")
		self.assertEqual(by_id["p2-first-critical-hit"]["actionOrder"][0], "p2")
		self.assertIn("miss", [event["type"] for event in by_id["p1-first-misses"]["events"]["p1"]])
		self.assertIn("miss", [event["type"] for event in by_id["p2-first-misses"]["events"]["p1"]])
		self.assertEqual(by_id["first-action-ko-skips-p2"]["actionOrder"], ["p1"])
		last_pp = by_id["last-pp-miss-then-opponent-acts"]
		self.assertEqual(last_pp["before"]["pp"], [1, MEGA_PUNCH_MAX_PP])
		self.assertEqual(last_pp["after"]["pp"][0], 0)
		self.assertEqual(last_pp["nextRequests"]["p1"]["own"]["move"]["id"], "struggle")
		self.assertEqual(last_pp["nextRequests"]["p1"]["boundary"], "struggle-no-pp")
		self.assertEqual(last_pp["after"]["terminal"], False)
		self.assertEqual(last_pp["actionOrder"], ["p1", "p2"])
		self.assertEqual(expected[-1].status, STATUS_OK)
		self.assertEqual(int(expected[-1].state.request_phase), PHASE_UNSUPPORTED)

		# The same compiled vmap includes successful native lanes and every runtime rejection class.
		states.extend([
			states[0]._replace(request_phase=jnp.asarray(PHASE_PENDING, dtype=jnp.int32)),
			states[0],
			states[0],
			states[0]._replace(pp=states[0].pp.at[0].set(0)),
			states[0]._replace(speed=jnp.asarray([int(states[0].speed[0]), int(states[0].speed[0])], dtype=jnp.int32)),
			states[0]._replace(variant_id=jnp.asarray(1, dtype=jnp.int32), move_id=jnp.asarray(33, dtype=jnp.int32)),
			states[0]._replace(variant_id=jnp.asarray(2, dtype=jnp.int32), move_id=jnp.asarray(33, dtype=jnp.int32)),
			states[0]._replace(variant_id=jnp.asarray(3, dtype=jnp.int32), move_id=jnp.asarray(1, dtype=jnp.int32)),
			states[0]._replace(variant_id=jnp.asarray(4, dtype=jnp.int32), move_id=jnp.asarray(5, dtype=jnp.int32)),
			expected[4].state,
			states[0],
		])
		choices = [[1, 1]] * 6 + [
			[1, 1],  # pending state
			[-1, 1],  # invalid negative choice
			[1, 2],  # unknown positive action
			[1, 1],  # final PP unsupported input
			[1, 1],  # speed tie
			[1, 1],  # v1 boundary
			[1, 1],  # v2 boundary
			[1, 1],  # v3 boundary
			[1, 1],  # v4 boundary
			[-1, 2],  # terminal absorption takes precedence
			[0, 1],  # pending choice
		]
		expected.extend([
			_empty_expected(states[6], STATUS_PENDING),
			_empty_expected(states[7], STATUS_INVALID_CHOICE),
			_empty_expected(states[8], STATUS_UNSUPPORTED),
			_empty_expected(states[9], STATUS_UNSUPPORTED),
			_empty_expected(states[10], STATUS_UNSUPPORTED),
			_empty_expected(states[11], STATUS_UNSUPPORTED),
			_empty_expected(states[12], STATUS_UNSUPPORTED),
			_empty_expected(states[13], STATUS_UNSUPPORTED),
			_empty_expected(states[14], STATUS_UNSUPPORTED),
			_empty_expected(states[15], STATUS_TERMINAL),
			_empty_expected(states[16], STATUS_PENDING),
		])
		compiled = jax.jit(jax.vmap(transition_mega_punch))(
			_stack_pytrees(states), jnp.asarray(choices, dtype=jnp.int32)
		)
		actual_native_lanes = []
		for lane, expected_lane in enumerate(expected):
			actual_lane = jax.tree_util.tree_map(lambda leaf: leaf[lane], compiled)
			_assert_tree_equal(self, actual_lane, expected_lane, f"compiled lane {lane}")
			if lane < len(fixture["scenarios"]):
				actual_native_lanes.append(actual_lane)
		for lane, scenario in enumerate(fixture["scenarios"]):
			for audience, side in ((0, "p1"), (1, "p2")):
				self.assertEqual(
					_projection_to_native_json(states[lane], actual_native_lanes[lane], audience),
					scenario["turn"]["views"][side],
					f"{scenario['id']} {side} recipient view",
				)
		self.assertEqual(int(compiled.status[6]), STATUS_PENDING)
		self.assertEqual(int(compiled.status[7]), STATUS_INVALID_CHOICE)
		self.assertEqual(int(compiled.status[8]), STATUS_UNSUPPORTED)
		self.assertEqual(int(compiled.status[15]), STATUS_TERMINAL)
		self.assertEqual(int(compiled.status[16]), STATUS_PENDING)
		self.assertTrue(bool(compiled.state.terminal[15]))

	def test_public_layout_annotations_and_fail_closed_boundaries(self):
		self.assertEqual(VARIANT_ID, 5)
		self.assertEqual(MOVE_MEGA_PUNCH, 5)
		self.assertEqual(VARIANT_NAME, "gen5-neutral-mega-punch-mew-1v1-v5")
		self.assertEqual(simulator.MEGA_PUNCH_DRAW_ACCURACY, DRAW_ACCURACY)
		self.assertEqual(simulator.MEGA_PUNCH_DRAW_CRITICAL, DRAW_CRITICAL)
		self.assertEqual(simulator.MEGA_PUNCH_DRAW_DAMAGE, DRAW_DAMAGE)
		state_hints = typing.get_type_hints(make_mega_punch_state)
		transition_hints = typing.get_type_hints(transition_mega_punch)
		projection_hints = typing.get_type_hints(project_mega_punch_view)
		self.assertIn("hp", state_hints)
		self.assertIn("rng", state_hints)
		self.assertEqual(state_hints["return"].__name__, "BattleState")
		self.assertEqual(set(transition_hints), {"state", "choices", "return"})
		self.assertEqual(transition_hints["return"], MegaPunchTransition)
		self.assertEqual(set(projection_hints), {"before", "result", "audience", "return"})
		self.assertEqual(projection_hints["return"], MegaPunchPlayerView)
		self.assertEqual(tuple(MegaPunchTransition._fields), ("state", "status", "events", "draws"))
		self.assertEqual(tuple(MegaPunchEvents._fields), ("executed", "actors", "hit", "missed", "critical", "damage", "target_hp", "target_max_hp"))
		self.assertEqual(tuple(MegaPunchDrawTrace._fields), ("count", "bounds", "values", "contexts", "before", "after"))

		state = _make_state(self.fixture["scenarios"][0]["turn"]["before"])
		self.assertEqual(state.hp.shape, (2,))
		self.assertEqual(state.rng.shape, (4,))
		self.assertEqual(state.variant_id.shape, ())
		self.assertEqual(state.move_id.shape, ())
		self.assertEqual(state.hp.dtype, jnp.int32)
		self.assertEqual(state.rng.dtype, jnp.uint16)
		self.assertEqual(int(state.variant_id), VARIANT_ID)
		self.assertEqual(int(state.move_id), MOVE_MEGA_PUNCH)
		result = _empty_expected(state, STATUS_UNSUPPORTED)
		with self.assertRaises(ValueError):
			transition_mega_punch(state, [1])
		with self.assertRaises(ValueError):
			transition_mega_punch(state._replace(hp=jnp.zeros((1,), dtype=jnp.int32)), [-1, 1])
		with self.assertRaises(ValueError):
			project_mega_punch_view(state, result, 2)
		with self.assertRaises(TypeError):
			jax.jit(lambda audience: project_mega_punch_view(state, result, audience))(jnp.asarray(0, dtype=jnp.int32))


if __name__ == "__main__":
	unittest.main()
