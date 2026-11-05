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
from simulator.accuracy_choice import (
	ACTION_MEGA_PUNCH,
	ACTION_TACKLE,
	AccuracyChoicePlayerView,
	AccuracyChoiceTransition,
	MEGA_PUNCH_MAX_PP,
	MOVE_IDS,
	VARIANT_ID,
	VARIANT_NAME,
	make_accuracy_choice_state,
	project_accuracy_choice_view,
	transition_accuracy_choice,
)
from simulator.neutral_mega_punch import (
	DRAW_ACCURACY,
	DRAW_CRITICAL,
	DRAW_DAMAGE,
	MegaPunchDrawTrace,
	MegaPunchEvents,
	MegaPunchTransition,
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
from simulator.neutral_two_move import make_two_move_state, project_two_move_view, transition_two_move

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "reference" / "accuracy-choice-oracle.cjs"
CONTEXT_IDS = {"accuracy": DRAW_ACCURACY, "critical": DRAW_CRITICAL, "damage": DRAW_DAMAGE}
MOVE_ID_TO_ACTION = {33: ACTION_TACKLE, 5: ACTION_MEGA_PUNCH}


def _side_index(side: str) -> int:
	return 0 if side == "p1" else 1


def _manual_damage(before, actor: int, power: int, critical: bool, roll: int) -> int:
	target = 1 - actor
	level_term = (2 * before["level"][actor]) // 5 + 2
	core = ((level_term * power * before["attack"][actor]) // before["defense"][target]) // 50
	pre_roll = core + 2
	if critical:
		pre_roll *= 2
	return max(1, ((pre_roll * (100 - roll)) // 100) & 0xFFFF)


def _phase(after):
	if after["terminal"]:
		return PHASE_TERMINAL
	if any(all(pp == 0 for pp in row) for row in after["pp"]):
		return PHASE_UNSUPPORTED
	return PHASE_MOVE_REQUEST


def _state(native, *, phase=None, variant_id=VARIANT_ID, move_ids=MOVE_IDS):
	return make_two_move_state(
		hp=native["hp"], max_hp=native["maxHp"], pp=native["pp"],
		level=native["level"], attack=native["attack"], defense=native["defense"],
		speed=native["speed"], rng=native["rng"], turn=native["turn"],
		request_phase=_phase(native) if phase is None else phase,
		terminal=native["terminal"], winner=native["winner"],
		variant_id=variant_id, move_ids=move_ids,
	)


def _native_expected(scenario):
	turn = scenario["turn"]
	before = turn["before"]
	after = turn["after"]
	choices = [scenario["choices"]["p1"], scenario["choices"]["p2"]]
	miss_actors = {event["actor"] for event in turn["events"]["p1"] if event["type"] == "miss"}
	critical_targets = {_side_index(event["target"]) for event in turn["events"]["p1"] if event["type"] == "critical"}
	executed = [False, False]
	actors = [-1, -1]
	hit = [False, False]
	missed = [False, False]
	critical = [False, False]
	damage = [0, 0]
	target_hp = [0, 0]
	target_max_hp = [0, 0]
	move_ids = [-1, -1]
	draw_cursor = 0
	for row, actor_side in enumerate(turn["actionOrder"]):
		actor = _side_index(actor_side)
		target = 1 - actor
		is_miss = actor_side in miss_actors
		executed[row] = True
		actors[row] = actor
		hit[row] = not is_miss
		missed[row] = is_miss
		critical[row] = target in critical_targets and not is_miss
		if not is_miss:
			damage_roll = turn["draws"][draw_cursor + 2]["value"]
			power = 50 if choices[actor] == ACTION_TACKLE else 80
			damage[row] = _manual_damage(before, actor, power, critical[row], damage_roll)
		target_hp[row] = before["hp"][target] if is_miss else after["hp"][target]
		target_max_hp[row] = before["maxHp"][target]
		move_ids[row] = MOVE_IDS[choices[actor] - 1]
		draw_cursor += 1 if is_miss else 3
	events = MegaPunchEvents(
		jnp.asarray(executed, dtype=jnp.bool_), jnp.asarray(actors, dtype=jnp.int32),
		jnp.asarray(hit, dtype=jnp.bool_), jnp.asarray(missed, dtype=jnp.bool_),
		jnp.asarray(critical, dtype=jnp.bool_), jnp.asarray(damage, dtype=jnp.int32),
		jnp.asarray(target_hp, dtype=jnp.int32), jnp.asarray(target_max_hp, dtype=jnp.int32),
	)
	draws_native = turn["draws"]
	count = len(draws_native)
	bounds = [0] * 6
	values = [-1] * 6
	contexts = [0] * 6
	seed_before = [[0] * 4 for _ in range(6)]
	seed_after = [[0] * 4 for _ in range(6)]
	for index, draw in enumerate(draws_native):
		bounds[index] = draw["bound"]
		values[index] = draw["value"]
		contexts[index] = CONTEXT_IDS[draw["context"]]
		seed_before[index] = draw["before"]
		seed_after[index] = draw["after"]
	draws = MegaPunchDrawTrace(
		jnp.asarray(count, dtype=jnp.int32), jnp.asarray(bounds, dtype=jnp.int32),
		jnp.asarray(values, dtype=jnp.int32), jnp.asarray(contexts, dtype=jnp.int32),
		jnp.asarray(seed_before, dtype=jnp.uint16), jnp.asarray(seed_after, dtype=jnp.uint16),
	)
	state = _state(after)
	return state, AccuracyChoiceTransition(
		state, jnp.asarray(STATUS_OK, dtype=jnp.int32), events, draws,
		jnp.asarray(move_ids, dtype=jnp.int32),
	)


def _assert_tree_equal(test, actual, expected, label):
	actual_leaves = jax.tree_util.tree_leaves(actual)
	expected_leaves = jax.tree_util.tree_leaves(expected)
	test.assertEqual(len(actual_leaves), len(expected_leaves), label)
	for index, (left, right) in enumerate(zip(actual_leaves, expected_leaves)):
		try:
			np.testing.assert_array_equal(np.asarray(left), np.asarray(right), err_msg=f"{label} leaf {index}")
		except AssertionError as error:
			test.fail(str(error))


def _stack_pytrees(trees):
	return jax.tree_util.tree_map(lambda *values: jnp.stack(values), *trees)


def _empty_result(state, status):
	return AccuracyChoiceTransition(
		state, jnp.asarray(status, dtype=jnp.int32),
		MegaPunchEvents(
			jnp.zeros((2,), dtype=jnp.bool_), jnp.full((2,), -1, dtype=jnp.int32),
			jnp.zeros((2,), dtype=jnp.bool_), jnp.zeros((2,), dtype=jnp.bool_),
			jnp.zeros((2,), dtype=jnp.bool_), jnp.zeros((2,), dtype=jnp.int32),
			jnp.zeros((2,), dtype=jnp.int32), jnp.zeros((2,), dtype=jnp.int32),
		),
		MegaPunchDrawTrace(
			jnp.asarray(0, dtype=jnp.int32), jnp.zeros((6,), dtype=jnp.int32),
			jnp.full((6,), -1, dtype=jnp.int32), jnp.zeros((6,), dtype=jnp.int32),
			jnp.zeros((6, 4), dtype=jnp.uint16), jnp.zeros((6, 4), dtype=jnp.uint16),
		),
		jnp.full((2,), -1, dtype=jnp.int32),
	)


class AccuracyChoiceTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		completed = subprocess.run(
			["node", str(ORACLE)], cwd=ROOT, check=True, capture_output=True, text=True, timeout=20,
		)
		cls.fixture = json.loads(completed.stdout)

	def _batch_inputs(self):
		native_cases = []
		for scenario in self.fixture["scenarios"]:
			_, expected = _native_expected(scenario)
			state = _state(scenario["turn"]["before"])
			native_cases.append((scenario["id"], state, scenario["choices"], expected, scenario))
			continuation = scenario.get("continuation")
			if continuation is not None:
				cont_state = _state(continuation["before"])
				_, cont_expected = _native_expected({**scenario, "choices": continuation["choices"], "turn": continuation})
				native_cases.append((f"{scenario['id']}-follow-up", cont_state, continuation["choices"], cont_expected, {**scenario, "turn": continuation, "choices": continuation["choices"]}))
		states = [case[1] for case in native_cases]
		choices = [jnp.asarray([case[2]["p1"], case[2]["p2"]], dtype=jnp.int32) for case in native_cases]
		statuses = [STATUS_OK] * len(states)
		labels = [case[0] for case in native_cases]
		expected = [case[3] for case in native_cases]

		base = states[0]
		pending = base._replace(request_phase=jnp.asarray(PHASE_PENDING, dtype=jnp.int32))
		invalid_zero_pp = base._replace(pp=base.pp.at[0, 0].set(0))
		all_pp_zero = base._replace(pp=base.pp.at[0].set(jnp.asarray([0, 0], dtype=jnp.int32)))
		tie = base._replace(speed=base.speed.at[1].set(base.speed[0]))
		bad_cap = base._replace(pp=base.pp.at[0, 1].set(MEGA_PUNCH_MAX_PP + 1))
		cross_variant = make_two_move_state(
			hp=base.hp, max_hp=base.max_hp, pp=base.pp, level=base.level, attack=base.attack,
			defense=base.defense, speed=base.speed, rng=base.rng, turn=base.turn,
			request_phase=base.request_phase, terminal=False, winner=-1, variant_id=4, move_ids=(33, 1),
		)
		terminal = base._replace(terminal=jnp.asarray(True), request_phase=jnp.asarray(PHASE_TERMINAL, dtype=jnp.int32), winner=jnp.asarray(1, dtype=jnp.int32))
		lanes = [
			("pending", pending, [1, 1], STATUS_PENDING),
			("invalid-negative", base, [-1, 1], STATUS_INVALID_CHOICE),
			("invalid-zero-pp-slot", invalid_zero_pp, [1, 1], STATUS_INVALID_CHOICE),
			("unsupported-all-pp-zero", all_pp_zero, [1, 1], STATUS_UNSUPPORTED),
			("unsupported-tie", tie, [1, 1], STATUS_UNSUPPORTED),
			("unsupported-pp-cap", bad_cap, [1, 1], STATUS_UNSUPPORTED),
			("cross-variant-v4", cross_variant, [1, 1], STATUS_UNSUPPORTED),
			("unsupported-positive-choice", base, [3, 1], STATUS_UNSUPPORTED),
			("terminal-absorbs-invalid", terminal, [-1, 3], STATUS_TERMINAL),
		]
		for label, state, action, status in lanes:
			labels.append(label)
			states.append(state)
			choices.append(jnp.asarray(action, dtype=jnp.int32))
			statuses.append(status)
			expected.append(_empty_result(state, status))
		return labels, states, choices, expected, statuses, native_cases

	def test_native_facts_and_fixed_scope(self):
		self.assertEqual(self.fixture["variant"], VARIANT_NAME)
		self.assertEqual(self.fixture["moveFacts"]["tackle"]["maxPp"], 56)
		self.assertEqual(self.fixture["moveFacts"]["megaPunch"]["maxPp"], 32)
		self.assertEqual(len(self.fixture["scenarios"]), 8)
		self.assertEqual(self.fixture["allPpZeroBoundary"]["nativeRequest"]["boundary"], "native-struggle-no-pp")
		self.assertIn("STATUS_UNSUPPORTED", self.fixture["allPpZeroBoundary"]["jaxBoundary"])
		ids = {scenario["id"] for scenario in self.fixture["scenarios"]}
		self.assertIn("last-mega-punch-misses-then-tackle-remains", ids)
		self.assertIn("last-tackle-pp-leaves-mega-punch", ids)
		critical = next(scenario for scenario in self.fixture["scenarios"] if scenario["id"] == "mega-punch-critical-seed-21")
		for side in ("p1", "p2"):
			self.assertIn("critical", [event["type"] for event in critical["turn"]["events"][side]])

	def test_heterogeneous_native_and_rejection_batch(self):
		labels, states, choices, expected, statuses, native_cases = self._batch_inputs()
		batched_states = _stack_pytrees(states)
		batched_choices = jnp.stack(choices)
		compiled = jax.jit(jax.vmap(transition_accuracy_choice))
		actual = compiled(batched_states, batched_choices)
		for index, (label, expected_result, status) in enumerate(zip(labels, expected, statuses)):
			actual_result = jax.tree_util.tree_map(lambda value: value[index], actual)
			_assert_tree_equal(self, actual_result, expected_result, label)
			self.assertEqual(int(actual_result.status), status, label)
			if status != STATUS_OK:
				_assert_tree_equal(self, actual_result.state, states[index], f"{label} rejected-state isolation")
				self.assertEqual(int(actual_result.draws.count), 0, label)
				self.assertEqual(np.asarray(actual_result.event_move_ids).tolist(), [-1, -1], label)
		for index, (label, state, action, expected_result, scenario) in enumerate(native_cases):
			batch_index = index
			result = jax.tree_util.tree_map(lambda value: value[batch_index], actual)
			self._assert_recipients(scenario, state, result)
			if label == "last-mega-punch-misses-then-tackle-remains":
				self.assertEqual(int(result.draws.values[0]), 93)
				self.assertEqual(bool(result.events.missed[0]), True)
				self.assertEqual(int(result.event_move_ids[0]), 5)
				self.assertEqual(int(result.state.pp[0, 1]), 0)
				self.assertEqual(int(result.state.pp[0, 0]), int(state.pp[0, 0]))
			if label == "first-action-ko-skips-other-choice":
				self.assertEqual(np.asarray(result.events.executed).tolist(), [True, False])
				self.assertEqual(np.asarray(result.event_move_ids).tolist(), [5, -1])
			if label.endswith("-follow-up"):
				row = np.flatnonzero(np.asarray(result.events.actors) == 0)[0]
				self.assertEqual(int(result.event_move_ids[row]), 33)

	def _assert_recipients(self, scenario, before_state, result):
		turn = scenario["turn"]

		def assert_request_view(player_view, request):
			own_request = request["own"]
			hp, max_hp = (int(value) for value in own_request["hpCondition"].split("/"))
			self.assertEqual(int(player_view.own_hp), hp)
			self.assertEqual(int(player_view.own_max_hp), max_hp)
			self.assertEqual(int(player_view.own_level), own_request["level"])
			self.assertEqual(int(player_view.own_attack), own_request["stats"]["atk"])
			self.assertEqual(int(player_view.own_defense), own_request["stats"]["def"])
			self.assertEqual(int(player_view.own_speed), own_request["stats"]["spe"])
			self.assertEqual(np.asarray(player_view.own_pp).tolist(), [row["pp"] for row in request["moves"]])
			self.assertEqual(np.asarray(player_view.own_move_ids).tolist(), [33, 5])
			self.assertEqual(np.asarray(player_view.legal_move_mask).tolist(), [not move["disabled"] for move in request["moves"]])

		for audience, side in ((0, "p1"), (1, "p2")):
			view = project_accuracy_choice_view(before_state, result, audience)
			self.assertIsInstance(view, AccuracyChoicePlayerView)
			request = turn["requests"][side]
			own = view.player_view
			native_events = turn["events"][side]
			native_moves = [event for event in native_events if event["type"] == "move"]
			native_misses = {event["actor"] for event in native_events if event["type"] == "miss"}
			native_criticals = {event["target"] for event in native_events if event["type"] == "critical"}
			native_faints = {event["target"] for event in native_events if event["type"] == "faint"}
			native_damage = {event["target"]: event["condition"] for event in native_events if event["type"] == "damage"}
			assert_request_view(own, request)
			self.assertEqual([event["actor"] for event in native_moves], turn["actionOrder"])
			expected_order = [int(event["actor"] == "p2") for event in native_moves] + [-1] * (2 - len(native_moves))
			expected_targets = [_side_index(event["target"]) for event in native_moves] + [-1] * (2 - len(native_moves))
			expected_move_ids = [{"tackle": 33, "mega punch": 5}[event["move"]] for event in native_moves] + [-1] * (2 - len(native_moves))
			expected_missed = [event["actor"] in native_misses for event in native_moves] + [False] * (2 - len(native_moves))
			self.assertEqual(np.asarray(own.action_order).tolist(), expected_order)
			self.assertEqual(np.asarray(own.target_player).tolist(), expected_targets)
			self.assertEqual(np.asarray(own.event_move_ids).tolist(), expected_move_ids)
			self.assertEqual(np.asarray(view.missed).tolist(), expected_missed)
			self.assertEqual(bool(own.turn_event), any(event["type"] == "turn" for event in native_events))
			turn_events = [event for event in native_events if event["type"] == "turn"]
			self.assertEqual(int(own.turn_number), turn_events[0]["number"] if turn_events else 0)
			self.assertEqual(bool(own.terminal), bool(turn["after"]["terminal"]))
			self.assertEqual(int(own.winner), turn["after"]["winner"])
			for row, missed in enumerate(np.asarray(view.missed).tolist()):
				executed = bool(result.events.executed[row])
				if executed:
					native_move = native_moves[row]
					target_side = native_move["target"]
					target = _side_index(target_side)
					self.assertEqual(int(own.critical[row]), target_side in native_criticals)
					self.assertEqual(int(own.critical[row]), bool(result.events.critical[row]))
					self.assertEqual(bool(own.fainted[row]), target_side in native_faints)
					self.assertEqual(bool(own.fainted[row]), int(result.state.hp[target]) == 0)
					if not missed:
						self.assertEqual(bool(own.target_hp_is_percent[row]), audience != target)
						condition = native_damage[target_side]
						if condition == "0 fnt":
							native_target_hp = 0
						else:
							native_target_hp = int(condition.split("/")[0])
							self.assertEqual(condition.endswith("/100"), audience != target)
						self.assertEqual(int(own.target_hp_value[row]), native_target_hp)
						if audience == target:
							self.assertTrue(bool(own.damage_visible[row]))
							self.assertEqual(int(own.damage_value[row]), int(before_state.hp[target] - result.state.hp[target]))
							self.assertEqual(int(own.target_hp_value[row]), int(result.state.hp[target]))
						else:
							self.assertFalse(bool(own.damage_visible[row]))
							self.assertEqual(int(own.damage_value[row]), -1)
							percent = (int(result.state.hp[target]) * 100 + int(result.state.max_hp[target]) - 1) // int(result.state.max_hp[target])
							if percent == 100 and int(result.state.hp[target]) < int(result.state.max_hp[target]):
								percent = 99
							self.assertEqual(int(own.target_hp_value[row]), percent)
				if missed:
					self.assertFalse(bool(own.critical[row]))
					self.assertEqual(int(own.damage_value[row]), -1)
					self.assertFalse(bool(own.damage_visible[row]))
					self.assertEqual(int(own.target_hp_value[row]), -1)
					self.assertFalse(bool(own.target_hp_is_percent[row]))
					self.assertFalse(bool(own.fainted[row]))
			self.assertNotIn("state", own._fields)
			self.assertNotIn("rng", own._fields)
			self.assertNotIn("opponent_pp", own._fields)
			next_request = turn["nextRequests"][side]
			if next_request is not None:
				next_view = project_accuracy_choice_view(result.state, _empty_result(result.state, STATUS_PENDING), audience).player_view
				assert_request_view(next_view, next_request)
			else:
				self.assertTrue(bool(result.state.terminal))

	def test_followup_is_native_chain_and_remaining_move_is_legal(self):
		scenario = next(item for item in self.fixture["scenarios"] if item["continuation"] is not None)
		first_state = _state(scenario["turn"]["before"])
		_, first_expected = _native_expected(scenario)
		compiled_transition = jax.jit(transition_accuracy_choice)
		first_choices = jnp.asarray([scenario["choices"]["p1"], scenario["choices"]["p2"]], dtype=jnp.int32)
		first_actual = compiled_transition(first_state, first_choices)
		_assert_tree_equal(self, first_actual, first_expected, "last Mega Punch native turn")
		follow = scenario["continuation"]
		follow_state = _state(follow["before"])
		_assert_tree_equal(self, first_actual.state, follow_state, "native ordinary-turn continuation state")
		self.assertEqual(int(follow_state.pp[0, 1]), 0)
		self.assertGreater(int(follow_state.pp[0, 0]), 0)
		_, follow_expected = _native_expected({**scenario, "choices": follow["choices"], "turn": follow})
		follow_choices = jnp.asarray([follow["choices"]["p1"], follow["choices"]["p2"]], dtype=jnp.int32)
		follow_actual = compiled_transition(first_actual.state, follow_choices)
		_assert_tree_equal(self, follow_actual, follow_expected, "remaining Tackle native turn")

	def test_validation_types_shapes_old_variant_and_static_audience(self):
		self.assertEqual(VARIANT_ID, 6)
		self.assertEqual(VARIANT_NAME, "gen5-accuracy-choice-mew-1v1-v6")
		state = _state(self.fixture["scenarios"][0]["turn"]["before"])
		self.assertEqual(simulator.ACCURACY_CHOICE_VARIANT_ID, 6)
		self.assertEqual(simulator.ACTION_MEGA_PUNCH, 1)
		self.assertEqual(simulator.ACCURACY_CHOICE_ACTION_TACKLE, 1)
		self.assertEqual(simulator.ACCURACY_CHOICE_ACTION_MEGA_PUNCH, 2)
		state_hints = typing.get_type_hints(make_accuracy_choice_state)
		transition_hints = typing.get_type_hints(transition_accuracy_choice)
		projection_hints = typing.get_type_hints(project_accuracy_choice_view)
		self.assertEqual(state_hints["return"], type(state))
		self.assertEqual(set(transition_hints), {"state", "choices", "return"})
		self.assertEqual(transition_hints["state"], type(state))
		self.assertEqual(transition_hints["return"], AccuracyChoiceTransition)
		self.assertEqual(set(projection_hints), {"before", "result", "audience", "return"})
		self.assertIs(projection_hints["return"], AccuracyChoicePlayerView)
		for choices, error in ((1, ValueError), ([1.0, 1.0], TypeError), ([[1, 1]], ValueError), ([True, False], TypeError)):
			with self.subTest(choices=choices):
				with self.assertRaises(error):
					transition_accuracy_choice(state, choices)
		with self.assertRaises(ValueError):
			transition_accuracy_choice(state._replace(pp=jnp.zeros((2, 3), dtype=jnp.int32)), [0, 1])
		with self.assertRaises(TypeError):
			transition_accuracy_choice(state._replace(pp=state.pp.astype(jnp.float32)), [0, 1])
		result = transition_accuracy_choice(state, [1, 1])
		for audience, error in ((2, ValueError), (0.0, TypeError)):
			with self.subTest(audience=audience):
				with self.assertRaises(error):
					project_accuracy_choice_view(state, result, audience)
		with self.assertRaises(TypeError):
			jax.jit(lambda aud: project_accuracy_choice_view(state, result, aud))(jnp.asarray(0, dtype=jnp.int32))
		bad_cap = make_accuracy_choice_state(
				hp=[1, 1], max_hp=[2, 2], pp=[[57, 1], [56, 32]], level=[50, 50],
				attack=[100, 100], defense=[100, 100], speed=[101, 100], rng=[1, 2, 3, 4],
			)
		self.assertEqual(int(transition_accuracy_choice(bad_cap, [1, 1]).status), STATUS_UNSUPPORTED)
		v6_to_v4 = transition_two_move(state, [1, 1])
		self.assertEqual(int(v6_to_v4.status), STATUS_UNSUPPORTED)
		self.assertEqual(int(v6_to_v4.state.variant_id), 6)
		legacy_view = project_two_move_view(state, v6_to_v4, 0)
		self.assertFalse(bool(legacy_view.legal_move_mask[0]))
		wrong_ids = state._replace(move_ids=jnp.asarray([33, 1], dtype=jnp.int32))
		self.assertEqual(int(transition_accuracy_choice(wrong_ids, [1, 1]).status), STATUS_UNSUPPORTED)


if __name__ == "__main__":
	unittest.main()
