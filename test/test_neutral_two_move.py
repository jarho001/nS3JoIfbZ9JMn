"""Focused native differential checks for the v4 two-move Mew turn."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from typing import get_type_hints

import jax
import jax.numpy as jnp
import numpy as np

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
	BattleState,
	TurnEvents,
	make_state,
	transition as transition_tackle,
)
from simulator.neutral_pound import transition_pound
from simulator.neutral_tackle_ties import transition_with_speed_ties
from simulator.neutral_two_move import (
	ACTION_POUND,
	ACTION_TACKLE,
	INVALID_MOVE_ID,
	MOVE_POUND,
	MOVE_TACKLE,
	VARIANT_ID,
	VARIANT_NAME,
	TwoMoveState,
	TwoMoveTransition,
	make_two_move_state,
	project_two_move_view,
	transition_two_move,
)


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "reference" / "neutral-two-move-oracle.cjs"
MOVE_NAME_BY_ID = {MOVE_TACKLE: "tackle", MOVE_POUND: "pound"}
MOVE_ID_BY_NAME = {"tackle": MOVE_TACKLE, "pound": MOVE_POUND}
MOVE_POWER_BY_ID = {MOVE_TACKLE: 50, MOVE_POUND: 40}
DRAW_CONTEXTS = {
	"accuracy": 1,
	"critical": 2,
	"damage": 3,
	"queue-sort": 4,
	"before-turn": 5,
	"turn-start-update": 7,
	"move-hit-update": 6,
	"action-update": 7,
	"residual-update": 8,
}


def _state(snapshot):
	phase = {
		"move": PHASE_MOVE_REQUEST,
		"pending": PHASE_PENDING,
		"unsupported": PHASE_UNSUPPORTED,
		"terminal": PHASE_TERMINAL,
	}[snapshot["requestPhase"]]
	return make_two_move_state(
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


def _hand_damage(level, attack, defense, critical, roll, base_power):
	level_term = (2 * level) // 5 + 2
	core = (level_term * base_power * attack // defense) // 50
	pre_roll = core + 2
	if critical:
		pre_roll *= 2
	return max(1, pre_roll * (100 - roll) // 100)


def _assert_state(test, actual, expected, label):
	for field, value in (
		("hp", actual.hp),
		("maxHp", actual.max_hp),
		("pp", actual.pp),
		("level", actual.level),
		("attack", actual.attack),
		("defense", actual.defense),
		("speed", actual.speed),
		("rng", actual.rng),
	):
		test.assertEqual(_as_list(value), expected[field], (label, field))
	for field, value in (("turn", actual.turn), ("terminal", actual.terminal), ("winner", actual.winner)):
		test.assertEqual(bool(value) if field == "terminal" else int(value), expected[field], (label, field))
	phase = {
		"move": PHASE_MOVE_REQUEST,
		"pending": PHASE_PENDING,
		"unsupported": PHASE_UNSUPPORTED,
		"terminal": PHASE_TERMINAL,
	}[expected["requestPhase"]]
	test.assertEqual(int(actual.request_phase), phase, (label, "requestPhase"))
	test.assertEqual(int(actual.variant_id), VARIANT_ID, (label, "variant id"))
	test.assertEqual(_as_list(actual.move_ids), [MOVE_TACKLE, MOVE_POUND], (label, "move ids"))


def _jax_view_projection(view):
	request = {
		"phase": "move",
		"legalActions": [f"move {index + 1}" for index, legal in enumerate(_as_list(view.legal_move_mask)) if legal],
		"own": {
			"species": "Mew",
			"level": int(view.own_level),
			"hpCondition": f"{int(view.own_hp)}/{int(view.own_max_hp)}",
			"stats": {"atk": int(view.own_attack), "def": int(view.own_defense), "spe": int(view.own_speed)},
			"moves": [
				{
					"slot": index + 1,
					"name": "Tackle" if int(move_id) == MOVE_TACKLE else "Pound",
					"id": MOVE_NAME_BY_ID[int(move_id)],
					"pp": int(pp),
					"disabled": int(pp) <= 0,
				}
				for index, (move_id, pp) in enumerate(zip(_as_list(view.own_move_ids), _as_list(view.own_pp)))
			],
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
		move_id = int(view.event_move_ids[index])
		events.append({"type": "move", "actor": actor_id, "move": MOVE_NAME_BY_ID[move_id], "target": target_id})
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


class NeutralTwoMoveTests(unittest.TestCase):
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
		self.assertEqual(result.draws.bounds.shape, (16,), (scenario_id, "draw bounds shape"))
		self.assertEqual(result.draws.values.shape, (16,), (scenario_id, "draw values shape"))
		self.assertEqual(result.draws.before.shape, (16, 4), (scenario_id, "draw limbs shape"))
		expected_actors = [0 if actor == "p1" else 1 for actor in turn["actionOrder"]]
		actual_actors = [int(actor) for actor in _as_list(result.events.actors) if actor >= 0]
		self.assertEqual(actual_actors, expected_actors, (scenario_id, "action order"))
		self.assertEqual(int(result.draws.count), len(turn["draws"]), (scenario_id, "draw count"))
		self.assertLessEqual(len(turn["draws"]), 16, scenario_id)
		action_draws = [draw for draw in turn["draws"] if draw["context"] != "speed-sort"]
		self.assertEqual(len(action_draws), len(expected_actors) * 3, (scenario_id, "action draw accounting"))
		for index, expected in enumerate(turn["draws"]):
			self.assertEqual(int(result.draws.bounds[index]), 2 if expected["context"] == "speed-sort" else expected["bound"], (scenario_id, index, "bound"))
			self.assertEqual(int(result.draws.values[index]), expected["value"], (scenario_id, index, "value"))
			phase = expected.get("phase", expected["context"])
			self.assertEqual(int(result.draws.contexts[index]), DRAW_CONTEXTS[phase], (scenario_id, index, "context"))
			self.assertEqual(_as_list(result.draws.before[index]), expected["before"], (scenario_id, index, "before limbs"))
			self.assertEqual(_as_list(result.draws.after[index]), expected["after"], (scenario_id, index, "after limbs"))
			if index:
				self.assertEqual(expected["before"], turn["draws"][index - 1]["after"], (scenario_id, index, "native RNG chain"))
		self.assertEqual(_as_list(result.events.executed), [index < len(expected_actors) for index in range(2)])
		expected_move_ids = [
			MOVE_ID_BY_NAME[move["move"]]
			for move in turn["events"]["p1"]
			if move["type"] == "move"
		]
		self.assertEqual(_as_list(result.event_move_ids), expected_move_ids + [INVALID_MOVE_ID] * (2 - len(expected_move_ids)))
		self.assertEqual(result.event_move_ids.dtype, jnp.int32)

		for audience, side in ((0, "p1"), (1, "p2")):
			view = project_two_move_view(_state(turn["before"]), result, audience)
			self.assertEqual(
				{"request": turn["requests"][side], "events": turn["events"][side]},
				_jax_view_projection(view),
				(scenario_id, "view", side),
			)
			for forbidden in ("rng", "draws", "state", "choices", "opponent_pp"):
				self.assertNotIn(forbidden, view._fields)
			for action_index, actor in enumerate(_as_list(view.action_order)):
				if actor < 0:
					self.assertEqual(int(view.event_move_ids[action_index]), INVALID_MOVE_ID)
				continue
				target = int(view.target_player[action_index])
				if target == audience:
					self.assertTrue(bool(view.damage_visible[action_index]), (scenario_id, side, action_index, "visible damage"))
				else:
					self.assertFalse(bool(view.damage_visible[action_index]), (scenario_id, side, action_index, "hidden damage"))
		self._assert_damage_and_critical(result, turn, scenario_id)

		after_state = _state(turn["after"])
		if turn["nextRequests"]["p1"] is None:
			self.assertTrue(bool(after_state.terminal), (scenario_id, "terminal requests"))
			self.assertIsNone(turn["nextRequests"]["p2"])
			idle_result = transition_two_move(after_state, jnp.asarray([1, 1], dtype=jnp.int32))
			self.assertEqual(int(idle_result.status), STATUS_TERMINAL)
			for audience in (0, 1):
				view = project_two_move_view(after_state, idle_result, audience)
				self.assertFalse(any(_as_list(view.legal_move_mask)))
		else:
			idle_result = transition_two_move(after_state, jnp.asarray([0, 0], dtype=jnp.int32))
			self.assertEqual(int(idle_result.status), STATUS_PENDING)
			for audience, side in ((0, "p1"), (1, "p2")):
				view = project_two_move_view(after_state, idle_result, audience)
				self.assertEqual(turn["nextRequests"][side], _jax_view_projection(view)["request"], (scenario_id, side, "next request"))
		if scenario_id == "last-tackle-pp":
			self.assertEqual(int(after_state.request_phase), PHASE_MOVE_REQUEST)
			self.assertEqual(_as_list(after_state.pp[0]), [0, turn["before"]["pp"][0][1]])
			p1_view = project_two_move_view(after_state, idle_result, 0)
			self.assertEqual(_as_list(p1_view.legal_move_mask), [False, True])
		if scenario_id == "first-action-ko":
			self.assertEqual(turn["actionOrder"], ["p2"])
			self.assertEqual(_as_list(after_state.pp[0]), turn["before"]["pp"][0], "KO victim did not spend its unexecuted move PP")

	def _assert_damage_and_critical(self, result, turn, scenario_id):
		actors = [int(actor) for actor in _as_list(result.events.actors) if actor >= 0]
		action_draws = [draw for draw in turn["draws"] if draw["context"] != "speed-sort"]
		for action_index, actor in enumerate(actors):
			choice = turn["choices"][actor]
			move_id = MOVE_TACKLE if choice == 1 else MOVE_POUND
			critical_draw = action_draws[action_index * 3 + 1]
			damage_draw = action_draws[action_index * 3 + 2]
			critical = critical_draw["value"] == 0
			self.assertEqual(bool(result.events.critical[action_index]), critical, (scenario_id, action_index, "critical"))
			expected_damage = _hand_damage(
				turn["before"]["level"][actor],
				turn["before"]["attack"][actor],
				turn["before"]["defense"][1 - actor],
				critical,
				damage_draw["value"],
				MOVE_POWER_BY_ID[move_id],
			)
			self.assertEqual(int(result.events.damage[action_index]), expected_damage, (scenario_id, action_index, MOVE_NAME_BY_ID[move_id], "damage power"))

	def test_native_turns_match_full_state_trace_move_identity_and_both_views(self):
		self.assertEqual(self.fixture["kind"], "gen5-restricted-neutral-two-move-turn-oracle")
		self.assertEqual(self.fixture["variant"], VARIANT_NAME)
		self.assertEqual(self.fixture["generation"], 5)
		self.assertEqual(self.fixture["upstream"]["commit"], "1242c1e2d9763ba5b85d33f13aecb89bae19ae3d")
		self.assertEqual(self.fixture["moveOrder"], ["tackle", "pound"])
		self.assertEqual(self.fixture["moveFacts"]["tackle"]["basePower"], 50)
		self.assertEqual(self.fixture["moveFacts"]["pound"]["basePower"], 40)
		self.assertEqual(set(self.scenarios), {
			"tackle-tackle-p2-first-critical",
			"tackle-pound-p1-first",
			"pound-tackle-p2-first",
			"pound-pound-p1-first",
			"tied-mixed-choice",
			"first-action-ko",
			"last-tackle-pp",
		})
		self.assertEqual({tuple(case["turn"]["choices"]) for case in self.fixture["scenarios"]} >= {(1, 1), (1, 2), (2, 1), (2, 2)}, True)
		critical_turn = self.scenarios["tackle-tackle-p2-first-critical"]["turn"]
		self.assertTrue(any(event["type"] == "critical" for event in critical_turn["events"]["p1"]))
		self.assertEqual(critical_turn["actionOrder"][0], "p2")
		self.assertEqual(self.scenarios["tackle-pound-p1-first"]["turn"]["actionOrder"][0], "p1")
		self.assertEqual(self.scenarios["pound-tackle-p2-first"]["turn"]["actionOrder"][0], "p2")
		tied_turn = self.scenarios["tied-mixed-choice"]["turn"]
		self.assertEqual(tied_turn["before"]["speed"][0], tied_turn["before"]["speed"][1])
		self.assertTrue(any(draw["context"] == "speed-sort" for draw in tied_turn["draws"]))
		boundary = self.scenarios["last-tackle-pp"]
		self.assertEqual(boundary["setupEvidence"]["rngBefore"], boundary["setupEvidence"]["rngAfter"])
		self.assertEqual(boundary["turn"]["nextRequests"]["p1"]["own"]["moves"][0]["disabled"], True)
		self.assertEqual(boundary["turn"]["nextRequests"]["p1"]["own"]["moves"][1]["disabled"], False)

		compiled = jax.jit(transition_two_move)
		compiled_view = jax.jit(project_two_move_view, static_argnums=2)
		for scenario in self.fixture["scenarios"]:
			turn = scenario["turn"]
			initial = _state(turn["before"])
			self.assertEqual(_as_list(initial.pp), turn["before"]["pp"])
			self.assertEqual(int(initial.variant_id), VARIANT_ID)
			self.assertEqual(_as_list(initial.move_ids), [MOVE_TACKLE, MOVE_POUND])
			choices = jnp.asarray(turn["choices"], dtype=jnp.int32)
			eager = transition_two_move(initial, choices)
			jit_result = compiled(initial, choices)
			self.assertTrue(_leaf_equal(eager, jit_result), (scenario["id"], "eager/JIT"))
			self._assert_turn(eager, turn, scenario["id"])
			for audience in (0, 1):
				self.assertTrue(
					_leaf_equal(
						compiled_view(initial, eager, audience),
						project_two_move_view(initial, eager, audience),
					),
					(scenario["id"], audience, "compiled projection"),
				)
			self.assertTrue(_leaf_equal(initial, _state(turn["before"])), (scenario["id"], "input unchanged"))

	def test_all_pp_exhaustion_is_unsupported_on_continuation(self):
		self.assertIs(get_type_hints(TwoMoveTransition)["events"], TurnEvents)
		before = _state(self.scenarios["tackle-pound-p1-first"]["turn"]["before"])
		before = before._replace(pp=before.pp.at[0].set(jnp.asarray([1, 0], dtype=jnp.int32)))
		choices = jnp.asarray([ACTION_TACKLE, ACTION_POUND], dtype=jnp.int32)
		compiled = jax.jit(transition_two_move)

		completed = compiled(before, choices)
		self.assertEqual(int(completed.status), STATUS_OK)
		self.assertEqual(_as_list(completed.events.executed), [True, True])
		self.assertFalse(bool(completed.state.terminal))
		self.assertEqual(int(completed.state.winner), -1)
		self.assertEqual(int(completed.state.request_phase), PHASE_UNSUPPORTED)
		self.assertEqual(_as_list(completed.state.pp[0]), [0, 0])
		self.assertTrue(all(int(hp) > 0 for hp in _as_list(completed.state.hp)), "normal native HP prevents a knockout")

		continuation = compiled(completed.state, choices)
		self.assertEqual(int(continuation.status), STATUS_UNSUPPORTED)
		self.assertTrue(_leaf_equal(continuation.state, completed.state), "unsupported continuation preserves the full state")
		self.assertTrue(np.array_equal(np.asarray(continuation.state.rng), np.asarray(completed.state.rng)))
		self.assertEqual(int(continuation.draws.count), 0)
		self.assertFalse(any(_as_list(continuation.events.executed)))
		self.assertEqual(_as_list(continuation.event_move_ids), [INVALID_MOVE_ID, INVALID_MOVE_ID])

	def test_boundaries_and_masking(self):
		base = _state(self.scenarios["tackle-pound-p1-first"]["turn"]["before"])
		choices = jnp.asarray([ACTION_TACKLE, ACTION_POUND], dtype=jnp.int32)
		for state, action, expected_status in (
			(base, [-1, 1], STATUS_INVALID_CHOICE),
			(base, [0, 1], STATUS_PENDING),
			(base, [3, 1], STATUS_UNSUPPORTED),
			(base._replace(request_phase=jnp.asarray(PHASE_PENDING, dtype=jnp.int32)), [1, 1], STATUS_PENDING),
			(base._replace(variant_id=jnp.asarray(VARIANT_ID + 1, dtype=jnp.int32)), [1, 1], STATUS_UNSUPPORTED),
			(base._replace(move_ids=jnp.asarray([MOVE_POUND, MOVE_TACKLE], dtype=jnp.int32)), [1, 1], STATUS_UNSUPPORTED),
			(base._replace(species_id=jnp.asarray([151, 25], dtype=jnp.int32)), [1, 1], STATUS_UNSUPPORTED),
			(base._replace(pp=base.pp.at[0, 0].set(57)), [1, 1], STATUS_UNSUPPORTED),
		):
			result = transition_two_move(state, np.asarray(action, dtype=np.int32))
			self.assertEqual(int(result.status), expected_status)
			self.assertTrue(_leaf_equal(result.state, state))
			self.assertEqual(int(result.draws.count), 0)
			self.assertFalse(any(_as_list(result.events.executed)))
			self.assertEqual(_as_list(result.event_move_ids), [INVALID_MOVE_ID, INVALID_MOVE_ID])

		no_tackle_pp = base._replace(pp=base.pp.at[0, 0].set(0))
		unavailable = transition_two_move(no_tackle_pp, choices)
		self.assertEqual(int(unavailable.status), STATUS_INVALID_CHOICE)
		self.assertTrue(_leaf_equal(unavailable.state, no_tackle_pp))
		self.assertEqual(int(unavailable.draws.count), 0)
		self.assertFalse(any(_as_list(unavailable.events.executed)))

		partial = transition_two_move(base, jnp.asarray([0, 1], dtype=jnp.int32))
		self.assertEqual(int(partial.status), STATUS_PENDING)
		for audience in (0, 1):
			view = project_two_move_view(base, partial, audience)
			self.assertEqual(_as_list(view.event_move_ids), [INVALID_MOVE_ID, INVALID_MOVE_ID])
			self.assertNotIn("rng", view._fields)
			self.assertNotIn("draws", view._fields)
			self.assertNotIn("opponent_pp", view._fields)

		exhausted = base._replace(pp=base.pp.at[0].set(jnp.asarray([0, 0], dtype=jnp.int32)))
		exhausted_result = transition_two_move(exhausted, jnp.asarray([2, 1], dtype=jnp.int32))
		self.assertEqual(int(exhausted_result.status), STATUS_UNSUPPORTED, "all PP exhausted stays outside Struggle support")
		self.assertTrue(_leaf_equal(exhausted_result.state, exhausted))
		self.assertEqual(int(exhausted_result.draws.count), 0)

		ko_after = _state(self.scenarios["first-action-ko"]["turn"]["after"])
		ko_terminal = transition_two_move(ko_after, choices)
		self.assertEqual(int(ko_terminal.status), STATUS_TERMINAL)
		self.assertTrue(_leaf_equal(ko_terminal.state, ko_after))
		self.assertEqual(int(ko_terminal.draws.count), 0)

		for malformed, exception, message in (
			(base._replace(pp=base.pp[:, :1]), ValueError, "state.pp must have shape"),
			(base._replace(pp=base.pp.astype(jnp.int16)), TypeError, "state.pp must have dtype"),
		):
			with self.assertRaisesRegex(exception, message):
				transition_two_move(malformed, choices)
		with self.assertRaises(ValueError):
			transition_two_move(base, [1, 1, 1])
		with self.assertRaises(TypeError):
			transition_two_move(base, [1.0, 1.0])
		old = make_state(
			hp=_as_list(base.hp), max_hp=_as_list(base.max_hp), pp=[56, 56],
			level=_as_list(base.level), attack=_as_list(base.attack), defense=_as_list(base.defense),
			speed=_as_list(base.speed), rng=_as_list(base.rng),
		)
		self.assertIsInstance(old, BattleState)
		with self.assertRaisesRegex(TypeError, "TwoMoveState"):
			transition_two_move(old, choices)
		for old_transition in (transition_tackle, transition_with_speed_ties, transition_pound):
			with self.assertRaises((TypeError, AttributeError)):
				old_transition(_state(self.scenarios["tackle-pound-p1-first"]["turn"]["before"]), jnp.asarray([1, 1], dtype=jnp.int32))

		valid = transition_two_move(base, choices)
		for audience in (0, 1):
			view = project_two_move_view(base, valid, audience)
			self.assertEqual(_as_list(view.own_move_ids), [MOVE_TACKLE, MOVE_POUND])
			self.assertEqual(_as_list(view.legal_move_mask), [True, True])
			self.assertEqual(_as_list(view.event_move_ids), _as_list(valid.event_move_ids))


	def test_jitted_vmap_isolates_heterogeneous_lanes(self):
		base = _state(self.scenarios["tackle-pound-p1-first"]["turn"]["before"])
		tied = _state(self.scenarios["tied-mixed-choice"]["turn"]["before"])
		last_pp = _state(self.scenarios["last-tackle-pp"]["turn"]["before"])
		unavailable = base._replace(pp=base.pp.at[0, 0].set(0))
		exhausted = base._replace(pp=base.pp.at[0].set(jnp.asarray([0, 0], dtype=jnp.int32)))
		ko_after = _state(self.scenarios["first-action-ko"]["turn"]["after"])
		states = [base, tied, base, unavailable, last_pp, exhausted, ko_after]
		lane_choices = jnp.asarray([[1, 2], [1, 2], [0, 1], [1, 2], [1, 2], [2, 1], [1, 1]], dtype=jnp.int32)
		batch = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *states)
		compiled_batch = jax.jit(jax.vmap(transition_two_move))(batch, lane_choices)
		self.assertEqual(
			_as_list(compiled_batch.status),
			[STATUS_OK, STATUS_OK, STATUS_PENDING, STATUS_INVALID_CHOICE, STATUS_OK, STATUS_UNSUPPORTED, STATUS_TERMINAL],
		)
		for lane, state in enumerate(states):
			separate = transition_two_move(state, lane_choices[lane])
			batched_lane = jax.tree_util.tree_map(lambda value: value[lane], compiled_batch)
			self.assertTrue(_leaf_equal(batched_lane, separate), (lane, "vmap lane equals scalar"))
			if int(separate.status) != STATUS_OK:
				self.assertTrue(_leaf_equal(batched_lane.state, state), (lane, "rejected lane frozen"))
				self.assertEqual(int(batched_lane.draws.count), 0, (lane, "rejected lane draws"))
				self.assertFalse(any(_as_list(batched_lane.events.executed)), (lane, "rejected lane events"))


if __name__ == "__main__":
	unittest.main()
