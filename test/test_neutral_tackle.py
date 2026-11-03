"""Differential tests for the restricted pinned Gen 5 Tackle turn."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from simulator.gen5_rng import next_state, sample_bounded
from simulator.neutral_tackle import (
	ACTION_TACKLE,
	DRAW_ACCURACY,
	DRAW_CRITICAL,
	DRAW_DAMAGE,
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
	make_state,
	project_view,
	transition,
)


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "reference" / "neutral-turn-oracle.cjs"
DRAW_CONTEXTS = {"accuracy": DRAW_ACCURACY, "critical": DRAW_CRITICAL, "damage": DRAW_DAMAGE}


def _state(snapshot):
	phase = PHASE_TERMINAL if snapshot["terminal"] else PHASE_MOVE_REQUEST
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


def _hand_damage(level, attack, defense, critical, roll):
	"""Independent ordinary Tackle arithmetic for this neutral Mew matchup."""
	level_term = (2 * level) // 5 + 2
	core = (level_term * 50 * attack // defense) // 50
	pre_roll = core + 2
	if critical:
		pre_roll *= 2  # Gen 5 critical multiplier is applied before randomizer.
	return max(1, pre_roll * (100 - roll) // 100)


def _jax_view_projection(view):
	viewer = int(view.audience)
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
			"move": {"name": "Tackle", "id": "tackle", "pp": int(view.own_pp), "disabled": False},
		},
	}
	events = []
	actors = _as_list(view.action_order)
	critical = _as_list(view.critical)
	targets = _as_list(view.target_player)
	hp_values = _as_list(view.target_hp_value)
	hp_percent = _as_list(view.target_hp_is_percent)
	fainted = _as_list(view.fainted)
	for index, actor in enumerate(actors):
		if actor < 0:
			continue
		actor_id = "p1" if actor == 0 else "p2"
		target_id = "p1" if targets[index] == 0 else "p2"
		events.append({"type": "move", "actor": actor_id, "move": "tackle", "target": target_id})
		if critical[index]:
			events.append({"type": "critical", "target": target_id})
		if hp_values[index] == 0:
			condition = "0 fnt"
		elif hp_percent[index]:
			condition = f"{hp_values[index]}/100"
		else:
			condition = f"{hp_values[index]}/{int(view.own_max_hp)}"
		events.append({"type": "damage", "target": target_id, "condition": condition})
		if fainted[index]:
			events.append({"type": "faint", "target": target_id})
	if bool(view.terminal):
		winner = int(view.winner)
		events.append({"type": "winner", "side": "p1" if winner == 1 else "p2" if winner == 2 else "tie"})
	elif bool(view.turn_event):
		events.append({"type": "turn", "number": int(view.turn_number)})
	return {"request": request, "events": events}


class NeutralTackleTests(unittest.TestCase):
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

	def test_live_pinned_rng_vectors_and_bounded_sampling(self):
		self.assertFalse(jax.config.read("jax_enable_x64"))
		compiled = jax.jit(sample_bounded)
		for vector in self.fixture["rngVectors"]:
			for row in vector["rows"]:
				seed = jnp.asarray(row["before"], dtype=jnp.uint16)
				for sampler in (sample_bounded, compiled):
					actual_seed, actual_value = sampler(seed, row["bound"])
					self.assertEqual(_as_list(actual_seed), row["after"])
					self.assertEqual(int(actual_value), row["value"])
		# Handwritten pinned vector: after seed [1,2,3,4], the first Gen 5
		# frame is [30982,33910,19571,50263], whose bounded-100 result is 47.
		seed, value = sample_bounded(jnp.asarray([1, 2, 3, 4], dtype=jnp.uint16), 100)
		self.assertEqual(_as_list(seed), [30982, 33910, 19571, 50263])
		self.assertEqual(int(value), 47)
		state, word = jax.jit(next_state)(jnp.asarray([1, 2, 3, 4], dtype=jnp.uint16))
		self.assertEqual(_as_list(state), [30982, 33910, 19571, 50263])
		self.assertEqual(int(word), (30982 << 16) | 33910)

	def test_actual_normal_request_turns_match_state_rng_and_both_views(self):
		compiled = jax.jit(transition)
		for scenario in self.fixture["scenarios"]:
			carried_state = None
			for turn_index, turn in enumerate(scenario["turns"]):
				if carried_state is None:
					before = _state(turn["before"])
				else:
					before = carried_state
					self.assertEqual(_as_list(before.hp), turn["before"]["hp"], (scenario["id"], turn_index, "carried hp"))
					self.assertEqual(_as_list(before.pp), turn["before"]["pp"], (scenario["id"], turn_index, "carried pp"))
					self.assertEqual(_as_list(before.rng), turn["before"]["rng"], (scenario["id"], turn_index, "carried rng"))
					self.assertEqual(int(before.turn), turn["before"]["turn"], (scenario["id"], turn_index, "carried turn"))
					self.assertEqual(int(before.request_phase), PHASE_MOVE_REQUEST, (scenario["id"], turn_index, "carried request phase"))
				choices = jnp.asarray([ACTION_TACKLE, ACTION_TACKLE], dtype=jnp.int32)
				eager = transition(before, choices)
				jit_result = compiled(before, choices)
				self.assertTrue(_leaf_equal(eager, jit_result), (scenario["id"], turn_index))
				self.assertEqual(int(eager.status), STATUS_OK)
				after = turn["after"]
				for name, actual in (
					("hp", eager.state.hp),
					("maxHp", eager.state.max_hp),
					("pp", eager.state.pp),
					("level", eager.state.level),
					("attack", eager.state.attack),
					("defense", eager.state.defense),
					("speed", eager.state.speed),
					("rng", eager.state.rng),
				):
					self.assertEqual(_as_list(actual), after[name], (scenario["id"], turn_index, name))
				for name, actual in (("turn", eager.state.turn), ("terminal", eager.state.terminal), ("winner", eager.state.winner)):
					self.assertEqual(bool(actual) if name == "terminal" else int(actual), after[name], (scenario["id"], turn_index, name))
				expected_phase = PHASE_TERMINAL if after["requestPhase"] == "terminal" else PHASE_MOVE_REQUEST
				self.assertEqual(int(eager.state.request_phase), expected_phase, (scenario["id"], turn_index, "next request phase"))
				self.assertEqual(after["requestsPresent"], [not after["terminal"], not after["terminal"]])
				self.assertEqual(int(eager.draws.count), len(turn["draws"]))
				actors = _as_list(eager.events.actors)
				self.assertEqual(["p1" if actor == 0 else "p2" for actor in actors if actor >= 0], turn["actionOrder"])
				for draw_index, draw in enumerate(turn["draws"]):
					self.assertEqual(int(eager.draws.bounds[draw_index]), draw["bound"])
					self.assertEqual(int(eager.draws.values[draw_index]), draw["value"])
					self.assertEqual(int(eager.draws.contexts[draw_index]), DRAW_CONTEXTS[draw["context"]])
					self.assertEqual(_as_list(eager.draws.before[draw_index]), draw["before"])
					self.assertEqual(_as_list(eager.draws.after[draw_index]), draw["after"])
					frame = " ".join(draw["stack"])
					self.assertIn(
						{"accuracy": "hitStepAccuracy", "critical": "getDamage", "damage": "randomizer"}[draw["context"]],
						frame,
					)
				# The reference choice is derived from each real activeRequest; the
				# projection contains only own request fields and split-visible events.
				for audience, side in ((0, "p1"), (1, "p2")):
					view = project_view(before, eager, audience)
					self.assertEqual(
						{"request": turn["views"][side]["request"], "events": turn["views"][side]["events"]},
						_jax_view_projection(view),
						(scenario["id"], turn_index, side),
					)
					self.assertNotIn("rng", view._fields)
					self.assertNotIn("opponent_hp", view._fields)
					self.assertNotIn("unprojected_packets", view._fields)
					for action_index, actor in enumerate(actors):
						if actor < 0:
							continue
						target = int(view.target_player[action_index])
						if target == audience:
							observed_loss = max(0, turn["before"]["hp"][target] - after["hp"][target])
							self.assertTrue(bool(view.damage_visible[action_index]))
							self.assertEqual(int(view.damage_value[action_index]), observed_loss)
						else:
							self.assertFalse(bool(view.damage_visible[action_index]))
							self.assertEqual(int(view.damage_value[action_index]), -1)
				# Handwritten Gen 5 formula independently checks every applied hit,
				# including a genuine critical and a KO overkill.
				for action_index, actor in enumerate(actors):
					if actor < 0:
						continue
					critical_roll = turn["draws"][action_index * 3 + 1]["value"]
					damage_roll = turn["draws"][action_index * 3 + 2]["value"]
					expected_damage = _hand_damage(
						turn["before"]["level"][actor],
						turn["before"]["attack"][actor],
						turn["before"]["defense"][1 - actor],
						critical_roll == 0,
						damage_roll,
					)
					self.assertEqual(int(eager.events.damage[action_index]), expected_damage)
				# A surviving continuation starts at the exact prior state/RNG.
				if turn_index + 1 < len(scenario["turns"]):
					self.assertEqual(after["rng"], scenario["turns"][turn_index + 1]["before"]["rng"])
					self.assertEqual(after["pp"], scenario["turns"][turn_index + 1]["before"]["pp"])
				carried_state = eager.state

	def test_critical_order_ko_skip_and_recipient_hp_privacy(self):
		critical_turn = self.scenarios["native-critical-hit"]["turns"][0]
		self.assertTrue(any(draw["context"] == "critical" and draw["value"] == 0 for draw in critical_turn["draws"]))
		self.assertTrue(any(event["type"] == "critical" for event in critical_turn["views"]["p1"]["events"]))
		first = self.scenarios["p2-first-surviving"]["turns"][0]
		p1_hp = next(event["condition"] for event in first["views"]["p1"]["events"] if event["type"] == "damage" and event["target"] == "p1")
		p2_hp = next(event["condition"] for event in first["views"]["p2"]["events"] if event["type"] == "damage" and event["target"] == "p1")
		self.assertEqual(p1_hp, "154/175")
		self.assertEqual(p2_hp, "88/100")
		self.assertNotEqual(p1_hp, p2_hp)
		ko = self.scenarios["first-action-ko-skip"]["turns"][0]
		self.assertEqual(ko["actionOrder"], ["p2"])
		self.assertEqual(len(ko["draws"]), 3)
		self.assertEqual(ko["after"]["hp"], [0, 341])
		self.assertEqual(ko["after"]["pp"], [56, 55])
		self.assertEqual(ko["after"]["turn"], 1)
		self.assertTrue(ko["after"]["terminal"])
		self.assertEqual(ko["after"]["winner"], 2)
		self.assertFalse(any("rng" in view or "trusted_state" in view for view in ko["views"].values()))
		ko_before = _state(ko["before"])
		ko_result = transition(ko_before, np.asarray([1, 1], dtype=np.int32))
		self.assertGreater(int(ko_result.events.damage[0]), 13, "trusted overkill must be distinguishable from HP removed")
		owner_view = project_view(ko_before, ko_result, 0)
		attacker_view = project_view(ko_before, ko_result, 1)
		self.assertEqual(int(owner_view.damage_value[0]), 13)
		self.assertTrue(bool(owner_view.damage_visible[0]))
		self.assertEqual(int(attacker_view.damage_value[0]), -1)
		self.assertFalse(bool(attacker_view.damage_visible[0]))

	def test_invalid_pending_unsupported_and_terminal_inputs_do_not_advance(self):
		initial = _state(self.scenarios["p2-first-surviving"]["turns"][0]["before"])
		for choices, expected_status in (
			([-1, 1], STATUS_INVALID_CHOICE),
			([0, 1], STATUS_PENDING),
			([2, 1], STATUS_UNSUPPORTED),
		):
			result = transition(initial, np.asarray(choices, dtype=np.int32))
			self.assertEqual(int(result.status), expected_status)
			self.assertTrue(_leaf_equal(result.state, initial))
			self.assertEqual(int(result.draws.count), 0)

		for state in (
			initial._replace(speed=jnp.asarray([120, 120], dtype=jnp.int32)),
			initial._replace(pp=jnp.asarray([0, 56], dtype=jnp.int32)),
			initial._replace(max_hp=jnp.asarray([405, 175], dtype=jnp.int32)),
			initial._replace(attack=jnp.asarray([300, 120], dtype=jnp.int32)),
			initial._replace(turn=jnp.asarray(0x7FFFFFFF, dtype=jnp.int32)),
			initial._replace(move_id=jnp.asarray(1, dtype=jnp.int32)),
			initial._replace(variant_id=jnp.asarray(2, dtype=jnp.int32)),
			initial._replace(request_phase=jnp.asarray(PHASE_UNSUPPORTED, dtype=jnp.int32)),
		):
			result = transition(state, np.asarray([1, 1], dtype=np.int32))
			self.assertEqual(int(result.status), STATUS_UNSUPPORTED)
			self.assertTrue(_leaf_equal(result.state, state))
			self.assertFalse(bool(project_view(state, result, 0).legal_move_mask[0]))
			self.assertEqual(int(result.draws.count), 0)

		pending = initial._replace(request_phase=jnp.asarray(PHASE_PENDING, dtype=jnp.int32))
		pending_result = transition(pending, np.asarray([1, 1], dtype=np.int32))
		self.assertEqual(int(pending_result.status), STATUS_PENDING)
		self.assertTrue(_leaf_equal(pending_result.state, pending))

		ko = self.scenarios["first-action-ko-skip"]["turns"][0]["after"]
		terminal = _state(ko)
		absorbed = transition(terminal, np.asarray([1, 1], dtype=np.int32))
		self.assertEqual(int(absorbed.status), STATUS_TERMINAL)
		self.assertTrue(_leaf_equal(absorbed.state, terminal))
		self.assertEqual(int(absorbed.draws.count), 0)
		self.assertFalse(bool(project_view(terminal, absorbed, 0).legal_move_mask[0]))

	def test_bad_types_and_audiences_fail_closed(self):
		initial = _state(self.scenarios["p2-first-surviving"]["turns"][0]["before"])
		for choices in ([1.2, 1], [1.0, 1.0]):
			with self.assertRaises(TypeError):
				transition(initial, choices)
		for kwargs in ({"turn": 1.5}, {"request_phase": 0.5}, {"winner": -1.5}, {"variant_id": 1.1}, {"move_id": 33.5}):
			with self.assertRaises(ValueError):
				make_state(
					hp=(175, 175), max_hp=(175, 175), pp=(56, 56), level=(50, 50),
					attack=(120, 120), defense=(120, 120), speed=(120, 152), rng=(1, 2, 3, 4), **kwargs,
				)
		result = transition(initial, np.asarray([1, 1], dtype=np.int32))
		for audience in (2, -1, True, 1.0, "p1"):
			with self.assertRaises((TypeError, ValueError)):
				project_view(initial, result, audience)
		with self.assertRaises(TypeError):
			jax.jit(project_view)(initial, result, 0)
		compiled_view = jax.jit(project_view, static_argnums=2)(initial, result, 0)
		self.assertEqual(int(compiled_view.audience), 0)

	def test_jit_vmap_lanes_match_individual_and_do_not_couple(self):
		p2_first = self.scenarios["p2-first-surviving"]["turns"][0]
		p1_first = self.scenarios["p1-first-surviving"]["turns"][0]
		ko = self.scenarios["first-action-ko-skip"]["turns"][0]
		states = [_state(p2_first["before"]), _state(p1_first["before"]), _state(ko["before"])]
		states.append(states[0]._replace(speed=jnp.asarray([120, 120], dtype=jnp.int32)))
		states.append(states[0]._replace(request_phase=jnp.asarray(PHASE_PENDING, dtype=jnp.int32)))
		terminal = _state(ko["after"])
		states.append(terminal)
		choices = jnp.asarray([[1, 1], [1, 1], [1, 1], [1, 1], [0, 1], [1, 1]], dtype=jnp.int32)
		batch_state = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *states)
		batched = jax.jit(jax.vmap(transition))(batch_state, choices)
		for lane, state in enumerate(states):
			separate = transition(state, choices[lane])
			self.assertTrue(_leaf_equal(jax.tree_util.tree_map(lambda value: value[lane], batched), separate), lane)
		self.assertEqual(_as_list(batched.status), [STATUS_OK, STATUS_OK, STATUS_OK, STATUS_UNSUPPORTED, STATUS_PENDING, STATUS_TERMINAL])
		self.assertEqual(_as_list(batched.state.rng[3]), _as_list(states[3].rng))
		self.assertEqual(_as_list(batched.state.rng[4]), _as_list(states[4].rng))
		self.assertEqual(_as_list(batched.state.rng[5]), _as_list(states[5].rng))
		# Reordering lanes changes only the output order.
		permutation = jnp.asarray([5, 2, 0, 4, 3, 1])
		reordered = jax.jit(jax.vmap(transition))(
			jax.tree_util.tree_map(lambda value: value[permutation], batch_state), choices[permutation]
		)
		for new_index, old_index in enumerate(np.asarray(permutation).tolist()):
			self.assertTrue(
				_leaf_equal(jax.tree_util.tree_map(lambda value: value[new_index], reordered), jax.tree_util.tree_map(lambda value: value[old_index], batched)),
				(old_index, new_index),
			)


if __name__ == "__main__":
	unittest.main()
