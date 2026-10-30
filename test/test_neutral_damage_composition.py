"""Pinned differential tests for the full neutral damage path."""

import json
from pathlib import Path
import subprocess
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from mechanics.neutral_damage import neutral_damage_after_roll, neutral_damage_before_roll
from mechanics.neutral_damage_composition import damage_from_effective_stats_neutral
from mechanics.damage_roll import apply_damage_roll


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "reference" / "fixtures" / "neutral-full-damage-v1.json"
ORACLE = ROOT / "reference" / "neutral-full-damage-oracle.cjs"


def run_oracle():
	result = subprocess.run(
		["node", str(ORACLE), "fixture"], cwd=ROOT, check=True,
		capture_output=True, text=True, timeout=60,
	)
	return json.loads(result.stdout)


def rows_from(fixture):
	return [
		(case, outcome)
		for group in ("ordinaryCases", "syntheticCases")
		for case in fixture[group]
		for outcome in case["outcomes"]
	]


class NeutralDamageCompositionTests(unittest.TestCase):
	def test_guarded_full_path_fixture_is_deterministic_and_ordered(self):
		fixture = json.loads(FIXTURE.read_text())
		self.assertEqual(run_oracle(), fixture)
		self.assertEqual(fixture["upstream"]["commit"], "1242c1e2d9763ba5b85d33f13aecb89bae19ae3d")
		self.assertEqual(len(fixture["ordinaryCases"]), 5)
		self.assertEqual(len(fixture["syntheticCases"]), 6)
		for case, row in rows_from(fixture):
			self.assertEqual([r["roll"] for r in case["outcomes"]], list(range(16)))
			self.assertEqual(row["weatherInput"], row["baseCore"] + 2)
			self.assertEqual(row["preRollDamage"], row["weatherInput"] & 0xFFFFF)
			self.assertEqual(row["input"]["preRollDamage"], row["preRollDamage"])
			self.assertEqual(row["input"]["roll"], row["roll"])
			self.assertEqual((row["randomizerCalls"], row["damageDraws"]), (1, 1))
			self.assertEqual((row["finalCalls"], row["finalModifierCalls"]), (1, 1))
			self.assertEqual(row["finalModifierObserved"], row["input"]["finalModifier"])
			self.assertEqual((row["baseCoreCalls"], row["weatherEventCalls"]), (1, 1))
			self.assertEqual(row["weatherHandlerCount"], 0)
			self.assertEqual(row["finalMoveCategory"] in ("Physical", "Special"), True)
			self.assertLessEqual(row["preRollDamage"], 1048575)
		guts = fixture["ordinaryCases"][3]["outcomes"][0]
		self.assertEqual(guts["attack"], 112, "Guts-modified event output is the observed core Attack")
		zero_defense = fixture["syntheticCases"][1]["outcomes"][0]
		self.assertEqual((zero_defense["defense"], zero_defense["baseCore"]), (0, 0))
		weather_boundaries = [fixture["syntheticCases"][index]["outcomes"][0] for index in (2, 3, 4)]
		self.assertEqual([row["weatherInput"] for row in weather_boundaries], [1048575, 1048576, 1048577])
		self.assertEqual([row["preRollDamage"] for row in weather_boundaries], [1048575, 0, 1])

	def test_prefix_and_full_composition_match_every_live_roll(self):
		fixture = json.loads(FIXTURE.read_text())
		rows = rows_from(fixture)
		int_fields = ("level", "basePower", "attack", "defense")
		core_inputs = tuple(jnp.asarray([row[key] for _, row in rows], dtype=jnp.int32) for key in int_fields)
		roll = jnp.asarray([row["roll"] for _, row in rows], dtype=jnp.int32)
		prefix = neutral_damage_before_roll(*core_inputs)
		self.assertEqual(jax.device_get(prefix).tolist(), [row["preRollDamage"] for _, row in rows])
		postroll = jax.device_get(neutral_damage_after_roll(*core_inputs, roll)).tolist()
		expected_postroll = [row["randomizerOutput"] for _, row in rows]
		self.assertEqual(postroll, expected_postroll)
		double_roll = jax.device_get(apply_damage_roll(
			jnp.asarray(expected_postroll, dtype=jnp.int32), roll
		)).tolist()
		witness = next(index for index, (once, twice) in enumerate(zip(postroll, double_roll)) if once != twice)
		self.assertNotEqual(postroll[witness], double_roll[witness], "the oracle witness distinguishes one roll from a duplicate roll")
		inputs = core_inputs + (roll,) + tuple(
			jnp.asarray([row["input"][key] for _, row in rows], dtype=jnp.bool_ if key in {
				"stabEligible", "adaptability", "typeless", "burned", "physical", "hasGuts"
			} else jnp.int32)
			for key in ("stabEligible", "adaptability", "typeless", "typeExponent", "burned", "physical", "hasGuts", "finalModifier")
		)
		expected = [row["result"] for _, row in rows]
		function = damage_from_effective_stats_neutral
		for result in (function(*inputs), jax.jit(function)(*inputs), jax.jit(jax.vmap(function))(*inputs)):
			self.assertEqual(result.dtype, jnp.int32)
			self.assertEqual(jax.device_get(result).tolist(), expected)

	def test_broadcasting_and_strict_input_dtypes(self):
		levels = jnp.asarray([[[50]], [[100]]], dtype=jnp.int32)
		powers = jnp.asarray([[[50], [255]]], dtype=jnp.int32)
		rolls = jnp.asarray([0, 8, 15], dtype=jnp.int32)
		result = damage_from_effective_stats_neutral(
			levels, powers, jnp.asarray(100, dtype=jnp.int32), jnp.asarray(50, dtype=jnp.int32),
			rolls, jnp.asarray(True), jnp.asarray(False), jnp.asarray(False),
			jnp.asarray(0, dtype=jnp.int32), jnp.asarray(False), jnp.asarray(True),
			jnp.asarray(False), jnp.asarray(4096, dtype=jnp.int32),
		)
		self.assertEqual(result.shape, (2, 2, 3))
		self.assertEqual(result.dtype, jnp.int32)
		self.assertEqual(jax.device_get(jax.jit(damage_from_effective_stats_neutral)(
			levels, powers, jnp.asarray(100, dtype=jnp.int32), jnp.asarray(50, dtype=jnp.int32),
			rolls, jnp.asarray(True), jnp.asarray(False), jnp.asarray(False),
			jnp.asarray(0, dtype=jnp.int32), jnp.asarray(False), jnp.asarray(True),
			jnp.asarray(False), jnp.asarray(4096, dtype=jnp.int32),
		)).tolist(), jax.device_get(result).tolist())
		with self.assertRaises(TypeError):
			damage_from_effective_stats_neutral(50, 50, 100, 50, jnp.asarray(1.0), True, False, False, 0, False, True, False, 4096)
		with self.assertRaises(TypeError):
			damage_from_effective_stats_neutral(50, 50, 100, 50, 1, 1, False, False, 0, False, True, False, 4096)


if __name__ == "__main__":
	unittest.main()
