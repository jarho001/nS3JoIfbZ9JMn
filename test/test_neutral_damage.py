"""Pinned Showdown differential tests for neutral damage after its random roll."""

import json
from pathlib import Path
import subprocess
import unittest

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from mechanics.neutral_damage import _neutral_weather_identity_damage, neutral_damage_after_roll


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "reference" / "fixtures" / "neutral-damage-v1.json"
ORACLE_PATH = ROOT / "reference" / "neutral-damage-oracle.cjs"


def run_oracle():
	result = subprocess.run(
		["node", str(ORACLE_PATH), "fixture"],
		cwd=ROOT,
		check=True,
		capture_output=True,
		text=True,
		timeout=60,
	)
	return json.loads(result.stdout)


def as_lists(value):
	return jax.device_get(value).tolist()


def flat_outcomes(fixture):
	return [
		(case, outcome)
		for case in fixture["ordinaryCases"] + fixture["syntheticCases"]
		for outcome in case["outcomes"]
	]


def arrays_from_outcomes(rows):
	return tuple(
		jnp.asarray([case[key] for case, _ in rows], dtype=jnp.int32)
		for key in ("level", "basePower", "attack", "defense")
	) + (jnp.asarray([outcome["roll"] for _, outcome in rows], dtype=jnp.int32),)


class NeutralDamageTests(unittest.TestCase):
	def test_jax_runtime_matches_pinned_dependencies(self):
		self.assertEqual(jax.__version__, "0.12.0.1")
		self.assertEqual(jaxlib.__version__, "0.12.0.1")

	def test_guarded_fixture_regenerates_and_records_composition_boundary(self):
		fixture = json.loads(FIXTURE_PATH.read_text())
		self.assertEqual(run_oracle(), fixture)
		self.assertEqual(fixture["upstream"]["commit"], "1242c1e2d9763ba5b85d33f13aecb89bae19ae3d")
		self.assertEqual(fixture["formatId"], "gen5randombattle")
		self.assertEqual(fixture["generation"], 5)
		self.assertEqual(fixture["battleSeed"], [101, 202, 303, 404])
		self.assertIn("sim/battle-actions.ts:1800", fixture["sourcePaths"])
		self.assertIn("sim/battle.ts:934-939", fixture["sourcePaths"])
		self.assertIn("sim/battle.ts:2348-2359", fixture["sourcePaths"])
		self.assertEqual(len(fixture["ordinaryCases"]), 8)
		self.assertEqual(len(fixture["syntheticCases"]), 13)
		self.assertEqual([case["roll"] for case in fixture["ordinaryCases"][0]["outcomes"]], list(range(16)))
		self.assertIn("modifyDamage", fixture["control"])
		self.assertIn("WeatherModifyDamage", fixture["neutralWeatherIdentity"])
		self.assertEqual(fixture["fixedpointIdentityBoundaries"], [
			{"input": 1048575, "output": 1048575},
			{"input": 1048576, "output": 0},
			{"input": 1048577, "output": 1},
		])

	def test_all_live_rolls_match_eager_jit_and_jit_vmap(self):
		fixture = json.loads(FIXTURE_PATH.read_text())
		rows = flat_outcomes(fixture)
		inputs = arrays_from_outcomes(rows)
		expected = [outcome["randomizerOutput"] for _, outcome in rows]
		for result in (
			neutral_damage_after_roll(*inputs),
			jax.jit(neutral_damage_after_roll)(*inputs),
			jax.jit(jax.vmap(neutral_damage_after_roll))(*inputs),
		):
			self.assertEqual(result.shape, (len(rows),))
			self.assertEqual(result.dtype, jnp.int32)
			self.assertEqual(as_lists(result), expected)

	def test_real_event_fixedpoint_identity_and_roll_order(self):
		fixture = json.loads(FIXTURE_PATH.read_text())
		for case in fixture["ordinaryCases"] + fixture["syntheticCases"]:
			for outcome in case["outcomes"]:
				self.assertEqual(outcome["modifyDamageInput"], case["baseCore"])
				self.assertEqual(outcome["weatherInput"], case["baseCore"] + 2)
				self.assertEqual(outcome["weatherOutput"], (case["baseCore"] + 2) & 0xFFFFF)
				self.assertEqual(outcome["randomizerInput"], outcome["weatherOutput"])
		boundary_values = jnp.asarray([1048575, 1048576, 1048577], dtype=jnp.int32)
		for result in (
			_neutral_weather_identity_damage(boundary_values),
			jax.jit(_neutral_weather_identity_damage)(boundary_values),
			jax.jit(jax.vmap(_neutral_weather_identity_damage))(boundary_values),
		):
			self.assertEqual(as_lists(result), [1048575, 0, 1])
		self.assertEqual(int(neutral_damage_after_roll(50, 1, 0, 1, 15)), 1)  # core 0; +2 before 85% roll

	def test_postroll_intermediate_is_not_final_gen5_damage(self):
		fixture = json.loads(FIXTURE_PATH.read_text())
		case = next(row for row in fixture["syntheticCases"] if row["label"] == "postroll-zero-before-gen5-minimum-one")
		outcome = case["outcomes"][0]
		self.assertEqual(case["baseCore"], 1048574)
		self.assertEqual(outcome["weatherOutput"], 0)
		self.assertEqual(outcome["randomizerOutput"], 0)
		self.assertEqual(outcome["finalDamage"], 1)

	def test_core_and_neutral_event_uint32_wrap_are_composed(self):
		fixture = json.loads(FIXTURE_PATH.read_text())
		case = next(row for row in fixture["syntheticCases"] if row["label"] == "core-product-wrap-edge")
		self.assertGreater(case["baseCore"], 1 << 20)
		self.assertEqual(case["outcomes"][0]["weatherOutput"], (case["baseCore"] + 2) & 0xFFFFF)
		inputs = tuple(jnp.asarray([case[key]], dtype=jnp.int32) for key in ("level", "basePower", "attack", "defense"))
		roll = jnp.asarray([0], dtype=jnp.int32)
		actual = neutral_damage_after_roll(*inputs, roll)
		self.assertEqual(as_lists(jax.jit(neutral_damage_after_roll)(*inputs, roll)), as_lists(actual))
		self.assertEqual(int(actual[0]), case["outcomes"][0]["randomizerOutput"])

	def test_broadcasting_including_roll_axis_and_dtype(self):
		levels = jnp.asarray([[[50]], [[100]]], dtype=jnp.int32)
		powers = jnp.asarray([[[1], [50], [255]]], dtype=jnp.int32)
		attacks = jnp.asarray(100, dtype=jnp.int32)
		defenses = jnp.asarray(50, dtype=jnp.int32)
		rolls = jnp.asarray([0, 8, 15, 4], dtype=jnp.int32)
		result = neutral_damage_after_roll(levels, powers, attacks, defenses, rolls)
		compiled = jax.jit(neutral_damage_after_roll)(levels, powers, attacks, defenses, rolls)
		self.assertEqual(result.shape, (2, 3, 4))
		self.assertEqual(result.dtype, jnp.int32)
		self.assertEqual(as_lists(compiled), as_lists(result))
		with self.assertRaises(TypeError):
			neutral_damage_after_roll(50, 100, 100, 100, jnp.asarray(1.0, dtype=jnp.float32))
		with self.assertRaises(TypeError):
			neutral_damage_after_roll(np.asarray(50, dtype=np.int64), 100, 100, 100, 1)


if __name__ == "__main__":
	unittest.main()
