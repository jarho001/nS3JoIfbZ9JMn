"""Pinned Showdown differential tests for Attack stat/stage composition."""

import json
from pathlib import Path
import subprocess
import unittest

import jax
import jax.numpy as jnp
import jaxlib

from mechanics.attack import derive_attack
from mechanics.stats import derive_stored_stats


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "reference" / "fixtures" / "attack-v1.json"
ORACLE_PATH = ROOT / "reference" / "attack-oracle.cjs"


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


def ordinary_inputs(cases):
	return (
		jnp.asarray([case["baseStats"] for case in cases], dtype=jnp.int32),
		jnp.asarray([case["level"] for case in cases], dtype=jnp.int32),
		jnp.asarray([case["ivs"] for case in cases], dtype=jnp.int32),
		jnp.asarray([case["evs"] for case in cases], dtype=jnp.int32),
		jnp.asarray([case["natureMultipliers"] for case in cases], dtype=jnp.int32),
	)


class AttackCompositionTests(unittest.TestCase):
	def test_jax_runtime_matches_pinned_dependencies(self):
		self.assertEqual(jax.__version__, "0.12.0.1")
		self.assertEqual(jaxlib.__version__, "0.12.0.1")

	def test_fixture_is_regenerated_by_guarded_pinned_showdown(self):
		committed = json.loads(FIXTURE_PATH.read_text())
		generated = run_oracle()
		self.assertEqual(generated, committed)
		self.assertEqual(generated["kind"], "gen5-randombattle-attack-stage-composition-oracle")
		self.assertEqual(generated["formatId"], "gen5randombattle")
		self.assertEqual(generated["sourcePaths"], [
			"sim/pokemon.ts:581-612", "sim/pokemon.ts:1408-1436", "sim/battle.ts:2362-2394",
		])
		self.assertEqual(len(generated["cases"]), 25)
		self.assertEqual(len({case["nature"] for case in generated["cases"]}), 25)
		self.assertEqual(generated["conditions"]["ordinaryCases"], "initialized Pokemon; storedStats are not overwritten")
		self.assertIn("synthetic-zero", generated["syntheticZeroCase"]["label"])

	def test_live_composition_matches_eager_jit_and_jit_vmap(self):
		fixture = json.loads(FIXTURE_PATH.read_text())
		cases = fixture["cases"]
		base, levels, ivs, evs, natures = ordinary_inputs(cases)
		stages = jnp.asarray(fixture["stageInputs"], dtype=jnp.int32)
		expected = [case["expected"] for case in cases]
		actual = derive_attack(base[:, None, :], levels[:, None], ivs[:, None, :], evs[:, None, :], natures[:, None, :], stages[None, :])
		compiled = jax.jit(derive_attack)(base[:, None, :], levels[:, None], ivs[:, None, :], evs[:, None, :], natures[:, None, :], stages[None, :])
		vectorized = jax.jit(jax.vmap(derive_attack, in_axes=(0, 0, 0, 0, 0, None)))(base, levels, ivs, evs, natures, stages)
		for result in (actual, compiled, vectorized):
			self.assertEqual(result.dtype, jnp.int32)
			self.assertEqual(as_lists(result), expected)
		self.assertEqual(len({case["nature"] for case in cases}), 25)
		self.assertEqual({case["level"] for case in cases}, {1, 17, 50, 99, 100})

	def test_synthetic_zero_survives_every_stage_in_eager_jit_and_vmap(self):
		case = json.loads(FIXTURE_PATH.read_text())["syntheticZeroCase"]
		self.assertEqual(case["storedAttack"], 0)
		self.assertEqual(case["attackBeforeNature"], 596)
		args = (
			jnp.asarray(case["baseStats"], dtype=jnp.int32), case["level"],
			jnp.asarray(case["ivs"], dtype=jnp.int32), jnp.asarray(case["evs"], dtype=jnp.int32),
			jnp.asarray(case["natureMultipliers"], dtype=jnp.int32),
		)
		stages = jnp.asarray(case["stages"], dtype=jnp.int32)
		expected = case["expected"]
		self.assertEqual(as_lists(derive_attack(*args, stages)), expected)
		self.assertEqual(as_lists(jax.jit(derive_attack)(*args, stages)), expected)
		vectorized = jax.jit(jax.vmap(derive_attack, in_axes=(None, None, None, None, None, 0)))(*args, stages)
		self.assertEqual(as_lists(vectorized), expected)

	def test_nature_rounding_precedes_stage_rounding(self):
		base = jnp.asarray([50, 50, 50, 50, 50, 50], dtype=jnp.int32)
		ivs = jnp.asarray([31, 31, 31, 31, 31, 31], dtype=jnp.int32)
		evs = jnp.asarray([0, 252, 0, 0, 0, 0], dtype=jnp.int32)
		atk_lowering_nature = jnp.asarray([10, 9, 11, 10, 10, 10], dtype=jnp.int32)
		stored = derive_stored_stats(base, 50, ivs, evs, atk_lowering_nature)
		self.assertEqual(int(stored[1]), 91)  # floor(102 * 90 / 100)
		self.assertEqual(int(derive_attack(base, 50, ivs, evs, atk_lowering_nature, 1)), 136)  # floor(91 * 3 / 2)
		self.assertEqual((102 * 3 // 2) * 90 // 100, 137)  # the reversed rounding order differs

	def test_shared_tables_stage_axes_and_multidimensional_broadcast(self):
		base = jnp.asarray([50, 50, 50, 50, 50, 50], dtype=jnp.int32)
		ivs = jnp.full((6,), 31, dtype=jnp.int32)
		evs = jnp.asarray([0, 252, 0, 0, 0, 0], dtype=jnp.int32)
		nature = jnp.asarray([10, 9, 11, 10, 10, 10], dtype=jnp.int32)
		levels = jnp.asarray([[1], [50], [100]], dtype=jnp.int32)
		stages = jnp.asarray([[-6, 0, 1, 6]], dtype=jnp.int32)
		result = derive_attack(base, levels, ivs, evs, nature, stages)
		compiled = jax.jit(derive_attack)(base, levels, ivs, evs, nature, stages)
		self.assertEqual(result.shape, (3, 4))
		self.assertEqual(result.dtype, jnp.int32)
		self.assertEqual(as_lists(compiled), as_lists(result))

		multi_base = jnp.broadcast_to(base, (2, 1, 6))
		multi_levels = jnp.asarray([[1, 50, 100]], dtype=jnp.int32)
		multi_stages = jnp.asarray([[[-6], [0]], [[1], [6]]], dtype=jnp.int32)
		multi = derive_attack(multi_base, multi_levels, ivs, evs, nature, multi_stages)
		self.assertEqual(multi.shape, (2, 2, 3))
		self.assertEqual(as_lists(jax.jit(derive_attack)(multi_base, multi_levels, ivs, evs, nature, multi_stages)), as_lists(multi))

	def test_rejects_non_int32_stage(self):
		base = jnp.full((6,), 50, dtype=jnp.int32)
		ivs = jnp.full((6,), 31, dtype=jnp.int32)
		evs = jnp.zeros((6,), dtype=jnp.int32)
		nature = jnp.full((6,), 10, dtype=jnp.int32)
		with self.assertRaises(TypeError):
			derive_attack(base, 50, ivs, evs, nature, jnp.asarray(1.0, dtype=jnp.float32))


if __name__ == "__main__":
	unittest.main()
