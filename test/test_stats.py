"""Pinned Showdown differential tests for Gen 5 stored-stat derivation."""

import json
from pathlib import Path
import subprocess
import unittest

import jax
import jax.numpy as jnp
import jaxlib

from mechanics.stats import derive_stored_stats


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "reference" / "fixtures" / "stats-v1.json"
ORACLE_PATH = ROOT / "reference" / "stats-oracle.cjs"


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


def arrays_from_cases(cases):
	return (
		jnp.asarray([case["baseStats"] for case in cases], dtype=jnp.int32),
		jnp.asarray([case["level"] for case in cases], dtype=jnp.int32),
		jnp.asarray([case["ivs"] for case in cases], dtype=jnp.int32),
		jnp.asarray([case["evs"] for case in cases], dtype=jnp.int32),
		jnp.asarray([case["natureMultipliers"] for case in cases], dtype=jnp.int32),
		jnp.asarray([case["shedinja"] for case in cases], dtype=jnp.bool_),
	)


def as_lists(value):
	return jax.device_get(value).tolist()


class StoredStatTests(unittest.TestCase):
	def test_jax_runtime_matches_pinned_dependencies(self):
		self.assertEqual(jax.__version__, "0.12.0.1")
		self.assertEqual(jaxlib.__version__, "0.12.0.1")

	def test_live_fixture_reproduction_and_target_format_facts(self):
		fixture = json.loads(FIXTURE_PATH.read_text())
		self.assertEqual(run_oracle(), fixture)
		self.assertEqual(fixture["schemaVersion"], 1)
		self.assertEqual(fixture["upstream"]["commit"], "1242c1e2d9763ba5b85d33f13aecb89bae19ae3d")
		self.assertEqual(fixture["formatId"], "gen5randombattle")
		self.assertEqual(fixture["formatFacts"]["overflowstatmod"], False)
		self.assertEqual(len({case["nature"] for case in fixture["cases"]}), 25)
		self.assertEqual({case["level"] for case in fixture["cases"]}, {1, 17, 50, 99, 100})
		self.assertGreater(sum(case["shedinja"] for case in fixture["cases"]), 0)
		self.assertIn("sim/pokemon.ts:1408-1436", fixture["sourcePaths"])
		self.assertIn("sim/battle.ts:2362-2394", fixture["sourcePaths"])
		self.assertIn("sim/dex.ts:365-367", fixture["sourcePaths"])

	def test_actual_pokemon_outputs_match_eager_jit_and_jit_vmap(self):
		fixture = json.loads(FIXTURE_PATH.read_text())
		cases = fixture["cases"]
		inputs = arrays_from_cases(cases)
		expected = [case["expected"] for case in cases]
		self.assertEqual(as_lists(derive_stored_stats(*inputs)), expected)
		compiled = jax.jit(derive_stored_stats)(*inputs)
		self.assertEqual(compiled.dtype, jnp.int32)
		self.assertEqual(as_lists(compiled), expected)
		vectorized = jax.jit(jax.vmap(derive_stored_stats))(*inputs)
		self.assertEqual(vectorized.dtype, jnp.int32)
		self.assertEqual(as_lists(vectorized), expected)

	def test_real_stat_modify_nature_wrap_thresholds(self):
		fixture = json.loads(FIXTURE_PATH.read_text())
		cases = fixture["syntheticStatModifyCases"]
		inputs = arrays_from_cases(cases)
		expected = [case["expected"] for case in cases]
		self.assertEqual(as_lists(derive_stored_stats(*inputs)), expected)
		self.assertEqual(as_lists(jax.jit(derive_stored_stats)(*inputs)), expected)
		self.assertEqual(as_lists(jax.jit(jax.vmap(derive_stored_stats))(*inputs)), expected)
		by_key = {(case["label"], case["nature"]): case for case in cases}
		self.assertEqual(by_key[("pre-nature-595", "Adamant")]["attackBeforeNature"], 595)
		self.assertEqual(by_key[("pre-nature-596", "Adamant")]["attackBeforeNature"], 596)
		self.assertEqual(by_key[("pre-nature-595", "Adamant")]["expected"][1], 654)
		self.assertEqual(by_key[("pre-nature-596", "Adamant")]["expected"][1], 0)
		self.assertEqual(by_key[("base-stat-upper-bound", "Adamant")]["expected"][1], 14)
		self.assertEqual(by_key[("base-stat-upper-bound", "Modest")]["expected"][1], 548)
		self.assertEqual(by_key[("base-stat-upper-bound", "Hardy")]["expected"][1], 609)

	def test_hand_anchors_floor_nature_rounding_and_shedinja_override(self):
		base = jnp.asarray([1, 50, 50, 50, 50, 50], dtype=jnp.int32)
		ivs = jnp.asarray([0, 31, 31, 31, 31, 31], dtype=jnp.int32)
		evs = jnp.asarray([0, 252, 252, 252, 252, 252], dtype=jnp.int32)
		nature = jnp.asarray([10, 11, 9, 10, 10, 10], dtype=jnp.int32)
		ordinary = derive_stored_stats(base, 50, ivs, evs, nature, False)
		self.assertEqual(as_lists(ordinary), [61, 112, 91, 102, 102, 102])

		base_hp_one = base.at[0].set(1)
		hp_ivs = ivs.at[0].set(31)
		hp_inputs = (base_hp_one, 100, hp_ivs, jnp.full((6,), 255, jnp.int32), jnp.full((6,), 10, jnp.int32))
		without_species_override = derive_stored_stats(*hp_inputs, False)
		with_shedinja_override = derive_stored_stats(*hp_inputs, True)
		self.assertEqual(int(without_species_override[0]), 206)
		self.assertEqual(int(with_shedinja_override[0]), 1)

	def test_shared_stat_tables_broadcast_over_batched_levels_and_shedinja_flags(self):
		base = jnp.asarray([1, 50, 50, 50, 50, 50], dtype=jnp.int32)
		ivs = jnp.asarray([31, 31, 31, 31, 31, 31], dtype=jnp.int32)
		evs = jnp.asarray([255, 252, 252, 252, 252, 252], dtype=jnp.int32)
		nature = jnp.asarray([10, 11, 9, 10, 10, 10], dtype=jnp.int32)
		levels = jnp.asarray([1, 50, 100], dtype=jnp.int32)
		shedinja = jnp.asarray([False, True, False], dtype=jnp.bool_)
		result = derive_stored_stats(base, levels, ivs, evs, nature, shedinja)
		self.assertEqual(result.shape, (3, 6))
		self.assertEqual(result.dtype, jnp.int32)
		self.assertEqual(int(result[1, 0]), 1)
		self.assertEqual(as_lists(jax.jit(derive_stored_stats)(base, levels, ivs, evs, nature, shedinja)), as_lists(result))

	def test_final_axis_six_batch_broadcast_and_dtype_contract(self):
		base = jnp.full((2, 1, 6), 50, dtype=jnp.int32)
		ivs = jnp.zeros((1, 3, 6), dtype=jnp.int32)
		evs = jnp.full((6,), 4, dtype=jnp.int32)
		nature = jnp.full((2, 3, 6), 10, dtype=jnp.int32)
		levels = jnp.asarray([[1], [100]], dtype=jnp.int32)
		shedinja = jnp.asarray([[False, True, False]], dtype=jnp.bool_)
		result = derive_stored_stats(base, levels, ivs, evs, nature, shedinja)
		self.assertEqual(result.shape, (2, 3, 6))
		self.assertEqual(result.dtype, jnp.int32)
		self.assertEqual(int(result[1, 1, 0]), 1)
		self.assertEqual(as_lists(jax.jit(derive_stored_stats)(base, levels, ivs, evs, nature, shedinja)), as_lists(result))

		with self.assertRaises(ValueError):
			derive_stored_stats(jnp.ones((5,), dtype=jnp.int32), 50, jnp.ones((5,), dtype=jnp.int32), evs, nature)
		with self.assertRaises(TypeError):
			derive_stored_stats(base.astype(jnp.float32), levels, ivs, evs, nature)
		with self.assertRaises(TypeError):
			derive_stored_stats(base, levels, ivs, evs, nature, jnp.zeros((2, 3), dtype=jnp.int32))


if __name__ == "__main__":
	unittest.main()
