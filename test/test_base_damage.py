"""Pinned Showdown differential tests for the base-damage integer core."""

import json
from pathlib import Path
import subprocess
import unittest

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from mechanics.base_damage import base_damage_core


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "reference" / "fixtures" / "base-damage-v1.json"
ORACLE_PATH = ROOT / "reference" / "base-damage-oracle.cjs"


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


def inputs_from_cases(cases):
	return tuple(
		jnp.asarray([case[key] for case in cases], dtype=jnp.int32)
		for key in ("level", "basePower", "attack", "defense")
	)


class BaseDamageTests(unittest.TestCase):
	def test_jax_runtime_matches_pinned_dependencies(self):
		self.assertEqual(jax.__version__, "0.12.0.1")
		self.assertEqual(jaxlib.__version__, "0.12.0.1")

	def test_guarded_fixture_regenerates_and_records_scope(self):
		committed = json.loads(FIXTURE_PATH.read_text())
		generated = run_oracle()
		self.assertEqual(generated, committed)
		self.assertEqual(generated["upstream"]["commit"], "1242c1e2d9763ba5b85d33f13aecb89bae19ae3d")
		self.assertEqual(generated["formatId"], "gen5randombattle")
		self.assertEqual(generated["method"], "BattleActions.getDamage")
		self.assertEqual(generated["sourcePaths"], [
			"sim/battle-actions.ts:1769-1775", "sim/battle-actions.ts:1778-1786", "sim/dex.ts:364-369",
		])
		self.assertIn("basePower * attack", generated["coreExpression"])
		self.assertIn("unsigned 32-bit conversion", generated["truncation"])
		self.assertEqual(generated["context"]["captureBoundary"].split()[0], "modifyDamage")
		self.assertEqual(len(generated["ordinaryCases"]), 8)
		self.assertEqual(len(generated["syntheticCases"]), 1218)
		self.assertEqual({case["level"] for case in generated["syntheticCases"]}, {1, 2, 4, 5, 17, 50, 99, 100})

	def test_ordinary_and_synthetic_oracle_cases_match_eager_jit_and_jit_vmap(self):
		fixture = json.loads(FIXTURE_PATH.read_text())
		cases = fixture["ordinaryCases"] + fixture["syntheticCases"]
		inputs = inputs_from_cases(cases)
		expected = [case["baseDamage"] for case in cases]
		for result in (
			base_damage_core(*inputs),
			jax.jit(base_damage_core)(*inputs),
			jax.jit(jax.vmap(base_damage_core))(*inputs),
		):
			self.assertEqual(result.shape, (len(cases),))
			self.assertEqual(result.dtype, jnp.int32)
			self.assertEqual(as_lists(result), expected)

	def test_zero_defense_matches_pinned_unsigned_conversion_without_invalid_division(self):
		fixture = json.loads(FIXTURE_PATH.read_text())
		cases = [case for case in fixture["syntheticCases"] if case["defense"] == 0]
		self.assertTrue(cases)
		inputs = inputs_from_cases(cases)
		for result in (base_damage_core(*inputs), jax.jit(base_damage_core)(*inputs), jax.jit(jax.vmap(base_damage_core))(*inputs)):
			self.assertEqual(as_lists(result), [0] * len(cases))

	def test_independent_floor_wrap_and_boundary_anchors(self):
		anchors = [
			(1, 1, 1, 1, 0),      # floor((2 * 1 * 1 / 1) / 50)
			(50, 100, 100, 100, 44),  # floor((22 * 100 * 100 / 100) / 50)
			(50, 3, 4, 2, 2),
			(100, 1560, 65535, 1, 85877064),
			(100, 1561, 65535, 1, 32767),  # product crosses uint32 boundary
			(100, 65535, 65535, 0, 0),
			(100, 65535, 0, 1, 0),
		]
		for level, power, attack, defense, expected in anchors:
			with self.subTest(level=level, power=power, attack=attack, defense=defense):
				self.assertEqual(int(base_damage_core(level, power, attack, defense)), expected)
		self.assertEqual((42 * 1561 * 65535) & 0xFFFFFFFF, 1638374)

	def test_broadcasting_and_int32_dtype(self):
		levels = jnp.asarray([[1], [50]], dtype=jnp.int32)
		powers = jnp.asarray([[1, 3, 100]], dtype=jnp.int32)
		attacks = jnp.asarray([[1], [100]], dtype=jnp.int32)
		defenses = jnp.asarray([[1], [100]], dtype=jnp.int32)
		result = base_damage_core(levels, powers, attacks, defenses)
		compiled = jax.jit(base_damage_core)(levels, powers, attacks, defenses)
		self.assertEqual(result.shape, (2, 3))
		self.assertEqual(result.dtype, jnp.int32)
		self.assertEqual(as_lists(compiled), as_lists(result))

	def test_rejects_non_int32_inputs(self):
		inputs = [1, 50, 10, 10]
		for index in range(4):
			bad = list(inputs)
			bad[index] = jnp.asarray(1.0, dtype=jnp.float32)
			with self.subTest(index=index), self.assertRaises(TypeError):
				base_damage_core(*bad)
		with self.assertRaises(TypeError):
			base_damage_core(np.asarray(1, dtype=np.int64), 50, 10, 10)


if __name__ == "__main__":
	unittest.main()
