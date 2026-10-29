"""Differential tests for the bounded Gen 5 type-effectiveness arithmetic."""

import json
from pathlib import Path
import subprocess
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from mechanics.type_effectiveness import apply_type_effectiveness


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "reference/fixtures/type-effectiveness-v1.json"


def as_lists(value):
	return jax.device_get(value).tolist()


class TypeEffectivenessTests(unittest.TestCase):
	def test_guarded_live_fixture_and_provenance(self):
		live = subprocess.run(
			["node", "reference/type-effectiveness-oracle.cjs", "fixture"],
			cwd=ROOT, check=True, capture_output=True, text=True, timeout=60,
		)
		self.assertEqual(live.stdout, FIXTURE.read_text())
		fixture = json.loads(live.stdout)
		self.assertEqual(fixture["upstream"]["commit"], "1242c1e2d9763ba5b85d33f13aecb89bae19ae3d")
		self.assertEqual(fixture["formatId"], "gen5randombattle")
		self.assertEqual(fixture["sourcePaths"], [
			"sim/battle-actions.ts:1849-1865",
			"sim/battle-actions.ts:1870-1877",
			"sim/battle-actions.ts:1640-1775",
		])
		self.assertEqual(fixture["inputDomain"], {
			"damage": [0, 33554431],
			"exponent": [-2147483648, 2147483647],
			"result": "int32 [0,2147483584]",
		})
		self.assertIn("no arithmetic or type lookup overrides", fixture["ordinaryContract"])
		self.assertIn("before burn", fixture["syntheticContract"])
		self.assertEqual({case["typeMod"] for case in fixture["ordinaryCases"]}, {-1, 0, 1})
		self.assertEqual(len(fixture["syntheticCases"]), 68)
		grid = fixture["syntheticCases"][:65]
		self.assertEqual({case["damage"] for case in grid}, {0, 1, 63, 65, 33554431})
		self.assertEqual({case["clampedExponent"] for case in grid}, set(range(-6, 7)))
		self.assertEqual(
			[case["label"] for case in fixture["syntheticCases"][65:]],
			["clamped-minimum-zero-result", "clamped-minimum-positive-result", "clamped-maximum"],
		)

	def test_live_ordinary_and_synthetic_results_eager_jit_vmap(self):
		fixture = json.loads(FIXTURE.read_text())
		ordinary = fixture["ordinaryCases"]
		synthetic = fixture["syntheticCases"]
		damage = jnp.asarray(
			[case["damageBeforeType"] for case in ordinary] + [case["damage"] for case in synthetic],
			dtype=jnp.int32,
		)
		exponent = jnp.asarray(
			[case["typeMod"] for case in ordinary] + [case["exponent"] for case in synthetic],
			dtype=jnp.int32,
		)
		expected = [case["damageAfterType"] for case in ordinary] + [case["result"] for case in synthetic]
		for result in (
			apply_type_effectiveness(damage, exponent),
			jax.jit(apply_type_effectiveness)(damage, exponent),
			jax.jit(jax.vmap(apply_type_effectiveness))(damage, exponent),
		):
			self.assertEqual(result.shape, (len(expected),))
			self.assertEqual(result.dtype, jnp.int32)
			self.assertEqual(as_lists(result), expected)

	def test_broadcast_extremes_and_maximum_domain_boundary(self):
		damage = jnp.asarray([0, 1, 3, 5, 127, 33554431], dtype=jnp.int32)[:, None]
		exponents = jnp.asarray([-2147483648, -6, -2, -1, 0, 1, 6, 2147483647], dtype=jnp.int32)[None, :]
		result = apply_type_effectiveness(damage, exponents)
		compiled = jax.jit(apply_type_effectiveness)(damage, exponents)
		self.assertEqual(result.shape, (6, 8))
		self.assertEqual(result.dtype, jnp.int32)
		self.assertEqual(as_lists(compiled), as_lists(result))
		self.assertEqual(as_lists(result[0]), [0] * 8)
		self.assertEqual(as_lists(result[1]), [0, 0, 0, 0, 1, 2, 64, 64])
		self.assertEqual(as_lists(result[2]), [0, 0, 0, 1, 3, 6, 192, 192])
		self.assertEqual(as_lists(result[3]), [0, 0, 1, 2, 5, 10, 320, 320])
		self.assertEqual(as_lists(result[4]), [1, 1, 31, 63, 127, 254, 8128, 8128])
		self.assertEqual(as_lists(result[5]), [524287, 524287, 8388607, 16777215, 33554431, 67108862, 2147483584, 2147483584])

	def test_rejects_non_int32_inputs(self):
		with self.assertRaises(TypeError):
			apply_type_effectiveness(jnp.asarray(3.0, dtype=jnp.float32), 1)
		with self.assertRaises(TypeError):
			apply_type_effectiveness(3, jnp.asarray(1.0, dtype=jnp.float32))
		with self.assertRaises(TypeError):
			apply_type_effectiveness(np.asarray(3, dtype=np.int64), 1)
