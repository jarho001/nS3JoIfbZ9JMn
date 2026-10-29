"""Differential tests for the bounded Gen 5 physical-burn slice."""

import itertools
import json
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from mechanics.burn import apply_burn_reduction
from mechanics.fixedpoint import apply_fixedpoint_modifier


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "reference/fixtures/burn-v1.json"


def as_lists(value):
	return jax.device_get(value).tolist()


class BurnReductionTests(unittest.TestCase):
	def test_guarded_fixture_and_provenance(self):
		fixture = json.loads(FIXTURE.read_text())
		self.assertEqual(fixture["kind"], "gen5-physical-burn-reduction-oracle")
		self.assertEqual(fixture["upstream"]["commit"], "1242c1e2d9763ba5b85d33f13aecb89bae19ae3d")
		self.assertEqual(fixture["formatId"], "gen5randombattle")
		self.assertEqual(fixture["inputDomain"], {
			"damage": [0, 2147483647], "eligibility": "boolean", "result": "int32",
		})
		self.assertIn("no Facade exemption", fixture["scope"])
		self.assertIn("transparent Battle.modify wrapper", fixture["syntheticContract"])
		ordinary = {case["label"]: case for case in fixture["ordinaryCases"]}
		self.assertEqual(ordinary["burned-physical"]["modifyCalls"], 1)
		self.assertEqual(ordinary["burned-special"]["modifyCalls"], 0)
		self.assertEqual(ordinary["burned-physical-guts"]["modifyCalls"], 0)
		self.assertEqual(ordinary["burned-facade-gen5"]["modifyCalls"], 1)
		self.assertEqual(ordinary["unburned-physical"]["modifyCalls"], 0)

	def test_synthetic_live_cases_eager_jit_and_vmap(self):
		fixture = json.loads(FIXTURE.read_text())
		cases = fixture["syntheticCases"]
		damage = jnp.asarray([case["damage"] for case in cases], dtype=jnp.int32)
		burned = jnp.ones(damage.shape, dtype=jnp.bool_)
		physical = jnp.ones(damage.shape, dtype=jnp.bool_)
		guts = jnp.zeros(damage.shape, dtype=jnp.bool_)
		expected = [case["burnOutput"] for case in cases]
		for result in (
			apply_burn_reduction(damage, burned, physical, guts),
			jax.jit(apply_burn_reduction)(damage, burned, physical, guts),
			jax.jit(jax.vmap(apply_burn_reduction))(damage, burned, physical, guts),
		):
			self.assertEqual(result.dtype, jnp.int32)
			self.assertEqual(as_lists(result), expected)
		self.assertEqual(cases[1]["burnOutput"], 0)
		self.assertEqual(cases[7]["burnOutput"], 0)
		self.assertEqual(cases[7]["postBurnAndGen5FinalDamage"], 1)

		ordinary = [case for case in fixture["ordinaryCases"] if case["modifyCalls"]]
		ordinary_damage = jnp.asarray([case["burnInput"] for case in ordinary], dtype=jnp.int32)
		ordinary_result = apply_burn_reduction(
			ordinary_damage,
			jnp.asarray([case["burned"] for case in ordinary], dtype=jnp.bool_),
			jnp.asarray([case["category"] == "Physical" for case in ordinary], dtype=jnp.bool_),
			jnp.asarray([case["ability"] == "guts" for case in ordinary], dtype=jnp.bool_),
		)
		self.assertEqual(as_lists(ordinary_result), [case["burnOutput"] for case in ordinary])
		self.assertIn("burned-facade-gen5", [case["label"] for case in ordinary])

	def test_predicate_truth_table_broadcast_and_int32_boundary(self):
		flags = list(itertools.product([False, True], repeat=3))
		damage = jnp.asarray([1, 2, 3, 4, 5, 2 ** 21, 2147483647], dtype=jnp.int32)[:, None]
		burned = jnp.asarray([row[0] for row in flags], dtype=jnp.bool_)[None, :]
		physical = jnp.asarray([row[1] for row in flags], dtype=jnp.bool_)[None, :]
		guts = jnp.asarray([row[2] for row in flags], dtype=jnp.bool_)[None, :]
		result = apply_burn_reduction(damage, burned, physical, guts)
		compiled = jax.jit(apply_burn_reduction)(damage, burned, physical, guts)
		self.assertEqual(result.shape, (7, 8))
		self.assertEqual(result.dtype, jnp.int32)
		self.assertEqual(as_lists(compiled), as_lists(result))
		values = [1, 2, 3, 4, 5, 2 ** 21, 2147483647]
		half = as_lists(apply_fixedpoint_modifier(damage[:, 0], jnp.asarray(2048, jnp.int32)))
		for column, (is_burned, is_physical, has_guts) in enumerate(flags):
			eligible = is_burned and is_physical and not has_guts
			self.assertEqual(as_lists(result[:, column]), half if eligible else values)

	def test_rejects_non_int32_damage_and_non_bool_eligibility(self):
		with self.assertRaises(TypeError):
			apply_burn_reduction(jnp.asarray(1.0, dtype=jnp.float32), True, True, False)
		for position in range(1, 4):
			args = [jnp.asarray(1, dtype=jnp.int32), True, True, False]
			args[position] = jnp.asarray(args[position], dtype=jnp.int32)
			with self.assertRaises(TypeError):
				apply_burn_reduction(*args)
		with self.assertRaises(TypeError):
			apply_burn_reduction(np.asarray(1, dtype=np.int64), True, True, False)


if __name__ == "__main__":
	unittest.main()
