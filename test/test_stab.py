"""Differential tests for the bounded Gen 5 STAB arithmetic slice."""

import itertools
import json
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from mechanics.fixedpoint import apply_fixedpoint_modifier
from mechanics.stab import apply_stab


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "reference/fixtures/stab-v1.json"


def as_lists(value):
	return jax.device_get(value).tolist()


class StabTests(unittest.TestCase):
	def test_guarded_live_fixture_and_provenance(self):
		fixture = json.loads(FIXTURE.read_text())
		self.assertEqual(fixture["kind"], "gen5-stab-arithmetic-oracle")
		self.assertEqual(fixture["upstream"]["commit"], "1242c1e2d9763ba5b85d33f13aecb89bae19ae3d")
		self.assertEqual(fixture["formatId"], "gen5randombattle")
		self.assertEqual(fixture["inputDomain"], {
			"damage": [0, 2147483647], "flags": "strict boolean", "result": "int32",
		})
		self.assertEqual(fixture["modifiers"], {"typedNonStab": 4096, "stab": 6144, "adaptability": 8192})
		self.assertEqual(fixture["sourcePaths"][0], "sim/battle-actions.ts:1809-1847")
		self.assertEqual(fixture["sourcePaths"][1], "data/abilities.ts:43-54")
		ordinary = {case["label"]: case for case in fixture["ordinaryCases"]}
		self.assertEqual(ordinary["ordinary-native-stab"]["modifier"], 6144)
		self.assertEqual(ordinary["ordinary-typed-non-stab"]["modifier"], 4096)
		self.assertEqual(ordinary["ordinary-adaptability-stab"]["modifier"], 8192)
		self.assertEqual(ordinary["ordinary-adaptability-non-stab"]["modifier"], 4096)
		self.assertEqual(ordinary["ordinary-force-stab"]["modifier"], 6144)
		self.assertFalse(ordinary["ordinary-native-stab"]["forceSTAB"])
		self.assertTrue(ordinary["ordinary-force-stab"]["forceSTAB"])
		synthetic = fixture["syntheticCases"]
		self.assertEqual(len(synthetic), 47)
		self.assertEqual({case["damage"] for case in synthetic if case["label"].startswith("typed-stab-")}
			, {0, 1, 2, 3, *fixture["wrapAnchors"], 2147483647})
		typeless = [case for case in synthetic if case["typeless"]]
		self.assertEqual(len(typeless), 5)
		self.assertTrue(all(case["modifyStabCalls"] == 0 and case["modifyCalls"] == 0
			and case["preTypeModifyCalls"] == 0 and case["modifiedOutput"] == case["damage"]
			for case in typeless))

	def test_live_ordinary_and_synthetic_oracle_eager_jit_vmap(self):
		fixture = json.loads(FIXTURE.read_text())
		cases = fixture["ordinaryCases"] + fixture["syntheticCases"]
		damage = jnp.asarray([case["damage"] for case in cases], dtype=jnp.int32)
		eligible = jnp.asarray([case["stabEligible"] for case in cases], dtype=jnp.bool_)
		adaptability = jnp.asarray([case["adaptability"] for case in cases], dtype=jnp.bool_)
		typeless = jnp.asarray([case["typeless"] for case in cases], dtype=jnp.bool_)
		expected = [case["modifiedOutput"] for case in cases]
		for result in (
			apply_stab(damage, eligible, adaptability, typeless),
			jax.jit(apply_stab)(damage, eligible, adaptability, typeless),
			jax.jit(jax.vmap(apply_stab))(damage, eligible, adaptability, typeless),
		):
			self.assertEqual(result.shape, damage.shape)
			self.assertEqual(result.dtype, jnp.int32)
			self.assertEqual(as_lists(result), expected)

	def test_all_flag_combinations_broadcast_and_maximum(self):
		flags = list(itertools.product([False, True], repeat=3))
		values = [0, 1, 2, 3, 524287, 524288, 524289, 699049, 699050, 699051,
			1048575, 1048576, 1048577, 2147483647]
		damage = jnp.asarray(values, dtype=jnp.int32)[:, None]
		eligible = jnp.asarray([row[0] for row in flags], dtype=jnp.bool_)[None, :]
		adaptability = jnp.asarray([row[1] for row in flags], dtype=jnp.bool_)[None, :]
		typeless = jnp.asarray([row[2] for row in flags], dtype=jnp.bool_)[None, :]
		result = apply_stab(damage, eligible, adaptability, typeless)
		compiled = jax.jit(apply_stab)(damage, eligible, adaptability, typeless)
		self.assertEqual(result.shape, (len(values), 8))
		self.assertEqual(result.dtype, jnp.int32)
		self.assertEqual(as_lists(compiled), as_lists(result))
		for column, (is_eligible, has_adaptability, is_typeless) in enumerate(flags):
			modifier = 8192 if is_eligible and has_adaptability else 6144 if is_eligible else 4096
			modified = as_lists(apply_fixedpoint_modifier(damage[:, 0], jnp.asarray(modifier, jnp.int32)))
			expected = values if is_typeless else modified
			self.assertEqual(as_lists(result[:, column]), expected)

	def test_rejects_non_int32_damage_and_non_bool_flags(self):
		with self.assertRaises(TypeError):
			apply_stab(jnp.asarray(1.0, dtype=jnp.float32), True, False, False)
		for position in range(1, 4):
			args = [jnp.asarray(1, dtype=jnp.int32), True, False, False]
			args[position] = jnp.asarray(args[position], dtype=jnp.int32)
			with self.assertRaises(TypeError):
				apply_stab(*args)
		with self.assertRaises(TypeError):
			apply_stab(np.asarray(1, dtype=np.int64), True, False, False)


if __name__ == "__main__":
	unittest.main()
