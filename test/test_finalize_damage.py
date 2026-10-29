"""Differential tests for the bounded Gen 5 damage-finalization tail."""

import json
from pathlib import Path
import subprocess
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from mechanics.finalize_damage import finalize_damage
from mechanics.neutral_damage import neutral_damage_after_roll


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "reference/fixtures/finalize-damage-v1.json"


class FinalizeDamageTests(unittest.TestCase):
	def test_guarded_live_fixture_and_provenance(self):
		live = subprocess.run(["node", "reference/finalize-damage-oracle.cjs", "fixture"],
			cwd=ROOT, check=True, capture_output=True, text=True, timeout=60)
		self.assertEqual(live.stdout, FIXTURE.read_text())
		fixture = json.loads(live.stdout)
		self.assertEqual(fixture["upstream"]["commit"], "1242c1e2d9763ba5b85d33f13aecb89bae19ae3d")
		self.assertEqual(fixture["generation"], 5)
		self.assertEqual(fixture["formatId"], "gen5randombattle")
		self.assertEqual(fixture["inputDomain"], {
			"damage": [0, 2147483647], "encodedModifier": [0, 65535], "result": "int32 [0,65535]",
		})
		self.assertIn("only randomizer output is replaced", fixture["syntheticContract"])
		self.assertIn("no arithmetic injection", fixture["ordinaryContract"])

	def test_synthetic_live_grid_eager_jit_vmap(self):
		grid = json.loads(FIXTURE.read_text())["controlledGrid"]
		values = jnp.asarray(grid["values"], dtype=jnp.int32)
		modifiers = jnp.asarray(grid["encodedModifiers"], dtype=jnp.int32)
		expected = [[case["result"] for case in row] for row in grid["cases"]]
		for result in (
			finalize_damage(values[:, None], modifiers),
			jax.jit(finalize_damage)(values[:, None], modifiers),
			jax.jit(jax.vmap(finalize_damage, in_axes=(0, None)))(values, modifiers),
		):
			self.assertEqual(result.dtype, jnp.int32)
			self.assertEqual(result.shape, (43, 10))
			self.assertEqual(jax.device_get(result).tolist(), expected)

	def test_ordinary_real_damage_and_life_orb(self):
		cases = json.loads(FIXTURE.read_text())["ordinaryCases"]
		self.assertEqual(len(cases), 8)
		for case in cases:
			self.assertEqual(case["encodedModifier"], 5324 if case["item"] else 4096)
		values = jnp.asarray([case["damage"] for case in cases], dtype=jnp.int32)
		modifiers = jnp.asarray([case["encodedModifier"] for case in cases], dtype=jnp.int32)
		self.assertEqual(jax.device_get(jax.jit(finalize_damage)(values, modifiers)).tolist(),
			[case["result"] for case in cases])

	def test_order_zero_rounding_and_wrap_anchors(self):
		anchors = [
			(0, 4096, 1), (0, 8192, 2),  # minimum before modifier
			(0, 0, 0), (0, 2048, 0), (0, 2049, 1),  # no second minimum
			(3, 2048, 1), (3, 2049, 2),  # half ties down
			(65535, 4096, 65535), (65536, 4096, 0), (65537, 4096, 1),
			(32768, 8192, 0),  # final truncation after modifier
			(1048576, 4096, 0), (1048577, 4096, 1),  # uint32 product wrap
			(65537, 65535, 0),  # rounding carry yields 2**20, then truncates
			(2147483647, 65535, 65520),
		]
		for value, modifier, expected in anchors:
			with self.subTest(value=value, modifier=modifier):
				self.assertEqual(int(finalize_damage(value, modifier)), expected)

	def test_neutral_damage_composition_matches_prior_oracle(self):
		fixture = json.loads((ROOT / "reference/fixtures/neutral-damage-v1.json").read_text())
		for case in fixture["ordinaryCases"] + fixture["syntheticCases"]:
			rolls = jnp.asarray([row["roll"] for row in case["outcomes"]], dtype=jnp.int32)
			intermediate = neutral_damage_after_roll(case["level"], case["basePower"], case["attack"], case["defense"], rolls)
			result = jax.jit(finalize_damage)(intermediate, 4096)
			self.assertEqual(jax.device_get(result).tolist(), [row["finalDamage"] for row in case["outcomes"]])

	def test_multiaxis_broadcast(self):
		values = jnp.asarray([0, 1, 65536], dtype=jnp.int32).reshape(3, 1, 1)
		modifiers = jnp.asarray([0, 2048, 4096, 8192], dtype=jnp.int32).reshape(1, 2, 2)
		result = jax.jit(finalize_damage)(values, modifiers)
		self.assertEqual(result.shape, (3, 2, 2))
		self.assertEqual(jax.device_get(result).tolist(), [[[0, 0], [1, 2]], [[0, 0], [1, 2]], [[0, 32768], [0, 0]]])

	def test_strict_dtype_on_both_arguments(self):
		for invalid in (jnp.asarray(1.0, dtype=jnp.float32), jnp.asarray(1, dtype=jnp.uint32),
			np.asarray(1, dtype=np.int64), True):
			with self.assertRaises(TypeError):
				finalize_damage(invalid, 4096)
			with self.assertRaises(TypeError):
				finalize_damage(1, invalid)


if __name__ == "__main__":
	unittest.main()
