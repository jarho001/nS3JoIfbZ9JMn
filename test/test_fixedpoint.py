"""Pinned Showdown differential tests for fixed-point modifier arithmetic."""

import json
from pathlib import Path
import subprocess
import unittest

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from mechanics.fixedpoint import apply_fixedpoint_modifier
from mechanics.neutral_damage import _neutral_weather_identity_damage


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "reference" / "fixtures" / "fixedpoint-v1.json"
ORACLE_PATH = ROOT / "reference" / "fixedpoint-oracle.cjs"


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


class FixedPointTests(unittest.TestCase):
	def test_jax_runtime_matches_pinned_dependencies(self):
		self.assertEqual(jax.__version__, "0.12.0.1")
		self.assertEqual(jaxlib.__version__, "0.12.0.1")

	def test_guarded_fixture_regenerates_and_records_target_contract(self):
		fixture = json.loads(FIXTURE_PATH.read_text())
		self.assertEqual(run_oracle(), fixture)
		self.assertEqual(fixture["schemaVersion"], 1)
		self.assertEqual(fixture["kind"], "gen5-fixedpoint-modifier-oracle")
		self.assertEqual(fixture["upstream"]["commit"], "1242c1e2d9763ba5b85d33f13aecb89bae19ae3d")
		self.assertEqual(fixture["formatId"], "gen5randombattle")
		self.assertEqual(fixture["method"], "Battle.modify(value, modifier, 4096)")
		self.assertEqual(fixture["sourcePaths"], ["sim/battle.ts:2348-2359", "sim/dex.ts:364-369"])
		self.assertEqual(fixture["inputDomain"], {
			"value": [0, 2147483647], "encodedModifier": [0, 65535], "result": "int32",
		})
		grid = fixture["controlledGrid"]
		self.assertEqual(grid["values"], list(range(1025)))
		self.assertEqual(grid["encodedModifiers"], [0, 1, 2047, 2048, 2049, 3072, 4095, 4096, 4097, 5324, 6144, 8192, 65535])
		self.assertIn("no arithmetic or event overrides", grid["purpose"])

	def test_full_live_grid_matches_eager_jit_and_jit_vmap(self):
		grid = json.loads(FIXTURE_PATH.read_text())["controlledGrid"]
		values = jnp.asarray(grid["values"], dtype=jnp.int32)[:, None]
		modifiers = jnp.asarray(grid["encodedModifiers"], dtype=jnp.int32)[None, :]
		expected = grid["results"]
		for result in (
			apply_fixedpoint_modifier(values, modifiers),
			jax.jit(apply_fixedpoint_modifier)(values, modifiers),
			jax.jit(jax.vmap(apply_fixedpoint_modifier, in_axes=(0, None)))(values[:, 0], modifiers[0]),
		):
			self.assertEqual(result.shape, (len(grid["values"]), len(grid["encodedModifiers"])))
			self.assertEqual(result.dtype, jnp.int32)
			self.assertEqual(as_lists(result), expected)

	def test_round_half_down_product_wrap_and_int32_anchors(self):
		anchors = [
			(0, 65535, 0),
			(1, 2047, 0),
			(1, 2048, 0),  # remainder exactly 2048 rounds down
			(1, 2049, 1),  # remainder above half rounds up
			(3, 2048, 1),
			(3, 2049, 2),
			(1, 6144, 1),  # 1.5 ties down
			(3, 6144, 4),  # 4.5 ties down
			(1, 8192, 2),
			(65537, 65535, 1048576),  # uint32 product is 2**32-1; the later +2047 does not wrap
			(2147483647, 65535, 524272),
		]
		for value, modifier, expected in anchors:
			with self.subTest(value=value, modifier=modifier):
				self.assertEqual(int(apply_fixedpoint_modifier(value, modifier)), expected)
		self.assertEqual((65537 * 65535) & 0xFFFFFFFF, 0xFFFFFFFF)

	def test_targeted_live_rounding_and_product_wrap_cases(self):
		fixture = json.loads(FIXTURE_PATH.read_text())
		cases = fixture["targetedCases"]
		values = jnp.asarray([case["value"] for case in cases], dtype=jnp.int32)
		modifiers = jnp.asarray([case["modifier"] for case in cases], dtype=jnp.int32)
		expected = [case["result"] for case in cases]
		for result in (
			apply_fixedpoint_modifier(values, modifiers),
			jax.jit(apply_fixedpoint_modifier)(values, modifiers),
			jax.jit(jax.vmap(apply_fixedpoint_modifier))(values, modifiers),
		):
			self.assertEqual(as_lists(result), expected)

	def test_broadcast_dtype_and_neutral_event_integration(self):
		values = jnp.asarray([[0], [1048575], [1048576], [1048577]], dtype=jnp.int32)
		modifiers = jnp.asarray([0, 2048, 4096], dtype=jnp.int32)
		result = apply_fixedpoint_modifier(values, modifiers)
		compiled = jax.jit(apply_fixedpoint_modifier)(values, modifiers)
		self.assertEqual(result.shape, (4, 3))
		self.assertEqual(result.dtype, jnp.int32)
		self.assertEqual(as_lists(compiled), as_lists(result))
		identity = jnp.asarray(4096, dtype=jnp.int32)
		self.assertEqual(
			as_lists(_neutral_weather_identity_damage(values[:, 0])),
			as_lists(apply_fixedpoint_modifier(values[:, 0], identity)),
		)
		with self.assertRaises(TypeError):
			apply_fixedpoint_modifier(jnp.asarray(1.0, dtype=jnp.float32), 4096)
		with self.assertRaises(TypeError):
			apply_fixedpoint_modifier(1, jnp.asarray(4096.0, dtype=jnp.float32))
		with self.assertRaises(TypeError):
			apply_fixedpoint_modifier(np.asarray(1, dtype=np.int64), 4096)


if __name__ == "__main__":
	unittest.main()
