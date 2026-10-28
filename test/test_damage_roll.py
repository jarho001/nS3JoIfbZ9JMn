"""Pinned Showdown differential tests for Gen 5 random damage rolls."""

import json
from pathlib import Path
import subprocess
import unittest

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from mechanics.damage_roll import apply_damage_roll


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "reference" / "fixtures" / "damage-roll-v1.json"
ORACLE_PATH = ROOT / "reference" / "damage-roll-oracle.cjs"


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


class DamageRollTests(unittest.TestCase):
	def test_jax_runtime_matches_pinned_dependencies(self):
		self.assertEqual(jax.__version__, "0.12.0.1")
		self.assertEqual(jaxlib.__version__, "0.12.0.1")

	def test_guarded_fixture_regenerates_and_records_target_contract(self):
		committed = json.loads(FIXTURE_PATH.read_text())
		generated = run_oracle()
		self.assertEqual(generated, committed)
		self.assertEqual(generated["upstream"]["commit"], "1242c1e2d9763ba5b85d33f13aecb89bae19ae3d")
		self.assertEqual(generated["formatId"], "gen5randombattle")
		self.assertEqual(generated["method"], "Battle.randomizer")
		self.assertEqual(generated["sourcePaths"], [
			"sim/battle.ts:2407-2410", "sim/dex.ts:364-369", "sim/battle-actions.ts:1809-1812",
		])
		grid = generated["controlledGrid"]
		self.assertEqual(grid["baseDamageValues"][:1025], list(range(1025)))
		self.assertEqual(grid["rolls"], list(range(16)))
		self.assertEqual(grid["callsPerResult"], 1)
		self.assertEqual(generated["liveSeeded"]["battleSeed"], [101, 202, 303, 404])
		self.assertIn("after Battle initialization", generated["liveSeeded"]["drawContract"])
		self.assertIn("not asserted to be the seed PRNG first draws", generated["liveSeeded"]["drawContract"])

	def test_full_controlled_grid_matches_eager_jit_and_jit_vmap(self):
		fixture = json.loads(FIXTURE_PATH.read_text())
		grid = fixture["controlledGrid"]
		damage = jnp.asarray(grid["baseDamageValues"], dtype=jnp.int32)
		rolls = jnp.asarray(grid["rolls"], dtype=jnp.int32)
		expected = grid["results"]
		actual = apply_damage_roll(damage[:, None], rolls[None, :])
		compiled = jax.jit(apply_damage_roll)(damage[:, None], rolls[None, :])
		vectorized = jax.jit(jax.vmap(apply_damage_roll, in_axes=(0, None)))(damage, rolls)
		for result in (actual, compiled, vectorized):
			self.assertEqual(result.shape, (len(grid["baseDamageValues"]), 16))
			self.assertEqual(result.dtype, jnp.int32)
			self.assertEqual(as_lists(result), expected)

	def test_live_seeded_draws_match_unmodified_reference_randomizer(self):
		fixture = json.loads(FIXTURE_PATH.read_text())
		cases = fixture["liveSeeded"]["cases"]
		damage = jnp.asarray([case["baseDamage"] for case in cases], dtype=jnp.int32)
		rolls = jnp.asarray([case["roll"] for case in cases], dtype=jnp.int32)
		expected = [case["result"] for case in cases]
		self.assertTrue(all(0 <= case["roll"] < 16 for case in cases))
		self.assertEqual(as_lists(apply_damage_roll(damage, rolls)), expected)
		self.assertEqual(as_lists(jax.jit(apply_damage_roll)(damage, rolls)), expected)
		self.assertEqual(as_lists(jax.jit(jax.vmap(apply_damage_roll))(damage, rolls)), expected)

	def test_independent_floor_and_unsigned_wrap_anchors(self):
		anchors = [
			(0, 0, 0),
			(1, 15, 0),  # floor(85 / 100); no minimum-one clamp here
			(100, 0, 100),
			(100, 15, 85),
			(42949672, 0, 42949672),
			(42949673, 0, 0),  # product wraps to 4, then floor(4 / 100)
			(50529027, 15, 42949672),
			(50529028, 15, 0),  # product wraps to 84
		]
		for base_damage, roll, expected in anchors:
			with self.subTest(base_damage=base_damage, roll=roll):
				self.assertEqual(int(apply_damage_roll(base_damage, roll)), expected)
		self.assertEqual((42949673 * 100) & 0xFFFFFFFF, 4)
		self.assertEqual((50529028 * 85) & 0xFFFFFFFF, 84)

	def test_broadcasting_and_int32_dtype_contract(self):
		damage = jnp.asarray([[0], [100], [42949673]], dtype=jnp.int32)
		rolls = jnp.asarray([0, 1, 15], dtype=jnp.int32)
		result = apply_damage_roll(damage, rolls)
		self.assertEqual(result.shape, (3, 3))
		self.assertEqual(result.dtype, jnp.int32)
		self.assertEqual(as_lists(jax.jit(apply_damage_roll)(damage, rolls)), as_lists(result))

	def test_rejects_non_int32_inputs(self):
		with self.assertRaises(TypeError):
			apply_damage_roll(jnp.asarray(1.0, dtype=jnp.float32), 0)
		with self.assertRaises(TypeError):
			apply_damage_roll(1, jnp.asarray(0.0, dtype=jnp.float32))
		with self.assertRaises(TypeError):
			apply_damage_roll(np.asarray(1, dtype=np.int64), 0)


if __name__ == "__main__":
	unittest.main()
