"""Differential tests for the bounded Gen 5 non-HP stage-scaling slice."""

import json
from pathlib import Path
import subprocess
import unittest

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from mechanics.stat_stage import scale_stat_stage


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "reference" / "fixtures" / "stat-stage-v1.json"
ORACLE_PATH = ROOT / "reference" / "stat-stage-oracle.cjs"


def run_oracle(mode):
	result = subprocess.run(
		["node", str(ORACLE_PATH), mode],
		cwd=ROOT,
		check=True,
		capture_output=True,
		text=True,
	)
	return json.loads(result.stdout)


class StatStageTests(unittest.TestCase):
	def test_jax_runtime_matches_pinned_dependencies(self):
		self.assertEqual(jax.__version__, "0.12.0.1")
		self.assertEqual(jaxlib.__version__, "0.12.0.1")

	def test_hand_computed_anchors_and_stage_clamping(self):
		anchors = [
			(1, 0, 1),
			(1, 6, 4),
			(3, -1, 2),
			(3, -6, 0),
			(5, -1, 3),
			(5, -2, 2),
			(5, -6, 1),
			(7, 1, 10),
			(65535, 6, 262140),
		]
		for stat, stage, expected in anchors:
			with self.subTest(stat=stat, stage=stage):
				self.assertEqual(int(scale_stat_stage(stat, stage)), expected)
		self.assertEqual(int(scale_stat_stage(5, -(2**31))), 1)
		self.assertEqual(int(scale_stat_stage(5, 2**31 - 1)), 20)

	def test_fixture_is_regenerated_by_pinned_showdown(self):
		committed = json.loads(FIXTURE_PATH.read_text())
		generated = run_oracle("fixture")
		self.assertEqual(generated, committed)
		self.assertEqual(generated["schemaVersion"], 1)
		self.assertEqual(generated["method"], "Pokemon.calculateStat")
		self.assertEqual(generated["sourcePath"], "sim/pokemon.ts:581-612")
		self.assertEqual(generated["conditions"]["modifyBoost"], "identity for the tested stage")

	def test_all_stages_for_stats_through_1024_match_live_pinned_oracle(self):
		oracle = run_oracle("grid")
		self.assertEqual(oracle["formatId"], "gen5customgame")
		self.assertEqual(oracle["sourcePath"], "sim/pokemon.ts:581-612")
		cases = oracle["cases"]
		self.assertEqual(len(cases), 1024 * 13 + 3 * 17)
		stats = jnp.asarray([case[0] for case in cases], dtype=jnp.int32)
		stages = jnp.asarray([case[1] for case in cases], dtype=jnp.int32)
		expected = [case[2] for case in cases]
		actual = jax.device_get(scale_stat_stage(stats, stages)).tolist()
		self.assertEqual(actual, expected)
		compiled = jax.device_get(jax.jit(scale_stat_stage)(stats, stages)).tolist()
		self.assertEqual(compiled, expected)
		vectorized = jax.device_get(jax.jit(jax.vmap(scale_stat_stage))(stats, stages)).tolist()
		self.assertEqual(vectorized, expected)

	def test_eager_jit_vmap_broadcast_and_int32_outputs(self):
		stats = jnp.asarray([1, 3, 5, 7, 15, 31, 63, 255, 1023, 65534, 65535], dtype=jnp.int32)
		stages = jnp.asarray([-6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6], dtype=jnp.int32)
		pairwise = scale_stat_stage(stats, stages)
		compiled = jax.jit(scale_stat_stage)(stats, stages)
		vectorized = jax.jit(jax.vmap(scale_stat_stage))(stats, stages)
		self.assertEqual(pairwise.dtype, jnp.int32)
		self.assertEqual(compiled.dtype, jnp.int32)
		self.assertEqual(vectorized.dtype, jnp.int32)
		self.assertEqual(jax.device_get(compiled).tolist(), jax.device_get(pairwise).tolist())
		self.assertEqual(jax.device_get(vectorized).tolist(), jax.device_get(pairwise).tolist())

		matrix = scale_stat_stage(stats[:, None], stages[None, :])
		self.assertEqual(matrix.shape, (11, 11))
		self.assertEqual(matrix.dtype, jnp.int32)
		for row, stat in enumerate(stats.tolist()):
			for column, stage in enumerate(stages.tolist()):
				self.assertEqual(int(matrix[row, column]), int(scale_stat_stage(stat, stage)))

	def test_rejects_non_int32_inputs_instead_of_converting_floats(self):
		with self.assertRaises(TypeError):
			scale_stat_stage(jnp.asarray(5.0, dtype=jnp.float32), 1)
		with self.assertRaises(TypeError):
			scale_stat_stage(5, jnp.asarray(1.0, dtype=jnp.float32))
		with self.assertRaises(TypeError):
			scale_stat_stage(np.asarray(5, dtype=np.int64), 1)


if __name__ == "__main__":
	unittest.main()
