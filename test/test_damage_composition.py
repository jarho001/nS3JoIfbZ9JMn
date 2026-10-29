"""Differential checks for the bounded post-core damage composition."""

import json
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from mechanics.damage_composition import damage_from_preroll
from mechanics.damage_roll import apply_damage_roll
from mechanics.stab import apply_stab
from mechanics.type_effectiveness import apply_type_effectiveness
from mechanics.burn import apply_burn_reduction
from mechanics.finalize_damage import finalize_damage


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "reference/fixtures/damage-composition-v1.json"


def as_lists(value):
	return jax.device_get(value).tolist()


def columns(cases):
	return tuple(jnp.asarray([case["input"][key] for case in cases], dtype=dtype)
		for key, dtype in (
			("preRollDamage", jnp.int32), ("roll", jnp.int32),
			("stabEligible", jnp.bool_), ("adaptability", jnp.bool_), ("typeless", jnp.bool_),
			("typeExponent", jnp.int32), ("burned", jnp.bool_), ("physical", jnp.bool_),
			("hasGuts", jnp.bool_), ("finalModifier", jnp.int32),
		))


class DamageCompositionTests(unittest.TestCase):
	def test_guarded_fixture_has_ordinary_and_synthetic_path_coverage(self):
		fixture = json.loads(FIXTURE.read_text())
		self.assertEqual(fixture["kind"], "gen5-bounded-post-core-damage-composition-oracle")
		self.assertEqual(fixture["upstream"]["commit"], "1242c1e2d9763ba5b85d33f13aecb89bae19ae3d")
		self.assertEqual(fixture["inputDomain"], {
			"preRollDamage": [0, 33554431], "roll": [0, 15],
			"typeExponent": [-2147483648, 2147483647], "finalModifier": [0, 65535],
		})
		ordinary = {case["label"]: case for case in fixture["ordinaryCases"]}
		self.assertTrue(ordinary["ordinary-stab-life-orb-special-burned"]["input"]["burned"])
		self.assertEqual(ordinary["ordinary-stab-life-orb-special-burned"]["input"]["finalModifier"], 5324)
		self.assertEqual(ordinary["ordinary-burned-physical"]["burnCalls"], 1)
		self.assertTrue(ordinary["ordinary-burned-physical"]["input"]["physical"])
		self.assertEqual(ordinary["ordinary-burned-physical-guts"]["burnCalls"], 0)
		self.assertTrue(ordinary["ordinary-burned-physical-guts"]["input"]["hasGuts"])
		self.assertTrue(ordinary["ordinary-adaptability-stab"]["input"]["adaptability"])
		self.assertFalse(ordinary["ordinary-typed-nonstab"]["input"]["stabEligible"])
		synthetic = fixture["syntheticCases"]
		self.assertEqual({case["roll"] for case in synthetic if case["label"].startswith("all-rolls-")}, set(range(16)))
		self.assertTrue(all(case["randomizerCalls"] == 1 and case["damageDraws"] == 1 for case in synthetic))
		self.assertTrue(all(case["finalCalls"] == 1 for case in fixture["ordinaryCases"] + synthetic))
		self.assertTrue(all(case["finalMoveType"] != "???" or case["input"]["typeless"] for case in synthetic))
		low_clamp = next(row for row in synthetic if row["label"] == "type-exponent-low-clamp")
		high_clamp = next(row for row in synthetic if row["label"] == "type-exponent-high-clamp")
		self.assertEqual(low_clamp["input"]["typeExponent"], -2147483648)
		self.assertEqual(low_clamp["afterType"], low_clamp["afterStab"] // 64)
		self.assertEqual(high_clamp["input"]["typeExponent"], 2147483647)
		self.assertEqual(high_clamp["afterType"], high_clamp["afterStab"] * 64)
		zero = next(row for row in synthetic if row["label"] == "zero-before-final-zero-modifier")
		self.assertEqual((zero["afterBurn"], zero["preFinal"], zero["result"]), (0, 1, 0))
		self.assertEqual(next(row for row in synthetic if row["label"] == "typeless-burned-physical-low")["burnCalls"], 1)
		self.assertEqual(next(row for row in synthetic if row["label"] == "typeless-uint16-zero-boundary")["result"], 0)
		self.assertEqual(next(row for row in synthetic if row["label"] == "typeless-max-preroll-and-type-exponent")["input"]["typeExponent"], 6)

	def test_actual_final_damage_matches_eager_jit_vmap_and_broadcast(self):
		fixture = json.loads(FIXTURE.read_text())
		cases = fixture["ordinaryCases"] + fixture["syntheticCases"]
		args = columns(cases)
		expected = [case["result"] for case in cases]
		for result in (
			damage_from_preroll(*args),
			jax.jit(damage_from_preroll)(*args),
			jax.jit(jax.vmap(damage_from_preroll))(*args),
		):
			self.assertEqual(result.dtype, jnp.int32)
			self.assertEqual(result.shape, (len(cases),))
			self.assertEqual(as_lists(result), expected)

		captured = [case for case in cases if case["burnCalls"] == 1]
		stage_args = columns(captured)
		rolled = apply_damage_roll(stage_args[0], stage_args[1])
		stabbed = apply_stab(rolled, stage_args[2], stage_args[3], stage_args[4])
		typed = apply_type_effectiveness(stabbed, stage_args[5])
		burned = apply_burn_reduction(typed, stage_args[6], stage_args[7], stage_args[8])
		self.assertEqual(as_lists(rolled), [case["randomizerOutput"] for case in captured])
		self.assertEqual(as_lists(stabbed), [case.get("afterStab", case["randomizerOutput"]) for case in captured])
		self.assertEqual(as_lists(typed), [case["afterType"] for case in captured])
		self.assertEqual(as_lists(burned), [case["afterBurn"] for case in captured])
		self.assertEqual(as_lists(finalize_damage(burned, stage_args[9])), [case["result"] for case in captured])

		rows = jnp.asarray([1, 31, 33554431], dtype=jnp.int32)[:, None]
		rolls = jnp.asarray([0, 15], dtype=jnp.int32)[None, :]
		result = damage_from_preroll(rows, rolls, False, False, False, 0, False, False, False, 4096)
		self.assertEqual(result.shape, (3, 2))
		self.assertEqual(as_lists(jax.jit(damage_from_preroll)(rows, rolls, False, False, False, 0, False, False, False, 4096)), as_lists(result))

	def test_order_anchor_differs_from_reversed_type_and_burn(self):
		case = next(row for row in json.loads(FIXTURE.read_text())["syntheticCases"] if row["label"] == "order-anchor-odd-low")
		actual = case["result"]
		rolled = apply_damage_roll(jnp.int32(case["input"]["preRollDamage"]), jnp.int32(case["roll"]))
		stabbed = apply_stab(rolled, True, False, False)
		typed = apply_type_effectiveness(stabbed, jnp.int32(case["input"]["typeExponent"]))
		burned = apply_burn_reduction(typed, True, True, False)
		self.assertEqual(int(finalize_damage(burned, jnp.int32(case["input"]["finalModifier"]))), actual)
		reversed = apply_type_effectiveness(apply_burn_reduction(stabbed, True, True, False), jnp.int32(case["input"]["typeExponent"]))
		self.assertNotEqual(int(finalize_damage(reversed, jnp.int32(case["input"]["finalModifier"]))), actual)

	def test_broadcasting_dtypes_and_precondition_only_contract(self):
		valid = [jnp.int32(1), jnp.int32(0), jnp.bool_(False), jnp.bool_(False), jnp.bool_(False),
			jnp.int32(0), jnp.bool_(False), jnp.bool_(False), jnp.bool_(False), jnp.int32(4096)]
		for index in (0, 1, 5, 9):
			invalid = list(valid)
			invalid[index] = np.asarray(1, dtype=np.int64)
			with self.assertRaises(TypeError):
				damage_from_preroll(*invalid)
		for index in (2, 3, 4, 6, 7, 8):
			invalid = list(valid)
			invalid[index] = jnp.int32(1)
			with self.assertRaises(TypeError):
				damage_from_preroll(*invalid)
		self.assertEqual(damage_from_preroll(jnp.int32(33554431), jnp.int32(0), False, False, True, 0,
			False, False, False, jnp.int32(4096)).dtype, jnp.int32)


if __name__ == "__main__":
	unittest.main()
