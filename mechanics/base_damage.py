"""Gen 5 integer base-damage core from the pinned Showdown simulator.

All inputs are int32 arrays/scalars. Caller preconditions are level in
``[1, 100]``, effective base power in ``[1, 65535]``, and effective Attack
and Defense in ``[0, 65535]``; ranges are not dynamically checked. The
formula truncates the product to unsigned 32 bits before dividing by Defense,
then floors once more by 50. A zero Defense produces 0, matching Showdown's
unsigned conversion of its Infinity/NaN division result. Safe positive
denominators are selected before division, so JAX never evaluates division by
zero. This is only the core before ``modifyDamage`` and the later ``+2``.
"""

import jax.numpy as jnp

from .stat_stage import _as_int32


def base_damage_core(level, base_power, attack, defense):
	"""Compute the nested-truncation core from effective battle values.

	Inputs broadcast together and the result is int32. The method deliberately
	does not derive or modify any of its four inputs.
	"""
	level_array = _as_int32(level, "level")
	power_array = _as_int32(base_power, "base_power")
	attack_array = _as_int32(attack, "attack")
	defense_array = _as_int32(defense, "defense")

	level_term = ((2 * level_array) // 5 + 2).astype(jnp.uint32)
	wrapped_product = level_term * power_array.astype(jnp.uint32) * attack_array.astype(jnp.uint32)
	safe_defense = jnp.where(defense_array == 0, 1, defense_array).astype(jnp.uint32)
	post_defense = wrapped_product // safe_defense
	post_defense = jnp.where(defense_array == 0, jnp.uint32(0), post_defense)
	return (post_defense // jnp.asarray(50, dtype=jnp.uint32)).astype(jnp.int32)
