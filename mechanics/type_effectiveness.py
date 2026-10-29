"""Pinned Gen 5 type-effectiveness exponent arithmetic.

Inputs are strict int32 arrays/scalars and broadcast together. ``damage`` has
the caller-precondition range [0, 33554431], which keeps six positive
doublings at or below 2147483584. ``exponent`` is any signed int32 and is
clamped to [-6, 6]. This module does not perform type lookup, immunity, STAB,
burn, modifier handling, or any other part of damage calculation. The damage
range is not checked dynamically.
"""

import jax.numpy as jnp

from .stat_stage import _as_int32


def apply_type_effectiveness(damage, exponent):
	"""Apply the pinned repeated doubling or truncating-halving type effect."""
	value = _as_int32(damage, "damage")
	stage = jnp.clip(_as_int32(exponent, "exponent"), -6, 6)

	positive_factor = jnp.left_shift(jnp.asarray(1, dtype=jnp.int32), jnp.maximum(stage, 0))
	boosted = value * positive_factor

	resisted = value
	for step in range(1, 7):
		resisted = jnp.where(stage <= -step, resisted // 2, resisted)

	return jnp.where(stage > 0, boosted, resisted).astype(jnp.int32)
