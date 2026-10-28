"""Gen 5 random damage-roll arithmetic from the pinned Showdown simulator.

Inputs are int32 arrays/scalars. ``base_damage`` must be in
``[0, 2147483647]`` and ``roll`` in ``[0, 15]``; these are caller
preconditions and are not dynamically checked. Multiplication is explicitly
performed modulo 2**32 before division, matching Showdown's unsigned ``tr``.
This applies only the random factor and does not generate a roll, clamp damage
to at least one, or apply later damage modifiers.
"""

import jax.numpy as jnp

from .stat_stage import _as_int32


def apply_damage_roll(base_damage, roll):
	"""Apply an explicit Gen 5 random roll with unsigned 32-bit wrap."""
	damage = _as_int32(base_damage, "base_damage")
	roll_array = _as_int32(roll, "roll")
	factor = (jnp.asarray(100, dtype=jnp.int32) - roll_array).astype(jnp.uint32)
	wrapped_product = damage.astype(jnp.uint32) * factor
	return (wrapped_product // jnp.asarray(100, dtype=jnp.uint32)).astype(jnp.int32)
