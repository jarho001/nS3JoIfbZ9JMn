"""Pinned Gen 5 STAB arithmetic with explicitly resolved eligibility flags.

``damage`` is strict int32 in the caller-precondition range
``[0, 2147483647]``. The three flags are strict booleans and broadcast with
damage. ``typeless`` bypasses the modifying call; typed inputs always use the
fixed-point modifier, including identity for non-STAB damage.
"""

import jax.numpy as jnp

from .fixedpoint import apply_fixedpoint_modifier
from .stat_stage import _as_int32
from .burn import _as_bool


def apply_stab(damage, stab_eligible, adaptability, typeless):
	"""Apply one pinned Gen 5 STAB modifier from explicit eligibility state."""
	value = _as_int32(damage, "damage")
	eligible_array = _as_bool(stab_eligible, "stab_eligible")
	adaptability_array = _as_bool(adaptability, "adaptability")
	typeless_array = _as_bool(typeless, "typeless")
	value, eligible_array, adaptability_array, typeless_array = jnp.broadcast_arrays(
		value, eligible_array, adaptability_array, typeless_array,
	)
	modifier = jnp.where(
		eligible_array,
		jnp.where(adaptability_array, 8192, 6144),
		4096,
	).astype(jnp.int32)
	modified = apply_fixedpoint_modifier(value, modifier)
	return jnp.where(typeless_array, value, modified).astype(jnp.int32)
