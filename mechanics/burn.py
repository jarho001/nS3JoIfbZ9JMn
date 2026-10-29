"""Pinned Gen 5 physical-burn damage reduction.

``damage`` is strict int32 in the caller-precondition range
``[0, 2147483647]``. The three eligibility inputs are strict booleans and
broadcast with damage. This is only the ``burned && physical && !Guts``
arithmetic step; callers resolve status, move category, and ability. Gen 5
does not exempt Facade. Eligible values use the pinned fixed-point 0.5
modifier, including its uint32 product wrap and half-down rounding.
"""

import jax.numpy as jnp

from .fixedpoint import apply_fixedpoint_modifier
from .stat_stage import _as_int32


def _as_bool(value, name):
	input_dtype = getattr(value, "dtype", None)
	if input_dtype is not None and jnp.dtype(input_dtype) != jnp.dtype(jnp.bool_):
		raise TypeError(f"{name} must have dtype bool")
	array = jnp.asarray(value)
	if array.dtype != jnp.dtype(jnp.bool_):
		raise TypeError(f"{name} must have dtype bool")
	return array


def apply_burn_reduction(damage, burned, physical, has_guts):
	"""Apply the Gen 5 half-damage modifier when explicit eligibility allows."""
	value = _as_int32(damage, "damage")
	burned_array = _as_bool(burned, "burned")
	physical_array = _as_bool(physical, "physical")
	guts_array = _as_bool(has_guts, "has_guts")
	value, burned_array, physical_array, guts_array = jnp.broadcast_arrays(
		value, burned_array, physical_array, guts_array,
	)
	reduced = apply_fixedpoint_modifier(value, jnp.asarray(2048, dtype=jnp.int32))
	return jnp.where(burned_array & physical_array & ~guts_array, reduced, value).astype(jnp.int32)
