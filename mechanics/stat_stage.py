"""Integer non-HP stat-stage scaling from Gen 5 Pokemon.calculateStat.

Inputs are int32 arrays/scalars. ``stored_stat`` must be a nonnegative stored
stat in [0, 65535], after Gen 5 field stat selection; zero is reachable through
Gen 5 nature-product truncation. This value-range precondition is documented
but not dynamically checked. ``stage`` is a signed int32
ModifyBoost result. The stage is clamped to [-6, 6]. This function implements
only the scaling substep and does not derive base stats, dispatch events, apply
field selection, or apply stat modifiers.
"""

import jax.numpy as jnp


def _as_int32(value, name):
	input_dtype = getattr(value, "dtype", None)
	if input_dtype is not None and jnp.dtype(input_dtype) != jnp.dtype(jnp.int32):
		raise TypeError(f"{name} must have dtype int32")
	array = jnp.asarray(value)
	if array.dtype != jnp.dtype(jnp.int32):
		raise TypeError(f"{name} must have dtype int32")
	return array


def scale_stat_stage(stored_stat, stage):
	"""Scale a stored non-HP stat in [0, 65535] by a clamped Gen 5 stage.

	For stage s >= 0, return floor(stat * (2 + s) / 2). For s < 0,
	return floor(stat * 2 / (2 - s)). Integer operations preserve the exact
	floor behavior; int32 intermediates are bounded by 65535 * 8.
	"""
	stat = _as_int32(stored_stat, "stored_stat")
	boost = jnp.clip(_as_int32(stage, "stage"), -6, 6)
	is_positive = boost >= 0
	numerator = jnp.where(is_positive, stat * (2 + boost), stat * 2)
	denominator = jnp.where(is_positive, 2, 2 - boost)
	return (numerator // denominator).astype(jnp.int32)
