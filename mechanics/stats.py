"""Gen 5 base-stat to stored-stat calculations, including nature scaling.

Every stat table uses final axis order ``hp, atk, def, spa, spd, spe``.
Inputs are int32 arrays with base stats in [1, 255], level in [1, 100], IVs
in [0, 31], EVs in [0, 255], and per-stat nature multipliers 9/10/11 for
minus/neutral/plus. These value ranges are caller preconditions and are not
dynamically checked. The ``shedinja`` bool is explicit: true forces HP to 1;
base HP 1 alone does not apply the species override.

This matches the pinned Gen 5 randombattle ``spreadModify`` / ``statModify``
path. That format does not have the ``overflowstatmod`` rule, so nature
multiplication uses unsigned low-16-bit truncation without the optional caps.
The function does not implement stat stages, form changes, event hooks, or team
legality.
"""

import jax.numpy as jnp

from .stat_stage import _as_int32


STAT_COUNT = 6
LOW_16_MASK = 0xFFFF


def _stat_table(value, name):
	array = _as_int32(value, name)
	if array.ndim < 1 or array.shape[-1] != STAT_COUNT:
		raise ValueError(f"{name} must have a final axis of six stats")
	return array


def _bool_array(value, name):
	input_dtype = getattr(value, "dtype", None)
	if input_dtype is not None and jnp.dtype(input_dtype) != jnp.dtype(jnp.bool_):
		raise TypeError(f"{name} must have dtype bool")
	array = jnp.asarray(value)
	if array.dtype != jnp.dtype(jnp.bool_):
		raise TypeError(f"{name} must have dtype bool")
	return array


def derive_stored_stats(base_stats, level, ivs, evs, nature_multipliers, shedinja=False):
	"""Derive HP and five non-HP stored stats with broadcast-compatible inputs.

	The four stat tables must broadcast together and end in six values ordered
	``hp, atk, def, spa, spd, spe``. ``level`` and ``shedinja`` broadcast over
	the preceding batch axes. Nature multipliers are integers: 9 (minus), 10
	(neutral), or 11 (plus); HP ignores nature. The Shedinja flag explicitly
	models ``species.maxHP === 1``.
	"""
	base = _stat_table(base_stats, "base_stats")
	individual = _stat_table(ivs, "ivs")
	effort = _stat_table(evs, "evs")
	nature = _stat_table(nature_multipliers, "nature_multipliers")
	base, individual, effort, nature = jnp.broadcast_arrays(base, individual, effort, nature)

	level_array = _as_int32(level, "level")
	shedinja_array = _bool_array(shedinja, "shedinja")
	_, level_array, shedinja_array = jnp.broadcast_arrays(base[..., 0], level_array, shedinja_array)
	batch_shape = level_array.shape
	base = jnp.broadcast_to(base, batch_shape + (STAT_COUNT,))
	individual = jnp.broadcast_to(individual, batch_shape + (STAT_COUNT,))
	effort = jnp.broadcast_to(effort, batch_shape + (STAT_COUNT,))
	nature = jnp.broadcast_to(nature, batch_shape + (STAT_COUNT,))
	level_array = level_array[..., None]
	shedinja_array = shedinja_array[..., None]

	ev_quarters = effort // 4
	hp = ((2 * base[..., 0] + individual[..., 0] + ev_quarters[..., 0] + 100) * level_array[..., 0]) // 100 + 10
	hp = jnp.where(shedinja_array[..., 0], 1, hp)

	non_hp_base = ((2 * base[..., 1:] + individual[..., 1:] + ev_quarters[..., 1:]) * level_array) // 100 + 5
	# Gen 5 uses Dex.trunc(product, 16) before dividing by 100 for nature
	# scaling. Select multipliers first; all evaluated intermediates stay int32.
	factor = jnp.where(nature[..., 1:] == 11, 110, jnp.where(nature[..., 1:] == 9, 90, 100))
	nature_product = (non_hp_base * factor) & LOW_16_MASK
	nature_scaled = nature_product // 100
	non_hp = jnp.where(nature[..., 1:] == 10, non_hp_base, nature_scaled)
	return jnp.concatenate((hp[..., None], non_hp), axis=-1).astype(jnp.int32)
