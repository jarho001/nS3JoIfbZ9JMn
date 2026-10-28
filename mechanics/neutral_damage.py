"""Compose Gen 5's neutral base-damage core and explicit random roll.

Inputs follow ``base_damage_core``: level ``[1, 100]``, base power
``[1, 65535]``, and effective Attack/Defense ``[0, 65535]``. Roll is an
integer in ``[0, 15]``. These are caller preconditions, not runtime checks.
The neutral ``WeatherModifyDamage`` event still applies Showdown's fixed-point
identity modifier, which wraps its value modulo ``2**20``. The result is the
intermediate after ``Battle.randomizer``; it is not final damage and excludes
downstream modifiers, Gen 5's minimum-one check, and final 16-bit truncation.
"""

import jax.numpy as jnp

from .base_damage import base_damage_core
from .damage_roll import apply_damage_roll
from .stat_stage import _as_int32


def _neutral_weather_identity_damage(value):
	"""Match ``Battle.modify(value, 1)`` in the no-handler weather event."""
	damage = _as_int32(value, "value")
	return (damage.astype(jnp.uint32) & jnp.asarray(0xFFFFF, dtype=jnp.uint32)).astype(jnp.int32)


def neutral_damage_after_roll(level, base_power, attack, defense, roll):
	"""Apply core, ``+2``, neutral event wrap, then an explicit random roll."""
	core = base_damage_core(level, base_power, attack, defense)
	base_damage = core + jnp.asarray(2, dtype=jnp.int32)
	weather_neutral = _neutral_weather_identity_damage(base_damage)
	return apply_damage_roll(weather_neutral, _as_int32(roll, "roll"))
