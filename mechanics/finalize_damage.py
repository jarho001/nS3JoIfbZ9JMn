"""Pinned Gen 5 damage tail, after type effectiveness and burn.

Strict int32 inputs broadcast: ``damage`` in [0, 2147483647] and an already
encoded final ``modifier`` in [0, 65535]. Ranges are caller preconditions.
Replace zero with one BEFORE applying the modifier, then truncate to 16 bits.
The result may be zero. This excludes event dispatch, modifier chaining,
bypassProtect, and every earlier damage step; it is not a damage calculator.
"""

import jax.numpy as jnp

from .fixedpoint import apply_fixedpoint_modifier
from .stat_stage import _as_int32


def finalize_damage(damage, modifier):
	"""Apply the Gen 5 zero check, encoded final modifier, and uint16 wrap."""
	value = _as_int32(damage, "damage")
	encoded = _as_int32(modifier, "modifier")
	minimum = jnp.where(value == 0, jnp.asarray(1, dtype=jnp.int32), value)
	return apply_fixedpoint_modifier(minimum, encoded) & jnp.asarray(65535, dtype=jnp.int32)
