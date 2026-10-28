"""Gen 5 fixed-point modifier arithmetic from the pinned Showdown simulator.

All inputs are strict int32 arrays/scalars and broadcast normally. Caller
preconditions are ``value`` in ``[0, 2147483647]`` and an already encoded
``modifier`` in ``[0, 65535]``; the kernel does not dynamically check ranges.
The product wraps modulo ``2**32``. Showdown then adds 2047 as a JavaScript
number and floors by 4096, so exact half ties round down. The result fits
int32. This does not encode floating fractions or chain modifiers.
"""

import jax.numpy as jnp

from .stat_stage import _as_int32


def apply_fixedpoint_modifier(value, modifier):
	"""Apply one pre-encoded fixed-point modifier with pinned uint32 wrap."""
	value_array = _as_int32(value, "value")
	modifier_array = _as_int32(modifier, "modifier")
	product = value_array.astype(jnp.uint32) * modifier_array.astype(jnp.uint32)
	quotient = product >> jnp.asarray(12, dtype=jnp.uint32)
	remainder = product & jnp.asarray(4095, dtype=jnp.uint32)
	round_up = (remainder > jnp.asarray(2048, dtype=jnp.uint32)).astype(jnp.uint32)
	return (quotient + round_up).astype(jnp.int32)
