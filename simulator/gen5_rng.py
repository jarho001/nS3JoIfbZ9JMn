"""Exact Gen 5 Showdown PRNG arithmetic for JAX's default 32-bit mode."""

from __future__ import annotations

import jax.numpy as jnp


_U32 = jnp.uint32
_U16 = jnp.uint16
_BYTE_MASK = _U32(0xFF)
_WORD_MASK = _U32(0xFFFF)

# Gen5RNG's big-endian 16-bit limbs are converted to little-endian bytes for
# convolution.  The multiplier is 0x5D588B656C078965 and the increment is
# 0x00269EC3, both taken from the pinned Showdown `Gen5RNG.nextFrame`.
_MULTIPLIER_LE_BYTES = (0x65, 0x89, 0x07, 0x6C, 0x65, 0x8B, 0x58, 0x5D)
_INCREMENT_LE_BYTES = (0xC3, 0x9E, 0x26, 0, 0, 0, 0, 0)


def _seed_bytes_le(seed):
	limbs = jnp.asarray(seed, dtype=_U32)
	return jnp.stack(
		(
			limbs[3] & _BYTE_MASK,
			limbs[3] >> 8,
			limbs[2] & _BYTE_MASK,
			limbs[2] >> 8,
			limbs[1] & _BYTE_MASK,
			limbs[1] >> 8,
			limbs[0] & _BYTE_MASK,
			limbs[0] >> 8,
		)
	).astype(_U32)


def next_state(seed):
	"""Return the next four uint16 limbs and its upper-32 output word.

	The base-256 convolution keeps each intermediate below 2**32: at most
	eight byte products (8 * 255**2), one carry, and one increment byte are
	added per column. This avoids int64 and is valid with JAX x64 disabled.
	"""
	state_bytes = _seed_bytes_le(seed)
	carry = _U32(0)
	result_bytes = []
	for output_index in range(8):
		total = carry + _U32(_INCREMENT_LE_BYTES[output_index])
		for state_index in range(output_index + 1):
			total = total + state_bytes[state_index] * _U32(_MULTIPLIER_LE_BYTES[output_index - state_index])
		result_bytes.append(total & _BYTE_MASK)
		carry = total >> 8

	byte = result_bytes
	new_seed = jnp.stack(
		(
			(byte[7] << 8) | byte[6],
			(byte[5] << 8) | byte[4],
			(byte[3] << 8) | byte[2],
			(byte[1] << 8) | byte[0],
		)
	).astype(_U16)
	upper32 = (new_seed[0].astype(_U32) << 16) | new_seed[1].astype(_U32)
	return new_seed, upper32


def sample_bounded(seed, bound):
	"""Advance once and return Showdown `random(bound)` for 1 <= bound <= 65535.

	For ``word = high16 * 2**16 + low16``, the upper half of
	``word * bound`` is computed without an overflowing uint32 multiply.
	"""
	new_seed, word = next_state(seed)
	bound_u32 = jnp.asarray(bound, dtype=_U32)
	low_product = (word & _WORD_MASK) * bound_u32
	high_product = (word >> 16) * bound_u32 + (low_product >> 16)
	return new_seed, high_product >> 16
