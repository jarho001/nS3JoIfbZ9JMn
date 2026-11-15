"""Pinned Eruption/Water Spout power from live source HP before power modifiers."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
MOVES=t.mv(lambda r:r['id'] in ('eruption','waterspout')).astype(jnp.bool_)


def power(s,actor,move,value):
    mon=s.active[actor]
    # Native clamps the callback's fractional relay to integer power before
    # Technician and other BasePower hooks. HP1 may clamp upward to1.
    scaled=jnp.maximum(1,value*s.hp[actor,mon]//jnp.maximum(1,s.stats[actor,mon,0]))
    return jnp.where(MOVES[move],scaled,value).astype(jnp.int32)
