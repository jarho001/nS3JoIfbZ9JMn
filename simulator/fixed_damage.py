"""Pinned reachable fixed damage, bypassing ordinary stats, criticals and damage RNG."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
SUPER_FANG=t.MOVE_IDS['superfang']
MOVE_IMPLEMENTED=t.mv(lambda r:r['id'] in ('superfang','seismictoss')).astype(jnp.bool_)


def amount(s,actor,target,move):
    half=jnp.maximum(1,s.hp[target,s.active[target]]//2)
    return jnp.where(move==SUPER_FANG,half,s.level[actor,s.active[actor]]).astype(jnp.int32)
