"""Guarded Gen5 weight callbacks in integer hectograms, before power modifiers."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
LOW_KICK=t.MOVE_IDS['lowkick'];GRASS_KNOT=t.MOVE_IDS['grassknot'];HEAVY_SLAM=t.MOVE_IDS['heavyslam']
WEIGHT_HG=jnp.asarray([0]+[int(t.SP_ROWS[t.g.to_id(name)]['weighthg']) for name in t.SPECIES[1:]],jnp.int32)


def power(s,actor,target,move,value):
    # Transform and Imposter change current species; unsupported weight modifiers
    # remain behind their ability/item/volatile capability gates.
    source=jnp.maximum(1,WEIGHT_HG[s.species[actor,s.active[actor]]])
    target_weight=jnp.maximum(1,WEIGHT_HG[s.species[target,s.active[target]]])
    absolute=jnp.where(target_weight>=2000,120,jnp.where(target_weight>=1000,100,jnp.where(target_weight>=500,80,jnp.where(target_weight>=250,60,jnp.where(target_weight>=100,40,20)))))
    ratio=jnp.where(source>=target_weight*5,120,jnp.where(source>=target_weight*4,100,jnp.where(source>=target_weight*3,80,jnp.where(source>=target_weight*2,60,40))))
    return jnp.where((move==LOW_KICK)|(move==GRASS_KNOT),absolute,jnp.where(move==HEAVY_SLAM,ratio,value))
