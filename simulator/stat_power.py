"""Pinned Gyro Ball live Speed queries and Stored Power raw boost count."""
import jax
import jax.numpy as jnp
from . import gen5_engine_tables as t
GYRO=t.MOVE_IDS['gyroball']
STORED=t.MOVE_IDS['storedpower']


def before_critical(c,actor,target,move,e):
    def gyro(c):
        # getStat(target), then getStat(source), before the critical draw.
        c=e.unaware_ops.pair(c,e)
        s,_=c;speeds=e._speeds(s)
        power=jnp.minimum(150,25*speeds[target,s.active[target]]//jnp.maximum(1,speeds[actor,s.active[actor]])+1)
        return c,power.astype(jnp.int32)
    return jax.lax.cond(move==GYRO,gyro,lambda c:(c,jnp.int32(0)),c)


def power(s,actor,move,value,gyro):
    positive=jnp.sum(jnp.maximum(0,s.boosts[actor,s.active[actor]]),dtype=jnp.int32)
    return jnp.where(move==GYRO,gyro,jnp.where(move==STORED,value+20*positive,value)).astype(jnp.int32)
