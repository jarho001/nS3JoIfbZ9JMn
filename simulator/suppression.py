"""Gastro Acid keeps the stored ability while disabling runtime callbacks."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
MOVE=t.MOVE_IDS['gastroacid'];SLOT=24
MOVE_IMPLEMENTED=jnp.asarray([k=='gastroacid'for k in t.MOVE_KEYS],jnp.bool_)

def abilities(s):
    return jnp.where(s.volatile_state[:,:,SLOT,0]>0,0,s.ability)

def try_hit(c,actor,target,move,move_row,e):
    s,_=c;denied=(move==MOVE)&e.ability_ops.CANT[s.ability[target,s.active[target]]]
    return e._when(denied,lambda c:e.ability_ops.fail(c,actor,move,move_row,e),c),denied

def hit(c,actor,target,move_row,e):
    s,_=c;mon=s.active[target];present=s.volatile_state[target,mon,SLOT,0]>0
    def install(c):
        c=e.effects.add_volatile(c,target,SLOT,actor,e.i32(MOVE),e)
        c=e._event(c,e.SUPPRESS_ABILITY,target,mon)
        c=e.ability_ops.end(c,target,e)
        return e.tracking.outcome(c,actor,e.i32(e.tracking.TRUE),MOVE)
    return e.jax.lax.cond(present,lambda c:e.ability_ops.fail(c,actor,MOVE,move_row,e),install,c)
