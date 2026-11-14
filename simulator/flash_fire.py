"""Gen5 Flash Fire interception and its owned stat-modifier condition."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
ABILITY=t.ABILITY_IDS['flashfire'];SLOT=27
ABILITY_IMPLEMENTED=jnp.asarray([k=='flashfire'for k in t.ABILITIES],jnp.bool_)

def intercepts(s,actor,target,move,e):
    mon=s.active[actor];tm=s.active[target]
    move_type=jnp.where(move==e.HIDDEN_POWER,s.hp_type[actor,mon],t.MOVE_TYPE[move])
    return (actor!=target)&(e.suppression_ops.abilities(s)[target,tm]==ABILITY)&(move_type==e.generic.TYPES['fire'])

def hit(c,actor,target,move,e):
    c=e.tracking.outcome(c,actor,e.i32(e.tracking.FALSE),move)
    def install(c):
        c=e.effects.add_volatile(c,target,SLOT,actor,e.i32(ABILITY),e,source_kind=e.phase_ops.SOURCE_ABILITY)
        return e._event(c,e.FLASH_START,target,c[0].active[target])
    s,_=c;mon=s.active[target]
    return e.jax.lax.cond(s.volatile_state[target,mon,SLOT,0]>0,lambda c:e._event(c,e.IMMUNE,target,mon,value=ABILITY,aux=4),install,c)

def end(c,side,e):
    s,_=c;mon=s.active[side]
    def remove(c):
        s,tr=c;s=s._replace(volatile_state=s.volatile_state.at[side,mon,SLOT].set(0))
        return e._event((s,tr),e.FLASH_END,side,mon)
    return e._when(s.volatile_state[side,mon,SLOT,0]>0,remove,c)
