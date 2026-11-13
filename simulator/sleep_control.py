"""Own Tempo confusion lifecycle and delayed Gen5 Yawn sleep."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
OWN=t.ABILITY_IDS['owntempo'];YAWN=t.MOVE_IDS['yawn'];SLOT=25
MOVE_IMPLEMENTED=jnp.asarray([k=='yawn'for k in t.MOVE_KEYS],jnp.bool_)
ABILITY_IMPLEMENTED=jnp.asarray([k=='owntempo'for k in t.ABILITIES],jnp.bool_)

def update(c,side,e):
    s,_=c;mon=s.active[side]
    allowed=(e.suppression_ops.abilities(s)[side,mon]==OWN)&(s.volatile_state[side,mon,2,0]>0)&(s.hp[side,mon]>0)
    def cure(c):
        c=e._event(c,e.STATUS_ABILITY,side,mon,value=OWN);c=e._event(c,e.CONF_END,side,mon)
        s,tr=c;return s._replace(volatile_state=s.volatile_state.at[side,mon,2].set(0)),tr
    return e._when(allowed,cure,c)

def try_hit(c,actor,target,move,move_row,e):
    s,_=c;denied=(move==YAWN)&(s.status[target,s.active[target]]!=e.NONE)
    return e._when(denied,lambda c:e.ability_ops.fail(c,actor,move,move_row,e),c),denied

def hit(c,actor,target,move_row,e):
    s,_=c;mon=s.active[target];ability=e.suppression_ops.abilities(s)[target,mon]
    def install(c):
        c=e.effects.add_volatile(c,target,SLOT,actor,e.i32(YAWN),e)
        s,tr=c;s=s._replace(volatile_state=s.volatile_state.at[target,mon,SLOT,5].set(2))
        c=e._event((s,tr),e.YAWN_START,target,mon,target=actor,targetmon=s.active[actor])
        return e.tracking.outcome(c,actor,e.i32(e.tracking.TRUE),YAWN)
    def available(c):
        return e.jax.lax.cond(e.immunity_ops.SLEEP[ability],lambda c:e._event(c,e.IMMUNE,target,mon,value=ability,aux=4),install,c)
    return e.jax.lax.cond(s.volatile_state[target,mon,SLOT,0]>0,lambda c:e.ability_ops.fail(c,actor,YAWN,move_row,e),available,c)

def tick(c,side,e):
    s,tr=c;mon=s.active[side];row=s.volatile_state[side,mon,SLOT].at[5].add(-1)
    s=s._replace(volatile_state=s.volatile_state.at[side,mon,SLOT].set(row))
    def end(c):
        s,tr=c;s=s._replace(volatile_state=s.volatile_state.at[side,mon,SLOT].set(0))
        c=e._event((s,tr),e.YAWN_END,side,mon)
        return e.effects.inflict(c,row[2],side,e.i32(e.SLP),e,secondary=True,source_mon=row[3])
    return e.jax.lax.cond(row[5]<=0,end,lambda c:c,(s,tr))
