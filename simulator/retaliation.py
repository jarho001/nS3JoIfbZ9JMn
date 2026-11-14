"""Singles Counter/Mirror Coat before-turn setup and per-hit retaliation state."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
from .phase_effects import VOLATILE_IDS
COUNTER=t.MOVE_IDS['counter'];MIRROR_COAT=t.MOVE_IDS['mirrorcoat']
COUNTER_SLOT=VOLATILE_IDS['counter'];COAT_SLOT=VOLATILE_IDS['mirrorcoat']
MOVE_IMPLEMENTED=t.mv(lambda r:r['id'] in ('counter','mirrorcoat')).astype(jnp.bool_)


def slot(move):return jnp.where(move==COUNTER,COUNTER_SLOT,COAT_SLOT)


def before_move(c,row,e):
    side,mon,action=row[1:4]
    move=jnp.where(action>=e.FORCED_MOVE_ACTION,action-e.FORCED_MOVE_ACTION,c[0].moves[side,mon,jnp.clip(action,0,3)])
    def prepare(c):
        kind=slot(move)
        def add(c):
            c=e.effects.add_volatile(c,side,kind,side,e.i32(0),e,source_kind=e.phase_ops.SOURCE_EMPTY)
            s,tr=c;return s._replace(volatile_state=s.volatile_state.at[side,mon,kind,5].set(1)),tr
        return e._when((c[0].active[side]==mon)&(c[0].hp[side,mon]>0),add,c)
    return e.jax.lax.cond(MOVE_IMPLEMENTED[move],prepare,lambda c:e.pursuit_ops.prepare(c,row,e),c)


def denied(s,actor,move):
    row=s.volatile_state[actor,s.active[actor],slot(move)]
    return MOVE_IMPLEMENTED[move]&((row[0]==0)|(row[8]==0))


def amount(s,actor,move):
    return jnp.maximum(1,s.volatile_state[actor,s.active[actor],slot(move),7])


def after_hit(c,actor,target,move,dealt,e):
    s,tr=c;mon=s.active[target];category=t.CATEGORY[move]
    kind=jnp.where(category==1,COUNTER_SLOT,COAT_SLOT);row=s.volatile_state[target,mon,kind]
    record=(actor!=target)&(category>0)&(row[0]>0)
    row=row.at[7].set(2*dealt).at[8].set(1)
    return s._replace(volatile_state=s.volatile_state.at[target,mon,kind].set(jnp.where(record,row,s.volatile_state[target,mon,kind]))),tr
