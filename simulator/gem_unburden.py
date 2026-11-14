"""Pinned typed gems and Unburden's explicit use/take/end condition lifecycle."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
from .phase_effects import VOLATILE_IDS
GEM=VOLATILE_IDS['gem'];UNBURDEN=VOLATILE_IDS['unburden'];ABILITY=t.ABILITY_IDS['unburden']
GEM_TYPE=jnp.asarray([{'flyinggem':t.g.TYPES['flying'],'fightinggem':t.g.TYPES['fighting']}.get(k,-1)for k in t.ITEMS],jnp.int32)


def activate(s,side,source,e):
    mon=s.active[side]
    allowed=(s.hp[side,mon]>0)&(e.suppression_ops.abilities(s)[side,mon]==ABILITY)
    return e._when(allowed,lambda s:e.effects.add_volatile_state(s,side,UNBURDEN,source,e.i32(ABILITY),e,source_kind=e.phase_ops.SOURCE_ABILITY),s)


def after_use(s,side,e):
    return activate(s,side,side,e)


def taken(s,side,source,e):
    return e._when(s.item[side,s.active[side]]>0,lambda s:activate(s,side,source,e),s)


def end(c,side,e):
    s,tr=c;mon=s.active[side]
    return e._when(s.ability[side,mon]==ABILITY,lambda c:(c[0]._replace(volatile_state=c[0].volatile_state.at[side,mon,UNBURDEN].set(0)),c[1]),c)


def speed(s,ability):
    return (s.volatile_state[:,:,UNBURDEN,0]>0)&(s.item==0)&(ability==ABILITY)


def modifier(s,side):
    return jnp.where(s.volatile_state[side,s.active[side],GEM,0]>0,6144,4096).astype(jnp.int32)


def try_hit(c,actor,target,move,e):
    s,_=c;mon=s.active[actor];item=s.item[actor,mon]
    move_type=jnp.where(move==e.HIDDEN_POWER,s.hp_type[actor,mon],t.MOVE_TYPE[move])
    allowed=(actor!=target)&(t.CATEGORY[move]>0)&(GEM_TYPE[item]==move_type)
    def use(c):
        c=e._event(c,e.END_ITEM,actor,mon,move=move,value=item,aux=1)
        s,tr=c;present=s.volatile_state[actor,mon,GEM,0]>0
        s=e.item_ops.used(s,actor,item,e,berry=False)
        s=e.effects.add_volatile_state(s,actor,GEM,actor,-item,e,source_kind=e.phase_ops.SOURCE_ITEM)
        row=s.volatile_state[actor,mon,GEM].at[5].set(jnp.where(present,s.volatile_state[actor,mon,GEM,5],1))
        return s._replace(volatile_state=s.volatile_state.at[actor,mon,GEM].set(row)),tr
    return e._when(allowed,use,c)
