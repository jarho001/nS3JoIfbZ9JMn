"""Illusion lifecycle and immutable event-time display identity."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
ABILITY=t.ABILITY_IDS['illusion']
ABILITY_IMPLEMENTED=jnp.asarray([k=='illusion'for k in t.ABILITIES],jnp.bool_)

def enter(s,side,e):
    mon=s.active[side];position=jnp.argmax(s.order[side]==mon)
    eligible=(jnp.arange(6)>position)&(s.species[side,s.order[side]]>0)&(~s.fainted[side,s.order[side]])
    last=jnp.max(jnp.where(eligible,jnp.arange(6),-1))
    target=jnp.where((s.ability[side,mon]==ABILITY)&(last>=0),s.order[side,jnp.maximum(0,last)],-1)
    return s._replace(illusion=s.illusion.at[side,mon].set(target))

def clear(s,side,e):
    return s._replace(illusion=s.illusion.at[side,s.active[side]].set(-1))

def reveal(c,side,e):
    def show(c):
        s,tr=c;s=clear(s,side,e)
        return e._event((s,tr),e.ILLUSION_END,side,s.active[side])
    return e._when(c[0].illusion[side,c[0].active[side]]>=0,show,c)

def event_identity(s,side,mon,target,tm):
    shown=jnp.where(s.illusion[side,mon]>=0,s.illusion[side,mon],mon)
    foe=jnp.where(s.illusion[target,tm]>=0,s.illusion[target,tm],tm)
    return (s.name[side,shown],s.name[target,foe],s.base_species[side,shown],s.level[side,shown],s.gender[side,shown],s.shiny[side,shown].astype(jnp.int32))
