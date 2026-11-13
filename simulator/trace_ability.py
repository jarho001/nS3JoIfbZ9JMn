"""Trace Start/Update copying with explicit saved seek state."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
TRACE=t.ABILITY_IDS['trace']
ABILITY_IMPLEMENTED=jnp.asarray([k=='trace'for k in t.ABILITIES],jnp.bool_)
NOTRACE=jnp.asarray([bool(next((r for r in t.g.DATA['abilities']if r['id']==k),{}).get('flags',{}).get('notrace'))for k in t.ABILITIES],jnp.bool_)

def copy(c,side,e,initial=False):
    s,_=c;mon=s.active[side];foe=1-side;other=s.active[foe];ability=s.ability[foe,other]
    allowed=(e.suppression_ops.abilities(s)[side,mon]==TRACE)&(s.ability_seek[side,mon]==1)&(s.hp[side,mon]>0)&(s.hp[foe,other]>0)&(ability>0)&(~NOTRACE[ability])
    def acquire(c):
        c,_=e._draw(c,0,1,e.INIT if initial else e.TRACE_TARGET)
        c=e.ability_ops.end(c,side,e);s,tr=c;old=s.ability[side,mon]
        s=e.ability_ops.assign(s,side,ability,e)
        c=e._event((s,tr),e.ABILITY_CHANGE,side,mon,target=foe,targetmon=other,value=ability,aux=old,status=2)
        return c
    return e._when(allowed,acquire,c),allowed

def update(c,side,e,initial=False):
    c,copied=copy(c,side,e,initial=initial)
    return e._when(copied,lambda c:e.hook_ops.ordinary_ability_start(c,side,e,initial=initial),c)

def start(c,side,e,initial=False):
    s,tr=c;mon=s.active[side]
    s=s._replace(ability_seek=s.ability_seek.at[side,mon].set(1))
    # ability_start owns the single ordinary Start dispatch after acquisition.
    return copy((s,tr),side,e,initial=initial)[0]
