"""Shed Skin residual cures and Poison Heal's Damage replacement."""
import jax.numpy as jnp
from . import gen5_engine_tables as tables

A=tables.ABILITY_IDS
ABILITY_IMPLEMENTED=jnp.asarray([key in ('shedskin','poisonheal')for key in tables.ABILITIES],jnp.bool_)


def shed_skin(c,side,e):
    s,_=c;mon=s.active[side]
    active=(e.suppression_ops.abilities(s)[side,mon]==A['shedskin'])&(s.hp[side,mon]>0)&(s.status[side,mon]!=e.NONE)&(s.status[side,mon]!=e.FNT)
    def attempt(c):
        c,value=e._draw(c,0,100,e.FIELD_SORT)
        def cure(c):
            c=e._event(c,e.STATUS_ABILITY,side,mon,value=A['shedskin'])
            return e.effects.cure(c,side,e)
        return e._when(value<33,cure,c)
    return e._when(active,attempt,c)


def poison_heal(c,side,e):
    s,_=c;mon=s.active[side];maximum=s.stats[side,mon,0]
    def heal(c):
        s,t=c;hp=jnp.minimum(maximum,s.hp[side,mon]+jnp.maximum(1,maximum//8))
        s=s._replace(hp=s.hp.at[side,mon].set(hp))
        return e._event((s,t),e.HEAL,side,mon,value=hp,aux=maximum,status=s.status[side,mon],move=9)
    return e._when((s.hp[side,mon]>0)&(s.hp[side,mon]<maximum),heal,c)
