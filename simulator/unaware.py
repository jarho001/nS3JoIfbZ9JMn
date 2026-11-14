"""Unaware stage views and ordered native ModifyBoost query draws."""
import jax.numpy as jnp
from . import gen5_engine_tables as tables
ABILITY=tables.ABILITY_IDS['unaware']


def effective(s,side,e):
    return e.suppression_ops.abilities(s)[side,s.active[side]]==ABILITY


def query(c,e,context=None):
    s,_=c;mon=s.active;raw=s.ability[jnp.arange(2),mon]
    tied=jnp.all(raw==ABILITY)&jnp.all(~s.fainted[jnp.arange(2),mon])&(s.speed[0,mon[0]]==s.speed[1,mon[1]])
    return e._when(tied,lambda c:e._draw(c,0,2,e.MODIFY_BOOST_SORT if context is None else context,2)[0],c)


def pair(c,e):
    return query(query(c,e),e)


def refresh_speed(c,e,context=None):
    values=e._speeds(c[0])
    for side in range(2):
        mon=c[0].active[side]
        def visit(c):
            c=query(c,e,context);s,tr=c
            return s._replace(speed=s.speed.at[side,mon].set(values[side,mon])),tr
        c=e._when(~c[0].fainted[side,mon],visit,c)
    return c
