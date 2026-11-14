"""Rain Dish weather healing and Hydration residual cure in native order."""
import jax.numpy as jnp
from . import gen5_engine_tables as tables

A=tables.ABILITY_IDS
ABILITY_IMPLEMENTED=jnp.asarray([key in ('raindish','hydration')for key in tables.ABILITIES],jnp.bool_)


def hydration(c,side,e):
    s,_=c;mon=s.active[side]
    active=(e.suppression_ops.abilities(s)[side,mon]==A['hydration'])&(s.weather_state[0]==1)&(s.status[side,mon]!=e.NONE)&(s.status[side,mon]!=e.FNT)
    def cure(c):
        c=e._event(c,e.STATUS_ABILITY,side,mon,value=A['hydration'])
        return e.effects.cure(c,side,e)
    return e._when(active,cure,c)
