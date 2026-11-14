"""Gen5 Facade's late chain modifier and Acrobatics' live item callback."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
FACADE=t.MOVE_IDS['facade'];ACROBATICS=t.MOVE_IDS['acrobatics']


def base_power(s,actor,move,value):
    empty=s.item[actor,s.active[actor]]==0
    return value*jnp.where((move==ACROBATICS)&empty,2,1)


def modifier(s,actor,move,e):
    status=s.status[actor,s.active[actor]]
    # The pinned Gen5 callback excludes sleep and retains ordinary burn damage
    # reduction. Guts and move category remain in their existing stat/final hooks.
    return jnp.where((move==FACADE)&(status!=e.NONE)&(status!=e.SLP),8192,4096).astype(jnp.int32)
