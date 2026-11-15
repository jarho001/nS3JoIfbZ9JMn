"""Gen5 Justified hit boosts and Speed Boost residual timing."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
JUSTIFIED=t.ABILITY_IDS['justified'];SPEED=t.ABILITY_IDS['speedboost']
ABILITY_IMPLEMENTED=jnp.asarray([key in ('justified','speedboost') for key in t.ABILITIES],jnp.bool_)


def after_hit(c,target,move_type,e):
    s,_=c;mon=s.active[target]
    allowed=(s.hp[target,mon]>0)&(e.suppression_ops.abilities(s)[target,mon]==JUSTIFIED)&(move_type==e.generic.TYPES['dark'])
    # Native can raise the surviving holder even after Helmet ended the game.
    return e._when(allowed,lambda c:e.boost_ability_ops.raise_stat(c,target,e.i32(0),e.i32(1),e.i32(JUSTIFIED),e),c)


def residual(c,side,e):
    s,_=c;mon=s.active[side]
    allowed=(e.suppression_ops.abilities(s)[side,mon]==SPEED)&(s.action_state[side,mon,e.tracking.ACTIVE_TURNS]>0)
    return e._when(allowed,lambda c:e.boost_ability_ops.raise_stat(c,side,e.i32(4),e.i32(1),e.i32(SPEED),e),c)
