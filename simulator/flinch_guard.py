"""Gen5 Inner Focus and Fake Out use existing flinch/action-count state."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
INNER=t.ABILITY_IDS['innerfocus'];FAKE=t.MOVE_IDS['fakeout']
ABILITY_IMPLEMENTED=jnp.asarray([key=='innerfocus' for key in t.ABILITIES],jnp.bool_)


def blocked(s,target,e):
    return e.suppression_ops.abilities(s)[target,s.active[target]]==INNER


def denied(s,actor,move,e):
    return (move==FAKE)&(s.action_state[actor,s.active[actor],e.tracking.ACTIONS]>1)


def hint(c,e):
    # Native Fake Out calls hint without once=true: repeat on every denial.
    return e._event(c,e.FAKE_OUT_HINT)
