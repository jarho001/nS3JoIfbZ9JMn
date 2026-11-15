"""Live Analytic queue predicate and Toxic Boost poison/category power."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
ANALYTIC=t.ABILITY_IDS['analytic'];TOXIC=t.ABILITY_IDS['toxicboost']
ABILITY_IMPLEMENTED=jnp.asarray([key in ('analytic','toxicboost') for key in t.ABILITIES],jnp.bool_)


def modifier(s,actor,target,physical,e):
    mon=s.active[actor];ability=e.suppression_ops.abilities(s)[actor,mon]
    # Native checks queued move presence, including status/recharge attempts,
    # rather than Speed or whether the opponent successfully moved.
    analytic=(ability==ANALYTIC)&(~e.phase_ops._will_move(s,target,s.active[target],e))
    toxic=(ability==TOXIC)&physical&((s.status[actor,mon]==e.PSN)|(s.status[actor,mon]==e.TOX))
    return jnp.where(analytic,5325,jnp.where(toxic,6144,4096)).astype(jnp.int32)
