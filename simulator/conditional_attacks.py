"""Queued-target predicates for Sucker Punch and Payback."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
SUCKER=t.MOVE_IDS['suckerpunch'];PAYBACK=t.MOVE_IDS['payback']


def queued(s,target,e):
    return (jnp.arange(e.QUEUE)<s.qcount)&(s.queue[:,0]==e.ATTACK)&(s.queue[:,1]==target)&(s.queue[:,2]==s.active[target])


def denied(s,target,move,e):
    actions=s.queue[:,3];mons=s.queue[:,2];sides=s.queue[:,1]
    selected=jnp.where(actions>=e.FORCED_MOVE_ACTION,actions-e.FORCED_MOVE_ACTION,jnp.where(actions==10,e.STRUGGLE,s.moves[jnp.clip(sides,0,1),jnp.clip(mons,0,5),jnp.clip(actions,0,3)]))
    selected=jnp.clip(selected,0,len(t.MOVE_KEYS)-1)
    # Me First is the pinned status-move exception; its own capability stays closed.
    attack=(t.CATEGORY[selected]>0)|(selected==t.MOVE_IDS['mefirst'])
    waiting=jnp.any(queued(s,target,e)&attack)
    return (move==SUCKER)&((~waiting)|(s.volatile_state[target,s.active[target],4,0]>0))


def power(s,target,move,value,e):
    switched=s.action_state[target,s.active[target],e.tracking.NEWLY_SWITCHED]>0
    double=(move==PAYBACK)&(~switched)&(~jnp.any(queued(s,target,e)))
    return value*jnp.where(double,2,1)
