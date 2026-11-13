"""Worry Seed/Simple Beam and Gen5 Simple/Contrary boost changes."""
import jax.numpy as jnp
from . import suppression as suppression_ops
from . import gen5_engine_tables as t
A=t.ABILITY_IDS;M=t.MOVE_IDS
WORRY=M['worryseed'];BEAM=M['simplebeam']
CREATED=t.mv(lambda r:A['insomnia']if r['id']=='worryseed'else A['simple']if r['id']=='simplebeam'else 0)
ABILITY_IMPLEMENTED=jnp.asarray([k in ('simple','contrary')for k in t.ABILITIES],jnp.bool_)

def delta(s,side,amount):
    ability=suppression_ops.abilities(s)[side,s.active[side]]
    return amount*jnp.where(ability==A['simple'],2,jnp.where(ability==A['contrary'],-1,1))

def immunity(s,target,move):
    ability=s.ability[target,s.active[target]]
    return (move==WORRY)&((ability==A['insomnia'])|(ability==A['truant']))

def denied(s,target,move,cant):
    ability=s.ability[target,s.active[target]]
    return jnp.where(CREATED[move]>0,cant[ability]|((move==BEAM)&((ability==A['simple'])|(ability==A['truant']))),False)
