"""Gen5 trapping truth, species queries and private request disclosure.

Known types only under the current capability masks; Transform/Illusion/type
mutation must extend these predicates before their handlers are enabled.
"""
import jax.numpy as jnp
from . import gen5_engine_tables as t

A=t.ABILITY_IDS
ABILITIES=('shadowtag','arenatrap','magnetpull','frisk')
ABILITY_IMPLEMENTED=jnp.asarray([key in ABILITIES for key in t.ABILITIES],jnp.bool_)
QUERY=jnp.asarray([[False,False,False]]+[
 [any(t.g.to_id(name)==ability and not(slot=='H' and t.SP_ROWS[t.g.to_id(species)].get('unreleasedHidden',False))
      for slot,name in t.SP_ROWS[t.g.to_id(species)]['abilities'].items())
  for ability in ('shadowtag','arenatrap','magnetpull')]
 for species in t.SPECIES[1:]],jnp.bool_)


def refresh(s,e):
    side=jnp.arange(2);mon=s.active;foe=1-side;fm=s.active[foe]
    ability=s.ability[side,mon];enemy=s.ability[foe,fm]
    types=s.types[side,mon]
    grounded=(~jnp.any(types==e.generic.TYPES['flying'],axis=1))&(ability!=A['levitate'])
    steel=jnp.any(types==e.generic.TYPES['steel'],axis=1)
    predicates=jnp.stack((ability!=A['shadowtag'],grounded,steel),axis=1)
    live=~s.fainted[foe,fm]
    actual=jnp.stack((enemy==A['shadowtag'],enemy==A['arenatrap'],enemy==A['magnetpull']),axis=1)
    trapped=jnp.any(actual&predicates,axis=1)&live
    maybe=jnp.any((QUERY[s.species[foe,fm]]|actual)&predicates,axis=1)&live
    state=s.action_state.at[side,mon,e.tracking.TRAPPED].set(jnp.where(trapped,2,0))
    state=state.at[side,mon,e.tracking.MAYBE_TRAPPED].set(maybe.astype(jnp.int32))
    return s._replace(action_state=state)


def has_bench(s,e):
    return jnp.any(e._live(s)&(jnp.arange(6)[None,:]!=s.active[:,None]),axis=1)


def selectable_switches(s,e):
    side=jnp.arange(2);mon=s.active
    hard=s.action_state[side,mon,e.tracking.TRAPPED]==1
    return (~hard)&(s.request_state[:,0]==0)


def private_error(s,choices,mask,e):
    """Return error codes1(disclose),2(known trap),3(cannot undo/replace)."""
    side=jnp.arange(2);mon=s.active
    incoming=choices>=0;undo=choices==-2
    locked=(incoming|undo)&(s.request_state[:,1]>0)
    # Check genuine roster targets before hidden trap truth. No disclosure on a
    # malformed or non-existent switch candidate.
    slot=jnp.clip(choices-4,0,5);target=s.order[side,slot]
    bench=e._live(s)[side,target]&(target!=mon)
    switching=(choices>=4)&(choices<10)&bench&(s.request==e.MOVE)
    truth=s.action_state[side,mon,e.tracking.TRAPPED]>0
    trap=switching&truth
    return jnp.where(locked,3,jnp.where(trap,jnp.where((s.request_state[:,0]>0)|(s.action_state[side,mon,e.tracking.TRAPPED]==1),2,1),0)).astype(jnp.int32)
