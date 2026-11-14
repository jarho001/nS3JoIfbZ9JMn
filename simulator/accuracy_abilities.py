"""Singles accuracy abilities, including native callback-sort draws."""
import jax.numpy as jnp
from . import gen5_engine_tables as tables
A=tables.ABILITY_IDS
NO_GUARD=A['noguard'];VICTORY=A['victorystar']
ABILITY_IMPLEMENTED=jnp.asarray([k in ('noguard','victorystar','tangledfeet','wonderskin')for k in tables.ABILITIES],jnp.bool_)


def sure_hit(s,actor,target,e):
    ability=e.suppression_ops.abilities(s)
    return (ability[actor,s.active[actor]]==NO_GUARD)|(ability[target,s.active[target]]==NO_GUARD)


def modify_sort(c,actor,e,target=None):
    s,_=c;mons=s.active;raw=s.ability[jnp.arange(2),mons]
    # runEvent collects raw handlers before skipping suppressed abilities.
    if target is None:target=actor
    source=(raw[actor]==A['compoundeyes'])|(raw[actor]==A['hustle'])
    tied=((raw[0]==VICTORY)&(raw[1]==VICTORY))|((raw[1-actor]==VICTORY)&source)
    target_tangle=raw[target]==A['tangledfeet']
    tangle_tie=target_tangle&(source|(raw[actor]==VICTORY)|(raw[1-actor]==VICTORY))
    tied=(tied|tangle_tie)&(s.speed[0,mons[0]]==s.speed[1,mons[1]])
    return e._when(tied,lambda c:e._draw(c,0,2,e.MODIFY_ACCURACY_SORT,2)[0],c)


def accuracy_sort(c,e):
    s,_=c;mons=s.active;raw=s.ability[jnp.arange(2),mons]
    tied=jnp.all(raw==NO_GUARD)&(s.speed[0,mons[0]]==s.speed[1,mons[1]])
    return e._when(tied,lambda c:e._draw(c,0,2,e.ACCURACY_SORT,2)[0],c)
