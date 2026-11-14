"""Default-order Weak Armor and Cursed Body DamagingHit callbacks."""
import jax.numpy as jnp
from . import gen5_engine_tables as tables
A=tables.ABILITY_IDS


def weak_armor(c,target,e):
    s,tr=c;mon=s.active[target]
    defense=jnp.maximum(-6,s.boosts[target,mon,1]-1);speed=jnp.minimum(6,s.boosts[target,mon,4]+1)
    drops=defense-s.boosts[target,mon,1];rises=speed-s.boosts[target,mon,4]
    s=s._replace(boosts=s.boosts.at[target,mon,1].set(defense).at[target,mon,4].set(speed))
    s=e.tracking.boosted(s,target,rises>0,drops<0);c=(s,tr)
    c=e._when((drops<0)|(rises>0),lambda c:e._event(c,e.REVEAL,target,mon,value=A['weakarmor'],aux=1),c)
    c=e._when(drops<0,lambda c:e._event(c,e.BOOST,target,mon,value=drops,aux=1,status=1),c)
    return e._when(rises>0,lambda c:e._event(c,e.BOOST,target,mon,value=rises,aux=4),c)


def disable(c,actor,target,e):
    s,_=c;mon=s.active[actor];last=s.move_history[actor,mon,0,0];kind=e.phase_ops.VOLATILE_IDS['disable']
    owns=jnp.any(s.moves[actor,mon]==last);has_pp=jnp.any((s.moves[actor,mon]==last)&(s.pp[actor,mon]>0))
    allowed=(last>0)&((~owns)|has_pp)
    c=e.effects.add_volatile(c,actor,kind,target,e.i32(A['cursedbody']),e,source_kind=e.phase_ops.SOURCE_ABILITY)
    s,tr=c;row=s.volatile_state[actor,mon,kind].at[5].set(4).at[7].set(last)
    # Native allocates effectOrder before Disable's Start rejects an empty-PP move.
    s=s._replace(volatile_state=s.volatile_state.at[actor,mon,kind].set(jnp.where(allowed,row,0)))
    return e._when(allowed,lambda c:e._event(c,e.VOL_START,actor,mon,move=last,value=kind,aux=1,target=target,targetmon=s.active[target]),(s,tr))


def after_hit(c,actor,target,move,e):
    s,_=c;mon=s.active[actor];tm=s.active[target];ability=e.suppression_ops.abilities(s)[target,tm]
    c=e._when((ability==A['weakarmor'])&(tables.CATEGORY[move]==1)&(s.hp[target,tm]>0),lambda c:weak_armor(c,target,e),c)
    allowed=(ability==A['cursedbody'])&(move!=e.STRUGGLE)&(s.volatile_state[actor,mon,e.phase_ops.VOLATILE_IDS['disable'],0]==0)
    def chance(c):
        c,roll=e._draw(c,0,10,e.CURSED_BODY)
        return e._when((roll<3)&(c[0].hp[actor,mon]>0),lambda c:disable(c,actor,target,e),c)
    return e._when(allowed,chance,c)
