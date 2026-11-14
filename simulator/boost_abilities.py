"""Download startup boosts and per-stat Defiant reactions."""
import jax.numpy as jnp
from . import gen5_engine_tables as tables
A=tables.ABILITY_IDS


def raise_stat(c,side,stat,amount,ability,e,*,emit_zero=False):
    s,tr=c;mon=s.active[side];old=s.boosts[side,mon,stat];new=jnp.minimum(6,old+amount);change=new-old
    s=s._replace(boosts=s.boosts.at[side,mon,stat].set(new))
    s=e.tracking.boosted(s,side,change>0,jnp.array(False));c=(s,tr)
    c=e._when(change>0,lambda c:e._event(c,e.REVEAL,side,mon,value=ability,aux=1),c)
    return e._when((change>0)|emit_zero,lambda c:e._event(c,e.BOOST,side,mon,value=change,aux=stat),c)


def after_drop(c,actor,target,change,e):
    s,_=c;ability=e.suppression_ops.abilities(s)[target,s.active[target]]
    return e._when((actor!=target)&(change<0)&(ability==A['defiant']),lambda c:raise_stat(c,target,e.i32(0),e.i32(2),e.i32(A['defiant']),e,emit_zero=True),c)


def download(c,side,e):
    s,_=c;foe=1-side;mon=s.active[foe]
    defense=e.field_ops.scale_stat_stage(s.stats[foe,mon,2],s.boosts[foe,mon,1])
    special=e.field_ops.scale_stat_stage(s.stats[foe,mon,4],s.boosts[foe,mon,3])
    stat=jnp.where(defense>=special,2,0)
    return e._when(s.hp[foe,mon]>0,lambda c:raise_stat(c,side,stat,e.i32(1),e.i32(A['download']),e),c)
