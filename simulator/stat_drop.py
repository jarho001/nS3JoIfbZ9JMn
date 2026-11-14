"""Clear Body TryBoost filtering and White Herb lifecycle callbacks."""
import jax.numpy as jnp
from . import gen5_engine_tables as tables
A=tables.ABILITY_IDS;I=tables.ITEM_IDS
HERB=I['whiteherb'];CLEAR=A['clearbody']
ITEM_IMPLEMENTED=jnp.asarray([k=='whiteherb'for k in tables.ITEMS],jnp.bool_)
ABILITY_IMPLEMENTED=jnp.asarray([k=='clearbody'for k in tables.ABILITIES],jnp.bool_)


def blocked(s,actor,target,change,e):
    return (actor!=target)&(e.suppression_ops.abilities(s)[target,s.active[target]]==CLEAR)&(change<0)


def fail(c,target,e):
    return e._event(c,e.BOOST_BLOCK,target,c[0].active[target])


def restore(c,side,e):
    s,_=c;mon=s.active[side]
    def consume(c):
        c=e._event(c,e.END_HELD_ITEM,side,mon,value=HERB)
        s,tr=c;s=s._replace(boosts=s.boosts.at[side,mon].set(jnp.maximum(0,s.boosts[side,mon])))
        c=e._event((s,tr),e.CLEAR_NEGATIVE,side,mon)
        s,tr=c;s=e.tracking.item_used(s,side,e.i32(HERB),berry=False)
        return s._replace(item=s.item.at[side,mon].set(0),item_order=s.item_order.at[side,mon].set(0)),tr
    return e._when((s.item[side,mon]==HERB)&(s.hp[side,mon]>0)&jnp.any(s.boosts[side,mon]<0),consume,c)


def broadcast(c,e,context,switch_first=None):
    s,_=c;mons=s.active;holders=(s.item[jnp.arange(2),mons]==HERB)&(~s.fainted[jnp.arange(2),mons])
    speed=s.speed[jnp.arange(2),mons];tie=jnp.all(holders)&(speed[0]==speed[1])&(switch_first is None)
    c,word=e.jax.lax.cond(tie,lambda c:e._draw(c,0,2,context,2),lambda c:(c,jnp.uint32(0)),c)
    first=jnp.where(tie,word.astype(jnp.int32),jnp.argmax(speed))
    if switch_first is not None:first=switch_first
    for k in range(2):
        side=jnp.where(k==0,first,1-first)
        c=e._when(holders[side],lambda c:restore(c,side,e),c)
    return c
