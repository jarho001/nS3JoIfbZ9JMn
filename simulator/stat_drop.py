"""External boost filtering and White Herb lifecycle callbacks."""
import jax.numpy as jnp
from . import gen5_engine_tables as tables
A=tables.ABILITY_IDS;I=tables.ITEM_IDS
HERB=I['whiteherb'];CLEAR=A['clearbody'];PECKS=A['bigpecks'];CUTTER=A['hypercutter']
ITEM_IMPLEMENTED=jnp.asarray([k=='whiteherb'for k in tables.ITEMS],jnp.bool_)
ABILITY_IMPLEMENTED=jnp.asarray([k in ('clearbody','bigpecks','hypercutter')for k in tables.ABILITIES],jnp.bool_)


def blocked(s,actor,target,change,e):
    ability=e.suppression_ops.abilities(s)[target,s.active[target]]
    protected=(ability==CLEAR)|((ability==PECKS)&(jnp.arange(7)==1))|((ability==CUTTER)&(jnp.arange(7)==0))
    # Native clamps a drop at -6 to zero before TryBoost handlers inspect it.
    return (actor!=target)&protected&(change<0)&(s.boosts[target,s.active[target]]>-6)


def fail(c,target,e):
    s,_=c;mon=s.active[target];ability=e.suppression_ops.abilities(s)[target,mon]
    specific=ability!=CLEAR
    return e._event(c,jnp.where(specific,e.STAT_GUARD_BLOCK,e.BOOST_BLOCK),target,mon,value=jnp.where(specific,ability,0),aux=jnp.where(specific,jnp.where(ability==CUTTER,0,1),0))


def restore(c,side,e):
    s,_=c;mon=s.active[side]
    def consume(c):
        c=e._event(c,e.END_HELD_ITEM,side,mon,value=HERB)
        s,tr=c;s=s._replace(boosts=s.boosts.at[side,mon].set(jnp.maximum(0,s.boosts[side,mon])))
        c=e._event((s,tr),e.CLEAR_NEGATIVE,side,mon)
        s,tr=c;s=e.item_ops.used(s,side,e.i32(HERB),e,berry=False)
        return s,tr
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
