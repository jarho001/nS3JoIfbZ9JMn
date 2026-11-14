"""Gen5 item exchange, removal and restoration with immutable initial provenance."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
TRICK,SWITCHEROO,RECYCLE,KNOCK=(t.MOVE_IDS[k]for k in ('trick','switcheroo','recycle','knockoff'))
EXCHANGE=t.mv(lambda r:r['id']in ('trick','switcheroo')).astype(jnp.bool_)
MOVE_IMPLEMENTED=jnp.asarray([k in ('trick','switcheroo','recycle','knockoff')for k in t.MOVE_KEYS],jnp.bool_)
ABILITY_IMPLEMENTED=jnp.asarray([k=='stickyhold'for k in t.ABILITIES],jnp.bool_)

def used(s,side,item,e,*,berry=True):
    s=e.tracking.item_used(s,side,item,berry=berry);mon=s.active[side]
    s=s._replace(item=s.item.at[side,mon].set(0),item_order=s.item_order.at[side,mon].set(0))
    return e.gem_ops.after_use(s,side,e)


def take(c,side,e,source=None):
    s,tr=c;mon=s.active[side]
    s=e.gem_ops.taken(s,side,side if source is None else source,e)
    return s._replace(item=s.item.at[side,mon].set(0),item_order=s.item_order.at[side,mon].set(0),item_provenance=s.item_provenance.at[side,mon,1].set(0)),tr


def give(c,side,item,e):
    s,tr=c;mon=s.active[side]
    s=s._replace(item=s.item.at[side,mon].set(item),item_order=s.item_order.at[side,mon].set(s.effect_order),effect_order=s.effect_order+1,item_provenance=s.item_provenance.at[side,mon,1].set(1))
    return e.hook_ops.item_start((s,tr),side,e)


def fail(c,actor,move_row,e):
    s,tr=c;tr=tr._replace(events=tr.events.at[move_row,7].set(2))
    return e._event(e.tracking.outcome((s,tr),actor,e.i32(e.tracking.FALSE),s.move_history[actor,s.active[actor],0,0]),e.FAIL,actor,s.active[actor])


def exchange(c,actor,target,move,move_row,e):
    s,_=c;mon=s.active[actor];tm=s.active[target];mine=s.item[actor,mon];yours=s.item[target,tm]
    def swap(c):
        c=take(c,target,e,source=actor);c=take(c,actor,e)
        c=e._event(c,e.ITEM_SWAP,actor,mon,move=move,target=target,targetmon=tm)
        for side,item,other in ((target,mine,yours),(actor,yours,mine)):
            def install(c):
                c=give(c,side,item,e)
                return e._event(c,e.ITEM_GAIN,side,c[0].active[side],move=move,value=item)
            def empty(c):return e._event(c,e.ITEM_LOSS,side,c[0].active[side],move=move,value=other,aux=1)
            c=e.jax.lax.cond(item>0,install,empty,c)
        return e.tracking.outcome(c,actor,e.i32(e.tracking.TRUE),move)
    return e.jax.lax.cond((mine>0)|(yours>0),swap,lambda c:fail(c,actor,move_row,e),c)


def recycle(c,actor,move_row,e):
    s,_=c;mon=s.active[actor];item=s.action_state[actor,mon,e.tracking.LAST_ITEM]
    def restore(c):
        s,tr=c;s=e.tracking.put(s,actor,e.tracking.LAST_ITEM,0)
        c=e._event((s,tr),e.ITEM_GAIN,actor,mon,move=RECYCLE,value=item)
        c=give(c,actor,item,e)
        return e.tracking.outcome(c,actor,e.i32(e.tracking.TRUE),RECYCLE)
    return e.jax.lax.cond((s.item[actor,mon]==0)&(item>0),restore,lambda c:fail(c,actor,move_row,e),c)


def knock(c,actor,target,move,e):
    s,_=c;mon=s.active[actor];tm=s.active[target];item=s.item[target,tm]
    def remove(c):
        sticky=(e.suppression_ops.abilities(s)[target,tm]==t.ABILITY_IDS['stickyhold'])&(s.hp[target,tm]>0)
        def held(c):return e._event(c,e.PHAZE_ABILITY,target,tm,value=t.ABILITY_IDS['stickyhold'])
        def lost(c):return e._event(take(c,target,e),e.ITEM_LOSS,target,tm,move=KNOCK,value=item,target=actor,targetmon=mon)
        return e.jax.lax.cond(sticky,held,lost,c)
    return e._when((move==KNOCK)&(s.hp[actor,mon]>0)&(item>0),remove,c)
