"""Custap's quarter-HP queue callback and exact tenths priority representation."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
ITEM=t.ITEM_IDS['custapberry'];PRIORITY_SCALE=10
# Gluttony/Unnerve/item suppression dependencies remain gated capabilities.


def consume(c,side,e):
    s,_=c;mon=s.active[side]
    eligible=(s.item[side,mon]==ITEM)&(s.hp[side,mon]>0)&(4*s.hp[side,mon]<=s.stats[side,mon,0])
    def eat(c):
        c=e._event(c,e.END_ITEM,side,mon,value=ITEM)
        s,tr=c;s=e.item_ops.used(s,side,e.i32(ITEM),e)
        c=e._event((s,tr),e.CUSTAP_ACTIVATE,side,mon,value=ITEM)
        return c,e.i32(1)
    return e.jax.lax.cond(eligible,eat,lambda c:(c,e.i32(0)),c)
