"""Chesto sleep cures and Sitrus threshold healing in native Update order."""
import jax.numpy as jnp
from . import gen5_engine_tables as tables
I=tables.ITEM_IDS
ITEM_IMPLEMENTED=jnp.asarray([key in ('chestoberry','sitrusberry')for key in tables.ITEMS],jnp.bool_)


def update(c,side,e):
    s,_=c;mon=s.active[side];item=s.item[side,mon];maximum=s.stats[side,mon,0]
    active=(s.hp[side,mon]>0)&(((item==I['chestoberry'])&(s.status[side,mon]==e.SLP))|((item==I['sitrusberry'])&(s.hp[side,mon]*2<=maximum)))
    def eat(c):
        c=e._event(c,e.END_ITEM,side,mon,value=item)
        def heal(c):
            s,t=c;hp=jnp.minimum(maximum,s.hp[side,mon]+jnp.maximum(1,maximum//4))
            s=s._replace(hp=s.hp.at[side,mon].set(hp))
            return e._event((s,t),e.HEAL,side,mon,value=hp,aux=maximum,status=s.status[side,mon],move=11)
        c=e.jax.lax.cond(item==I['chestoberry'],lambda c:e.effects.cure(c,side,e),heal,c)
        s,t=c;s=e.item_ops.used(s,side,item,e)
        return s,t
    return e._when(active,eat,c)
