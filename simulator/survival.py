"""Gen5 direct-Move lethal damage limits, before actual damage accounting."""
import jax.numpy as jnp
from . import gen5_engine_tables as tables
A=tables.ABILITY_IDS;I=tables.ITEM_IDS
STURDY=A['sturdy'];SASH=I['focussash']
ABILITY_IMPLEMENTED=jnp.asarray([k=='sturdy'for k in tables.ABILITIES],jnp.bool_)
ITEM_IMPLEMENTED=jnp.asarray([k=='focussash'for k in tables.ITEMS],jnp.bool_)


def limit(c,side,damage,e):
    s,_=c;mon=s.active[side];hp=s.hp[side,mon]
    full=hp==s.stats[side,mon,0]
    sturdy=full&(damage>=hp)&(e.suppression_ops.abilities(s)[side,mon]==STURDY)
    c=e._when(sturdy,lambda c:e._event(c,e.REVEAL,side,mon,value=STURDY),c)
    damage=jnp.where(sturdy,hp-1,damage)
    sash=full&(damage>=hp)&(s.item[side,mon]==SASH)
    def consume(c):
        c=e._event(c,e.END_HELD_ITEM,side,mon,value=SASH)
        s,tr=c;s=e.item_ops.used(s,side,e.i32(SASH),e,berry=False)
        return s,tr
    c=e._when(sash,consume,c)
    return c,jnp.where(sash,hp-1,damage)
