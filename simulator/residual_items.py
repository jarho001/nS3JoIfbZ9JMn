"""Black Sludge and status Orb callbacks on the shared residual scheduler."""
import jax.numpy as jnp
from . import gen5_engine_tables as tables
I=tables.ITEM_IDS
ITEM_IMPLEMENTED=jnp.asarray([key in ('blacksludge','toxicorb','flameorb')for key in tables.ITEMS],jnp.bool_)


def tick(c,side,e):
    s,_=c;mon=s.active[side];item=s.item[side,mon]
    def sludge(c):
        s,t=c;maximum=s.stats[side,mon,0]
        def heal(c):
            s,t=c;hp=jnp.minimum(maximum,s.hp[side,mon]+jnp.maximum(1,maximum//16))
            s=s._replace(hp=s.hp.at[side,mon].set(hp))
            return e._event((s,t),e.HEAL,side,mon,value=hp,aux=maximum,status=s.status[side,mon],move=10)
        def damage(c):
            c=e._damage(c,side,maximum//8,cause=16)
            c=e._when(c[0].hp[side,mon]==0,lambda c:e._faint(c,side),c)
            return e._win(c,side)
        return e.jax.lax.cond(jnp.any(s.types[side,mon]==e.generic.TYPES['poison']),lambda c:e._when(s.hp[side,mon]<maximum,heal,c),damage,c)
    def orb(c):
        status=jnp.where(item==I['toxicorb'],e.TOX,e.BRN)
        return e.effects.inflict(c,side,side,status,e,reflect=False,secondary=True,item=item)
    return e.jax.lax.cond(item==I['blacksludge'],sludge,orb,c)
