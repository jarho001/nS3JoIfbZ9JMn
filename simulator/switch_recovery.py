"""Singles switch-out recovery, after Pursuit and before volatile/base restoration."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
A=t.ABILITY_IDS
ABILITY_IMPLEMENTED=jnp.asarray([k in ('naturalcure','regenerator')for k in t.ABILITIES],jnp.bool_)

def out(c,side,e):
    s,_=c;mon=s.active[side];ability=e.suppression_ops.abilities(s)[side,mon];alive=s.hp[side,mon]>0
    # Natural Cure CheckShow returns immediately in singles. Its hidden-cure
    # doubles branch cannot occur, so every actual cure has native silent attribution.
    cure=alive&(ability==A['naturalcure'])&(s.status[side,mon]!=e.NONE)&(s.status[side,mon]!=e.FNT)
    c=e._when(cure,lambda c:e.effects.cure(c,side,e,ability=A['naturalcure'],silent=True),c)
    s,tr=c;hp=jnp.minimum(s.stats[side,mon,0],s.hp[side,mon]+s.stats[side,mon,0]//3)
    hp=jnp.where(alive&(ability==A['regenerator']),hp,s.hp[side,mon])
    # Pokemon.heal here changes HP without adding a public heal message.
    return s._replace(hp=s.hp.at[side,mon].set(hp)),tr
