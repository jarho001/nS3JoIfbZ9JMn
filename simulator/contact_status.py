"""Native default-order Static/Flame Body contact-status chances."""
import jax.numpy as jnp
from . import gen5_engine_tables as tables
A=tables.ABILITY_IDS


def after_hit(c,actor,target,move,e):
    s,_=c;ability=e.suppression_ops.abilities(s)[target,s.active[target]]
    allowed=e.hit_tables.CONTACT[move]&((ability==A['static'])|(ability==A['flamebody']))
    def chance(c):
        c,roll=e._draw(c,0,10,e.CONTACT_STATUS)
        status=jnp.where(ability==A['static'],e.PAR,e.BRN)
        return e._when(roll<3,lambda c:e.effects.inflict(c,target,actor,status,e,secondary=True,source_ability=ability),c)
    return e._when(allowed,chance,c)
