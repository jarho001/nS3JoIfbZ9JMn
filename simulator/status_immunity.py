"""Status-immunity SetStatus/Immunity and acquisition-time Update hooks."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
A=t.ABILITY_IDS
KEYS=('immunity','limber','waterveil','magmaarmor','insomnia','vitalspirit')
ABILITY_IMPLEMENTED=jnp.asarray([k in KEYS for k in t.ABILITIES],jnp.bool_)
SLEEP=jnp.asarray([k in ('insomnia','vitalspirit')for k in t.ABILITIES],jnp.bool_)

def blocks(ability,status,e):
    return (((status==e.PSN)|(status==e.TOX))&(ability==A['immunity']))|((status==e.PAR)&(ability==A['limber']))|((status==e.BRN)&(ability==A['waterveil']))|((status==e.FRZ)&(ability==A['magmaarmor']))|((status==e.SLP)&SLEEP[ability])

def update(c,side,e):
    s,_=c;mon=s.active[side];ability=e.suppression_ops.abilities(s)[side,mon]
    def cure(c):
        c=e._event(c,e.STATUS_ABILITY,side,mon,value=ability)
        return e.effects.cure(c,side,e)
    return e._when(blocks(ability,s.status[side,mon],e)&(s.hp[side,mon]>0),cure,c)

def rest_fail(c,side,e):
    s,_=c;mon=s.active[side]
    c=e._event(c,e.REST_ABILITY,side,mon,value=e.suppression_ops.abilities(s)[side,mon])
    return e.tracking.outcome(c,side,e.i32(e.tracking.FALSE),e.REST)
