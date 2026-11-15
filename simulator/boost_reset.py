"""Raw boost resets bypass stage-change abilities and raised/lowered flags."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
HAZE=t.MOVE_IDS['haze'];SMOG=t.MOVE_IDS['clearsmog']
MOVE_IMPLEMENTED=t.mv(lambda r:r['id']=='haze' and r.get('onHitField')=={'$callback':'moves:haze:onHitField'} and r['target']=='all' and r['category']=='Status' and r['accuracy'] is True and not any(t.has_callback(v) for k,v in r.items() if k!='onHitField')).astype(jnp.bool_)


def field(c,actor,move,e):
    s,tr=c
    c=e._event((s,tr),e.CLEAR_BOOST,aux=1)
    s,tr=c
    for side in range(2):s=s._replace(boosts=s.boosts.at[side,s.active[side]].set(0))
    return e.tracking.outcome((s,tr),actor,e.i32(e.tracking.TRUE),move)


def hit(c,actor,target,move,e):
    def reset(c):
        s,tr=c;mon=s.active[target]
        s=s._replace(boosts=s.boosts.at[target,mon].set(0))
        return e._event((s,tr),e.CLEAR_BOOST,target,mon)
    return e._when(move==SMOG,reset,c)
