"""Relic Song and Zen Mode form triggers in the shared scheduler."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
ZEN=t.ABILITY_IDS['zenmode'];SONG=t.MOVE_IDS['relicsong'];S=t.SPECIES_IDS
MELOETTA=jnp.asarray([k.startswith('Meloetta')for k in t.SPECIES],jnp.bool_)
DARMANITAN=jnp.asarray([k in ('Darmanitan','Darmanitan-Zen')for k in t.SPECIES],jnp.bool_)
SPECIES_IMPLEMENTED=MELOETTA|DARMANITAN
MOVE_IMPLEMENTED=jnp.asarray([k=='relicsong'for k in t.MOVE_KEYS],jnp.bool_)
ABILITY_IMPLEMENTED=jnp.asarray([k=='zenmode'for k in t.ABILITIES],jnp.bool_)
SLOT=23

def valid_pair(s):
    return (MELOETTA[s.base_species]&MELOETTA[s.species])|(DARMANITAN[s.base_species]&DARMANITAN[s.species])

def after_hit(c,side,move,e):
    s,_=c;mon=s.active[side]
    allowed=(move==SONG)&MELOETTA[s.base_species[side,mon]]&(~s.transformed[side,mon])
    species=jnp.where(s.species[side,mon]==S['meloettapirouette'],S['meloetta'],S['meloettapirouette'])
    return e._when(allowed,lambda c:e.form_ops.set_form(c,side,species,e,message=1),c)

def residual(c,side,e):
    s,_=c;mon=s.active[side];allowed=DARMANITAN[s.base_species[side,mon]]&(~s.transformed[side,mon])
    low=s.hp[side,mon]*2<=s.stats[side,mon,0];zen=s.species[side,mon]==S['darmanitanzen']
    def enter(c):
        c=e.effects.add_volatile(c,side,SLOT,side,e.i32(ZEN),e,source_kind=e.phase_ops.SOURCE_ABILITY)
        return e.form_ops.set_form(c,side,S['darmanitanzen'],e,message=2)
    def leave(c):
        c=e.effects.add_volatile(c,side,SLOT,side,e.i32(ZEN),e,source_kind=e.phase_ops.SOURCE_ABILITY)
        s,tr=c;s=s._replace(volatile_state=s.volatile_state.at[side,mon,SLOT].set(0))
        return e.form_ops.set_form((s,tr),side,S['darmanitan'],e,message=2)
    c=e._when(allowed&low&(~zen),enter,c)
    return e._when(allowed&(~low)&zen,leave,c)

def end(c,side,e):
    s,_=c;mon=s.active[side]
    allowed=(s.ability[side,mon]==ZEN)&(s.volatile_state[side,mon,SLOT,0]>0)&(s.hp[side,mon]>0)
    def clear(c):
        s,tr=c;s=s._replace(volatile_state=s.volatile_state.at[side,mon,SLOT].set(0),transformed=s.transformed.at[side,mon].set(False))
        return e.form_ops.set_form((s,tr),side,S['darmanitan'],e,ability=ZEN,message=3)
    return e._when(allowed,clear,c)
