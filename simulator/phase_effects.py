"""Pure JAX forced recharge phase; other move locks remain unsupported."""
import jax.numpy as jnp
from . import gen5_engine_tables as t

RECHARGE_SLOT=4
# Internal queued action, distinct from the public eleven-action request space.
RECHARGE_ACTION=11
RECHARGE=t.mv(lambda row:row.get('self')=={'volatileStatus':'mustrecharge'}).astype(jnp.bool_)


def supported(row):
    if row.get('self')!={'volatileStatus':'mustrecharge'}:return False
    clean=dict(row);clean.pop('self')
    clean['flags']={k:v for k,v in row.get('flags',{}).items() if k!='recharge'}
    return t.plain(clean)


MOVE_IMPLEMENTED=jnp.asarray([False]+[supported(t.MV_ROWS[key]) for key in t.MOVE_KEYS[1:]],jnp.bool_)


def locked(s):
    return s.volatile_state[jnp.arange(2),s.active,RECHARGE_SLOT,0]>0


def install(c,actor,move,e):
    def apply(c):
        c=e.effects.add_volatile(c,actor,RECHARGE_SLOT,actor,move,e)
        s,trace=c;mon=s.active[actor]
        s=s._replace(volatile_state=s.volatile_state.at[actor,mon,RECHARGE_SLOT,5].set(2))
        return e._event((s,trace),e.MUST_RECHARGE,actor,mon)
    return e._when(RECHARGE[move]&(c[0].hp[actor,c[0].active[actor]]>0),apply,c)


def consume(c,actor,e):
    # Native BeforeMove priority11 precedes sleep/freeze/confusion/paralysis.
    s,trace=c;mon=s.active[actor]
    s=s._replace(volatile_state=s.volatile_state.at[actor,mon,RECHARGE_SLOT].set(0))
    return e._event((s,trace),e.RECHARGE_CANT,actor,mon)


def tick(c,side,mon):
    s,trace=c;row=s.volatile_state[side,mon,RECHARGE_SLOT]
    row=row.at[5].add(-1)
    return s._replace(volatile_state=s.volatile_state.at[side,mon,RECHARGE_SLOT].set(jnp.where(row[5]>0,row,0))),trace


def valid(s):
    row=s.volatile_state[:,:,RECHARGE_SLOT];present=row[:,:,0]>0
    source=row[:,:,4];safe=jnp.clip(source,0,len(t.MOVE_KEYS)-1)
    metadata=(row[:,:,1]<s.effect_order)&(row[:,:,2]==jnp.arange(2)[:,None])&(row[:,:,3]==jnp.arange(6)[None,:])
    metadata=metadata&(source>0)&(source<len(t.MOVE_KEYS))&MOVE_IMPLEMENTED[safe]&(row[:,:,5]>=1)&(row[:,:,5]<=2)
    active=(s.active[:,None]==jnp.arange(6)[None,:])&(~s.fainted)&(s.species>0)
    return jnp.all(jnp.where(present,metadata&active,jnp.all(row==0,axis=-1)))
