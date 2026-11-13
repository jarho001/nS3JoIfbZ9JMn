"""Gen5 force-switch flags and synchronous drag entry through the shared kernel."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
MOVES=t.mv(lambda r:bool(r.get('forceSwitch'))).astype(jnp.bool_)
STATUS=MOVES&(t.CATEGORY==0)
SOUND=t.mv(lambda r:bool(r.get('flags',{}).get('sound'))).astype(jnp.bool_)
ABILITY_IMPLEMENTED=jnp.asarray([k in ('suctioncups','soundproof')for k in t.ABILITIES],jnp.bool_)

def supported(row):
    if row['id']in ('roar','whirlwind'):return True
    if not row.get('forceSwitch'):return False
    clean=dict(row);clean.pop('forceSwitch');clean['flags']={k:v for k,v in clean.get('flags',{}).items()if k not in ('noassist','failcopycat')}
    return t.plain(clean)
MOVE_IMPLEMENTED=jnp.asarray([False]+[supported(t.MV_ROWS[k])for k in t.MOVE_KEYS[1:]],jnp.bool_)


def mark(c,actor,target,move,e):
    s,_=c;tm=s.active[target]
    can=(s.hp[target,tm]>0)&(s.hp[actor,s.active[actor]]>0)&e.trap_ops.has_bench(s,e)[target]
    def attempt(c):
        def held(c):return e._event(c,e.PHAZE_ABILITY,target,tm,value=t.ABILITY_IDS['suctioncups'])
        def flag(c):
            s,tr=c;return s,tr._replace(force_switch=tr.force_switch.at[target].set(True))
        return e.jax.lax.cond(e.suppression_ops.abilities(s)[target,tm]==t.ABILITY_IDS['suctioncups'],held,flag,c)
    return e._when(MOVES[move]&can,attempt,c)


def status_hit(c,actor,target,move,move_row,e):
    can=e.trap_ops.has_bench(c[0],e)[target]
    def success(c):
        c=mark(c,actor,target,move,e)
        c=e.tracking.attacked(c,target,actor,move,e.i32(0),e.i32(e.tracking.TRUE),e.i32(0),e)
        c=e.tracking.outcome(c,actor,e.i32(e.tracking.TRUE),move)
        return e._update(e._update(c,e.HIT_UPDATE),e.HIT_UPDATE)
    def fail(c):
        s,tr=c;tr=tr._replace(events=tr.events.at[move_row,7].set(2))
        return e._event((s,tr),e.FAIL,actor,s.active[actor])
    return e.jax.lax.cond(can,success,fail,c)


def resolve(c,e):
    for side in range(2):
        def drag(c):
            s,tr=c;mon=s.active[side];mask=e._live(s)[side,s.order[side]]&(s.order[side]!=mon)
            count=jnp.sum(mask).astype(jnp.int32)
            def select(c):
                c,draw=e._draw(c,0,count,e.PHAZE_SELECTION)
                slots=jnp.nonzero(mask,size=6,fill_value=0)[0];slot=slots[draw.astype(jnp.int32)]
                row=e.i32([e.INSTANT,side,0,0,0,0,0]).at[2].set(mon).at[3].set(slot+4)
                return e._switch(c,row,drag=True)
            return e._when((s.hp[side,mon]>0)&(count>0),select,c)
        c=e._when(c[1].force_switch[side],drag,c)
    s,tr=c
    return s,tr._replace(force_switch=jnp.zeros(2,jnp.bool_))
