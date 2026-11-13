"""Gen5 binding-move timers, residuals and disclosed trapping, with native provenance."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
SLOT=21
MOVES=t.mv(lambda r:r.get('volatileStatus')=='partiallytrapped').astype(jnp.bool_)
ITEM_IMPLEMENTED=jnp.asarray([key in ('bindingband','gripclaw')for key in t.ITEMS],jnp.bool_)

def supported(row):
    if row.get('volatileStatus')!='partiallytrapped':return False
    clean=dict(row);clean.pop('volatileStatus')
    return t.plain(clean)

MOVE_IMPLEMENTED=jnp.asarray([False]+[supported(t.MV_ROWS[key])for key in t.MOVE_KEYS[1:]],jnp.bool_)


def install(c,actor,target,move,e):
    s,_=c;mon=s.active[actor];tm=s.active[target]
    def create(c):
        c=e.effects.add_volatile(c,target,SLOT,actor,move,e)
        fixed=c[0].item[actor,mon]==t.ITEM_IDS['gripclaw']
        c,duration=e.jax.lax.cond(fixed,lambda c:(c,jnp.uint32(8)),lambda c:e._draw(c,5,7,e.VOLATILE_DURATION,2),c)
        s,trace=c;row=s.volatile_state[target,tm,SLOT].at[5].set(duration.astype(jnp.int32)).at[7].set(jnp.where(s.item[actor,mon]==t.ITEM_IDS['bindingband'],8,16))
        s=s._replace(volatile_state=s.volatile_state.at[target,tm,SLOT].set(row))
        return e._event((s,trace),e.PARTIAL_START,target,tm,move=move,target=actor,targetmon=mon)
    return e._when(MOVES[move]&(s.hp[target,tm]>0)&(s.volatile_state[target,tm,SLOT,0]==0),create,c)


def end(c,side,e,silent=False):
    s,trace=c;mon=s.active[side];row=s.volatile_state[side,mon,SLOT]
    def apply(c):
        s,trace=c;s=s._replace(volatile_state=s.volatile_state.at[side,mon,SLOT].set(0))
        return e._event((s,trace),e.PARTIAL_END,side,mon,move=row[4],aux=int(silent))
    return e._when(row[0]>0,apply,c)


def tick(c,side,e):
    s,trace=c;mon=s.active[side];row=s.volatile_state[side,mon,SLOT].at[5].add(-1)
    s=s._replace(volatile_state=s.volatile_state.at[side,mon,SLOT].set(row));c=(s,trace)
    source,sm=row[2],row[3]
    gone=(s.active[source]!=sm)|(s.hp[source,sm]<=0)|(s.action_state[source,sm,e.tracking.ACTIVE_TURNS]==0)
    def damage(c):
        previous=c[1].ecount
        c=e._damage(c,side,s.stats[side,mon,0]//row[7],cause=15,source_side=source,source_mon=sm)
        state,trace=c;ix=trace.ecount-1
        c=e._when(trace.ecount>previous,lambda c:(c[0],c[1]._replace(events=c[1].events.at[ix,0].set(e.PARTIAL_DAMAGE).at[ix,3].set(row[4]))),c)
        c=e._when(c[0].hp[side,mon]==0,lambda c:e._faint(c,side),c)
        return e._win(c,side)
    return e.jax.lax.cond(row[5]<=0,lambda c:end(c,side,e),lambda c:e.jax.lax.cond(gone,lambda c:end(c,side,e,silent=True),damage,c),c)
