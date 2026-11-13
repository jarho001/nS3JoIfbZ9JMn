"""Gen5 Substitute creation and per-hit primary interception, using shared hooks."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
MOVE=t.MOVE_IDS['substitute']
SLOT=20
BYPASS=t.mv(lambda row:bool(row.get('flags',{}).get('bypasssub'))).astype(jnp.bool_)


def intercepts(s,target,move):
    return (s.volatile_state[target,s.active[target],SLOT,0]>0)&(~BYPASS[move])


def create(c,actor,move_row,e):
    s,_=c;mon=s.active[actor];maxhp=s.stats[actor,mon,0]
    present=s.volatile_state[actor,mon,SLOT,0]>0;weak=(s.hp[actor,mon]*4<=maxhp)|(maxhp==1)
    def fail(c):
        return e._event(c,e.FAIL,actor,mon,aux=jnp.where(present,2,3))
    def success(c):
        c=e.effects.add_volatile(c,actor,SLOT,actor,MOVE,e)
        s,trace=c;cost=maxhp//4
        s=s._replace(volatile_state=s.volatile_state.at[actor,mon,SLOT,7].set(cost),hp=s.hp.at[actor,mon].add(-cost))
        c=e._event((s,trace),e.VOL_START,actor,mon,value=SLOT)
        c=e.partial_ops.end(c,actor,e,silent=True)
        # Native directDamage changes neither hurtThisTurn nor lastDamage.
        c=e._event(c,e.DAMAGE,actor,mon,value=s.hp[actor,mon],aux=maxhp,status=s.status[actor,mon])
        return e.tracking.outcome(c,actor,e.i32(e.tracking.TRUE),MOVE)
    return e.jax.lax.cond(present|weak,fail,success,c)


def block_status(c,actor,move,move_row,e):
    s,trace=c;trace=trace._replace(events=trace.events.at[move_row,7].set(2))
    # The native primary-hit null result still completes the outer move loop.
    return e._event((s,trace),e.FAIL,actor,s.active[actor])


def absorb(c,actor,target,move,damage,e):
    s,trace=c;mon=s.active[actor];tm=s.active[target];row=s.volatile_state[target,tm,SLOT]
    dealt=jnp.minimum(row[7],damage);hp=row[7]-dealt
    row=row.at[7].set(hp)
    s=s._replace(volatile_state=s.volatile_state.at[target,tm,SLOT].set(jnp.where(hp>0,row,0)))
    s=e.tracking.put(s,actor,e.tracking.LAST_DAMAGE,dealt)
    c=e.jax.lax.cond(hp>0,lambda c:e._event(c,e.SUB_DAMAGE,target,tm),lambda c:e._event(c,e.VOL_END,target,tm,value=SLOT),(s,trace))
    regular=(e.hit_tables.RECOIL_NUM[move]>0)&(e.suppression_ops.abilities(s)[actor,mon]!=t.ABILITY_IDS['rockhead'])
    def recoil(c):
        amount=jnp.where(move==e.STRUGGLE,(s.stats[actor,mon,0]+2)//4,jnp.maximum(1,e.hit_ops.rounded_fraction(dealt,e.hit_tables.RECOIL_NUM[move],e.hit_tables.RECOIL_DEN[move])))
        return e._damage(c,actor,amount,cause=jnp.where(move==e.STRUGGLE,3,10))
    c=e._when(((move==e.STRUGGLE)|regular)&(dealt>0),recoil,c)
    c=e.hit_ops.drain(c,actor,target,move,dealt,e,substitute=True)
    c=e.phase_ops.install(c,actor,move,e)
    c=e._when((move==e.RAPID_SPIN)&(c[0].hp[actor,mon]>0),lambda c:e.hit_ops.spin(c,actor,e),c)
    chance=jnp.maximum(e.tables.SECONDARY_CHANCE[move],e.phase_ops.FLINCH_CHANCE[move])
    def secondary_draw(c):
        c,_=e._draw(c,0,100,e.SECONDARY)
        return c
    c=e._when(chance>0,secondary_draw,c)
    return c,dealt
