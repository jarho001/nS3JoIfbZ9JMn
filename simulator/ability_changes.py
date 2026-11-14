"""Gen5 Role Play, Skill Swap and Entrainment ability lifecycle."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
A=t.ABILITY_IDS;M=t.MOVE_IDS
ROLE=M['roleplay'];SWAP=M['skillswap'];ENTRAIN=M['entrainment']
MOVE_IMPLEMENTED=jnp.asarray([k in ('roleplay','skillswap','entrainment','worryseed','simplebeam')for k in t.MOVE_KEYS],jnp.bool_)
PROTECT=t.mv(lambda r:bool(r.get('flags',{}).get('protect'))).astype(jnp.bool_)
ROWS={r['id']:r for r in t.g.DATA['abilities']}
def flag(key):return jnp.asarray([bool(ROWS.get(k,{}).get('flags',{}).get(key))for k in t.ABILITIES],jnp.bool_)
CANT=flag('cantsuppress');ROLE_FAIL=flag('failroleplay');SWAP_FAIL=flag('failskillswap');NO_ENTRAIN=flag('noentrain')

def fail(c,actor,move,move_row,e):
    s,tr=c;tr=tr._replace(events=tr.events.at[move_row,7].set(2))
    c=e._event((s,tr),e.FAIL,actor,s.active[actor])
    return e.tracking.outcome(c,actor,e.i32(e.tracking.FALSE),move)

def try_hit(c,actor,target,move,move_row,e):
    s,_=c;old=s.ability[actor,s.active[actor]];other=s.ability[target,s.active[target]]
    denied=jnp.where(move==ROLE,(old==other)|ROLE_FAIL[other]|CANT[old],jnp.where(move==ENTRAIN,(old==other)|CANT[other]|(other==A['truant'])|NO_ENTRAIN[old],e.creation_ops.denied(s,target,move,CANT)))
    return e._when(denied,lambda c:fail(c,actor,move,move_row,e),c),denied

def end(c,side,e):
    s,_=c;mon=s.active[side]
    c=e._when(s.ability[side,mon]==A['illusion'],lambda c:e.illusion_ops.reveal(c,side,e),c)
    c=e.flash_ops.end(c,side,e)
    return e.trigger_form_ops.end(c,side,e)

def assign(s,side,ability,e):
    mon=s.active[side]
    return s._replace(ability_seek=s.ability_seek.at[side,mon].set(0),ability=s.ability.at[side,mon].set(ability),ability_effect_id=s.ability_effect_id.at[side,mon].set(ability),ability_order=s.ability_order.at[side,mon].set(s.effect_order),effect_order=s.effect_order+1)

def hit(c,actor,target,move,move_row,e):
    s,_=c;old=s.ability[actor,s.active[actor]];other=s.ability[target,s.active[target]]
    denied=jnp.where(move==SWAP,SWAP_FAIL[old]|SWAP_FAIL[other]|(old==other),jnp.where(e.creation_ops.CREATED[move]>0,CANT[other],CANT[old]|CANT[other]))
    def swap(c):
        s,_=c;c=e._event(c,e.ABILITY_SWAP,actor,s.active[actor],target=target,targetmon=s.active[target],value=other,aux=old)
        c=end(c,actor,e);c=end(c,target,e);s,tr=c
        s=assign(s,actor,other,e);s=assign(s,target,old,e)
        s=s._replace(ability_staleness=s.ability_staleness.at[actor,s.active[actor]].set(0).at[target,s.active[target]].set(1))
        c=e.hook_ops.ability_start((s,tr),target,e)
        return e.hook_ops.ability_start(c,actor,e)
    def single(c):
        receiver=jnp.where(move==ROLE,actor,target);source=1-receiver;ability=jnp.where(e.creation_ops.CREATED[move]>0,e.creation_ops.CREATED[move],jnp.where(move==ROLE,other,old))
        c=end(c,receiver,e);s,tr=c;previous=s.ability[receiver,s.active[receiver]]
        s=assign(s,receiver,ability,e)
        c=e._event((s,tr),e.ABILITY_CHANGE,receiver,s.active[receiver],move=move,target=source,targetmon=s.active[source],value=ability,aux=previous,status=(e.creation_ops.CREATED[move]>0).astype(jnp.int32))
        c=e.hook_ops.ability_start(c,receiver,e)
        c=e._when((move==e.creation_ops.WORRY)&(c[0].status[receiver,c[0].active[receiver]]==e.SLP),lambda c:e.effects.cure(c,receiver,e,source=actor),c)
        s,tr=c;s=s._replace(ability_staleness=s.ability_staleness.at[receiver,s.active[receiver]].set(jnp.where(move==ENTRAIN,1,s.ability_staleness[receiver,s.active[receiver]])))
        return s,tr
    def success(c):
        c=e.jax.lax.cond(move==SWAP,swap,single,c)
        return e.tracking.outcome(c,actor,e.i32(e.tracking.TRUE),move)
    return e.jax.lax.cond(denied,lambda c:fail(c,actor,move,move_row,e),success,c)
