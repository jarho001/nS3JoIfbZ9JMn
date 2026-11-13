"""Explicit active-move frames sharing the ordinary useMove kernel."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
SLEEP_TALK=t.MOVE_IDS['sleeptalk'];FRAMES=8;WIDTH=16
# Move, actor/mon, target/mon, source kind/id, original location, selected
# slot, accounting mode, return phase, root move/slot, lock, event tag, flags.
MOVE,ACTOR,MON,TARGET,TM,SOURCE_KIND,SOURCE_ID,LOCATION,SLOT,MODE,PHASE,ROOT_MOVE,ROOT_SLOT,LOCKED,TAG,FLAGS=range(WIDTH)
FIELDS=('call_frames','call_depth')
MOVE_IMPLEMENTED=jnp.asarray([k=='sleeptalk'for k in t.MOVE_KEYS],jnp.bool_)
ELIGIBLE=t.mv(lambda r:not(r.get('flags',{}).get('nosleeptalk')or r.get('flags',{}).get('charge'))).astype(jnp.bool_).at[0].set(False)

def empty(e):return jnp.zeros((FRAMES,WIDTH),jnp.int32)

def push(c,frame,e):
    def add(c):
        s,tr=c;return s._replace(call_frames=s.call_frames.at[s.call_depth].set(frame),call_depth=s.call_depth+1),tr
    def overflow(c):
        s,tr=c;return s,tr._replace(overflow=jnp.array(True))
    return e.jax.lax.cond(c[0].call_depth<FRAMES,add,overflow,c)

def pop(c):
    s,tr=c;n=s.call_depth-1
    return s._replace(call_frames=s.call_frames.at[n].set(0),call_depth=n),tr

def root(c,actor,mon,move,location,slot,locked,tag,e):
    s,_=c;target=jnp.where(e.MOVE_ACTOR_TARGET[move],actor,1-actor)
    frame=jnp.stack([e.i32(v)for v in (move,actor,mon,target,s.active[target],0,0,location,slot,0,0,move,slot,locked,tag,0)])
    c=push(c,frame,e)
    def condition(c):return(c[0].call_depth>0)&(~c[1].overflow)
    def step(c):
        s,tr=c;depth=s.call_depth;frame=s.call_frames[depth-1]
        def enter(c):
            s,tr=c;s=s._replace(call_frames=s.call_frames.at[depth-1,PHASE].set(1));c=e._use_move((s,tr),frame)
            c=e.phaze_ops.resolve(c,e)
            # An ordinary frame has completed; a successful caller pushed its
            # child and leaves its own return phase underneath it.
            return e.jax.lax.cond(c[0].call_depth==depth,pop,lambda c:c,c)
        def resume(c):
            c=e._update(e._update(c,e.HIT_UPDATE),e.HIT_UPDATE)
            s,tr=c;s=s._replace(battle_success=s.battle_move[0])
            return pop((s,tr))
        return e.jax.lax.cond(frame[PHASE]==0,enter,resume,c)
    return e.jax.lax.while_loop(condition,step,c)

def enter(c,frame,e):
    s,tr=c;actor=frame[ACTOR];mon=frame[MON];move=frame[MOVE]
    # useMove updates lastMoveUsed/active move, but not moveUsed or root action
    # counters, target commitment, original slot use or original PP.
    move_type=jnp.where(move==e.HIDDEN_POWER,s.hp_type[actor,mon],t.MOVE_TYPE[move]);move_type=jnp.where(move==e.STRUGGLE,0,move_type)
    snapshot=jnp.stack((move,move_type,t.CATEGORY[move],t.POWER[move]))
    s=s._replace(move_history=s.move_history.at[actor,mon,2].set(snapshot),battle_move=snapshot)
    s=e.tracking.put(s,actor,e.tracking.RESULT,e.tracking.FALSE)
    def pressure(s):
        enemy=1-actor;em=s.active[enemy]
        extra=e.hook_tables.PRESSURE_TARGET[move].astype(jnp.bool_)&(e.suppression_ops.abilities(s)[enemy,em]==e.hook_tables.A['pressure'])&(~s.fainted[enemy,em])
        slot=frame[ROOT_SLOT];s=s._replace(pp=s.pp.at[actor,mon,slot].set(jnp.maximum(0,s.pp[actor,mon,slot]-extra.astype(jnp.int32))))
        return e.identity_ops.sync(s,actor,e)
    s=e.jax.lax.cond(frame[MODE]>0,pressure,lambda s:s,s)
    return s,tr

def sleep_talk(c,frame,move_row,e):
    s,_=c;actor=frame[ACTOR];mon=frame[MON];moves=s.moves[actor,mon]
    eligible=(moves>0)&ELIGIBLE[moves];count=jnp.sum(eligible).astype(jnp.int32)
    def call(c):
        c,index=e._draw(c,0,count,e.CALLED_SELECTION);s,tr=c
        slots=jnp.nonzero(eligible,size=4,fill_value=0)[0];slot=slots[index.astype(jnp.int32)];move=moves[slot]
        target=jnp.where(e.MOVE_ACTOR_TARGET[move],actor,1-actor)
        child=frame.at[MOVE].set(move).at[TARGET].set(target).at[TM].set(s.active[target]).at[SOURCE_KIND].set(1).at[SOURCE_ID].set(SLEEP_TALK).at[SLOT].set(slot).at[MODE].set(1).at[PHASE].set(0).at[LOCKED].set(0).at[TAG].set(-(SLEEP_TALK+1))
        return push((s,tr),child,e)
    allowed=(s.status[actor,mon]==e.SLP)&(count>0)
    return e.jax.lax.cond(allowed,call,lambda c:e.ability_ops.fail(c,actor,SLEEP_TALK,move_row,e),c)
