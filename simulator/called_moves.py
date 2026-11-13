"""Explicit active-move frames sharing the ordinary useMove kernel."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
SLEEP_TALK=t.MOVE_IDS['sleeptalk'];NATURE_POWER=t.MOVE_IDS['naturepower'];COPYCAT=t.MOVE_IDS['copycat'];MIRROR_MOVE=t.MOVE_IDS['mirrormove'];FRAMES=8;WIDTH=20
# Move, actor/mon, target/mon, source kind/id, original location, selected
# slot, accounting mode, return phase, root move/slot, lock, event tag, flags.
MOVE,ACTOR,MON,TARGET,TM,SOURCE_KIND,SOURCE_ID,LOCATION,SLOT,MODE,PHASE,ROOT_MOVE,ROOT_SLOT,LOCKED,TAG,FLAGS=range(16)
PREVIOUS_MOVE=16
FIELDS=('call_frames','call_depth')
MOVE_IMPLEMENTED=jnp.asarray([k in ('sleeptalk','naturepower','copycat','mirrormove','magiccoat')for k in t.MOVE_KEYS],jnp.bool_)
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

def root(c,actor,mon,move,location,slot,locked,tag,previous,e,prankster=False):
    s,_=c;target=jnp.where(e.MOVE_ACTOR_TARGET[move],actor,1-actor)
    frame=jnp.stack([e.i32(v)for v in (move,actor,mon,target,s.active[target],0,0,location,slot,0,0,move,slot,locked,tag,e.i32(prankster)*2,*previous)])
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
            mirror=(frame[PHASE]==2)|(frame[PHASE]==3)
            c=e._when(~mirror,lambda c:e._update(e._update(c,e.HIT_UPDATE),e.HIT_UPDATE),c)
            s,tr=c;s=s._replace(battle_success=jnp.where(mirror,0,s.battle_move[0]))
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
    s=e.jax.lax.cond(frame[MODE]==1,pressure,lambda s:s,s)
    return s,tr

def sleep_talk(c,frame,move_row,e):
    s,_=c;actor=frame[ACTOR];mon=frame[MON];moves=s.moves[actor,mon]
    eligible=(moves>0)&ELIGIBLE[moves];count=jnp.sum(eligible).astype(jnp.int32)
    def call(c):
        c,index=e._draw(c,0,count,e.CALLED_SELECTION);s,tr=c
        slots=jnp.nonzero(eligible,size=4,fill_value=0)[0];slot=slots[index.astype(jnp.int32)];move=moves[slot]
        return child((s,tr),frame,move,slot,e)
    allowed=(s.status[actor,mon]==e.SLP)&(count>0)
    return e.jax.lax.cond(allowed,call,lambda c:e.ability_ops.fail(c,actor,SLEEP_TALK,move_row,e),c)


def child(c,frame,move,slot,e):
    s,tr=c;actor=frame[ACTOR];source=frame[MOVE]
    target=jnp.where(e.MOVE_ACTOR_TARGET[move],actor,1-actor)
    nested=frame.at[MOVE].set(move).at[TARGET].set(target).at[TM].set(s.active[target]).at[SOURCE_KIND].set(1).at[SOURCE_ID].set(source).at[SLOT].set(slot).at[MODE].set(1).at[PHASE].set(0).at[LOCKED].set(0).at[TAG].set(-(source+1))
    return push((s,tr),nested,e)


NO_COPY=t.mv(lambda r:bool(r.get('flags',{}).get('failcopycat'))).astype(jnp.bool_)
MIRRORABLE=t.mv(lambda r:bool(r.get('flags',{}).get('mirror'))).astype(jnp.bool_)


def copy_move(c,frame,move_row,e,mirror=False):
    s,_=c;actor=frame[ACTOR];target=frame[TARGET]
    move=s.move_history[target,s.active[target],0,0]if mirror else frame[PREVIOUS_MOVE]
    allowed=(move>0)&(MIRRORABLE[move]if mirror else ~NO_COPY[move])
    def use(c):
        if mirror:
            s,tr=c;s=s._replace(call_frames=s.call_frames.at[s.call_depth-1,PHASE].set(2));c=(s,tr)
        return child(c,frame,move,e.i32(-1),e)
    return e.jax.lax.cond(allowed,use,lambda c:e.ability_ops.fail(c,actor,frame[MOVE],move_row,e),c)


def call(c,frame,move_row,e):
    return e.jax.lax.cond(frame[MOVE]==NATURE_POWER,lambda c:child(c,frame,e.i32(t.MOVE_IDS['earthquake']),e.i32(-1),e),lambda c:e.jax.lax.cond(frame[MOVE]==COPYCAT,lambda c:copy_move(c,frame,move_row,e),lambda c:sleep_talk(c,frame,move_row,e),c),c)


MAGIC_COAT=t.MOVE_IDS['magiccoat'];COAT_SLOT=26;CONDITION_TAG=2000
MAGIC_BOUNCE=t.ABILITY_IDS['magicbounce']
ABILITY_IMPLEMENTED=jnp.asarray([k=='magicbounce'for k in t.ABILITIES],jnp.bool_)
REFLECTABLE=t.mv(lambda r:bool(r.get('flags',{}).get('reflectable'))).astype(jnp.bool_)
ABILITY_TAG=1000

def reflect(c,frame,e):
    s,tr=c;actor=frame[ACTOR];target=frame[TARGET];move=frame[MOVE];mon=s.active[target]
    side_move=(move==e.STEALTH_ROCK)|(move==e.SPIKES)
    semi=jnp.any(s.volatile_state[target,mon,11:15,0]>0)
    protected=s.protected[target,mon]&e.ability_ops.PROTECT[move]
    _,type_immune=e.generic.type_matchup(t.MOVE_TYPE[move],s.types[target,mon])
    denied=(~side_move)&(protected|((move==e.THUNDER_WAVE)&type_immune))
    coat=s.volatile_state[target,mon,COAT_SLOT,0]>0
    enabled=coat|(e.suppression_ops.abilities(s)[target,mon]==MAGIC_BOUNCE)
    allowed=(~denied)&(s.hp[target,mon]>0)&(actor!=target)&REFLECTABLE[move]&((frame[FLAGS]&1)==0)&enabled&(side_move|(~semi))
    def bounce(c):
        s,tr=c;s=s._replace(call_frames=s.call_frames.at[s.call_depth-1,PHASE].set(3))
        nested=frame.at[ACTOR].set(target).at[MON].set(mon).at[TARGET].set(actor).at[TM].set(frame[MON]).at[SOURCE_KIND].set(jnp.where(coat,3,2)).at[SOURCE_ID].set(jnp.where(coat,COAT_SLOT,MAGIC_BOUNCE)).at[LOCATION].set(1).at[SLOT].set(-1).at[MODE].set(2).at[PHASE].set(0).at[LOCKED].set(0).at[TAG].set(jnp.where(coat,CONDITION_TAG+COAT_SLOT,ABILITY_TAG+MAGIC_BOUNCE)).at[FLAGS].set(1+jnp.where(coat&(~side_move),2*s.volatile_state[target,mon,COAT_SLOT,10],0))
        return push((s,tr),nested,e)
    return e._when(allowed,bounce,c),allowed


def coat(c,frame,e):
    actor=frame[ACTOR];mon=frame[MON]
    c=e.effects.add_volatile(c,actor,COAT_SLOT,actor,MAGIC_COAT,e)
    s,tr=c;row=s.volatile_state[actor,mon,COAT_SLOT].at[5].set(1).at[10].set((frame[FLAGS]&2)//2)
    c=(s._replace(volatile_state=s.volatile_state.at[actor,mon,COAT_SLOT].set(row)),tr)
    c=e._event(c,e.SINGLE,actor,mon,move=MAGIC_COAT)
    c=e.tracking.outcome(c,actor,e.i32(e.tracking.TRUE),MAGIC_COAT)
    return e._update(e._update(c,e.HIT_UPDATE),e.HIT_UPDATE)
