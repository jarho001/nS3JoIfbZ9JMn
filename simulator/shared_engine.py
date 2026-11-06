"""Bounded shared Gen 5 singles engine, independent of the native oracle.

Fixed-shape int32 storage; exact four-limb PRNG; stable mon identity and a
separate command roster. Strings/protocol rendering belong to the host adapter.
The allowlist is a restricted custom game, not full gen5randombattle coverage.
"""
from __future__ import annotations

from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

from .gen5_rng import next_state, sample_bounded
from mechanics.base_damage import base_damage_core
from mechanics.damage_composition import damage_from_preroll
from mechanics.stats import derive_stored_stats

SCHEMA = 1
ROSTER, SLOTS, QUEUE, EVENTS, DRAWS, HISTORY = 6, 4, 8, 128, 128, 2048
OK, PENDING, INVALID, UNSUPPORTED, CAPACITY, TERMINAL, STALE = range(7)
MOVE, FORCED, WAIT, END = range(4)
NONE, BRN, PAR, TOX, FNT = range(5)
# Species and moves use local table indices, explicitly versioned here.
MEW, SNORLAX, GENGAR = 1, 2, 3
TACKLE, POUND, MEGA_PUNCH, EARTHQUAKE, U_TURN, WISP, TOXIC, THUNDER_WAVE, RECOVER, PROTECT, STRUGGLE = range(1,12)
SPECIES = ('', 'Mew', 'Snorlax', 'Gengar')
MOVE_NAMES = ('', 'Tackle', 'Pound', 'Mega Punch', 'Earthquake', 'U-turn', 'Will-O-Wisp', 'Toxic', 'Thunder Wave', 'Recover', 'Protect', 'Struggle')
MOVE_KEYS = ('', 'tackle', 'pound', 'megapunch', 'earthquake', 'uturn', 'willowisp', 'toxic', 'thunderwave', 'recover', 'protect', 'struggle')
PP = jnp.array([0,56,56,32,16,32,24,16,32,16,16,1],jnp.int32)
POWER = jnp.array([0,50,40,80,100,70,0,0,0,0,0,50],jnp.int32)
ACCURACY = jnp.array([0,100,100,85,100,100,75,90,100,0,0,0],jnp.int32)
BASE_STATS = jnp.array([[1]*6,[100]*6,[160,110,65,65,110,30],[60,65,60,130,75,110]],jnp.int32)
ITEMS = ('','leftovers','lumberry')
# Event fields: kind, side, stable mon, move, target side, target mon, value,
# auxiliary, status, reserved. HP events retain maxHP for exact projection.
BLANK, EV_MOVE, DAMAGE, HEAL, CRIT, IMMUNE, MISS, FAINT, WIN, TURN, UPKEEP, SWITCH, STATUS, ABILITY, END_ITEM, CURE, FAIL, CANT, ACTIVATE, SINGLE = range(20)
# Draw contexts; range form stores actual native arguments including offsets.
INIT, QUEUE_SORT, BEFORE_TURN, UPDATE, HIT_UPDATE, SWITCH_SORT, FIELD_SORT, ACC, CRITICAL, ROLL, PARALYSIS, STALL = range(12)
DRAW_CONTEXTS=('initialization','queue-sort','before-turn','update','hit-update','switch-order','residual-handlers','accuracy','critical','damage','paralysis','stall')
# Queue kinds, order values mirror the pinned scheduler.
BEFORE, ATTACK, SWAP, INSTANT, RUN_SWITCH, RESIDUAL = range(1,7)
QORDER=jnp.array([999,4,200,103,3,101,300],jnp.int32)


class State(NamedTuple):
    species: jax.Array
    level: jax.Array
    stats: jax.Array
    speed: jax.Array
    gender: jax.Array
    hp: jax.Array
    fainted: jax.Array
    moves: jax.Array
    pp: jax.Array
    item: jax.Array
    status: jax.Array
    toxic: jax.Array
    stall: jax.Array
    stall_duration: jax.Array
    protected: jax.Array
    active: jax.Array
    order: jax.Array
    pending: jax.Array
    request: jax.Array
    pivot: jax.Array
    queue: jax.Array
    qcount: jax.Array
    midturn: jax.Array
    rng: jax.Array
    turn: jax.Array
    winner: jax.Array
    generation: jax.Array
    episode: jax.Array
    history: jax.Array
    hcount: jax.Array


class Trace(NamedTuple):
    events: jax.Array
    ecount: jax.Array
    args: jax.Array
    values: jax.Array
    context: jax.Array
    before: jax.Array
    after: jax.Array
    dcount: jax.Array
    overflow: jax.Array


class Result(NamedTuple):
    state: State
    code: jax.Array
    trace: Trace


def i32(value):
    return jnp.asarray(value,jnp.int32)


def empty_trace():
    return Trace(jnp.zeros((EVENTS,10),jnp.int32),i32(0),jnp.zeros((DRAWS,3),jnp.uint32),
                 jnp.zeros(DRAWS,jnp.uint32),jnp.zeros(DRAWS,jnp.int32),
                 jnp.zeros((DRAWS,4),jnp.uint16),jnp.zeros((DRAWS,4),jnp.uint16),i32(0),jnp.array(False))


def _event(c, kind, side=0, mon=0, move=0, target=0, targetmon=0, value=0, aux=0, status=0):
    s,t=c
    row=jnp.stack([i32(x) for x in (kind,side,mon,move,target,targetmon,value,aux,status,0)])
    return s,t._replace(events=t.events.at[t.ecount].set(row),ecount=t.ecount+1,
                        overflow=t.overflow | (t.ecount>=EVENTS))


def _draw(c, lo, hi, context, form=1, full=False):
    """form 1=random(bound), 2=random(from,to); full tags the 2**32 bound.

    uint32 word is the exact random(2**32) result; its bound is represented
    by an explicit full-width flag rather than overflowing an int32.
    """
    s,t=c
    before=s.rng
    if full:
        rng,value=next_state(before)
    else:
        rng,v=sample_bounded(before,i32(hi)-i32(lo))
        value=v+jnp.asarray(lo,jnp.uint32)
    args=jnp.stack((jnp.asarray(lo,jnp.uint32),jnp.asarray(hi,jnp.uint32),jnp.uint32(3 if full else form)))
    t=t._replace(args=t.args.at[t.dcount].set(args),values=t.values.at[t.dcount].set(value),
                 context=t.context.at[t.dcount].set(context),before=t.before.at[t.dcount].set(before),
                 after=t.after.at[t.dcount].set(rng),dcount=t.dcount+1,overflow=t.overflow|(t.dcount>=DRAWS))
    return (s._replace(rng=rng),t),value


def _when(pred,fn,c):
    return jax.lax.cond(pred,fn,lambda x:x,c)


def _live(s):
    return (s.species>0)&(s.hp>0)


def _speeds(s):
    return jnp.maximum(1,jnp.where(s.status==PAR,s.stats[:,:,5]//4,s.stats[:,:,5]))


def _refresh_speed(s):
    effective=_speeds(s)
    for side in range(2):
        mon=s.active[side]
        s=s._replace(speed=s.speed.at[side,mon].set(jnp.where(s.fainted[side,mon],s.speed[side,mon],effective[side,mon])))
    return s


def _update(c,context=UPDATE,sort_all=False):
    s,t=c
    active_speed=s.speed[jnp.arange(2),s.active]
    alive=~s.fainted[jnp.arange(2),s.active]
    tie=(active_speed[0]==active_speed[1]) & (jnp.all(alive)|sort_all)
    c=_when(tie,lambda x:_draw(x,0,2,context,2)[0],c)
    # Lum's onUpdate handler fires in native speed order, no random cure draw.
    for side in range(2):
        def cure(c):
            s,t=c;m=s.active[side];old=s.status[side,m]
            c=_event(c,END_ITEM,side,m)
            c=_event(c,CURE,side,m,status=old)
            s,t=c
            return s._replace(item=s.item.at[side,m].set(0),status=s.status.at[side,m].set(0),toxic=s.toxic.at[side,m].set(0)),t
        s,t=c;m=s.active[side]
        c=_when((s.item[side,m]==2)&(s.status[side,m]>0)&(s.status[side,m]<FNT)&(s.hp[side,m]>0),cure,c)
    return c


def _qappend(s,row):
    return s._replace(queue=s.queue.at[s.qcount].set(row),qcount=s.qcount+1)


def _qsort(c):
    """Pinned speedSort promotion plus Fisher-Yates; preserve native offsets."""
    # Queue has at most two equally-ranked player actions. Other actions have
    # distinct orders. Stable sorting first matches native promotion for these
    # two-player queues (no three-way queue group is reachable in this slice).
    s,t=c
    q=s.queue
    valid=jnp.arange(QUEUE)<s.qcount
    order=jnp.where(valid,QORDER[jnp.clip(q[:,0],0,6)],10000)
    idx=jnp.lexsort((jnp.arange(QUEUE),-q[:,5],-q[:,4],order))
    s=s._replace(queue=q[idx]);c=(s,t)
    def body(k,c):
        s,t=c;q=s.queue
        tie=(k+1<s.qcount)&(QORDER[q[k,0]]==QORDER[q[k+1,0]])&(q[k,4]==q[k+1,4])&(q[k,5]==q[k+1,5])
        def shuffle(c):
            c,v=_draw(c,k,k+2,QUEUE_SORT,2);s,t=c;q=s.queue
            a=q[k];b=q[v.astype(jnp.int32)]
            q=q.at[k].set(b).at[v.astype(jnp.int32)].set(a)
            return s._replace(queue=q),t
        return _when(tie,shuffle,c)
    return jax.lax.fori_loop(0,QUEUE-1,body,c)


def _faint(c,side):
    s,t=c;m=s.active[side]
    c=_event(c,FAINT,side,m);s,t=c
    return s._replace(fainted=s.fainted.at[side,m].set(True),speed=s.speed.at[side,m].set(s.stats[side,m,5]),status=s.status,
                      stall=s.stall.at[side,m].set(0),stall_duration=s.stall_duration.at[side,m].set(0),
                      protected=s.protected.at[side,m].set(False)),t


def _win(c,last_side):
    s,t=c;alive=jnp.any(_live(s),axis=1)
    winner=jnp.where(jnp.all(~alive),last_side+1,jnp.where(~alive[0],2,1))
    def end(c):
        c=_event(c,BLANK);c=_event(c,WIN,value=winner);s,t=c
        return s._replace(winner=winner,request=jnp.full(2,END,jnp.int32)),t
    return _when(jnp.any(~alive),end,c)


def _damage(c,side,amount,cause=0):
    s,t=c;m=s.active[side];hp=jnp.maximum(0,s.hp[side,m]-jnp.maximum(1,amount))
    s=s._replace(hp=s.hp.at[side,m].set(hp))
    return _event((s,t),DAMAGE,side,m,value=hp,aux=s.stats[side,m,0],status=s.status[side,m],move=cause)


def _status(c,source,target,status,reflect=True):
    s,t=c;m=s.active[target];sp=s.species[target,m]
    immunity=(status==TOX)&((sp==SNORLAX)|(sp==GENGAR))
    failed=(s.status[target,m]!=NONE)
    def blocked(c):
        return _event(c,IMMUNE,target,m,aux=jnp.where(sp==SNORLAX,1,0))
    def attempt(c):
        def fail(c):
            same=s.status[target,m]==status
            def different(c):
                s,t=c
                last=jnp.max(jnp.where((jnp.arange(EVENTS)<t.ecount)&(t.events[:,0]==EV_MOVE),jnp.arange(EVENTS),0))
                t=t._replace(events=t.events.at[last,7].set(2))
                return _event((s,t),FAIL,source,s.active[source])
            return jax.lax.cond(same,lambda c:_event(c,FAIL,target,m,status=status),different,c)
        def apply(c):
            s,t=c;s=s._replace(status=s.status.at[target,m].set(status),toxic=s.toxic.at[target,m].set(0))
            c=_event((s,t),STATUS,target,m,status=status)
            if reflect:
                def sync(c):
                    c=_event(c,ABILITY,target,m)
                    return _status(c,target,source,status,False)
                c=_when((sp==MEW)&(source!=target),sync,c)
            # Native Synchronize recursively attempts reflection once more,
            # generating activation/failure before the item cures.
            elif True:
                def back(c):
                    s,t=c;other=s.active[source]
                    c=_event(c,ABILITY,target,m)
                    return _event(c,FAIL,source,other,status=status)
                c=_when((sp==MEW)&(source!=target),back,c)
            # Items update immediately after successful application.
            s,t=c
            def lum(c):
                c=_event(c,END_ITEM,target,m);c=_event(c,CURE,target,m,status=status);s,t=c
                return s._replace(item=s.item.at[target,m].set(0),status=s.status.at[target,m].set(0),toxic=s.toxic.at[target,m].set(0)),t
            return _when(s.item[target,m]==2,lum,c)
        return jax.lax.cond(failed,fail,apply,c)
    return jax.lax.cond(immunity&(~failed),blocked,attempt,c)


def _hit(c,actor,move):
    s,t=c;mon=s.active[actor];target=1-actor;tm=s.active[target]
    damaging=POWER[move]>0
    def physical(c):
        c,crit=_draw(c,0,16,CRITICAL);c,roll=_draw(c,0,16,ROLL)
        s,t=c
        core=base_damage_core(s.level[actor,mon],POWER[move],s.stats[actor,mon,1],s.stats[target,tm,2])+2
        preroll=core*jnp.where(crit==0,2,1)
        sp=s.species[actor,mon];ts=s.species[target,tm]
        normal=(move==TACKLE)|(move==POUND)|(move==MEGA_PUNCH)
        exponent=jnp.where((move==U_TURN)&(ts==MEW),1,jnp.where((move==U_TURN)&(ts==GENGAR),-2,0)).astype(jnp.int32)
        damage=damage_from_preroll(preroll,roll.astype(jnp.int32),(sp==SNORLAX)&normal,jnp.array(False),move==STRUGGLE,
                                  exponent,s.status[actor,mon]==BRN,jnp.array(True),jnp.array(False),i32(4096))
        # Native effectiveness precedes the critical disclosure.
        c=_when(exponent!=0,lambda c:_event(c,ACTIVATE,target,tm,aux=exponent,move=-1),c)
        c=_when(crit==0,lambda c:_event(c,CRIT,target,tm),c)
        c=_damage(c,target,damage)
        c=_update(c,HIT_UPDATE)
        s,t=c;ko=s.hp[target,tm]==0
        c=_when(ko,lambda c:_faint(c,target),c)
        if_recoil=move==STRUGGLE
        def recoil(c):
            s,t=c
            c=_damage(c,actor,(s.stats[actor,mon,0]+2)//4,cause=3)
            return c
        c=_when(if_recoil,recoil,c)
        # Recoil's faint is queued until after the second hit-loop Update;
        # it must still participate in that speed sort before resolution.
        s,t=c
        c=_update(c,HIT_UPDATE)
        s,t=c
        c=_when(if_recoil&(s.hp[actor,mon]==0),lambda c:_faint(c,actor),c)
        c=_win(c,jnp.where(if_recoil,actor,target))
        s,t=c
        def pivot(c):
            s,t=c
            return s._replace(pivot=s.pivot.at[actor].set(True)),t
        return _when((move==U_TURN)&(s.winner<0)&jnp.any(_live(s)[actor]&(jnp.arange(ROSTER)!=mon)),pivot,c)
    def status_hit(c):
        s,t=c
        success=(s.status[target,tm]==NONE)&(~((move==TOXIC)&((s.species[target,tm]==SNORLAX)|(s.species[target,tm]==GENGAR))))
        c=_status(c,actor,target,jnp.where(move==WISP,BRN,jnp.where(move==TOXIC,TOX,PAR)))
        return _when(success,lambda c:_update(_update(c,HIT_UPDATE),HIT_UPDATE),c)
    return jax.lax.cond(damaging,physical,status_hit,c)


def _attack(c,row):
    s,t=c;actor=row[1];mon=row[2];action=row[3]
    move=jnp.where(action==10,STRUGGLE,s.moves[actor,mon,jnp.clip(action,0,3)])
    target=jnp.where((move==RECOVER)|(move==PROTECT),actor,1-actor)
    tm=s.active[target]
    def execute(c):
        # Paralysis occurs before PP spend and move event.
        def par(c):
            c,value=_draw(c,0,4,PARALYSIS)
            return c,value==0
        c,full=jax.lax.cond(s.status[actor,mon]==PAR,par,lambda c:(c,jnp.array(False)),c)
        def cannot(c):return _event(c,CANT,actor,mon,status=PAR)
        def act(c):
            s,t=c;slot=jnp.clip(action,0,3)
            s=s._replace(pp=s.pp.at[actor,mon,slot].add(jnp.where(move==STRUGGLE,0,-1)))
            c=(s,t)
            c=_when(move==STRUGGLE,lambda c:_event(c,ACTIVATE,actor,mon,move=STRUGGLE),c)
            # Reserve move row so the native miss/fail annotation can be filled
            # after the outcome without reordering the event stream.
            move_row=t.ecount+jnp.where(move==STRUGGLE,1,0)
            c=_event(c,EV_MOVE,actor,mon,move,target,tm)
            def self_move(c):
                def recover(c):
                    s,t=c;hp=s.hp[actor,mon];maxhp=s.stats[actor,mon,0]
                    def fail(c):
                        s,t=c;t=t._replace(events=t.events.at[move_row,7].set(2))
                        return _event((s,t),FAIL,actor,mon,aux=1)
                    def heal(c):
                        s,t=c;newhp=jnp.minimum(maxhp,hp+(maxhp+1)//2)
                        s=s._replace(hp=s.hp.at[actor,mon].set(newhp))
                        return _event((s,t),HEAL,actor,mon,value=newhp,aux=maxhp,status=s.status[actor,mon])
                    return jax.lax.cond(hp==maxhp,fail,heal,c)
                def protect(c):
                    s,t=c;counter=s.stall[actor,mon]
                    # Native Protect requires a remaining non-residual action.
                    will_act=jnp.any((jnp.arange(QUEUE)<s.qcount)&((s.queue[:,0]==ATTACK)|(s.queue[:,0]==SWAP)|(s.queue[:,0]==INSTANT)))
                    def check(c):
                        return jax.lax.cond(counter>=256,lambda c:_draw(c,0,0,STALL,full=True),
                                            lambda c:_draw(c,0,jnp.maximum(counter,1),STALL),c)
                    c,value=jax.lax.cond((counter>0)&will_act,check,lambda c:(c,jnp.uint32(0)),c)
                    success=will_act&(value==0)
                    def success_fn(c):
                        s,t=c;s=s._replace(protected=s.protected.at[actor,mon].set(True),
                         stall=s.stall.at[actor,mon].set(jnp.minimum(256,jnp.maximum(2,counter*2))),
                         stall_duration=s.stall_duration.at[actor,mon].set(2))
                        return _event((s,t),SINGLE,actor,mon)
                    def fail_fn(c):
                        s,t=c;t=t._replace(events=t.events.at[move_row,7].set(2))
                        return _event((s,t),FAIL,actor,mon)
                    return jax.lax.cond(success,success_fn,fail_fn,c)
                c=jax.lax.cond(move==RECOVER,recover,protect,c)
                # Successful self moves run both hit-loop Updates.
                s,t=c;failed=t.events[move_row,7]==2
                c=_when(~failed,lambda c:_update(_update(c,HIT_UPDATE),HIT_UPDATE),c)
                return c
            def foe_move(c):
                s,t=c;ts=s.species[target,tm]
                normal=(move==TACKLE)|(move==POUND)|(move==MEGA_PUNCH)
                immune=(normal&(ts==GENGAR))|((move==EARTHQUAKE)&(ts==GENGAR))
                blocked=s.protected[target,tm]&(move!=STRUGGLE)
                def block(c):return _event(c,ACTIVATE,target,tm,move=PROTECT)
                def try_hit(c):
                    def immunity(c):return _event(c,IMMUNE,target,tm,aux=jnp.where(move==EARTHQUAKE,2,0))
                    def accuracy(c):
                        c,roll=jax.lax.cond(ACCURACY[move]>0,lambda c:_draw(c,0,100,ACC),lambda c:(c,jnp.uint32(0)),c)
                        hit=(ACCURACY[move]==0)|(roll<ACCURACY[move].astype(jnp.uint32))
                        def miss(c):
                            s,t=c;t=t._replace(events=t.events.at[move_row,7].set(1))
                            return _event((s,t),MISS,actor,mon,move,target,tm)
                        def hit_fn(c):
                            s,t=c
                            existing=(POWER[move]==0)&(s.status[target,tm]!=NONE)&(s.status[target,tm]!=jnp.where(move==WISP,BRN,jnp.where(move==TOXIC,TOX,PAR)))
                            t=t._replace(events=t.events.at[move_row,7].set(jnp.where(existing,2,t.events[move_row,7])))
                            return _hit((s,t),actor,move)
                        return jax.lax.cond(hit,hit_fn,miss,c)
                    return jax.lax.cond(immune,immunity,accuracy,c)
                return jax.lax.cond(blocked,block,try_hit,c)
            def no_target(c):
                s,t=c;t=t._replace(events=t.events.at[move_row,7].set(3))
                return _event((s,t),FAIL,actor,mon)
            return jax.lax.cond((target!=actor)&(s.hp[target,tm]==0),no_target,
                                lambda c:jax.lax.cond(target==actor,self_move,foe_move,c),c)
        return jax.lax.cond(full,cannot,act,c)
    # A queued actor remains tied to its original mon, and cannot act after KO.
    return _when((s.active[actor]==mon)&(s.hp[actor,mon]>0),execute,c)


def _switch(c,row):
    s,t=c;side=row[1];old=s.active[side];slot=row[3]-4;incoming=s.order[side,slot]
    c=_when((row[0]==SWAP)&(s.hp[side,old]>0),lambda c:_update(c,UPDATE),c)
    s,t=c
    oldslot=jnp.argmax(s.order[side]==old)
    order=s.order.at[side,oldslot].set(incoming).at[side,slot].set(old)
    s=s._replace(active=s.active.at[side].set(incoming),order=order,speed=s.speed.at[side,old].set(s.stats[side,old,5]),
                 protected=s.protected.at[side,old].set(False),stall=s.stall.at[side,old].set(0),
                 stall_duration=s.stall_duration.at[side,old].set(0),
                 status=s.status.at[side,old].set(jnp.where(s.status[side,old]==FNT,NONE,s.status[side,old])),
                 toxic=s.toxic.at[side,incoming].set(0),pivot=s.pivot.at[side].set(False))
    c=_event((s,t),SWITCH,side,incoming,value=s.hp[side,incoming],aux=s.stats[side,incoming,0],
             status=s.status[side,incoming],move=row[6])
    s,t=c
    # runSwitch precedes ordinary moves but follows remaining simultaneous swaps.
    s=_qappend(s,jnp.array([RUN_SWITCH,0,0,0,0,0,0],jnp.int32).at[1].set(side).at[2].set(incoming).at[5].set(s.speed[side,incoming]))
    return _qsort((s,t))


def _residual(c):
    c=_event(c,BLANK);s,t=c;s=_refresh_speed(s);c=(s,t)
    # Native field handlers: status, protect, stall, item in insertion order.
    # Duration-only handlers sort last (false order), and still consume shuffle
    # draws even when their callback does nothing.
    rows=[];valid=[]
    for side in range(2):
        m=s.active[side];live=~s.fainted[side,m];status=s.status[side,m]
        rows.extend([jnp.array([1,side,0,0,0,0],jnp.int32).at[2].set(m).at[3].set(jnp.where(status==TOX,9,10)).at[4].set(s.speed[side,m]),
                     jnp.array([2,side,0,10000,0,2],jnp.int32).at[2].set(m).at[4].set(s.speed[side,m]),
                     jnp.array([3,side,0,10000,0,2],jnp.int32).at[2].set(m).at[4].set(s.speed[side,m]),
                     jnp.array([4,side,0,5,0,4],jnp.int32).at[2].set(m).at[4].set(s.speed[side,m])])
        valid.extend([live&((status==TOX)|(status==BRN)),live&s.protected[side,m],live&(s.stall_duration[side,m]>0),live&(s.item[side,m]==1)])
    rows=jnp.stack(rows);valid=jnp.stack(valid);count=jnp.sum(valid).astype(jnp.int32)
    order=jnp.where(valid,rows[:,3],20000)
    idx=jnp.lexsort((jnp.arange(8),rows[:,5],-rows[:,4],order));rows=rows[idx]
    def shuffle_body(k,carry):
        c,rows=carry
        # Resolve each equal group once, using native shuffle(from,to) calls.
        key=rows[k,3:6]
        eq=jnp.all(rows[:,3:6]==key,axis=1)&(jnp.arange(8)<count)
        first=jnp.min(jnp.where(eq,jnp.arange(8),8));end=jnp.max(jnp.where(eq,jnp.arange(8),-1))+1
        def shuffle(carry):
            c,rows=carry;c,value=_draw(c,k,end,FIELD_SORT,2);v=value.astype(jnp.int32)
            a,b=rows[k],rows[v];rows=rows.at[k].set(b).at[v].set(a)
            return c,rows
        return jax.lax.cond((k>=first)&(k+1<end)&(k<count),shuffle,lambda x:x,(c,rows))
    c,rows=jax.lax.fori_loop(0,7,shuffle_body,(c,rows))
    def effect(k,c):
        s,t=c;row=rows[k];kind,side,m=row[:3]
        def apply(c):
            def status_tick(c):
                s,t=c;tox=s.status[side,m]==TOX
                stage=jnp.minimum(15,s.toxic[side,m]+1)
                amount=jnp.where(tox,jnp.maximum(1,s.stats[side,m,0]//16)*stage,s.stats[side,m,0]//8)
                s=s._replace(toxic=s.toxic.at[side,m].set(jnp.where(tox,stage,s.toxic[side,m])))
                c=_damage((s,t),side,amount,cause=jnp.where(tox,2,1));s,t=c
                c=_when(s.hp[side,m]==0,lambda c:_faint(c,side),c)
                return _win(c,side)
            def duration(c):
                s,t=c
                def protect(c):
                    s,t=c;return s._replace(protected=s.protected.at[side,m].set(False)),t
                def stall(c):
                    s,t=c;duration=jnp.maximum(0,s.stall_duration[side,m]-1)
                    return s._replace(stall_duration=s.stall_duration.at[side,m].set(duration),
                                     stall=s.stall.at[side,m].set(jnp.where(duration==0,0,s.stall[side,m]))),t
                return jax.lax.cond(kind==2,protect,stall,c)
            def leftovers(c):
                s,t=c;hp=s.hp[side,m];maxhp=s.stats[side,m,0]
                def heal(c):
                    s,t=c;newhp=jnp.minimum(maxhp,hp+jnp.maximum(1,maxhp//16));s=s._replace(hp=s.hp.at[side,m].set(newhp))
                    return _event((s,t),HEAL,side,m,value=newhp,aux=maxhp,status=s.status[side,m],move=1)
                return _when(hp<maxhp,heal,c)
            return jax.lax.cond(kind==1,status_tick,lambda c:jax.lax.cond(kind==4,leftovers,duration,c),c)
        return _when((k<count)&(~s.fainted[side,m])&(s.winner<0),apply,c)
    c=jax.lax.fori_loop(0,8,effect,c)
    s,t=c
    return _when(s.winner<0,lambda c:_event(c,UPKEEP),c)


def _new_request(c,request):
    s,t=c
    for side in range(2):
        mon=s.active[side]
        status=jnp.where((request[side]==FORCED)&s.fainted[side,mon],FNT,s.status[side,mon])
        s=s._replace(status=s.status.at[side,mon].set(status))
    return s._replace(request=request,pending=jnp.full(2,-1,jnp.int32),episode=s.episode+1),t


def _finish(c):
    s,t=c
    s=s._replace(midturn=jnp.array(False),turn=s.turn+1,queue=jnp.zeros_like(s.queue),qcount=i32(0))
    c=_event((s,t),TURN,value=s.turn)
    return _new_request(c,jnp.full(2,MOVE,jnp.int32))


def _run_queue(c):
    def condition(c):
        s,t=c
        return (s.qcount>0)&(s.winner<0)&(~jnp.any(s.pivot))&(~t.overflow)
    def step(c):
        s,t=c;row=s.queue[0];kind=row[0]
        s=s._replace(queue=jnp.concatenate((s.queue[1:],jnp.zeros((1,7),jnp.int32))),qcount=s.qcount-1)
        c=(s,t)
        def before(c):return _update(c,BEFORE_TURN)
        def move(c):return _attack(c,row)
        def swap(c):return _switch(c,row)
        def run_switch(c):
            # Coalesce queued simultaneous runSwitch callbacks into one native
            # speedSort over all active mons, including fainted slots.
            s,t=c;q=s.queue
            run=(jnp.arange(QUEUE)<s.qcount)&(q[:,0]==RUN_SWITCH)
            # runSwitch actions are contiguous at the front after all swaps.
            count=jnp.sum(run).astype(jnp.int32)
            q=jnp.where((jnp.arange(QUEUE)+count)[:,None]<s.qcount,q[jnp.minimum(jnp.arange(QUEUE)+count,QUEUE-1)],0)
            s=s._replace(queue=q,qcount=s.qcount-count)
            return _update((s,t),SWITCH_SORT,True)
        c=jax.lax.switch(jnp.clip(kind-1,0,5),(before,move,swap,swap,run_switch,_residual),c)
        s,t=c
        # A skipped fainted action returns before the native per-action Update.
        skipped=(kind==ATTACK)&((s.active[row[1]]!=row[2])|s.fainted[row[1],row[2]])
        next_instant=(s.qcount>0)&(s.queue[0,0]==INSTANT)
        c=_when((s.winner<0)&(~skipped)&(~next_instant),lambda c:_update(c,UPDATE),c)
        s,t=c
        c=_when(jnp.any(s.pivot)&(s.winner<0),lambda c:_update(c,UPDATE),c)
        return c
    c=jax.lax.while_loop(condition,step,c)
    s,t=c
    def ongoing(c):
        s,t=c
        needs=jnp.where(jnp.any(s.pivot),s.pivot,s.fainted[jnp.arange(2),s.active])
        def interruption(c):
            return _new_request(c,jnp.where(needs,FORCED,WAIT).astype(jnp.int32))
        return jax.lax.cond(jnp.any(needs),interruption,_finish,c)
    return _when((s.winner<0)&(~t.overflow),ongoing,c)


def legal_actions(s):
    side=jnp.arange(2);mon=s.active
    pp=s.pp[side,mon];moves=s.moves[side,mon]
    ordinary=(pp>0)&(moves>0)&(s.request[:,None]==MOVE)
    bench=_live(s)[side[:,None],s.order]&(s.order!=mon[:,None])
    switches=bench&((s.request[:,None]==MOVE)|(s.request[:,None]==FORCED))
    struggle=jnp.all(~((pp>0)&(moves>0)),axis=1)&(s.request==MOVE)
    return jnp.concatenate((ordinary,switches,struggle[:,None]),axis=1)


def _supported(s):
    occupied=s.species>0
    base=derive_stored_stats(BASE_STATS[jnp.clip(s.species,0,3)],s.level,jnp.full((2,6,6),31,jnp.int32),jnp.zeros((2,6,6),jnp.int32),jnp.full((2,6,6),10,jnp.int32))
    speed_ev=((2*BASE_STATS[jnp.clip(s.species,0,3),5]+31+63)*s.level)//100+5
    stat_ok=jnp.all(jnp.where(occupied[:,:,None],s.stats[:,:,:5]==base[:,:,:5],s.stats[:,:,:5]==0))
    stat_ok=stat_ok&jnp.all(jnp.where(occupied,(s.stats[:,:,5]==base[:,:,5])|(s.stats[:,:,5]==speed_ev),s.stats[:,:,5]==0))
    slots=jnp.arange(4)
    duplicate=(s.moves[:,:,:,None]==s.moves[:,:,None,:])&(s.moves[:,:,:,None]>0)&(slots[:,None]<slots[None,:])
    valid_order=jnp.all(jnp.sort(s.order,axis=1)==jnp.arange(ROSTER))
    return (stat_ok&valid_order&(~jnp.any(duplicate))&jnp.all((s.hcount>=0)&(s.hcount<=HISTORY))
            &jnp.all((s.qcount>=0)&(s.qcount<=QUEUE))&(s.episode>=0)&(s.generation>=0)
            &jnp.all(jnp.where(occupied,jnp.any(s.moves>0,axis=-1),jnp.all(s.moves==0,axis=-1)))
            &jnp.all((s.species>=0)&(s.species<=3))&jnp.all(jnp.sum(occupied,axis=1)<=3)&jnp.all(jnp.any(occupied,axis=1))
            &jnp.all((s.moves>=0)&(s.moves<=PROTECT))&jnp.all((s.item>=0)&(s.item<=2))
            &jnp.all((s.hp>=0)&(s.hp<=s.stats[:,:,0]))&jnp.all((s.pp>=0)&(s.pp<=PP[s.moves]))
            &jnp.all((s.status>=0)&(s.status<=FNT))&jnp.all((s.toxic>=0)&(s.toxic<=15))
            &jnp.all((~occupied)|(s.level>=1))
            &jnp.all((~occupied)|(s.level<=100))&jnp.all((s.request>=MOVE)&(s.request<=END))
            &jnp.all((s.active>=0)&(s.active<ROSTER))&jnp.all(s.species[jnp.arange(2),s.active]>0))



def _validate_structure(s):
    if not isinstance(s,State):raise ValueError('expected shared-engine State')
    roster_fields={'species','level','speed','gender','hp','item','status','toxic','stall','stall_duration'}
    vector_fields={'active','pending','request','pivot'}
    bool_fields={'fainted','protected','pivot','midturn'}
    for name,value in zip(State._fields,s):
        shape=(2,6) if name in roster_fields or name in {'fainted','protected'} else (2,) if name in vector_fields else (2,6,6) if name=='stats' else (2,6,4) if name in {'moves','pp'} else (2,6) if name=='order' else (QUEUE,7) if name=='queue' else (4,) if name=='rng' else (HISTORY,10) if name=='history' else ()
        dtype=jnp.bool_ if name in bool_fields else jnp.uint16 if name=='rng' else jnp.int32
        if getattr(value,'shape',None)!=shape or getattr(value,'dtype',None)!=jnp.dtype(dtype):
            raise ValueError(f'{name} must have shape {shape} and dtype {jnp.dtype(dtype)}')


def advance(s,choices,token=None):
    """Commit/replace one or both actions; -1 means no submission, -2 undo.

    An invalid combined submission is atomic. Tokens are local generation and
    request-episode numbers, never native rqid. Events/draws are successful
    prefixes only; any overflow rolls the entire advance back.
    """
    _validate_structure(s)
    choices=jnp.asarray(choices)
    if choices.shape!=(2,) or choices.dtype!=jnp.int32:raise ValueError('choices must be int32[2]')
    if token is not None:
        token=jnp.asarray(token)
        if token.shape!=(2,) or token.dtype!=jnp.int32:raise ValueError('token must be int32[2]')
    t=empty_trace();mask=legal_actions(s)
    incoming=choices>=0
    safe=jnp.clip(choices,0,10)
    invalid=jnp.any((choices < -2)|(choices>10)|(incoming&(~mask[jnp.arange(2),safe]))|((choices==-2)&(s.request>=WAIT)))
    stale=jnp.array(False) if token is None else jnp.any(token!=jnp.stack((s.generation,s.episode)))
    def accepted(c):
        s,t=c
        pending=jnp.where(choices==-2,-1,jnp.where(incoming,choices,s.pending))
        s=s._replace(pending=pending)
        ready=jnp.all((s.request>=WAIT)|(pending>=0))
        def resolve(c):
            s,t=c;c=_event(c,BLANK);s,t=c;s=_refresh_speed(s)
            normal=jnp.all(s.request==MOVE)
            def normal_queue(c):
                s,t=c;s=s._replace(midturn=jnp.array(True))
                return s,t
            c=_when(normal,normal_queue,(s,t));s,t=c
            for side in range(2):
                def add(c):
                    s,t=c;action=pending[side];mon=s.active[side];move=s.moves[side,mon,jnp.clip(action,0,3)]
                    kind=jnp.where(action>=4,jnp.where(action==10,ATTACK,jnp.where(normal,SWAP,INSTANT)),ATTACK)
                    priority=jnp.where((kind==ATTACK)&(move==PROTECT)&(action!=10),4,0)
                    source=jnp.where(s.pivot[side],U_TURN,0)
                    row=jnp.stack([kind,i32(side),mon,action,priority,s.speed[side,mon],source])
                    return _qappend(s,row),t
                c=_when(s.request[side]<WAIT,add,c);s,t=c
            c=_when(normal,lambda c:(_qappend(c[0],i32([RESIDUAL,0,0,0,0,1,0])),c[1]),c)
            s,t=c;s=s._replace(pivot=jnp.zeros(2,jnp.bool_))
            c=_qsort((s,t));s,t=c
            def prepend(c):
                s,t=c
                return s._replace(queue=jnp.concatenate((i32([[BEFORE,0,0,0,0,1,0]]),s.queue[:-1])),qcount=s.qcount+1),t
            s,t=_when(normal,prepend,(s,t))
            t=t._replace(overflow=t.overflow|(s.qcount>QUEUE))
            return _run_queue((s,t))
        s,t=jax.lax.cond(ready,resolve,lambda c:c,(s,t))
        overflow=t.overflow|(s.hcount+t.ecount>HISTORY)|(s.episode<0)
        def success(_):
            def add(k,s):return s._replace(history=s.history.at[s.hcount+k].set(t.events[k]))
            s2=jax.lax.fori_loop(0,t.ecount,add,s)
            return Result(s2._replace(hcount=s.hcount+t.ecount),jnp.where(ready,OK,PENDING),t)
        return jax.lax.cond(overflow,lambda _:Result(original,i32(CAPACITY),empty_trace()),success,None)
    original=s
    code=jnp.where(stale,STALE,jnp.where(s.winner>=0,TERMINAL,jnp.where(~_supported(s),UNSUPPORTED,jnp.where(invalid,INVALID,-1))))
    return jax.lax.cond(code>=0,lambda _:Result(s,code,empty_trace()),lambda _:accepted((s,t)),None)


def make_state(teams,seed=(1,2,3,4),generation=0):
    """Host schema validation plus pure JAX initialization from declared sets.

    Sets accept species, moves, level, speed_evs and item only. Unknown fields
    are rejected, preventing silently ignored effects. Trusted boundary fixtures
    are installed explicitly by tests, never treated as ordinary initial sets.
    """
    if len(teams)!=2 or any(not 1<=len(team)<=3 for team in teams):raise ValueError('two teams of one to three mons required')
    species=np.zeros((2,6),np.int32);level=np.zeros((2,6),np.int32);moves=np.zeros((2,6,4),np.int32)
    item=np.zeros((2,6),np.int32);evs=np.zeros((2,6,6),np.int32)
    for side,team in enumerate(teams):
        for mon,desc in enumerate(team):
            if set(desc)-{'species','moves','level','speed_evs','item'}:raise ValueError('unsupported set field')
            if type(desc.get('level',50)) is not int:raise ValueError('level must be an integer')
            species[side,mon]=SPECIES.index(desc['species']);level[side,mon]=desc.get('level',50)
            if species[side,mon]==0 or not 1<=level[side,mon]<=100:raise ValueError('unsupported species/level')
            speed=desc.get('speed_evs',0)
            if type(speed) is not int or speed not in (0,252):raise ValueError('unsupported Speed EVs')
            evs[side,mon,5]=speed
            if not 1<=len(desc['moves'])<=4 or len(set(desc['moves']))!=len(desc['moves']):raise ValueError('one to four distinct moves required')
            for slot,name in enumerate(desc['moves']):
                mid=MOVE_NAMES.index(name)
                if mid==0 or mid>PROTECT:raise ValueError('unsupported move')
                moves[side,mon,slot]=mid
            item[side,mon]=ITEMS.index(desc.get('item',''))
    seed=np.asarray(seed)
    if seed.shape!=(4,) or seed.dtype.kind not in 'iu' or np.any(seed<0) or np.any(seed>65535):raise ValueError('seed must have four 16-bit limbs')
    if not isinstance(generation,int) or not 0<=generation<2**31-1:raise ValueError('invalid generation')
    return initialize(i32(species),i32(level),i32(moves),i32(item),i32(evs),jnp.asarray(seed,jnp.uint16),i32(generation))


@jax.jit
def initialize(species,level,moves,item,evs,seed,generation):
    occupied=species>0
    stats=derive_stored_stats(BASE_STATS[species],level,jnp.full((2,6,6),31,jnp.int32),evs,jnp.full((2,6,6),10,jnp.int32))
    stats=jnp.where(occupied[:,:,None],stats,0)
    zeros=jnp.zeros((2,6),jnp.int32)
    s=State(species,level,stats,stats[:,:,5],zeros,stats[:,:,0],jnp.zeros((2,6),jnp.bool_),moves,PP[moves],item,
            zeros,zeros,zeros,zeros,jnp.zeros((2,6),jnp.bool_),jnp.zeros(2,jnp.int32),
            jnp.tile(jnp.arange(6,dtype=jnp.int32),(2,1)),jnp.full(2,-1,jnp.int32),jnp.full(2,MOVE,jnp.int32),
            jnp.zeros(2,jnp.bool_),jnp.zeros((QUEUE,7),jnp.int32),i32(0),jnp.array(False),seed,i32(1),i32(-1),generation,i32(0),
            jnp.zeros((HISTORY,10),jnp.int32),i32(0))
    c=(s,empty_trace())
    for side in range(2):
        for mon in range(3):
            def gender(c):
                c,value=_draw(c,0,2,INIT);s,t=c
                return s._replace(gender=s.gender.at[side,mon].set(value.astype(jnp.int32)+1)),t
            c=_when(species[side,mon]>MEW,gender,c)
    # Native startup: runSwitch all-active speed sort, then its Update, followed
    # by the residual action's Update. No battle RNG is re-seeded after setup.
    for _ in range(3):c=_update(c,INIT)
    for side in range(2):
        s,t=c
        c=_event(c,SWITCH,side,0,value=s.hp[side,0],aux=s.stats[side,0,0])
    c=_event(c,TURN,value=1)
    s,t=c
    s=s._replace(history=s.history.at[:3].set(t.events[:3]),hcount=i32(3))
    return Result(s,i32(OK),t)


def reset(state,replacement,token):
    """Reset a terminal lane with a supplied initialized battle; live/stale reset
    is rejected atomically. No native reset-pool or throughput claim is implied.
    """
    _validate_structure(state)
    _validate_structure(replacement)
    if token.shape!=(2,) or token.dtype!=jnp.int32:raise ValueError('token must be int32[2]')
    valid=_supported(replacement)&(replacement.winner<0)&(replacement.turn==1)&(~replacement.midturn)&jnp.all(replacement.pending==-1)&(replacement.qcount==0)&(state.winner>=0)&jnp.all(token==jnp.stack((state.generation,state.episode)))&(state.generation<2**31-1)
    def apply(_):
        s=replacement._replace(generation=state.generation+1,pending=jnp.full(2,-1,jnp.int32),
                               episode=i32(0))
        return Result(s,i32(OK),empty_trace())
    return jax.lax.cond(valid,apply,lambda _:Result(state,jnp.where(state.generation==2**31-1,CAPACITY,INVALID),empty_trace()),None)


def rewards(state):
    """Terminal zero-sum outcome; live/truncated work carries no invented win."""
    winner=state.winner
    return jnp.where(winner<=0,jnp.zeros(2,jnp.int32),jnp.where(jnp.arange(2)+1==winner,1,-1)).astype(jnp.int32)
