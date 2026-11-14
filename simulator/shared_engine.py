"""Bounded shared Gen 5 singles engine, independent of the native oracle.

Fixed-shape int32 storage; exact four-limb PRNG; stable mon identity and a
separate command roster. Strings/protocol rendering belong to the host adapter.
Schema 12 carries Substitute interception, private retry disclosure, move phases and reactive effects through the same scheduler. Handler masks
remain incomplete and reject unsupported effects; full conformance is pending.
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
from . import gen5_tables as generic
from . import gen5_engine_tables as tables
from . import status_effects as effects
from . import field_tables
from . import field_effects as field_ops
from . import hit_tables
from . import hit_effects as hit_ops
from . import phase_effects as phase_ops
from . import battle_tracking as tracking
from . import trapping as trap_ops
from . import substitute as sub_ops
from . import partial_trap as partial_ops
from . import phazing as phaze_ops
from . import pursuit as pursuit_ops
from . import item_lifecycle as item_ops
from . import identity as identity_ops
from . import illusion as illusion_ops
from . import forms as form_ops
from . import trigger_forms as trigger_form_ops
from . import ability_changes as ability_ops
from . import status_immunity as immunity_ops
from . import ability_creation as creation_ops
from . import suppression as suppression_ops
from . import sleep_control as sleep_ops
from . import called_moves as called_ops
from . import trace_ability as trace_ops
from . import mummy as mummy_ops
from . import hook_tables
from . import hook_effects as hook_ops
import sys

def _engine():
    return sys.modules[__name__]

SCHEMA = 38
ROSTER, SLOTS, QUEUE, EVENTS, DRAWS, HISTORY = 6, 4, 8, 128, 128, 2048
OK, PENDING, INVALID, UNSUPPORTED, CAPACITY, TERMINAL, STALE = range(7)
RETRY=7
MOVE, FORCED, WAIT, END = range(4)
NONE, BRN, PAR, TOX, FNT, PSN, SLP, FRZ = range(8)
STATUS_NAMES=('','brn','par','tox','fnt','psn','slp','frz')
# Species and moves use local table indices, explicitly versioned here.
MEW, SNORLAX, GENGAR = 1, 2, 3
TACKLE, POUND, MEGA_PUNCH, EARTHQUAKE, U_TURN, WISP, TOXIC, THUNDER_WAVE, RECOVER, PROTECT, STRUGGLE = range(1,12)
STEALTH_ROCK, SPIKES, RAPID_SPIN = range(12,15)
ITEMS = tables.ITEMS
LEGACY_SPECIES,LEGACY_ITEMS,LEGACY_MOVES=tables.LEGACY_SPECIES,tables.LEGACY_ITEMS,tables.LEGACY_MOVES
SPECIES,MOVE_NAMES,MOVE_KEYS=tables.SPECIES,tables.MOVE_NAMES,tables.MOVE_KEYS
BASE_STATS,PP,POWER,ACCURACY=tables.BASE_STATS,tables.PP,tables.POWER,tables.ACCURACY
ABILITIES=tables.ABILITIES
SPORE,CONFUSE_RAY,POISON_POWDER,REST=(tables.MOVE_IDS[k] for k in ('spore','confuseray','poisonpowder','rest'))
HIDDEN_POWER,RETURN,FRUSTRATION=(tables.MOVE_IDS[k] for k in ('hiddenpower','return','frustration'))
# Event fields: kind, side, stable mon, move, target side, target mon, value,
# auxiliary, status, reserved. HP events retain maxHP for exact projection.
BLANK, EV_MOVE, DAMAGE, HEAL, CRIT, IMMUNE, MISS, FAINT, WIN, TURN, UPKEEP, SWITCH, STATUS, ABILITY, END_ITEM, CURE, FAIL, CANT, ACTIVATE, SINGLE = range(20)
SIDE_START, SIDE_END = 20, 21
CONF_START, CONF_END, CONF_ACTIVATE, SLEEP_CLAUSE = range(22,26)
BOOST, WEATHER = 26, 27
REVEAL = 28
HIT_COUNT = 29
MUST_RECHARGE, RECHARGE_CANT = 30,31
PREPARE,ANIM,END_HELD_ITEM,FLINCH_CANT=32,33,34,35
VOL_START,VOL_END,VOL_CANT=36,37,38
TURN_WARNING,TURN_LIMIT=39,40
FRISK_ITEM=41
SUB_DAMAGE=42
PARTIAL_START,PARTIAL_END,PARTIAL_DAMAGE=43,44,45
DRAG,PHAZE_ABILITY=46,47
PURSUIT_HINT=48
ITEM_SWAP,ITEM_GAIN,ITEM_LOSS=49,50,51
TRANSFORM=52
ILLUSION_END=53
FORM_CHANGE=54
ABILITY_CHANGE,ABILITY_SWAP=55,56
STATUS_ABILITY,REST_ABILITY=57,58
SUPPRESS_ABILITY=59
YAWN_START,YAWN_END=60,61
EVENT_COLUMNS=16
# Draw contexts; range form stores actual native arguments including offsets.
INIT, QUEUE_SORT, BEFORE_TURN, UPDATE, HIT_UPDATE, SWITCH_SORT, FIELD_SORT, ACC, CRITICAL, ROLL, PARALYSIS, STALL = range(12)
SLEEP_DURATION, FREEZE, CONF_DURATION, CONFUSION, CONF_DAMAGE, SECONDARY = range(12,18)
WEATHER_CHANGE, WEATHER_ORDER = 18,19
DAMAGE_MODIFIERS = 20
MULTIHIT_COUNT = 21
RAMPAGE_DURATION = 22
FRISK_TARGET=23
VOLATILE_DURATION=24
PHAZE_SELECTION=25
CALLED_SELECTION=26
TRACE_TARGET=27
DAMAGE_CALC_SORT=28
MOVE_ACTOR_TARGET=field_tables.ACTOR_TARGET
DRAW_CONTEXTS=('initialization','queue-sort','before-turn','update','hit-update','switch-order','residual-handlers','accuracy','critical','damage','paralysis','stall','sleep-duration','freeze','confusion-duration','confusion','confusion-damage','secondary','weather-change','weather-order','damage-modifiers','multihit-count','rampage-duration','frisk-target','volatile-duration','phazing-selection','sleep-talk-selection','trace-target','damage-calculation-sort')
# Queue kinds, order values mirror the pinned scheduler.
BEFORE, ATTACK, SWAP, INSTANT, RUN_SWITCH, RESIDUAL, BEFORE_MOVE, FINISH_SWITCH = range(1,9)
FORCED_MOVE_ACTION=12
QORDER=jnp.array([999,4,200,103,3,101,300,5,3],jnp.int32)


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
    # Per side: Rock present, Spikes layers, first installed hazard (1/2).
    # Native onSwitchIn callbacks retain insertion order, even on layer restart.
    hazards: jax.Array
    ivs: jax.Array
    evs: jax.Array
    nature: jax.Array
    ability: jax.Array
    types: jax.Array
    hp_type: jax.Array
    hp_power: jax.Array
    shiny: jax.Array
    happiness: jax.Array
    name: jax.Array
    boosts: jax.Array
    effect_order: jax.Array
    # id, effectOrder, source side/mon, target present, time, startTime.
    status_state: jax.Array
    # Protect/stall/confusion: present, order, source side/mon, move, time.
    volatile_state: jax.Array
    ability_order: jax.Array
    item_order: jax.Array
    weather_state: jax.Array
    screens: jax.Array
    hazard_meta: jax.Array
    sleep_clause_target: jax.Array
    move_history: jax.Array
    action_state: jax.Array
    attacked_by: jax.Array
    side_faints: jax.Array
    battle_move: jax.Array
    battle_success: jax.Array
    battle_damage: jax.Array
    request_state: jax.Array
    item_provenance: jax.Array
    base_species: jax.Array
    base_ability: jax.Array
    base_stats: jax.Array
    base_moves: jax.Array
    base_pp: jax.Array
    base_used: jax.Array
    transformed: jax.Array
    known_type: jax.Array
    apparent_types: jax.Array
    ability_effect_id: jax.Array
    illusion: jax.Array
    ability_staleness: jax.Array
    call_frames: jax.Array
    call_depth: jax.Array
    ability_seek: jax.Array



# Same mechanical fields, without the compatibility history allocation.
CoreState = NamedTuple('CoreState', [(name, jax.Array) for name in State._fields if name not in ('history','hcount')])

def core_state(state):
    _validate_structure(state)
    return CoreState(*(getattr(state,name) for name in CoreState._fields))


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
    private_errors: jax.Array
    force_switch: jax.Array


class Result(NamedTuple):
    state: State
    code: jax.Array
    trace: Trace


def i32(value):
    return jnp.asarray(value,jnp.int32)


def empty_trace():
    return Trace(jnp.zeros((EVENTS,EVENT_COLUMNS),jnp.int32),i32(0),jnp.zeros((DRAWS,3),jnp.uint32),
                 jnp.zeros(DRAWS,jnp.uint32),jnp.zeros(DRAWS,jnp.int32),
                 jnp.zeros((DRAWS,4),jnp.uint16),jnp.zeros((DRAWS,4),jnp.uint16),i32(0),jnp.array(False),jnp.zeros(2,jnp.int32),jnp.zeros(2,jnp.bool_))


def _event(c, kind, side=0, mon=0, move=0, target=0, targetmon=0, value=0, aux=0, status=0):
    s,t=c
    row=jnp.stack([i32(x) for x in (kind,side,mon,move,target,targetmon,value,aux,status,0,*illusion_ops.event_identity(s,side,mon,target,targetmon))])
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
    speed=field_ops.scale_stat_stage(s.stats[:,:,5],s.boosts[:,:,4])
    return hook_ops.speeds(s,speed,_engine())


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
    c,word=jax.lax.cond(tie,lambda c:_draw(c,0,2,context,2),lambda c:(c,jnp.uint32(0)),c)
    first=jnp.where(tie,word.astype(jnp.int32),jnp.argmax(active_speed))
    # Keep one compiled callback body, visiting holders in native speed/shuffle
    # order. Python unrolling duplicates Trace and its ordinary Start subtree
    # at every Update call site.
    def visit(k,c):
        side=jnp.where(k==0,first,1-first)
        c=immunity_ops.update(c,side,_engine())
        c=sleep_ops.update(c,side,_engine())
        c=trace_ops.update(c,side,_engine(),initial=context==INIT)
        def cure(c):
            s,t=c;m=s.active[side];old=s.status[side,m]
            c=_event(c,END_ITEM,side,m)
            s,t=c
            return effects.lum((s,t),side,_engine(),emit_item=False)
        s,t=c;m=s.active[side]
        return _when((s.item[side,m]==2)&((s.status[side,m]!=NONE)&(s.status[side,m]!=FNT)|(s.volatile_state[side,m,2,0]>0))&(s.hp[side,m]>0),cure,c)
    return jax.lax.fori_loop(0,2,visit,c)


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
    order=jnp.where(valid,QORDER[jnp.clip(q[:,0],0,8)],10000)
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
    s=illusion_ops.clear(s,side,_engine())
    s=s._replace(ability_staleness=s.ability_staleness.at[side,s.active[side]].set(0))
    s=identity_ops.restore(s,side,_engine())
    s=tracking.faint(s,side)
    s=s._replace(volatile_state=s.volatile_state.at[side,m].set(0),boosts=s.boosts.at[side,m].set(0))
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


def _damage(c,side,amount,cause=0,minimum=1,source_side=0,source_mon=0):
    # Cause0 is a Move. Struggle and confusion use native directDamage and
    # bypass Damage callbacks. Every other currently implemented cause is a
    # Condition, Item or Ability and Gen5 Magic Guard silently cancels it.
    s,_=c;m=s.active[side]
    blocked=(suppression_ops.abilities(s)[side,m]==tables.ABILITY_IDS['magicguard'])&(cause!=0)&(cause!=3)&(cause!=6)
    def apply(c):
        s,t=c;hp=jnp.maximum(0,s.hp[side,m]-jnp.maximum(minimum,amount))
        s=tracking.damaged(s,side,s.hp[side,m]-hp,hp,source_side,cause)
        s=s._replace(hp=s.hp.at[side,m].set(hp))
        return _event((s,t),DAMAGE,side,m,value=hp,aux=s.stats[side,m,0],status=s.status[side,m],move=cause,target=source_side,targetmon=source_mon)
    return _when(~blocked,apply,c)


def _status(c,source,target,status,reflect=True,move=0,secondary=False,overwrite=False):
    return effects.inflict(c,source,target,status,_engine(),reflect=reflect,move=move,
                           secondary=secondary,overwrite=overwrite)


def _hit(c,actor,move):
    s,tr=c;c=(tracking.put(s,actor,tracking.LAST_DAMAGE,i32(0)),tr)
    s,t=c;mon=s.active[actor];target=1-actor;tm=s.active[target]
    damaging=tables.CATEGORY[move]>0
    def strike(c):
        s,t=c
        bound=hook_ops.critical_denominator(move,suppression_ops.abilities(s)[actor,mon],s.item[actor,mon])
        c,crit=jax.lax.cond(bound>0,lambda c:_draw(c,0,bound,CRITICAL),lambda c:(c,jnp.uint32(1)),c)
        crit=jnp.where(hook_ops.critical_blocked(suppression_ops.abilities(s)[target,tm]),jnp.uint32(1),crit)
        s,t=c
        physical=tables.CATEGORY[move]==1
        attack=s.stats[actor,mon,jnp.where(physical,1,3)]
        defense=s.stats[target,tm,jnp.where(physical,2,4)]
        c,attack,defense=field_ops.damage_parameters(c,actor,target,move,crit,physical,attack,defense,_engine());s,t=c
        power=jnp.where(move==HIDDEN_POWER,s.hp_power[actor,mon],
                        jnp.where(move==RETURN,s.happiness[actor,mon]*10//25,
                                  jnp.where(move==FRUSTRATION,(255-s.happiness[actor,mon])*10//25,POWER[move])))
        power=jnp.maximum(1,power)*jnp.where(pursuit_ops.boosted(s,target,move,_engine()),2,1)
        move_type=jnp.where(move==HIDDEN_POWER,s.hp_type[actor,mon],tables.MOVE_TYPE[move])
        power=hook_ops.power(power,suppression_ops.abilities(s)[actor,mon],s.item[actor,mon],physical,move_type,_engine(),phase_ops.power_modifier(s,target,move))
        c,attack,defense=hook_ops.stats(c,attack,defense,actor,target,physical,move_type,_engine())
        c,roll=_draw(c,0,16,ROLL);s,t=c
        core=base_damage_core(s.level[actor,mon],power,attack,defense)+2
        preroll=field_ops.weather_damage(core,move_type,s,_engine())*jnp.where(crit==0,2,1)
        exponent,_=generic.type_matchup(move_type,s.types[target,tm])
        stab=jnp.any(s.types[actor,mon]==move_type)
        # Native effectiveness precedes the critical disclosure.
        c=_when((exponent!=0)&(move!=STRUGGLE),lambda c:_event(c,ACTIVATE,target,tm,aux=exponent,move=-1),c)
        c=_when(crit==0,lambda c:_event(c,CRIT,target,tm),c)
        c,final_modifier=hook_ops.damage_modifier(c,actor,target,physical,crit==0,jnp.where(move==STRUGGLE,0,exponent),_engine(),move)
        damage=damage_from_preroll(preroll,roll.astype(jnp.int32),stab,suppression_ops.abilities(s)[actor,mon]==hook_tables.A['adaptability'],move==STRUGGLE,
                                  jnp.where(move==STRUGGLE,0,exponent),s.status[actor,mon]==BRN,physical,suppression_ops.abilities(s)[actor,mon]==hook_tables.A['guts'],final_modifier)
        intercepted=sub_ops.intercepts(s,target,move)
        def normal(c):
            dealt=jnp.minimum(s.hp[target,tm],damage)
            c=_damage(c,target,damage,minimum=0,source_side=actor,source_mon=mon)
            c=hit_ops.drain(c,actor,target,move,dealt,_engine())
            c=partial_ops.install(c,actor,target,move,_engine())
            c=phase_ops.install(c,actor,move,_engine())
            c=effects.secondary(c,actor,move,_engine())
            c=phaze_ops.mark(c,actor,target,move,_engine())
            c=hit_ops.contact(c,actor,target,move,_engine())
            c=illusion_ops.reveal(c,target,_engine())
            # Native DamagingHit resolves ordered contact callbacks before the
            # default-order freeze hook, even when a callback has ended the game.
            c=_when((c[0].status[target,tm]==FRZ)&(move_type==generic.TYPES['fire'])&(c[0].hp[target,tm]>0),
                    lambda c:effects.cure(c,target,_engine(),source=actor),c)
            # Default-order status hooks precede Mummy in native stable
            # left-to-right DamagingHit order, after ordered contact damage.
            c=mummy_ops.contact(c,actor,target,move,_engine())
            c=item_ops.knock(c,actor,target,move,_engine())
            # Gen5 Spin is AfterHit, and is skipped if contact damage killed it.
            c=_when((move==RAPID_SPIN)&(c[0].hp[actor,mon]>0),lambda c:hit_ops.spin(c,actor,_engine()),c)
            return c,dealt
        c,dealt=jax.lax.cond(intercepted,lambda c:sub_ops.absorb(c,actor,target,move,damage,_engine()),normal,c)
        c=_update(c,HIT_UPDATE)
        return c,dealt,intercepted
    def physical(c):
        c,count=hit_ops.hit_count(c,move,actor,_engine())
        def more(carry):
            (s,_),hits,total,last,last_sub=carry
            return (hits<count)&(s.hp[target,tm]>0)&((s.hp[actor,mon]>0)|((hits==0)&hit_tables.SELF_DESTRUCT[move]))
        def hit(carry):
            c,hits,total,last,last_sub=carry;c,dealt,sub=strike(c)
            return c,hits+1,total+jnp.where(sub,0,dealt),dealt,sub
        c,hits,total,last,last_sub=jax.lax.while_loop(more,hit,(c,i32(0),i32(0),i32(0),jnp.array(False)))
        source_ko=c[0].hp[actor,mon]==0
        # Gen5 resolves the last queued faint first when the source has also
        # fainted. Contact double-KO therefore differs from later recoil KO.
        c=_when(source_ko&(~c[0].fainted[actor,mon]),lambda c:_faint(c,actor),c)
        c=_when((c[0].hp[target,tm]==0)&(~c[0].fainted[target,tm]),lambda c:_faint(c,target),c)
        c=_when(source_ko&(c[0].winner<0),lambda c:_win(c,target),c)
        c=_when(hit_tables.MAX_HITS[move]>1,lambda c:_event(c,HIT_COUNT,target,tm,value=hits,aux=c[0].fainted[target,tm].astype(jnp.int32)),c)
        regular=(hit_tables.RECOIL_NUM[move]>0)&(suppression_ops.abilities(c[0])[actor,mon]!=tables.ABILITY_IDS['rockhead'])
        if_recoil=(move==STRUGGLE)|regular
        def recoil(c):
            s,t=c
            amount=jnp.where(move==STRUGGLE,(s.stats[actor,mon,0]+2)//4,
                jnp.maximum(1,hit_ops.rounded_fraction(total,hit_tables.RECOIL_NUM[move],hit_tables.RECOIL_DEN[move])))
            return _damage(c,actor,amount,cause=jnp.where(move==STRUGGLE,3,10))
        c=_when(if_recoil&(total>0)&(c[0].hp[actor,mon]>0),recoil,c)
        c=_when(~last_sub,lambda c:tracking.attacked(c,target,actor,move,last,i32(tracking.NUMBER),hits,_engine()),c)
        c=tracking.outcome(c,actor,i32(tracking.TRUE),move)
        c=_update(c,HIT_UPDATE)
        c=_when((c[0].hp[actor,mon]==0)&(~c[0].fainted[actor,mon]),lambda c:_faint(c,actor),c)
        c=trigger_form_ops.after_hit(c,actor,move,_engine())
        def life_orb(c):
            s,t=c;c=_damage(c,actor,s.stats[actor,mon,0]//10,cause=9)
            return _when(c[0].hp[actor,mon]==0,lambda c:_faint(c,actor),c)
        orb=(c[0].item[actor,mon]==hook_tables.I['lifeorb'])&(c[0].hp[actor,mon]>0)
        c=_when(orb,life_orb,c)
        # Native AfterMove fatigue precedes the ordinary final win announcement.
        c=_when(phase_ops.RAMPAGE[move],lambda c:phase_ops.after_move(c,actor,_engine()),c)
        c=_when(c[0].winner<0,lambda c:_win(c,jnp.where(if_recoil|orb,actor,target)),c)
        s,t=c
        def pivot(c):
            s,t=c
            return s._replace(pivot=s.pivot.at[actor].set(True)),t
        return _when((move==U_TURN)&(s.winner<0)&(s.hp[actor,mon]>0)&jnp.any(_live(s)[actor]&(jnp.arange(ROSTER)!=mon)),pivot,c)
    def status_hit(c):
        def confuse(c):return effects.confuse(c,actor,target,move,_engine())
        def status(c):
            wanted=jnp.where(move==WISP,BRN,jnp.where(move==TOXIC,TOX,jnp.where(move==POISON_POWDER,PSN,jnp.where(move==SPORE,SLP,PAR))))
            return jax.lax.cond(field_tables.IS_BOOST[move],lambda c:field_ops.boost(c,actor,target,move,_engine()),
                lambda c:_status(c,actor,target,wanted,move=move),c)
        before_count=c[1].ecount
        move_row=jnp.max(jnp.where((jnp.arange(EVENTS)<c[1].ecount)&(c[1].events[:,0]==EV_MOVE),jnp.arange(EVENTS),0))
        c,preblocked=phase_ops.try_hit(c,actor,target,move,move_row,_engine())
        c,ability_denied=ability_ops.try_hit(c,actor,target,move,move_row,_engine())
        preblocked=preblocked|ability_denied
        c,suppression_denied=suppression_ops.try_hit(c,actor,target,move,move_row,_engine())
        preblocked=preblocked|suppression_denied
        c,sleep_denied=sleep_ops.try_hit(c,actor,target,move,move_row,_engine())
        preblocked=preblocked|sleep_denied
        sub_block=(~preblocked)&sub_ops.intercepts(c[0],target,move)
        c=_when(sub_block,lambda c:sub_ops.block_status(c,actor,move,move_row,_engine()),c)
        c=jax.lax.cond(preblocked|sub_block,lambda c:c,
            lambda c:jax.lax.cond(move==sleep_ops.YAWN,lambda c:sleep_ops.hit(c,actor,target,move_row,_engine()),lambda c:jax.lax.cond(move==suppression_ops.MOVE,lambda c:suppression_ops.hit(c,actor,target,move_row,_engine()),lambda c:jax.lax.cond(ability_ops.MOVE_IMPLEMENTED[move],lambda c:ability_ops.hit(c,actor,target,move,move_row,_engine()),lambda c:jax.lax.cond(move==identity_ops.MOVE,lambda c:identity_ops.move(c,actor,target,move_row,_engine()),lambda c:jax.lax.cond(item_ops.EXCHANGE[move],lambda c:item_ops.exchange(c,actor,target,move,move_row,_engine()),lambda c:jax.lax.cond(phaze_ops.STATUS[move],lambda c:phaze_ops.status_hit(c,actor,target,move,move_row,_engine()),lambda c:jax.lax.cond(phase_ops.RESTRICTION[move]>0,lambda c:phase_ops.status_move(c,actor,target,move,_engine()),
                lambda c:jax.lax.cond(move==CONFUSE_RAY,confuse,status,c),c),c),c),c),c),c),c),c)
        # Only actual successful application runs the hit-loop updates. Lum can
        # immediately cure it, so inspect the emitted status/start event.
        event=c[1].events[:,0];valid=(jnp.arange(EVENTS)>=before_count)&(jnp.arange(EVENTS)<c[1].ecount)
        success=jnp.any(valid&((event==STATUS)|(event==CONF_START)|(event==VOL_START)))
        tracked_success=success|((field_tables.IS_BOOST[move]|item_ops.EXCHANGE[move]|ability_ops.MOVE_IMPLEMENTED[move]|suppression_ops.MOVE_IMPLEMENTED[move]|sleep_ops.MOVE_IMPLEMENTED[move])&(c[0].action_state[actor,mon,tracking.RESULT]==tracking.TRUE))
        c=_when(tracked_success,lambda c:tracking.attacked(c,target,actor,move,i32(0),i32(tracking.TRUE),i32(0),_engine()),c)
        c=_when((~phaze_ops.STATUS[move])&((move!=identity_ops.MOVE)|sub_block),lambda c:tracking.outcome(c,actor,jnp.where(tracked_success|sub_block,tracking.TRUE,tracking.FALSE),move),c)
        return _when((success|sub_block|((item_ops.EXCHANGE[move]|ability_ops.MOVE_IMPLEMENTED[move]|suppression_ops.MOVE_IMPLEMENTED[move]|sleep_ops.MOVE_IMPLEMENTED[move])&tracked_success))&(~phaze_ops.STATUS[move])&((move!=identity_ops.MOVE)|sub_block),lambda c:_update(_update(c,HIT_UPDATE),HIT_UPDATE),c)
    def mirror(c):
        s,tr=c;frame=s.call_frames[s.call_depth-1]
        move_row=jnp.max(jnp.where((jnp.arange(EVENTS)<tr.ecount)&(tr.events[:,0]==EV_MOVE),jnp.arange(EVENTS),0))
        return called_ops.copy_move(c,frame,move_row,_engine(),mirror=True)
    return jax.lax.cond(move==called_ops.MIRROR_MOVE,mirror,lambda c:jax.lax.cond(damaging,physical,status_hit,c),c)


def _attack(c,row):
    s,t=c;actor=row[1];mon=row[2];action=phase_ops.override_action(s,actor,row[3])
    selected=jnp.where(row[3]>=FORCED_MOVE_ACTION,row[3]-FORCED_MOVE_ACTION,jnp.where(row[3]==10,STRUGGLE,s.moves[actor,mon,jnp.clip(row[3],0,3)]))
    target_loc=jnp.where(MOVE_ACTOR_TARGET[selected],-1,1)
    move=jnp.where(action>=FORCED_MOVE_ACTION,action-FORCED_MOVE_ACTION,jnp.where(action==10,STRUGGLE,s.moves[actor,mon,jnp.clip(action,0,3)]))
    locked_before=phase_ops.forced_move(s)[actor]>0
    oldloc=s.action_state[actor,mon,tracking.TARGET_LOC]
    oldloc=jnp.where(oldloc==-2,0,oldloc)
    charge=s.volatile_state[actor,mon,6,0]>0
    child=phase_ops.CHARGE_SLOT[move];storedloc=s.volatile_state[actor,mon,child,9]
    target_loc=jnp.where(locked_before,jnp.where(charge&(storedloc!=0),storedloc,oldloc),target_loc)
    target=jnp.where(MOVE_ACTOR_TARGET[move],actor,1-actor)
    tm=s.active[target]
    def execute(c):
        s,t=c;c=(tracking.attempted(s,actor),t)
        c,full=effects.before_move(c,actor,move,_engine())
        c=_when(full,lambda c:phase_ops.abort_move(c,actor,_engine()),c)
        def cannot(c):return _when(~c[0].fainted[actor,mon],lambda c:tracking.result(c,actor,i32(tracking.FALSE)),c)
        def act(c):
            s,t=c;slot=jnp.clip(action,0,3);previous=s.battle_move
            s=tracking.used(s,actor,move,slot,_engine(),target_loc,mark_slot=(~locked_before)&(s.moves[actor,mon,slot]==move))
            s=s._replace(battle_success=i32(0))
            pressured=hook_tables.PRESSURE_TARGET[move].astype(jnp.bool_)&(suppression_ops.abilities(s)[1-actor,s.active[1-actor]]==hook_tables.A['pressure'])&(~s.fainted[1-actor,s.active[1-actor]])
            cost=1+pressured.astype(jnp.int32)
            s=s._replace(pp=s.pp.at[actor,mon,slot].set(jnp.where((move==STRUGGLE)|locked_before,s.pp[actor,mon,slot],jnp.maximum(0,s.pp[actor,mon,slot]-cost))))
            s=identity_ops.sync(s,actor,_engine())
            tag=jnp.where(row[6]==pursuit_ops.MOVE,2,locked_before.astype(jnp.int32))
            return called_ops.root((s,t),actor,mon,move,target_loc,slot,locked_before,tag,previous,_engine(),prankster=(row[4]>tables.PRIORITY[selected])&(tables.CATEGORY[selected]==0))
        return jax.lax.cond(full,cannot,act,c)
    # A queued actor remains tied to its original mon, and cannot act after KO.
    return _when((s.active[actor]==mon)&(s.hp[actor,mon]>0),
                 lambda c:jax.lax.cond(action==phase_ops.RECHARGE_ACTION,lambda c:phase_ops.consume((tracking.attempted(c[0],actor),c[1]),actor,_engine()),execute,c),c)



def _use_move(c,frame):
    actor=frame[called_ops.ACTOR];mon=frame[called_ops.MON];move=frame[called_ops.MOVE]
    locked_before=frame[called_ops.LOCKED]>0;target=frame[called_ops.TARGET];tm=frame[called_ops.TM]
    c=called_ops.enter(c,frame,_engine());s,t=c
    c=_when((s.status[actor,mon]==FRZ)&hit_tables.DEFROST[move],
            lambda c:effects.cure(c,actor,_engine(),source=target,move=move),(s,t))
    s,t=c
    c=hook_ops.modify_move(c,actor,target,move,_engine())
    c=_when(move==STRUGGLE,lambda c:_event(c,ACTIVATE,actor,mon,move=STRUGGLE),c)
    # Reserve move row so the native miss/fail annotation can be filled
    # after the outcome without reordering the event stream.
    move_row=t.ecount+jnp.where(move==STRUGGLE,1,0)
    side_move=(move==STEALTH_ROCK)|(move==SPIKES)
    c=_event(c,EV_MOVE,actor,mon,move,target,tm,aux=jnp.where(side_move&(s.hp[target,tm]==0),4,0))
    s,t=c;c=(s,t._replace(events=t.events.at[move_row,9].set(frame[called_ops.TAG])))
    def sacrifice(c):
        s,t=c
        return s._replace(hp=s.hp.at[actor,mon].set(0)),t
    c=_when(hit_tables.SELF_DESTRUCT[move],sacrifice,c)
    def self_move(c):
        original=c[0]
        def recover(c):
            s,t=c;hp=s.hp[actor,mon];maxhp=s.stats[actor,mon,0]
            def fail(c):
                s,t=c;t=t._replace(events=t.events.at[move_row,7].set(2))
                return _event((s,t),FAIL,actor,mon,aux=1)
            def heal(c):
                c=tracking.outcome(c,actor,i32(tracking.TRUE),move)
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
                c=effects.add_volatile((s,t),actor,0,actor,move,_engine())
                c=effects.add_volatile(c,actor,1,actor,move,_engine())
                c=tracking.outcome(c,actor,i32(tracking.TRUE),move)
                return _event(c,SINGLE,actor,mon)
            def fail_fn(c):
                s,t=c;t=t._replace(events=t.events.at[move_row,7].set(2))
                return _event((s,t),FAIL,actor,mon)
            return jax.lax.cond(success,success_fn,fail_fn,c)
        c=jax.lax.cond(move==REST,lambda c:effects.rest(c,actor,move_row,_engine()),
            lambda c:jax.lax.cond(move==item_ops.RECYCLE,lambda c:item_ops.recycle(c,actor,move_row,_engine()),lambda c:jax.lax.cond(move==sub_ops.MOVE,lambda c:sub_ops.create(c,actor,move_row,_engine()),lambda c:jax.lax.cond(move==RECOVER,recover,protect,c),c),c),c)
        # Successful self moves run both hit-loop Updates.
        s,t=c;failed=(t.events[move_row,7]==2)|((move==REST)&((original.hp[actor,mon]==original.stats[actor,mon,0])|immunity_ops.SLEEP[suppression_ops.abilities(original)[actor,mon]]))|((move==sub_ops.MOVE)&(s.action_state[actor,mon,tracking.RESULT]!=tracking.TRUE))
        c=_when(~failed,lambda c:_update(_update(c,HIT_UPDATE),HIT_UPDATE),c)
        return c
    def foe_move(c):
        s,t=c
        move_type=jnp.where(move==HIDDEN_POWER,s.hp_type[actor,mon],tables.MOVE_TYPE[move])
        _,type_immune=generic.type_matchup(move_type,s.types[target,tm])
        levitate=(suppression_ops.abilities(s)[target,tm]==3)&(move_type==generic.TYPES['ground'])
        damaging=tables.CATEGORY[move]>0
        immune=(item_ops.EXCHANGE[move]&(suppression_ops.abilities(s)[target,tm]==tables.ABILITY_IDS['stickyhold']))|((move!=STRUGGLE)&(((damaging|(move==THUNDER_WAVE))&type_immune)|(damaging&levitate)))
        blocked=s.protected[target,tm]&ability_ops.PROTECT[move]
        def block(c):return _event(tracking.outcome(c,actor,i32(tracking.NULL),move),ACTIVATE,target,tm,move=PROTECT)
        def immunity(c):return _event(c,IMMUNE,target,tm,aux=jnp.where(levitate,2,0))
        def try_hit(c):
            def accuracy(c):
                c,roll=jax.lax.cond((ACCURACY[move]>0)&(~pursuit_ops.boosted(c[0],target,move,_engine())),lambda c:_draw(c,0,100,ACC),lambda c:(c,jnp.uint32(0)),c)
                base_accuracy=hook_ops.accuracy(ACCURACY[move],c[0],actor,tables.CATEGORY[move]==1)
                threshold=field_ops.accuracy(base_accuracy,c[0],actor,target,_engine())
                hit=pursuit_ops.boosted(c[0],target,move,_engine())|(ACCURACY[move]==0)|(roll<threshold.astype(jnp.uint32))
                def miss(c):
                    s,t=c;t=t._replace(events=t.events.at[move_row,7].set(1))
                    return _event((s,t),MISS,actor,mon,move,target,tm)
                def hit_fn(c):
                    s,t=c
                    existing=(~called_ops.MOVE_IMPLEMENTED[move])&(phase_ops.RESTRICTION[move]==0)&(move!=sleep_ops.YAWN)&(move!=suppression_ops.MOVE)&(~ability_ops.MOVE_IMPLEMENTED[move])&(move!=identity_ops.MOVE)&(~item_ops.EXCHANGE[move])&(~phaze_ops.STATUS[move])&(tables.CATEGORY[move]==0)&(~field_tables.IS_BOOST[move])&(move!=CONFUSE_RAY)&(s.status[target,tm]!=NONE)&(s.status[target,tm]!=jnp.where(move==WISP,BRN,jnp.where(move==TOXIC,TOX,jnp.where(move==POISON_POWDER,PSN,jnp.where(move==SPORE,SLP,PAR)))))
                    t=t._replace(events=t.events.at[move_row,7].set(jnp.where(existing,2,t.events[move_row,7])))
                    return _hit((s,t),actor,move)
                return jax.lax.cond(hit,hit_fn,miss,c)
            return jax.lax.cond(blocked,block,lambda c:jax.lax.cond(immunity_ops.absorbs(c[0],actor,target,move,_engine()),lambda c:immunity_ops.absorb(c,actor,target,move,_engine()),lambda c:jax.lax.cond(immunity_ops.wonder_guard(c[0],actor,target,move,_engine()),lambda c:_event(c,IMMUNE,target,tm,value=tables.ABILITY_IDS['wonderguard'],aux=4),lambda c:jax.lax.cond(creation_ops.immunity(c[0],target,move),lambda c:_event(c,IMMUNE,target,tm),accuracy,c),c),c),c)
        # Pinned Gen 5 resolves type/Levitate immunity before Protect;
        # status failure checks still occur after Protect and accuracy.
        def semi_miss(c):
            s,t=c;t=t._replace(events=t.events.at[move_row,7].set(1))
            return _event((s,t),MISS,actor,mon,move,target,tm)
        sound=phaze_ops.SOUND[move]&(suppression_ops.abilities(s)[target,tm]==tables.ABILITY_IDS['soundproof'])
        return jax.lax.cond(phase_ops.invulnerable(s,target,move)&(~pursuit_ops.boosted(s,target,move,_engine())),semi_miss,lambda c:jax.lax.cond(immune,immunity,lambda c:jax.lax.cond(sound,lambda c:_event(c,IMMUNE,target,tm,aux=3),try_hit,c),c),c)
    def no_target(c):
        s,t=c;t=t._replace(events=t.events.at[move_row,7].set(3))
        return _event((s,t),FAIL,actor,mon)
    def hazard(c):
        s,t=c;index=jnp.where(move==STEALTH_ROCK,0,1)
        limit=jnp.where(index==0,1,3);layers=s.hazards[target,index]
        def success(c):
            s,t=c
            hazards=s.hazards.at[target,index].set(layers+1).at[target,2].set(
                jnp.where(s.hazards[target,2]==0,index+1,s.hazards[target,2]))
            new=layers==0
            metadata=jnp.stack((s.effect_order,actor,mon))
            s=s._replace(hazards=hazards,hazard_meta=s.hazard_meta.at[target,index].set(jnp.where(new,metadata,s.hazard_meta[target,index])),
                         effect_order=s.effect_order+new.astype(jnp.int32))
            c=_event((s,t),SIDE_START,target,move=move)
            return tracking.outcome(c,actor,i32(tracking.TRUE),move)
        def fail(c):
            s,t=c;t=t._replace(events=t.events.at[move_row,7].set(2))
            return _event((s,t),FAIL,actor,mon)
        return jax.lax.cond(layers<limit,success,fail,c)
    def ordinary(c):
        return jax.lax.cond((move==STEALTH_ROCK)|(move==SPIKES),hazard,
        lambda c:jax.lax.cond((target!=actor)&(s.hp[target,tm]==0),no_target,
            lambda c:jax.lax.cond(target==actor,self_move,foe_move,c),c),c)
    def resolve_move(c):
        return jax.lax.cond((field_tables.WEATHER[move]>0)|(field_tables.SCREEN[move]>0),
        lambda c:field_ops.move(c,actor,move,move_row,_engine()),
        lambda c:jax.lax.cond(field_tables.IS_BOOST[move]&(target==actor),
            lambda c:field_ops.boost(c,actor,target,move,_engine()),lambda c:jax.lax.cond(move==called_ops.MAGIC_COAT,lambda c:called_ops.coat(c,frame,_engine()),ordinary,c),c),c)
    c,charging=jax.lax.cond(phase_ops.CHARGE[move],lambda c:phase_ops.charge(c,actor,move,move_row,_engine(),target_loc=jnp.where(target==actor,-1,1)),lambda c:(c,jnp.array(False)),c)
    move_row=jnp.max(jnp.where((jnp.arange(EVENTS)<c[1].ecount)&((c[1].events[:,0]==EV_MOVE)|(c[1].events[:,0]==ANIM)),jnp.arange(EVENTS),0))
    blocked_rest=(move==REST)&(c[0].status[actor,mon]==SLP)
    c=_when(blocked_rest,lambda c:ability_ops.fail(c,actor,move,move_row,_engine()),c)
    c,reflected=called_ops.reflect(c,frame,_engine())
    c=_when((~charging)&(~blocked_rest)&(~reflected),lambda c:jax.lax.cond((called_ops.MOVE_IMPLEMENTED[move]&(move!=called_ops.MIRROR_MOVE)&(move!=called_ops.MAGIC_COAT)),lambda c:called_ops.call(c,frame,move_row,_engine()),resolve_move,c),c)
    c=_when(hit_tables.SELF_DESTRUCT[move]&(~c[0].fainted[actor,mon]),lambda c:_faint(c,actor),c)
    c=_when(hit_tables.SELF_DESTRUCT[move]&(c[0].winner<0),lambda c:_win(c,actor),c)
    return phase_ops.after_move(c,actor,_engine())


def _switch(c,row,drag=False):
    if drag:return _switch_entry(c,row,drag=True)
    side=row[1];old=c[0].active[side]
    return jax.lax.cond(c[0].hp[side,old]>0,lambda c:pursuit_ops.before_switch(c,row,_engine()),lambda c:_switch_entry(c,row),c)


def _finish_switch(c,row):
    side=row[1];old=c[0].active[side]
    return jax.lax.cond(c[0].hp[side,old]==0,lambda c:_event(c,PURSUIT_HINT),lambda c:_switch_entry(c,row),c)


def _switch_entry(c,row,drag=False):
    s,t=c;side=row[1];old=s.active[side];slot=row[3]-4;incoming=s.order[side,slot]
    s,t=c
    if drag:
        keep=(jnp.arange(QUEUE)<s.qcount)&(~((s.queue[:,0]==ATTACK)&(s.queue[:,1]==side)&(s.queue[:,2]==old)))
        ix=jnp.nonzero(keep,size=QUEUE,fill_value=0)[0];count=jnp.sum(keep).astype(jnp.int32)
        s=s._replace(queue=jnp.where((jnp.arange(QUEUE)<count)[:,None],s.queue[ix],0),qcount=count)
    c=trigger_form_ops.end((s,t),side,_engine());s,t=c
    s=s._replace(ability_staleness=s.ability_staleness.at[side,s.active[side]].set(0))
    s=identity_ops.restore(s,side,_engine())
    s=jax.lax.cond(s.hp[side,old]>0,lambda s:tracking.clear_volatile(s,side),lambda s:s,s)
    oldslot=jnp.argmax(s.order[side]==old)
    order=s.order.at[side,oldslot].set(incoming).at[side,slot].set(old)
    s=s._replace(active=s.active.at[side].set(incoming),order=order,speed=s.speed.at[side,old].set(s.stats[side,old,5]),
                 protected=s.protected.at[side,old].set(False),stall=s.stall.at[side,old].set(0),
                 stall_duration=s.stall_duration.at[side,old].set(0),
                 status=s.status.at[side,old].set(jnp.where(s.status[side,old]==FNT,NONE,s.status[side,old])),
                 toxic=s.toxic.at[side,incoming].set(0),pivot=s.pivot.at[side].set(False))
    s=s._replace(volatile_state=s.volatile_state.at[side,old].set(0),boosts=s.boosts.at[side,old].set(0))
    s=tracking.switched(s,side,old,incoming)
    s=s._replace(base_used=s.base_used.at[side,incoming].set(0))
    c=effects.activate((s,t),side,_engine());s,t=c
    s=illusion_ops.enter(s,side,_engine())
    c=_event((s,t),DRAG if drag else SWITCH,side,incoming,value=s.hp[side,incoming],aux=s.stats[side,incoming,0],
             status=s.status[side,incoming],move=row[6])
    s,t=c
    # runSwitch precedes ordinary moves and remaining simultaneous swaps.
    # Native queue.insertChoice updates the incoming Pokemon's Speed before
    # resolving/sorting its runSwitch callback, including forced replacements.
    s=s._replace(speed=s.speed.at[side,incoming].set(_speeds(s)[side,incoming]))
    if drag:return _run_switch((s,t),jnp.arange(2)==side)
    s=_qappend(s,jnp.array([RUN_SWITCH,0,0,0,0,0,0],jnp.int32).at[1].set(side).at[2].set(incoming).at[5].set(s.speed[side,incoming]))
    return _qsort((s,t))


def _run_switch(c,entrants):
    s,t=c
    speeds=s.speed[jnp.arange(2),s.active]
    c,value=jax.lax.cond(speeds[0]==speeds[1],lambda c:_draw(c,0,2,SWITCH_SORT,2),
                         lambda c:(c,jnp.uint32(0)),(s,t))
    first=jnp.where(speeds[0]==speeds[1],value.astype(jnp.int32),jnp.argmax(speeds))
    for k in range(2):
        side=jnp.where(k==0,first,1-first)
        def entry(c):
            s,t=c;m=s.active[side]
            c=effects.sleep_switch(c,side,_engine());s,t=c
            for n in range(2):
                index=jnp.where(n==0,s.hazards[side,2]-1,2-s.hazards[side,2])
                index=jnp.clip(index,0,1)
                layers=s.hazards[side,index]
                grounded=(suppression_ops.abilities(s)[side,m]!=3)&(~jnp.any(s.types[side,m]==generic.TYPES['flying']))
                rock_exp,_=generic.type_matchup(i32(generic.TYPES['rock']),s.types[side,m])
                rock=s.stats[side,m,0]*jnp.left_shift(i32(1),jnp.maximum(0,rock_exp))//(8*jnp.left_shift(i32(1),jnp.maximum(0,-rock_exp)))
                amount=jnp.where(index==0,rock,
                                 s.stats[side,m,0]//jnp.maximum(1,10-2*layers))
                c=_when((layers>0)&((index==0)|grounded)&(c[0].hp[side,m]>0),
                        lambda c:_damage(c,side,jnp.maximum(1,amount),cause=jnp.where(index==0,4,5)),c)
            return c
        c=_when(entrants[side],entry,c)
        c=_when(entrants[side]&(c[0].hp[side,c[0].active[side]]>0),lambda c:hook_ops.start(c,side,_engine()),c)
    # Native resolves entry faint/win after the SwitchIn callbacks,
    # before the ordinary per-action Update and pending attacks.
    for k in range(2):
        side=jnp.where(k==0,first,1-first);s,t=c;m=s.active[side]
        c=_when(entrants[side]&(s.hp[side,m]==0)&(~s.fainted[side,m]),lambda c:_win(_faint(c,side),side),c)
    return c


def _residual(c):
    c=_event(c,BLANK);s,t=c;s=_refresh_speed(s);c=(s,t)
    # Native field handlers: status, protect, stall, item in insertion order.
    # Duration-only handlers sort last (false order), and still consume shuffle
    # draws even when their callback does nothing.
    rows=[];valid=[]
    for side in range(2):
        m=s.active[side];live=~s.fainted[side,m];status=s.status[side,m]
        rows.extend([jnp.array([1,side,0,0,0,0],jnp.int32).at[2].set(m).at[3].set(jnp.where((status==TOX)|(status==PSN),9,10)).at[4].set(s.speed[side,m]),
                     jnp.array([2,side,0,10000,0,2],jnp.int32).at[2].set(m).at[4].set(s.speed[side,m]),
                     jnp.array([3,side,0,10000,0,2],jnp.int32).at[2].set(m).at[4].set(s.speed[side,m]),
                     jnp.array([4,side,0,5,0,4],jnp.int32).at[2].set(m).at[4].set(s.speed[side,m])])
        # fieldEvent collects status/item callbacks before checking fainted holders.
        # A fainted holder still participates in handler shuffle draws, then
        # effect() below skips its callback. Fainting clears volatiles; native
        # checkFainted changes status to fnt only after the residual action.
        valid.extend([(status==TOX)|(status==BRN)|(status==PSN),live&s.protected[side,m],live&(s.stall_duration[side,m]>0),s.item[side,m]==1])
        rows.append(i32([8,side,0,10000,0,2]).at[2].set(m).at[4].set(s.speed[side,m]));valid.append(live&(s.volatile_state[side,m,5,0]>0))
        rows.append(i32([7,side,0,10000,0,2]).at[2].set(m).at[4].set(s.speed[side,m]));valid.append(live&(s.volatile_state[side,m,phase_ops.RECHARGE_SLOT,0]>0))
    rows.append(i32([5,0,0,1,0,0]));valid.append(s.weather_state[0]>0)
    for side in range(2):
        for index in range(2):
            rows.append(i32([6,side,index,26,0,index+1]));valid.append(s.screens[side,index,0]>0)
    rows=[jnp.pad(row,(0,1)) for row in rows]
    for side in range(2):
        mon=s.active[side]
        for kind in range(6,len(phase_ops.VOLATILES)):
            rows.append(i32([9,side,0,10000,0,0 if kind==partial_ops.SLOT else 2,kind]).at[3].set(phase_ops.RESIDUAL_ORDER[kind]).at[2].set(mon).at[4].set(s.speed[side,mon]))
            valid.append((~s.fainted[side,mon])&(s.volatile_state[side,mon,kind,0]>0)&(s.volatile_state[side,mon,kind,5]>0))
    for side in range(2):
        mon=s.active[side]
        rows.append(i32([10,side,0,29,0,0,0]).at[2].set(mon).at[4].set(s.speed[side,mon]));valid.append(suppression_ops.abilities(s)[side,mon]==trigger_form_ops.ZEN)
    handlers=len(rows)
    rows=jnp.stack(rows);valid=jnp.stack(valid);count=jnp.sum(valid).astype(jnp.int32)
    order=jnp.where(valid,rows[:,3],20000)
    volatile_kind=jnp.where(rows[:,0]==2,0,jnp.where(rows[:,0]==3,1,jnp.where(rows[:,0]==7,4,jnp.where(rows[:,0]==8,5,rows[:,6]))))
    is_volatile=(rows[:,0]==2)|(rows[:,0]==3)|(rows[:,0]==7)|(rows[:,0]==8)|(rows[:,0]==9)
    creation=jnp.where(is_volatile,s.volatile_state[rows[:,1],rows[:,2],volatile_kind,1],jnp.arange(handlers))
    idx=jnp.lexsort((creation,rows[:,1],rows[:,5],-rows[:,4],order));rows=rows[idx]
    def shuffle_body(k,carry):
        c,rows=carry
        # Resolve each equal group once, using native shuffle(from,to) calls.
        key=rows[k,3:6]
        eq=jnp.all(rows[:,3:6]==key,axis=1)&(jnp.arange(handlers)<count)
        first=jnp.min(jnp.where(eq,jnp.arange(handlers),handlers));end=jnp.max(jnp.where(eq,jnp.arange(handlers),-1))+1
        def shuffle(carry):
            c,rows=carry;c,value=_draw(c,k,end,FIELD_SORT,2);v=value.astype(jnp.int32)
            a,b=rows[k],rows[v];rows=rows.at[k].set(b).at[v].set(a)
            return c,rows
        return jax.lax.cond((k>=first)&(k+1<end)&(k<count),shuffle,lambda x:x,(c,rows))
    c,rows=jax.lax.fori_loop(0,handlers-1,shuffle_body,(c,rows))
    def effect(k,c):
        s,t=c;row=rows[k];kind,side,m=row[:3]
        def apply(c):
            def status_tick(c):
                s,t=c;tox=s.status[side,m]==TOX
                stage=jnp.minimum(15,s.toxic[side,m]+1)
                amount=jnp.where(tox,jnp.maximum(1,s.stats[side,m,0]//16)*stage,s.stats[side,m,0]//8)
                s=s._replace(toxic=s.toxic.at[side,m].set(jnp.where(tox,stage,s.toxic[side,m])))
                c=_damage((s,t),side,amount,cause=jnp.where(tox|(s.status[side,m]==PSN),2,1));s,t=c
                c=_when(s.hp[side,m]==0,lambda c:_faint(c,side),c)
                return _win(c,side)
            def duration(c):
                s,t=c
                def protect(c):
                    s,t=c;return s._replace(protected=s.protected.at[side,m].set(False),volatile_state=s.volatile_state.at[side,m,0].set(0)),t
                def stall(c):
                    s,t=c;duration=jnp.maximum(0,s.stall_duration[side,m]-1)
                    return s._replace(stall_duration=s.stall_duration.at[side,m].set(duration),
                                     stall=s.stall.at[side,m].set(jnp.where(duration==0,0,s.stall[side,m])),
                                     volatile_state=s.volatile_state.at[side,m,1].set(jnp.where(duration==0,0,s.volatile_state[side,m,1]))),t
                return jax.lax.cond(kind==2,protect,stall,c)
            def leftovers(c):
                s,t=c;hp=s.hp[side,m];maxhp=s.stats[side,m,0]
                def heal(c):
                    s,t=c;newhp=jnp.minimum(maxhp,hp+jnp.maximum(1,maxhp//16));s=s._replace(hp=s.hp.at[side,m].set(newhp))
                    return _event((s,t),HEAL,side,m,value=newhp,aux=maxhp,status=s.status[side,m],move=1)
                return _when(hp<maxhp,heal,c)
            return jax.lax.cond(kind==10,lambda c:trigger_form_ops.residual(c,side,_engine()),lambda c:jax.lax.cond(kind==9,lambda c:jax.lax.cond(row[6]==partial_ops.SLOT,lambda c:partial_ops.tick(c,side,_engine()),lambda c:jax.lax.cond(row[6]==sleep_ops.SLOT,lambda c:sleep_ops.tick(c,side,_engine()),lambda c:phase_ops.extra_tick(c,side,row[6],_engine()),c),c),
                lambda c:jax.lax.cond(kind==8,lambda c:phase_ops.tick_phase(c,side,5,_engine()),
                lambda c:jax.lax.cond(kind==7,lambda c:phase_ops.tick(c,side,m),
                lambda c:jax.lax.cond(kind==5,lambda c:field_ops.weather_tick(c,_engine()),
                lambda c:jax.lax.cond(kind==6,lambda c:field_ops.screen_tick(c,side,m,_engine()),
                    lambda c:jax.lax.cond(kind==1,status_tick,lambda c:jax.lax.cond(kind==4,leftovers,duration,c),c),c),c),c),c),c),c)
        return _when((k<count)&((kind==5)|(kind==6)|(~s.fainted[side,m]))&(s.winner<0),apply,c)
    c=jax.lax.fori_loop(0,handlers,effect,c)
    s,t=c
    return _when(s.winner<0,lambda c:_event(c,UPKEEP),c)


def _new_request(c,request):
    s,t=c;s=hook_ops.normalize_locks(s)
    for side in range(2):
        mon=s.active[side]
        status=jnp.where((request[side]==FORCED)&s.fainted[side,mon],FNT,s.status[side,mon])
        s=s._replace(status=s.status.at[side,mon].set(status))
    for side in range(2):
        m=s.active[side]
        value=jnp.where((request[side]==MOVE)&phase_ops.locked(s)[side],1,s.action_state[side,m,tracking.TRAPPED])
        s=tracking.put(s,side,tracking.TRAPPED,value)
        s=tracking.put(s,side,tracking.MAYBE_TRAPPED,jnp.where((request[side]==MOVE)&phase_ops.locked(s)[side],0,s.action_state[side,m,tracking.MAYBE_TRAPPED]))
    return s._replace(request_state=jnp.zeros((2,2),jnp.int32),request=request,pending=jnp.full(2,-1,jnp.int32),episode=s.episode+1),t


def _finish(c):
    s,t=c
    s=s._replace(midturn=jnp.array(False),turn=s.turn+1,queue=jnp.zeros_like(s.queue),qcount=i32(0))
    s=trap_ops.refresh(tracking.new_turn(s),_engine())
    c=(s,t)
    warning=((s.turn>=500)&(s.turn%100==0))|((s.turn>=900)&(s.turn%10==0))|(s.turn>=990)
    def terminal(c):
        c=_event(c,TURN_LIMIT);c=_event(c,BLANK);c=_event(c,WIN,value=0)
        s,t=c
        return s._replace(winner=i32(0),request=jnp.full(2,END,jnp.int32)),t
    def live(c):
        c=_when(warning,lambda c:_event(c,TURN_WARNING,value=s.turn),c)
        return _new_request(_event(c,TURN,value=s.turn),jnp.full(2,MOVE,jnp.int32))
    return jax.lax.cond(s.turn>1000,terminal,live,c)


def _run_queue(c):
    def condition(c):
        s,t=c
        intercept=(s.queue[0,0]==ATTACK)&(s.queue[0,6]==pursuit_ops.MOVE)
        return (s.qcount>0)&((s.winner<0)|(s.queue[0,0]==FINISH_SWITCH))&((~jnp.any(s.pivot))|intercept)&(~t.overflow)
    def step(c):
        s,t=c;row=s.queue[0];kind=row[0]
        s=s._replace(queue=jnp.concatenate((s.queue[1:],jnp.zeros((1,7),jnp.int32))),qcount=s.qcount-1)
        started_move=(s.active[row[1]]==row[2])&(~s.fainted[row[1],row[2]])
        scheduling=((kind==SWAP)|(kind==INSTANT))&pursuit_ops.eligible(s,row[1],_engine())
        intercepted=(kind==ATTACK)&(row[6]==pursuit_ops.MOVE)
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
            entrants=jnp.any(run[:,None]&(q[:,1,None]==jnp.arange(2)),axis=0).at[row[1]].set(True)
            q=jnp.where((jnp.arange(QUEUE)+count)[:,None]<s.qcount,q[jnp.minimum(jnp.arange(QUEUE)+count,QUEUE-1)],0)
            s=s._replace(queue=q,qcount=s.qcount-count)
            return _run_switch((s,t),entrants)
        c=jax.lax.switch(jnp.clip(kind-1,0,7),(before,move,swap,swap,run_switch,_residual,lambda c:pursuit_ops.prepare(c,row,_engine()),lambda c:_finish_switch(c,row)),c)
        c=phaze_ops.resolve(c,_engine())
        s,t=c
        # A skipped fainted action returns before the native per-action Update.
        skipped=(kind==ATTACK)&((~started_move)|s.fainted[row[1],row[2]])
        next_instant=(s.qcount>0)&(s.queue[0,0]==INSTANT)
        c=_when((s.winner<0)&(~skipped)&(~next_instant)&(~scheduling),lambda c:_update(c,UPDATE),c)
        s,t=c
        c=_when(intercepted,lambda c:pursuit_ops.complete(c,1-row[1],_engine()),c)
        for side in range(2):
            c=_when(c[0].pivot[side]&(c[0].winner<0),lambda c:pursuit_ops.pivot(c,i32(side),_engine()),c)
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
    available=(~(hook_ops.disabled(s)|phase_ops.disabled(s))[side,mon])&(moves>0)
    recharge=phase_ops.locked(s)
    ordinary=jnp.where(recharge[:,None],jnp.arange(SLOTS)[None,:]==0,available)&(s.request[:,None]==MOVE)
    bench=_live(s)[side[:,None],s.order]&(s.order!=mon[:,None])
    switches=bench&(((s.request[:,None]==MOVE)&(~recharge[:,None])&trap_ops.selectable_switches(s,_engine())[:,None])|(s.request[:,None]==FORCED))
    struggle=jnp.all(~available,axis=1)&(~recharge)&(s.request==MOVE)
    return jnp.concatenate((ordinary,switches,struggle[:,None]),axis=1)


def _supported(s,initializing=False):
    occupied=s.species>0
    safe_species=jnp.clip(s.species,0,len(SPECIES)-1)
    base=derive_stored_stats(BASE_STATS[safe_species],s.level,s.ivs,s.evs,
                            generic.NATURE_MULTIPLIERS[s.nature],tables.SHEDINJA[safe_species])
    stat_ok=(s.call_depth==0)&jnp.all(s.call_frames==0)&jnp.all((s.ability_staleness>=0)&(s.ability_staleness<=1))&identity_ops.valid(s,_engine())&jnp.all((s.illusion>=-1)&(s.illusion<ROSTER))&jnp.all((s.ability_effect_id>=0)&(s.ability_effect_id<len(ABILITIES)))
    slots=jnp.arange(4)
    duplicate=(s.moves[:,:,:,None]==s.moves[:,:,None,:])&(s.moves[:,:,:,None]>0)&(slots[:,None]<slots[None,:])
    valid_order=jnp.all(jnp.sort(s.order,axis=1)==jnp.arange(ROSTER))
    weather,source,source_mon,source_move,duration,target=s.weather_state
    weather_sources=(source>=0)&(source<2)&(source_mon>=0)&(source_mon<ROSTER)&(source_move>=0)&(source_move<len(MOVE_KEYS))
    weather_sources=weather_sources&(s.species[jnp.clip(source,0,1),jnp.clip(source_mon,0,ROSTER-1)]>0)
    move_weather=(source_move>0)&(field_tables.WEATHER[jnp.clip(source_move,0,len(MOVE_KEYS)-1)]==weather)&(duration>=1)
    # Native permanent weather retains its creator after that mon loses the
    # weather ability. Creation-time source occupancy survives ability mutation.
    ability_weather=(source_move==0)&(duration==0)
    weather_ok=jnp.where(weather>0,weather_sources&(move_weather|ability_weather),
                         (source==-1)&(source_mon==-1)&(source_move==0)&(duration==0))
    screen_side=s.screens[:,:,1];screen_mon=s.screens[:,:,2];screen_move=s.screens[:,:,3]
    screen_sources=(screen_side==jnp.arange(2)[:,None])&(screen_mon>=0)&(screen_mon<ROSTER)
    screen_sources=screen_sources&(screen_move==i32(field_tables.SCREEN_MOVES)[None,:])
    screen_sources=screen_sources&(s.species[jnp.clip(screen_side,0,1),jnp.clip(screen_mon,0,ROSTER-1)]>0)
    screen_sources=screen_sources&(s.screens[:,:,4]>=0)&(s.screens[:,:,4]<s.effect_order)
    screen_ok=jnp.all(jnp.where(s.screens[:,:,0]>0,screen_sources,jnp.all(s.screens==0,axis=-1)))
    hazard_side=s.hazard_meta[:,:,1];hazard_mon=s.hazard_meta[:,:,2]
    hazard_sources=(hazard_side>=0)&(hazard_side<2)&(hazard_mon>=0)&(hazard_mon<ROSTER)
    hazard_sources=hazard_sources&(s.species[jnp.clip(hazard_side,0,1),jnp.clip(hazard_mon,0,ROSTER-1)]>0)
    hazard_sources=hazard_sources&(s.hazard_meta[:,:,0]<s.effect_order)
    hazard_ok=jnp.all(jnp.where(s.hazards[:,:2]>0,hazard_sources,jnp.all(s.hazard_meta==0,axis=-1)))
    lock=s.volatile_state[:,:,3];source_item=-lock[:,:,4]
    lock_ok=(lock[:,:,4]<0)&hook_tables.CHOICE[jnp.clip(source_item,0,len(ITEMS)-1)]
    lock_ok=lock_ok&(source_item==s.item)&(lock[:,:,5]>0)&(lock[:,:,5]<len(MOVE_KEYS))
    lock_ok=lock_ok&(lock[:,:,1]<s.effect_order)
    lock_ok=jnp.all((lock[:,:,0]==0)|lock_ok)
    move_mask=(tables.MOVE_IMPLEMENTED|field_tables.MOVE_IMPLEMENTED|hit_tables.MOVE_IMPLEMENTED|phase_ops.MOVE_IMPLEMENTED|partial_ops.MOVE_IMPLEMENTED|phaze_ops.MOVE_IMPLEMENTED|pursuit_ops.MOVE_IMPLEMENTED|item_ops.MOVE_IMPLEMENTED|identity_ops.MOVE_IMPLEMENTED|trigger_form_ops.MOVE_IMPLEMENTED|ability_ops.MOVE_IMPLEMENTED|suppression_ops.MOVE_IMPLEMENTED|sleep_ops.MOVE_IMPLEMENTED|called_ops.MOVE_IMPLEMENTED)
    cached=jnp.concatenate((s.move_history[:,:,:,0].reshape(-1),s.battle_move[:1]))
    cached_ok=jnp.all((cached==0)|(cached==STRUGGLE)|move_mask[jnp.clip(cached,0,len(MOVE_KEYS)-1)])
    literal=s.queue[:,3]-FORCED_MOVE_ACTION
    used_literal=(jnp.arange(QUEUE)<s.qcount)&(s.queue[:,0]==ATTACK)&(s.queue[:,3]>=FORCED_MOVE_ACTION)
    literal_ok=jnp.all((~used_literal)|((literal>0)&(literal<len(MOVE_KEYS))&move_mask[jnp.clip(literal,0,len(MOVE_KEYS)-1)]))
    return (jnp.all((s.ability_seek>=0)&(s.ability_seek<=1)&((s.ability_seek==0)|(s.ability_effect_id==trace_ops.TRACE)))&jnp.all((s.item_provenance[:,:,0]>=0)&(s.item_provenance[:,:,0]<len(ITEMS)))&jnp.all((s.item_provenance[:,:,1]>=0)&(s.item_provenance[:,:,1]<=2))&jnp.all((s.request_state>=0)&(s.request_state<=1))&tracking.valid(s,_engine())&stat_ok&cached_ok&literal_ok&valid_order&weather_ok&screen_ok&hazard_ok&lock_ok&(~jnp.any(duplicate))&(jnp.all((s.hcount>=0)&(s.hcount<=HISTORY)) if isinstance(s,State) else jnp.array(True))
            &jnp.all((s.qcount>=0)&(s.qcount<=QUEUE))&(s.episode>=0)&(s.generation>=0)
            &jnp.all(jnp.where(occupied,jnp.any(s.moves>0,axis=-1),jnp.all(s.moves==0,axis=-1)))
            &jnp.all((s.species>=0)&(s.species<len(SPECIES)))&jnp.all(jnp.any(occupied,axis=1))
            &jnp.all((~occupied)|(tables.SPECIES_IMPLEMENTED|form_ops.SPECIES_IMPLEMENTED|trigger_form_ops.SPECIES_IMPLEMENTED)[safe_species])
            &jnp.all((s.moves>=0)&(s.moves<len(MOVE_KEYS))&(s.moves!=STRUGGLE))
            &jnp.all((s.moves==0)|move_mask[jnp.clip(s.moves,0,len(MOVE_KEYS)-1)])
            &jnp.all((s.item>=0)&(s.item<len(ITEMS)))
            &jnp.all((hook_tables.ITEM_IMPLEMENTED|hit_tables.ITEM_IMPLEMENTED|phase_ops.ITEM_IMPLEMENTED|partial_ops.ITEM_IMPLEMENTED)[jnp.clip(s.item,0,len(ITEMS)-1)])
            &jnp.all((s.ability>=0)&(s.ability<len(ABILITIES)))&jnp.all((hook_tables.ABILITY_IMPLEMENTED|hit_tables.ABILITY_IMPLEMENTED|trap_ops.ABILITY_IMPLEMENTED|phaze_ops.ABILITY_IMPLEMENTED|item_ops.ABILITY_IMPLEMENTED|identity_ops.ABILITY_IMPLEMENTED|illusion_ops.ABILITY_IMPLEMENTED|form_ops.ABILITY_IMPLEMENTED|trigger_form_ops.ABILITY_IMPLEMENTED|immunity_ops.ABILITY_IMPLEMENTED|creation_ops.ABILITY_IMPLEMENTED|sleep_ops.ABILITY_IMPLEMENTED|called_ops.ABILITY_IMPLEMENTED|trace_ops.ABILITY_IMPLEMENTED)[jnp.clip(s.ability,0,len(ABILITIES)-1)])
            &jnp.all((s.ivs>=0)&(s.ivs<=31))&jnp.all((s.evs>=0)&(s.evs<=255))
            &jnp.all((s.nature>=1)&(s.nature<=len(generic.NATURES)))
            &jnp.all(s.types==tables.SPECIES_TYPES[safe_species])&jnp.all((s.boosts>=-6)&(s.boosts<=6))
            &jnp.all((s.happiness>=0)&(s.happiness<=255))
            &jnp.all((s.gender>=(-1 if initializing else 0))&(s.gender<=2))
            &jnp.all((s.name>=0)&(s.name<len(SPECIES)))&jnp.all((~occupied)|(s.name>0))
            &jnp.all((~occupied)|((s.hp_type>=1)&(s.hp_type<=len(generic.TYPES))))
            &jnp.all(s.hp_power==jnp.where(occupied,generic.hidden_power(s.ivs)[1],0))
            &jnp.all((s.hazards>=0)&(s.hazards<=i32([1,3,2])))
            &jnp.all((s.hazards[:,2]==0)==jnp.all(s.hazards[:,:2]==0,axis=1))
            &jnp.all((s.hazards[:,2]!=1)|(s.hazards[:,0]>0))
            &jnp.all((s.hazards[:,2]!=2)|(s.hazards[:,1]>0))
            &jnp.all((s.hp>=0)&(s.hp<=s.stats[:,:,0]))&jnp.all((s.pp>=0)&(s.pp<=identity_ops.maximum_pp(s)))
            &jnp.all((s.status>=0)&(s.status<=FRZ))
            &(s.effect_order>=0)&phase_ops.valid(s)&phase_ops.valid_phases(s)
            &jnp.all((s.action_state[:,:,tracking.ATTACK_COUNT]>=0)&(s.action_state[:,:,tracking.ATTACK_COUNT]<=tracking.ATTACK_RECORDS))
            &(s.weather_state[0]>=0)&(s.weather_state[0]<=4)&(s.weather_state[4]>=0)&(s.weather_state[4]<=8)
            &(s.weather_state[1]>=-1)&(s.weather_state[1]<=1)&(s.weather_state[2]>=-1)&(s.weather_state[2]<ROSTER)
            &(s.weather_state[5]>=0)&(s.weather_state[5]<=1)
            &jnp.all((s.screens[:,:,0]>=0)&(s.screens[:,:,0]<=8))
            &jnp.all(s.hazard_meta[:,:,0]>=0)&jnp.all(s.ability_order>=0)&jnp.all(s.item_order>=0)
            &jnp.all((s.status_state[:,:,0]>=0)&(s.status_state[:,:,0]<=FRZ))
            &jnp.all(s.status_state[:,:,1]>=0)
            &jnp.all((s.status_state[:,:,2]>=-1)&(s.status_state[:,:,2]<=1))
            &jnp.all((s.status_state[:,:,3]>=-1)&(s.status_state[:,:,3]<ROSTER))
            &jnp.all((s.status_state[:,:,4]>=0)&(s.status_state[:,:,4]<=1))
            &jnp.all((s.status_state[:,:,5:]>=-1)&(s.status_state[:,:,5:]<=4))
            &jnp.all((s.volatile_state[:,:,:,0]>=0)&(s.volatile_state[:,:,:,0]<=1))
            &jnp.all(s.volatile_state[:,:,:,1]>=0)
            &jnp.all((s.volatile_state[:,:,:,2]>=0)&(s.volatile_state[:,:,:,2]<=1))
            &jnp.all((s.volatile_state[:,:,:,3]>=0)&(s.volatile_state[:,:,:,3]<ROSTER))
            &jnp.all((s.volatile_state[:,:,:3,4]>=0)&(s.volatile_state[:,:,:3,4]<len(MOVE_KEYS)))
            &jnp.all((s.volatile_state[:,:,:3,5]>=0)&(s.volatile_state[:,:,:3,5]<=5))
            &jnp.all((s.volatile_state[:,:,3,4]>-len(ITEMS))&(s.volatile_state[:,:,3,4]<=0))
            &jnp.all((s.volatile_state[:,:,3,5]>=0)&(s.volatile_state[:,:,3,5]<len(MOVE_KEYS)))&jnp.all((s.toxic>=0)&(s.toxic<=15))
            &jnp.all((~occupied)|(s.level>=1))
            &jnp.all((~occupied)|(s.level<=100))&jnp.all((s.request>=MOVE)&(s.request<=END))
            &jnp.all((s.active>=0)&(s.active<ROSTER))&jnp.all(s.species[jnp.arange(2),s.active]>0))



def _validate_structure(s):
    if not isinstance(s,(State,CoreState)):raise ValueError('expected shared-engine State or CoreState')
    roster_fields={'species','level','speed','gender','hp','item','status','toxic','stall','stall_duration',
                   'ability_seek','ability_staleness','illusion','ability_effect_id','base_species','base_ability','base_used','transformed','known_type','nature','ability','hp_type','hp_power','shiny','happiness','name','ability_order','item_order'}
    vector_fields={'active','pending','request','pivot'}
    bool_fields={'fainted','protected','pivot','midturn','shiny','sleep_clause_target','transformed','known_type'}
    for name,value in zip(type(s)._fields,s):
        shape=(called_ops.FRAMES,called_ops.WIDTH) if name=='call_frames' else (2,6) if name in roster_fields or name in {'fainted','protected'} else (2,) if name in vector_fields else (2,3) if name=='hazards' else (2,6,6) if name in {'stats','ivs','evs','base_stats'} else (2,6,2) if name in ('types','item_provenance','apparent_types') else (2,6,7) if name in {'boosts','status_state'} else (2,6,len(phase_ops.VOLATILES),phase_ops.VOLATILE_WIDTH) if name=='volatile_state' else (2,6,3,4) if name=='move_history' else (2,6,tracking.COLUMNS) if name=='action_state' else (2,6,tracking.ATTACK_RECORDS,7) if name=='attacked_by' else (2,2) if name in ('side_faints','request_state') else (4,) if name=='battle_move' else (6,) if name=='weather_state' else (2,2,5) if name=='screens' else (2,2,3) if name=='hazard_meta' else (2,6,4) if name in {'moves','pp','base_moves','base_pp'} else (2,6) if name=='order' else (QUEUE,7) if name=='queue' else (4,) if name=='rng' else (HISTORY,EVENT_COLUMNS) if name=='history' else ()
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
        switching=(choices>=4)&(choices<10)&(s.request==MOVE)
        lock=switching&(s.action_state[jnp.arange(2),s.active,tracking.MAYBE_TRAPPED]>0)
        cant=jnp.where((choices>=0)|(choices==-2),lock.astype(jnp.int32),s.request_state[:,1])
        s=s._replace(pending=pending,request_state=s.request_state.at[:,1].set(cant))
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
                    forced=phase_ops.forced_move(s)[side]
                    actual_slot=jnp.argmax(s.moves[side,mon]==forced)
                    owns_forced=jnp.any(s.moves[side,mon]==forced)
                    forced_action=jnp.where(owns_forced,actual_slot,FORCED_MOVE_ACTION+forced)
                    action=jnp.where((kind==ATTACK)&(forced>0),forced_action,action)
                    move=jnp.where((kind==ATTACK)&(forced>0),forced,move)
                    recharge=(kind==ATTACK)&(forced<0)
                    action=jnp.where(recharge,phase_ops.RECHARGE_ACTION,action)
                    priority=jnp.where((kind==ATTACK)&(action!=10)&(~recharge),hook_ops.priority(move,suppression_ops.abilities(s)[side,mon],_engine()),0)
                    source=jnp.where(s.pivot[side],U_TURN,0)
                    row=jnp.stack([kind,i32(side),mon,action,priority,s.speed[side,mon],source])
                    s=_qappend(s,row)
                    def prepare(s):return _qappend(s,i32([BEFORE_MOVE,0,0,0,0,0,0]).at[1].set(side).at[2].set(mon).at[3].set(action).at[5].set(s.speed[side,mon]))
                    s=jax.lax.cond(normal&(kind==ATTACK)&(move==pursuit_ops.MOVE)&(~recharge),prepare,lambda s:s,s)
                    return s,t
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
        overflow=t.overflow|(s.episode<0)
        if isinstance(s,State):overflow=overflow|(s.hcount+t.ecount>HISTORY)
        def success(_):
            if isinstance(s,CoreState):return Result(s,jnp.where(ready,OK,PENDING),t)
            def add(k,s):return s._replace(history=s.history.at[s.hcount+k].set(t.events[k]))
            s2=jax.lax.fori_loop(0,t.ecount,add,s)
            return Result(s2._replace(hcount=s.hcount+t.ecount),jnp.where(ready,OK,PENDING),t)
        return jax.lax.cond(overflow,lambda _:Result(original,i32(CAPACITY),empty_trace()),success,None)
    original=s
    errors=trap_ops.private_error(s,choices,mask,_engine())
    any_error=jnp.any(errors>0)
    # Native retry episodes isolate a side's rejected command. Keep ambiguous
    # combined invalid/valid submissions atomic, rather than inventing ordering.
    isolated=(jnp.sum((choices>=0)|(choices==-2))==1)&jnp.all((choices>=-2)&(choices<=10))
    retry=any_error&isolated
    def rejected(_):
        clear=(errors==1)|(errors==2)
        pending=jnp.where(clear,-1,s.pending)
        disclosed=jnp.where(errors==1,1,s.request_state[:,0])
        changed=jnp.any(disclosed!=s.request_state[:,0])|jnp.any(pending!=s.pending)
        state=s._replace(pending=pending,request_state=s.request_state.at[:,0].set(disclosed),episode=s.episode+changed.astype(jnp.int32))
        return jax.lax.cond(state.episode<0,lambda _:Result(s,i32(CAPACITY),empty_trace()),
            lambda _:Result(state,i32(RETRY),empty_trace()._replace(private_errors=errors)),None)
    code=jnp.where(stale,STALE,jnp.where(s.winner>=0,TERMINAL,jnp.where(~_supported(s),UNSUPPORTED,jnp.where((invalid|any_error)&(~retry),INVALID,-1))))
    return jax.lax.cond(code>=0,lambda _:Result(s,code,empty_trace()),
        lambda _:jax.lax.cond(retry,rejected,lambda _:accepted((s,t)),None),None)



def advance_core(s,choices,token=None):
    if not isinstance(s,CoreState):raise ValueError('expected CoreState')
    return advance(s,choices,token)


def make_state(teams,seed=(1,2,3,4),generation=0):
    """Host schema validation plus pure JAX initialization from declared sets.

    Sets accept species, moves, level, speed_evs and item only. Unknown fields
    are rejected, preventing silently ignored effects. Trusted boundary fixtures
    are installed explicitly by tests, never treated as ordinary initial sets.
    """
    if len(teams)!=2 or any(not 1<=len(team)<=ROSTER for team in teams):raise ValueError('two teams of one to six mons required')
    species=np.zeros((2,6),np.int32);level=np.zeros((2,6),np.int32);moves=np.zeros((2,6,4),np.int32)
    item=np.zeros((2,6),np.int32);evs=np.zeros((2,6,6),np.int32)
    for side,team in enumerate(teams):
        for mon,desc in enumerate(team):
            if set(desc)-{'species','moves','level','speed_evs','item'}:raise ValueError('unsupported set field')
            if type(desc.get('level',50)) is not int:raise ValueError('level must be an integer')
            species[side,mon]=SPECIES.index(desc['species']);level[side,mon]=desc.get('level',50)
            if not 1<=species[side,mon]<=3 or not 1<=level[side,mon]<=100:raise ValueError('unsupported species/level')
            speed=desc.get('speed_evs',0)
            if type(speed) is not int or speed not in (0,252):raise ValueError('unsupported Speed EVs')
            evs[side,mon,5]=speed
            if not 1<=len(desc['moves'])<=4 or len(set(desc['moves']))!=len(desc['moves']):raise ValueError('one to four distinct moves required')
            for slot,name in enumerate(desc['moves']):
                mid=MOVE_NAMES.index(name)
                if mid==0 or mid==STRUGGLE or mid>RAPID_SPIN:raise ValueError('unsupported move')
                moves[side,mon,slot]=mid
            item[side,mon]=ITEMS.index(desc.get('item',''))
            if item[side,mon]>2:raise ValueError('unsupported legacy item')
    seed=np.asarray(seed)
    if seed.shape!=(4,) or seed.dtype.kind not in 'iu' or np.any(seed<0) or np.any(seed>65535):raise ValueError('seed must have four 16-bit limbs')
    if not isinstance(generation,int) or not 0<=generation<2**31-1:raise ValueError('invalid generation')
    return initialize(i32(species),i32(level),i32(moves),i32(item),i32(evs),jnp.asarray(seed,jnp.uint16),i32(generation))


def initialize(species,level,moves,item,evs,seed,generation):
    """Compatibility constructor for schema 1/2 declared custom sets."""
    ivs=jnp.full((2,6,6),31,jnp.int32)
    nature=jnp.full((2,6),generic.NATURES['hardy'],jnp.int32)
    hp_type=generic.HP_TYPE_IDS[generic.hidden_power(ivs)[0]]
    return _initialize(species,level,moves,item,evs,seed,generation,ivs,nature,
                       tables.SPECIES_ABILITY[species],tables.SPECIES_GENDER[species],
                       jnp.zeros((2,6),jnp.bool_),jnp.full((2,6),255,jnp.int32),
                       tables.SPECIES_NAME[species],hp_type)


def make_gen5_state(teams,seed=(1,2,3,4),generation=0):
    """General set/data schema, with explicit current handler preflight.

    Unsupported initialization returns UNSUPPORTED with unconsumed input RNG
    and no public events. No generated team is filtered or silently simulated
    with missing callbacks. Callback coverage remains in the G4 ledger.
    """
    sets=generic.decode_sets(teams)
    seed=np.asarray(seed)
    if seed.shape!=(4,) or seed.dtype.kind not in 'iu' or np.any(seed<0) or np.any(seed>65535):raise ValueError('seed must have four 16-bit limbs')
    if type(generation) is not int or not 0<=generation<2**31-1:raise ValueError('invalid generation')
    species=tables.G_SPECIES[sets.species]
    names=np.asarray(tables.SPECIES_NAME[species]).copy()
    for side,team in enumerate(teams):
        for mon,desc in enumerate(team):
            if desc.get('name'):
                name=generic.to_id(desc['name'])
                if name not in tables.SPECIES_IDS:raise ValueError('arbitrary nickname encoding pending')
                names[side,mon]=tables.SPECIES_IDS[name]
    values=generic.derive_set_stats(sets)
    return _initialize(species,sets.level,tables.G_MOVES[sets.moves],tables.G_ITEMS[sets.item],
                        sets.evs,jnp.asarray(seed,jnp.uint16),i32(generation),sets.ivs,sets.nature,
                        tables.G_ABILITIES[sets.ability],sets.gender,sets.shiny,sets.happiness,i32(names),values.hp_type)


@jax.jit
def _initialize(species,level,moves,item,evs,seed,generation,ivs,nature,ability,gender,shiny,happiness,name,hp_type):
    occupied=species>0
    stats=derive_stored_stats(BASE_STATS[species],level,ivs,evs,generic.NATURE_MULTIPLIERS[nature],tables.SHEDINJA[species])
    stats=jnp.where(occupied[:,:,None],stats,0)
    zeros=jnp.zeros((2,6),jnp.int32)
    s=State(species,level,stats,stats[:,:,5],gender,stats[:,:,0],jnp.zeros((2,6),jnp.bool_),moves,PP[moves],item,
            zeros,zeros,zeros,zeros,jnp.zeros((2,6),jnp.bool_),jnp.zeros(2,jnp.int32),
            jnp.tile(jnp.arange(6,dtype=jnp.int32),(2,1)),jnp.full(2,-1,jnp.int32),jnp.full(2,MOVE,jnp.int32),
            jnp.zeros(2,jnp.bool_),jnp.zeros((QUEUE,7),jnp.int32),i32(0),jnp.array(False),seed,i32(1),i32(-1),generation,i32(0),
            jnp.zeros((HISTORY,EVENT_COLUMNS),jnp.int32),i32(0),jnp.zeros((2,3),jnp.int32),
            ivs,evs,nature,ability,tables.SPECIES_TYPES[species],jnp.where(occupied,hp_type,0),
            jnp.where(occupied,generic.hidden_power(ivs)[1],0),shiny,happiness,name,jnp.zeros((2,6,7),jnp.int32),i32(0),effects.empty_status(),
            jnp.zeros((2,6,len(phase_ops.VOLATILES),phase_ops.VOLATILE_WIDTH),jnp.int32),zeros,zeros,field_ops.empty_weather(),jnp.zeros((2,2,5),jnp.int32),jnp.zeros((2,2,3),jnp.int32),jnp.array(False),jnp.zeros((2,6,3,4),jnp.int32),
            tracking.empty().at[:,:,tracking.NEWLY_SWITCHED].set(occupied.astype(jnp.int32)),
            jnp.zeros((2,6,tracking.ATTACK_RECORDS,7),jnp.int32),jnp.full((2,2),-1,jnp.int32),jnp.zeros(4,jnp.int32),i32(0),i32(0),jnp.zeros((2,2),jnp.int32),jnp.stack((item,zeros),axis=-1),species,ability,stats,moves,PP[moves],zeros,jnp.zeros((2,6),jnp.bool_),occupied,tables.SPECIES_TYPES[species],ability,jnp.full((2,6),-1,jnp.int32),zeros,called_ops.empty(_engine()),i32(0),zeros)
    return jax.lax.cond(_supported(s,initializing=True),_initialize_events,lambda s:Result(s,i32(UNSUPPORTED),empty_trace()),s)


def _initialize_events(s):
    c=(s,empty_trace())
    for side in range(2):c=effects.activate(c,i32(side),_engine())
    for side in range(2):
        for mon in range(ROSTER):
            def gender(c):
                c,value=_draw(c,0,2,INIT);s,t=c
                return s._replace(gender=s.gender.at[side,mon].set(value.astype(jnp.int32)+1)),t
            c=_when((s.species[side,mon]>0)&(s.gender[side,mon]<0),gender,c)
    # Initial queue insertion and runSwitch each perform their own all-active
    # sort. The second draw determines SwitchIn callback order. Keep cached
    # speed until the first commitChoices, even after weather changes.
    s,t=c;s=_refresh_speed(s);c=(s,t)
    for side in range(2):
        s=illusion_ops.enter(c[0],i32(side),_engine());c=(s,c[1])
        c=_event(c,SWITCH,side,0,value=s.hp[side,0],aux=s.stats[side,0,0])
    speeds=s.speed[:,0];tie=speeds[0]==speeds[1]
    c=_when(tie,lambda c:_draw(c,0,2,INIT,2)[0],c)
    c,word=jax.lax.cond(tie,lambda c:_draw(c,0,2,INIT,2),lambda c:(c,jnp.uint32(0)),c)
    first=jnp.where(tie,word.astype(jnp.int32),jnp.argmax(speeds))
    for k in range(2):
        side=jnp.where(k==0,first,1-first)
        c=hook_ops.start(c,side,_engine(),initial=True)
    c=_update(c,INIT)
    s,t=c
    s=s._replace(action_state=s.action_state.at[:,0,tracking.SWITCH_COUNT].set(1))
    c=(trap_ops.refresh(tracking.new_turn(s),_engine()),t)
    c=_event(c,TURN,value=1)
    s,t=c
    s=s._replace(history=s.history.at[:EVENTS].set(t.events),hcount=t.ecount)
    return Result(s,i32(OK),t)


def reset(state,replacement,token):
    """Reset a terminal lane with a supplied initialized battle; live/stale reset
    is rejected atomically. No native reset-pool or throughput claim is implied.
    """
    _validate_structure(state)
    _validate_structure(replacement)
    if type(state) is not type(replacement):raise ValueError('reset state types must match')
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
