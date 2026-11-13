"""Numeric causal history used by move restrictions and trusted differential tests.

This is hidden simulator state. The actor adapter never exports it. Ordered
attack rows have an explicit bound/cursor; overflow rejects the whole command.
"""
import jax
import jax.numpy as jnp
from . import gen5_engine_tables as t

UNDEFINED,NULL,FALSE,TRUE,NUMBER=range(5)
(TARGET_LOC,LAST_DAMAGE,ACTIVE_TURNS,ACTIONS,MOVE_THIS_TURN,RESULT,PREV_RESULT,
 HURT,TIMES_ATTACKED,TRAPPED,MAYBE_TRAPPED,MAYBE_DISABLED,NEWLY_SWITCHED,
 USED_ITEM,LAST_ITEM,RAISED,LOWERED,ATTACK_COUNT,USED_SLOTS,ATE_BERRY,SWITCH_COUNT)=range(21)
CALLED_BACK,PURSUIT_ACTIVE,SKIP_PURSUIT=21,22,23
COLUMNS=24
ATTACK_RECORDS=1024


def empty():
    a=jnp.zeros((2,6,COLUMNS),jnp.int32)
    return a.at[:,:,TARGET_LOC].set(-2).at[:,:,HURT].set(-1)


def put(s,side,column,value,mon=None):
    mon=s.active[side] if mon is None else mon
    return s._replace(action_state=s.action_state.at[side,mon,column].set(value))


def attempted(s,side):
    mon=s.active[side]
    return put(s,side,ACTIONS,s.action_state[side,mon,ACTIONS]+1)


def result(c,side,value):
    s,trace=c
    return put(s,side,RESULT,value),trace


def used(s,side,move,slot,e,target_loc,mark_slot=True):
    mon=s.active[side]
    move_type=jnp.where(move==e.HIDDEN_POWER,s.hp_type[side,mon],t.MOVE_TYPE[move])
    move_type=jnp.where(move==e.STRUGGLE,0,move_type)
    row=jnp.stack((move,move_type,t.CATEGORY[move],t.POWER[move]))
    history=s.move_history.at[side,mon,0].set(row).at[side,mon,2].set(row)
    a=s.action_state.at[side,mon,TARGET_LOC].set(target_loc).at[side,mon,MOVE_THIS_TURN].set(move)
    a=a.at[side,mon,LAST_DAMAGE].set(0).at[side,mon,RESULT].set(FALSE)
    bit=jnp.left_shift(jnp.int32(1),jnp.clip(slot,0,3))
    bits=jnp.where((move==e.STRUGGLE)|(~jnp.asarray(mark_slot)),a[side,mon,USED_SLOTS],a[side,mon,USED_SLOTS]|bit)
    a=a.at[side,mon,USED_SLOTS].set(bits)
    # clearActiveMove records attempted used moves even on miss/immunity.
    return s._replace(move_history=history,action_state=a,battle_move=row)


def damaged(s,side,amount,hp,source,cause):
    mon=s.active[side]
    # directDamage(Struggle recoil) bypasses spreadDamage's hurtThisTurn.
    value=jnp.where((amount>0)&(cause!=3),hp,s.action_state[side,mon,HURT])
    s=put(s,side,HURT,value)
    move_damage=(cause==0)|(cause==6)
    value=jnp.where(move_damage,amount,s.action_state[source,s.active[source],LAST_DAMAGE])
    return put(s,source,LAST_DAMAGE,value)


def attacked(c,target,source,move,amount,value_kind,hits,e):
    s,trace=c;mon=s.active[target];source_mon=s.active[source]
    count=s.action_state[target,mon,ATTACK_COUNT]
    row=jnp.stack((source,source_mon,move,amount,jnp.int32(1),value_kind,amount))
    attacks=s.attacked_by.at[target,mon,count].set(row)
    a=s.action_state.at[target,mon,ATTACK_COUNT].add(1)
    a=a.at[target,mon,TIMES_ATTACKED].add(jnp.where(value_kind==NUMBER,hits,0))
    return s._replace(action_state=a,attacked_by=attacks),trace._replace(overflow=trace.overflow|(count>=ATTACK_RECORDS))


def clear_volatile(s,side):
    mon=s.active[side];a=s.action_state
    for k in (LAST_DAMAGE,MOVE_THIS_TURN,RESULT,PREV_RESULT,ATTACK_COUNT):a=a.at[side,mon,k].set(0)
    a=a.at[side,mon,HURT].set(-1).at[side,mon,NEWLY_SWITCHED].set(1).at[side,mon,CALLED_BACK].set(0)
    return s._replace(action_state=a,move_history=s.move_history.at[side,mon].set(0),
                      attacked_by=s.attacked_by.at[side,mon].set(0))


def faint(s,side):
    s=clear_volatile(s,side)
    return s._replace(side_faints=s.side_faints.at[side,0].set(s.active[side]))


def switched(s,side,old,incoming):
    # Preserve lifetime counters and the fainted holder's final attack records.
    for k in (USED_ITEM,RAISED,LOWERED,PURSUIT_ACTIVE,SKIP_PURSUIT):s=put(s,side,k,0,old)
    a=s.action_state.at[side,incoming,ACTIVE_TURNS].set(0).at[side,incoming,ACTIONS].set(0)
    a=a.at[side,incoming,USED_SLOTS].set(0).at[side,incoming,SWITCH_COUNT].add(1)
    return s._replace(action_state=a)


def new_turn(s):
    def one(side,s):
        mon=s.active[side];a=s.action_state;row=a[side,mon]
        row=row.at[MOVE_THIS_TURN].set(0).at[NEWLY_SWITCHED].set(0)
        row=row.at[PREV_RESULT].set(row[RESULT]).at[RESULT].set(UNDEFINED)
        for k in (USED_ITEM,RAISED,LOWERED):row=row.at[k].set(jnp.where(s.turn==1,row[k],0))
        row=row.at[HURT].set(jnp.where(s.turn==1,row[HURT],-1))
        for k in (TRAPPED,MAYBE_TRAPPED,MAYBE_DISABLED):row=row.at[k].set(0)
        row=row.at[ACTIVE_TURNS].add((~s.fainted[side,mon]).astype(jnp.int32))
        attacks=s.attacked_by[side,mon]
        valid=jnp.arange(ATTACK_RECORDS)<row[ATTACK_COUNT]
        source=jnp.clip(attacks[:,0],0,1);sm=jnp.clip(attacks[:,1],0,5)
        valid=valid&(s.active[source]==sm)&(~s.fainted[source,sm])
        ix=jnp.nonzero(valid,size=ATTACK_RECORDS,fill_value=0)[0];count=jnp.sum(valid).astype(jnp.int32)
        attacks=attacks[ix].at[:,4].set(0)
        attacks=jnp.where((jnp.arange(ATTACK_RECORDS)<count)[:,None],attacks,0)
        row=row.at[ATTACK_COUNT].set(count)
        return s._replace(action_state=a.at[side,mon].set(row),attacked_by=s.attacked_by.at[side,mon].set(attacks))
    s=jax.lax.fori_loop(0,2,one,s)
    return s._replace(side_faints=s.side_faints.at[:,1].set(s.side_faints[:,0]).at[:,0].set(-1),battle_success=jnp.int32(0))


def item_used(s,side,item,*,berry=True):
    s=put(s,side,USED_ITEM,1);s=put(s,side,LAST_ITEM,item)
    return put(s,side,ATE_BERRY,jnp.where(berry,1,s.action_state[side,s.active[side],ATE_BERRY]))


def boosted(s,side,raised,lowered):
    mon=s.active[side]
    s=put(s,side,RAISED,s.action_state[side,mon,RAISED]|raised.astype(jnp.int32))
    return put(s,side,LOWERED,s.action_state[side,mon,LOWERED]|lowered.astype(jnp.int32))


def outcome(c,side,value,move):
    s,trace=result(c,side,value)
    return s._replace(battle_success=jnp.where(value==TRUE,move,0)),trace


def valid(s,e):
    a=s.action_state
    mask=jnp.ones(COLUMNS,jnp.bool_).at[TARGET_LOC].set(False).at[HURT].set(False)
    good=jnp.all(jnp.where(mask,a>=0,True))
    good=good&jnp.all((a[:,:,TARGET_LOC]>=-2)&(a[:,:,TARGET_LOC]<=1))
    good=good&jnp.all((a[:,:,HURT]>=-1)&(a[:,:,HURT]<=s.stats[:,:,0]))
    for k in (RESULT,PREV_RESULT):good=good&jnp.all(a[:,:,k]<=TRUE)
    for k in (MAYBE_TRAPPED,MAYBE_DISABLED,NEWLY_SWITCHED,USED_ITEM,RAISED,LOWERED,ATE_BERRY,CALLED_BACK,PURSUIT_ACTIVE,SKIP_PURSUIT):good=good&jnp.all(a[:,:,k]<=1)
    good=good&jnp.all(a[:,:,TRAPPED]<=2)&jnp.all(a[:,:,USED_SLOTS]<=15)
    good=good&jnp.all(a[:,:,MOVE_THIS_TURN]<len(t.MOVE_KEYS))&jnp.all(a[:,:,LAST_ITEM]<len(t.ITEMS))
    good=good&jnp.all(a[:,:,ATTACK_COUNT]<=ATTACK_RECORDS)
    def moves_valid(rows):
        present=rows[...,0]>0
        fields=(rows[...,0]<len(t.MOVE_KEYS))&(rows[...,1]>=0)&(rows[...,1]<=len(e.generic.TYPES))
        fields=fields&(rows[...,2]>=0)&(rows[...,2]<=2)&(rows[...,3]>=0)
        return jnp.all((rows[...,0]>=0)&jnp.where(present,fields,jnp.all(rows==0,axis=-1)))
    good=good&moves_valid(s.move_history)&moves_valid(s.battle_move)
    good=good&jnp.all((s.side_faints>=-1)&(s.side_faints<6))
    good=good&(s.battle_success>=0)&(s.battle_success<len(t.MOVE_KEYS))&(s.battle_damage>=0)
    row=s.attacked_by;live=jnp.arange(ATTACK_RECORDS)[None,None,:]<a[:,:,ATTACK_COUNT,None]
    source=(row[...,0]>=0)&(row[...,0]<2)&(row[...,1]>=0)&(row[...,1]<6)
    fields=(row[...,2]>0)&(row[...,2]<len(t.MOVE_KEYS))&(row[...,3]>=0)&(row[...,4]>=0)&(row[...,4]<=1)
    fields=fields&(row[...,5]>=0)&(row[...,5]<=NUMBER)&(row[...,6]>=0)
    fields=fields&jnp.where(row[...,5]==NUMBER,row[...,3]==row[...,6],(row[...,3]==0)&(row[...,6]==0))
    return good&jnp.all(jnp.where(live,source&fields,jnp.all(row==0,axis=-1)))
