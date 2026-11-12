"""Pure JAX forced recharge phase; other move locks remain unsupported."""
import jax.numpy as jnp
from . import gen5_engine_tables as t

VOLATILES=('protect','stall','confusion','choicelock','mustrecharge','lockedmove','twoturnmove','solarbeam','skyattack','skullbash','razorwind','fly','dig','dive','bounce','taunt','encore','disable','torment','flinch','substitute','partiallytrapped','pursuit','zenmode')
VOLATILE_IDS={key:i for i,key in enumerate(VOLATILES)}
VOLATILE_WIDTH=12
SOURCE_MOVE,SOURCE_ITEM,SOURCE_CONDITION,SOURCE_EMPTY,SOURCE_ABILITY=1,2,3,4,5
RECHARGE_SLOT=4
# Internal queued action, distinct from the public eleven-action request space.
RECHARGE_ACTION=11
RECHARGE=t.mv(lambda row:row.get('self')=={'volatileStatus':'mustrecharge'}).astype(jnp.bool_)


def supported(row):
    if row.get('self')!={'volatileStatus':'mustrecharge'}:return False
    clean=dict(row);clean.pop('self')
    clean['flags']={k:v for k,v in row.get('flags',{}).items() if k!='recharge'}
    return t.plain(clean)


RAMPAGE=t.mv(lambda row:row.get('self')=={'volatileStatus':'lockedmove'}).astype(jnp.bool_)


def rampage_supported(row):
    if row.get('self')!={'volatileStatus':'lockedmove'}:return False
    clean=dict(row);clean.pop('self');clean['target']='normal'
    clean['flags']={k:v for k,v in row.get('flags',{}).items() if k not in ('failinstruct','dance')}
    return t.plain(clean)


CHARGE_KEYS=('solarbeam','skyattack','skullbash','razorwind','fly','dig','dive','bounce')
CHARGE=t.mv(lambda row:row['id'] in CHARGE_KEYS).astype(jnp.bool_)
CHARGE_SLOT=t.mv(lambda row:VOLATILE_IDS.get(row['id'],0) if row['id'] in CHARGE_KEYS else 0)
FLINCH_CHANCE=t.mv(lambda row:(row.get('secondary') or {}).get('chance',0) if (row.get('secondary') or {}).get('volatileStatus')=='flinch' else 0)
RESTRICTIONS=('taunt','encore','disable','torment')
RESTRICTION=t.mv(lambda row:VOLATILE_IDS.get(row['id'],0) if row['id'] in RESTRICTIONS else 0)
ENCORE_BLOCKED=t.mv(lambda row:bool(row.get('flags',{}).get('failencore'))).astype(jnp.bool_)
RESIDUAL_ORDER=jnp.asarray([15 if key=='taunt' else 16 if key=='encore' else 17 if key=='disable' else 13 if key=='partiallytrapped' else 10000 for key in VOLATILES],jnp.int32)
POWER_HERB=t.ITEM_IDS['powerherb']
ITEM_IMPLEMENTED=jnp.asarray([key=='powerherb' for key in t.ITEMS],jnp.bool_)

def flinch_supported(row):
    secondary=row.get('secondary') or {}
    if set(secondary)!={'chance','volatileStatus'} or secondary['volatileStatus']!='flinch':return False
    if row.get('secondaries') not in (None,[secondary]):return False
    clean=dict(row);clean.pop('secondary',None);clean.pop('secondaries',None)
    return t.plain(clean)


MOVE_IMPLEMENTED=jnp.asarray([False]+[supported(t.MV_ROWS[key]) or rampage_supported(t.MV_ROWS[key]) or key in CHARGE_KEYS or key in RESTRICTIONS or key=='substitute' or flinch_supported(t.MV_ROWS[key]) for key in t.MOVE_KEYS[1:]],jnp.bool_)



def forced_move(s):
    active=s.volatile_state[jnp.arange(2),s.active]
    move=jnp.where(active[:,VOLATILE_IDS['lockedmove'],0]>0,active[:,VOLATILE_IDS['lockedmove'],7],0)
    move=jnp.where(active[:,VOLATILE_IDS['twoturnmove'],0]>0,active[:,VOLATILE_IDS['twoturnmove'],7],move)
    return jnp.where(active[:,RECHARGE_SLOT,0]>0,-1,move)


def locked(s):
    return forced_move(s)!=0


def install(c,actor,move,e):
    def apply(c):
        c=e.effects.add_volatile(c,actor,RECHARGE_SLOT,actor,move,e)
        s,trace=c;mon=s.active[actor]
        s=s._replace(volatile_state=s.volatile_state.at[actor,mon,RECHARGE_SLOT,5].set(2))
        return e._event((s,trace),e.MUST_RECHARGE,actor,mon)
    c=e._when(RECHARGE[move]&(c[0].hp[actor,c[0].active[actor]]>0),apply,c)
    return e._when(RAMPAGE[move]&(c[0].hp[actor,c[0].active[actor]]>0),lambda c:install_rampage(c,actor,move,e),c)


def consume(c,actor,e):
    # Native BeforeMove priority11 precedes sleep/freeze/confusion/paralysis.
    s,trace=c;mon=s.active[actor]
    s=s._replace(volatile_state=s.volatile_state.at[actor,mon,RECHARGE_SLOT].set(0))
    s=e.tracking.put(s,actor,e.tracking.RESULT,e.tracking.NULL)
    return e._event((s,trace),e.RECHARGE_CANT,actor,mon)


def tick(c,side,mon):
    s,trace=c;row=s.volatile_state[side,mon,RECHARGE_SLOT]
    row=row.at[5].add(-1)
    return s._replace(volatile_state=s.volatile_state.at[side,mon,RECHARGE_SLOT].set(jnp.where(row[5]>0,row,0))),trace


def valid(s):
    row=s.volatile_state[:,:,RECHARGE_SLOT];present=row[:,:,0]>0
    source=row[:,:,4];safe=jnp.clip(source,0,len(t.MOVE_KEYS)-1)
    metadata=(row[:,:,1]<s.effect_order)&(row[:,:,2]==jnp.arange(2)[:,None])&(row[:,:,3]==jnp.arange(6)[None,:])
    metadata=metadata&(source>0)&(source<len(t.MOVE_KEYS))&RECHARGE[safe]&(row[:,:,6]==SOURCE_MOVE)&(row[:,:,5]>=1)&(row[:,:,5]<=2)
    active=(s.active[:,None]==jnp.arange(6)[None,:])&(~s.fainted)&(s.species>0)
    return jnp.all(jnp.where(present,metadata&active,jnp.all(row==0,axis=-1)))


def install_rampage(c,actor,move,e):
    kind=VOLATILE_IDS['lockedmove'];s,trace=c;mon=s.active[actor]
    def create(c):
        c=e.effects.add_volatile(c,actor,kind,actor,move,e)
        c,value=e._draw(c,2,4,e.RAMPAGE_DURATION,2);s,trace=c
        row=s.volatile_state[actor,mon,kind].at[5].set(2).at[7].set(move).at[8].set(value.astype(jnp.int32))
        return s._replace(volatile_state=s.volatile_state.at[actor,mon,kind].set(row)),trace
    def restart(c):
        s,trace=c;row=s.volatile_state[actor,mon,kind]
        row=row.at[5].set(jnp.where(row[8]>=2,2,row[5]))
        return s._replace(volatile_state=s.volatile_state.at[actor,mon,kind].set(row)),trace
    return e.jax.lax.cond(s.volatile_state[actor,mon,kind,0]>0,restart,create,c)


def end_rampage(c,side,e):
    kind=VOLATILE_IDS['lockedmove'];s,trace=c;mon=s.active[side];row=s.volatile_state[side,mon,kind]
    s=s._replace(volatile_state=s.volatile_state.at[side,mon,kind].set(0))
    return e._when((row[8]<=1)&(s.volatile_state[side,mon,2,0]==0),
        lambda c:e.effects.confuse(c,side,side,e.i32(kind),e,fatigue=True),(s,trace))


def after_move(c,side,e):
    row=c[0].volatile_state[side,c[0].active[side],VOLATILE_IDS['lockedmove']]
    return e._when((row[0]>0)&(row[5]==1),lambda c:end_rampage(c,side,e),c)


def tick_phase(c,side,kind,e):
    s,trace=c;mon=s.active[side];row=s.volatile_state[side,mon,kind].at[5].add(-1)
    s=s._replace(volatile_state=s.volatile_state.at[side,mon,kind].set(row));c=(s,trace)
    def expired(c):return end_rampage(c,side,e)
    def active(c):
        s,trace=c;row=s.volatile_state[side,mon,kind].at[8].add(-1)
        row=jnp.where(s.status[side,mon]==e.SLP,0,row)
        return s._replace(volatile_state=s.volatile_state.at[side,mon,kind].set(row)),trace
    return e.jax.lax.cond(row[5]<=0,expired,active,c)



PARTIAL_MOVES=t.mv(lambda row:row.get('volatileStatus')=='partiallytrapped').astype(jnp.bool_)


def valid_phases(s):
    head=s.volatile_state[:,:,:5];present=head[:,:,:,0]>0
    active=(s.active[:,None]==jnp.arange(6)[None,:])&(~s.fainted)
    metadata=(head[:,:,:,1]<s.effect_order)&active[:,:,None]
    source_kind=head[:,:,:,6];effect=head[:,:,:,4]
    simple=((source_kind==SOURCE_MOVE)&(effect>0))|((source_kind==SOURCE_EMPTY)&(effect==0))
    condition=(source_kind[:,:,2]==SOURCE_CONDITION)&(effect[:,:,2]==VOLATILE_IDS['lockedmove'])
    simple=simple.at[:,:,2].set(simple[:,:,2]|condition)
    simple=simple.at[:,:,3].set((source_kind[:,:,3]==SOURCE_ITEM)&(effect[:,:,3]<0))
    metadata=metadata&simple&jnp.all(head[:,:,:,7:]==0,axis=-1)
    good=jnp.all(jnp.where(present,metadata,jnp.all(head==0,axis=-1)))
    for kind in range(5,len(VOLATILES)):
        row=s.volatile_state[:,:,kind];present=row[:,:,0]>0
        move=jnp.clip(row[:,:,7],0,len(t.MOVE_KEYS)-1)
        common=(row[:,:,1]<s.effect_order)&(s.active[:,None]==jnp.arange(6)[None,:])&(~s.fainted)
        if kind==5:
            fields=RAMPAGE[move]&(row[:,:,4]==row[:,:,7])&(row[:,:,6]==SOURCE_MOVE)
            fields=fields&(row[:,:,5]>=1)&(row[:,:,5]<=2)&(row[:,:,8]>=1)&(row[:,:,8]<=3)
        elif kind==6:
            fields=CHARGE[move]&(row[:,:,4]==row[:,:,7])&(row[:,:,6]==SOURCE_MOVE)&(row[:,:,5]>=1)&(row[:,:,5]<=2)
        elif kind in (15,16,17,18):
            fields=(row[:,:,6]==SOURCE_MOVE)&(row[:,:,4]==t.MOVE_IDS[VOLATILES[kind]])
            fields=fields&((row[:,:,5]>=1)&(row[:,:,5]<=5) if kind!=18 else row[:,:,5]==0)
            if kind in (16,17):fields=fields&(row[:,:,7]>0)&(row[:,:,7]<len(t.MOVE_KEYS))
        elif kind==VOLATILE_IDS['zenmode']:
            fields=(row[:,:,6]==SOURCE_ABILITY)&(row[:,:,4]==t.ABILITY_IDS['zenmode'])&(row[:,:,5]==0)&jnp.all(row[:,:,7:]==0,axis=-1)
            fields=fields&(row[:,:,2]==jnp.arange(2)[:,None])&(row[:,:,3]==jnp.arange(6)[None,:])
        elif kind==VOLATILE_IDS['pursuit']:
            fields=(row[:,:,6]==SOURCE_MOVE)&(row[:,:,4]==t.MOVE_IDS['pursuit'])&(row[:,:,5]==1)&jnp.all(row[:,:,7:]==0,axis=-1)
            fields=fields&(row[:,:,2]==jnp.arange(2)[:,None])&(row[:,:,3]==jnp.arange(6)[None,:])
        elif kind==VOLATILE_IDS['partiallytrapped']:
            effect=jnp.clip(row[:,:,4],0,len(t.MOVE_KEYS)-1)
            fields=(row[:,:,6]==SOURCE_MOVE)&PARTIAL_MOVES[effect]&(row[:,:,4]>0)&(row[:,:,4]<len(t.MOVE_KEYS))
            fields=fields&(row[:,:,5]>=1)&(row[:,:,5]<=8)&((row[:,:,7]==8)|(row[:,:,7]==16))&jnp.all(row[:,:,8:]==0,axis=-1)
            source=jnp.clip(row[:,:,2],0,1);sm=jnp.clip(row[:,:,3],0,5)
            fields=fields&(s.species[source,sm]>0)
        elif kind==VOLATILE_IDS['substitute']:
            fields=(row[:,:,6]==SOURCE_MOVE)&(row[:,:,4]==t.MOVE_IDS['substitute'])&(row[:,:,5]==0)
            fields=fields&(row[:,:,7]>0)&(row[:,:,7]<=s.stats[:,:,0]//4)&jnp.all(row[:,:,8:]==0,axis=-1)
            fields=fields&(row[:,:,2]==jnp.arange(2)[:,None])&(row[:,:,3]==jnp.arange(6)[None,:])
        elif kind==VOLATILE_IDS['flinch']:
            fields=(row[:,:,5]==1)&(row[:,:,6]==SOURCE_MOVE)&(row[:,:,4]>0)&(row[:,:,4]<len(t.MOVE_KEYS))
        else:
            fields=(row[:,:,6]==SOURCE_CONDITION)&(row[:,:,4]==6)&(row[:,:,9]>=-1)&(row[:,:,9]<=1)
            fields=fields&(s.volatile_state[:,:,6,0]>0)&(s.volatile_state[:,:,6,7]==t.MOVE_IDS[VOLATILES[kind]])
            duration=VOLATILES[kind] in ('fly','dig','dive','bounce')
            fields=fields&((row[:,:,5]>=1)&(row[:,:,5]<=2) if duration else row[:,:,5]==0)
        good=good&jnp.all(jnp.where(present,common&fields,jnp.all(row==0,axis=-1)))
    forced=jnp.sum(s.volatile_state[:,:,[4,5,6],0]>0,axis=-1)
    children=jnp.sum(s.volatile_state[:,:,7:15,0]>0,axis=-1)
    return good&jnp.all(forced<=1)&jnp.all(children<=1)


def charge(c,actor,move,move_row,e):
    """Return (carry, charging). The real move is still used on charge turns."""
    s,trace=c;mon=s.active[actor];child=CHARGE_SLOT[move]
    def release(c):
        s,trace=c
        return (s._replace(volatile_state=s.volatile_state.at[actor,mon,child].set(0)),trace),jnp.array(False)
    def prepare(c):
        c=e._event(c,e.PREPARE,actor,mon,move=move)
        def skull(c):
            s,trace=c;old=s.boosts[actor,mon,1];new=jnp.minimum(6,old+1)
            s=s._replace(boosts=s.boosts.at[actor,mon,1].set(new))
            s=e.tracking.boosted(s,actor,new>old,jnp.array(False))
            return e._event((s,trace),e.BOOST,actor,mon,value=new-old,aux=1)
        c=e._when(move==t.MOVE_IDS['skullbash'],skull,c)
        sun=(move==t.MOVE_IDS['solarbeam'])&(c[0].weather_state[0]==2)
        herb=(c[0].item[actor,mon]==POWER_HERB)&(~sun)
        def immediate(c):
            s,trace=c;trace=trace._replace(events=trace.events.at[move_row,7].set(2));c=(s,trace)
            def consume(c):
                s,trace=c;s=e.tracking.item_used(s,actor,POWER_HERB,berry=False)
                s=s._replace(item=s.item.at[actor,mon].set(0),item_order=s.item_order.at[actor,mon].set(0))
                return e._event((s,trace),e.END_HELD_ITEM,actor,mon,value=POWER_HERB)
            c=e._when(herb,consume,c)
            return e._event(c,e.ANIM,actor,mon,move=move,target=1-actor,targetmon=c[0].active[1-actor]),jnp.array(False)
        def install(c):
            c=e.effects.add_volatile(c,actor,6,1-actor,move,e)
            s,trace=c;row=s.volatile_state[actor,mon,6].at[5].set(2).at[7].set(move)
            s=s._replace(volatile_state=s.volatile_state.at[actor,mon,6].set(row));c=(s,trace)
            c=e.effects.add_volatile(c,actor,child,1-actor,e.i32(6),e,source_kind=SOURCE_CONDITION)
            s,trace=c;row=s.volatile_state[actor,mon,child].at[9].set(s.action_state[actor,mon,e.tracking.TARGET_LOC])
            semi=(child>=VOLATILE_IDS['fly'])&(child<=VOLATILE_IDS['bounce'])
            row=row.at[5].set(jnp.where(semi,2,0))
            s=s._replace(volatile_state=s.volatile_state.at[actor,mon,child].set(row))
            trace=trace._replace(events=trace.events.at[move_row,7].set(2))
            return e.tracking.outcome((s,trace),actor,e.i32(e.tracking.NULL),move),jnp.array(True)
        return e.jax.lax.cond(sun|herb,immediate,install,c)
    return e.jax.lax.cond(s.volatile_state[actor,mon,child,0]>0,release,prepare,c)


def abort_move(c,side,e):
    s,trace=c;mon=s.active[side];move=s.volatile_state[side,mon,6,7]
    child=CHARGE_SLOT[jnp.clip(move,0,len(t.MOVE_KEYS)-1)]
    def clear(c):
        s,trace=c
        return s._replace(volatile_state=s.volatile_state.at[side,mon,6].set(0).at[side,mon,child].set(0)),trace
    return e._when(s.volatile_state[side,mon,6,0]>0,clear,c)


def flinch(c,actor,target,move,e):
    s,trace=c;mon=s.active[target]
    def install(c):
        c=e.effects.add_volatile(c,target,VOLATILE_IDS['flinch'],actor,move,e)
        s,trace=c
        return s._replace(volatile_state=s.volatile_state.at[target,mon,VOLATILE_IDS['flinch'],5].set(1)),trace
    return e._when(s.hp[target,mon]>0,install,c)


def before_move(c,side,move,e):
    s,trace=c;mon=s.active[side]
    def stopped(c):return e._event(c,e.FLINCH_CANT,side,mon),jnp.array(True)
    c,blocked=e.jax.lax.cond(s.volatile_state[side,mon,VOLATILE_IDS['flinch'],0]>0,stopped,lambda c:(c,jnp.array(False)),c)
    def restriction(c):
        s,trace=c;disabled=s.volatile_state[side,mon,VOLATILE_IDS['disable']]
        disable=(disabled[0]>0)&(move==disabled[7]);taunt=(s.volatile_state[side,mon,VOLATILE_IDS['taunt'],0]>0)&(e.tables.CATEGORY[move]==0)&(move!=t.MOVE_IDS['mefirst'])
        kind=jnp.where(disable,VOLATILE_IDS['disable'],VOLATILE_IDS['taunt'])
        c=e._when(disable|taunt,lambda c:e._event(c,e.VOL_CANT,side,mon,move=move,value=kind),c)
        return c,disable|taunt
    return e.jax.lax.cond(~blocked,restriction,lambda c:(c,blocked),c)


def end_effect(c,side,kind,e):
    s,trace=c;mon=s.active[side];row=s.volatile_state[side,mon,kind]
    v=s.volatile_state.at[side,mon,kind].set(0)
    child=CHARGE_SLOT[jnp.clip(row[7],0,len(t.MOVE_KEYS)-1)]
    v=e.jax.lax.cond(kind==6,lambda v:v.at[side,mon,child].set(0),lambda v:v,v)
    c=(s._replace(volatile_state=v),trace)
    return e._when((kind>=15)&(kind<=18),lambda c:e._event(c,e.VOL_END,side,mon,value=kind),c)


def extra_tick(c,side,kind,e):
    def locked(c):return tick_phase(c,side,kind,e)
    def ordinary(c):
        s,trace=c;mon=s.active[side];row=s.volatile_state[side,mon,kind].at[5].add(-1)
        s=s._replace(volatile_state=s.volatile_state.at[side,mon,kind].set(row));c=(s,trace)
        exhausted=(kind==VOLATILE_IDS['encore'])&(~jnp.any((s.moves[side,mon]==row[7])&(s.pp[side,mon]>0)))
        return e.jax.lax.cond((row[5]<=0)|exhausted,lambda c:end_effect(c,side,kind,e),lambda c:c,c)
    return e._when(c[0].volatile_state[side,c[0].active[side],kind,0]>0,
        lambda c:e.jax.lax.cond(kind==5,locked,ordinary,c),c)


def _member(move,keys):
    ids=jnp.asarray([t.MOVE_IDS[k] for k in keys if k in t.MOVE_IDS],jnp.int32)
    return jnp.any(move==ids)


def invulnerable(s,target,move):
    active=s.volatile_state[target,s.active[target]]
    air=(active[VOLATILE_IDS['fly'],0]>0)|(active[VOLATILE_IDS['bounce'],0]>0)
    allowed_air=_member(move,('gust','twister','skyuppercut','thunder','hurricane','smackdown'))
    ground=(active[VOLATILE_IDS['dig'],0]>0)&(~_member(move,('earthquake','magnitude')))
    water=(active[VOLATILE_IDS['dive'],0]>0)&(~_member(move,('surf','whirlpool')))
    return (air&(~allowed_air))|ground|water


def power_modifier(s,target,move):
    active=s.volatile_state[target,s.active[target]]
    bounce=(active[VOLATILE_IDS['bounce'],0]>0)&_member(move,('gust','twister'))
    solar=(move==t.MOVE_IDS['solarbeam'])&((s.weather_state[0]==1)|(s.weather_state[0]==3)|(s.weather_state[0]==4))
    return jnp.where(solar,2048,jnp.where(bounce,8192,4096)).astype(jnp.int32)


def damage_modifier(s,target,move):
    active=s.volatile_state[target,s.active[target]]
    fly=active[VOLATILE_IDS['fly'],0]>0;dig=active[VOLATILE_IDS['dig'],0]>0;dive=active[VOLATILE_IDS['dive'],0]>0
    doubled=(fly&_member(move,('gust','twister')))|(dig&_member(move,('earthquake','magnitude')))|(dive&_member(move,('surf','whirlpool')))
    return fly|dig|dive,jnp.where(doubled,8192,4096).astype(jnp.int32)



def override_action(s,side,action):
    mon=s.active[side];encore=s.volatile_state[side,mon,VOLATILE_IDS['encore']]
    slot=jnp.argmax(s.moves[side,mon]==encore[7])
    return jnp.where((encore[0]>0)&(action<4)&jnp.any(s.moves[side,mon]==encore[7]),slot,action)


def disabled(s):
    v=s.volatile_state;taunt=v[:,:,VOLATILE_IDS['taunt'],0]>0
    disable=v[:,:,VOLATILE_IDS['disable']];encore=v[:,:,VOLATILE_IDS['encore']]
    torment=v[:,:,VOLATILE_IDS['torment'],0]>0;last=s.move_history[:,:,0,0]
    mask=taunt[:,:,None]&(t.CATEGORY[s.moves]==0)&(s.moves!=t.MOVE_IDS['mefirst'])
    mask=mask|((disable[:,:,0,None]>0)&(s.moves==disable[:,:,7,None]))
    owns=jnp.any(s.moves==encore[:,:,7,None],axis=-1)
    mask=mask|((encore[:,:,0,None]>0)&owns[:,:,None]&(s.moves!=encore[:,:,7,None]))
    return mask|(torment[:,:,None]&(last[:,:,None]>0)&(last[:,:,None]!=t.MOVE_IDS['struggle'])&(s.moves==last[:,:,None]))


def _will_move(s,side,mon,e):
    return jnp.any((jnp.arange(e.QUEUE)<s.qcount)&(s.queue[:,0]==e.ATTACK)&(s.queue[:,1]==side)&(s.queue[:,2]==mon))


def try_hit(c,actor,target,move,move_row,e):
    s,trace=c;last=s.move_history[target,s.active[target],0,0]
    blocked=(move==t.MOVE_IDS['disable'])&((last==0)|(last==t.MOVE_IDS['struggle']))
    def fail(c):
        s,trace=c;trace=trace._replace(events=trace.events.at[move_row,7].set(2))
        return e._event((s,trace),e.FAIL,actor,s.active[actor])
    return e._when(blocked,fail,c),blocked


def status_move(c,actor,target,move,e):
    s,trace=c;mon=s.active[target];kind=RESTRICTION[move]
    present=s.volatile_state[target,mon,kind,0]>0;last=s.move_history[target,mon,0,0]
    will=_will_move(s,target,mon,e);owns=jnp.any(s.moves[target,mon]==last)
    has_pp=jnp.any((s.moves[target,mon]==last)&(s.pp[target,mon]>0))
    encore=(last>0)&(~ENCORE_BLOCKED[last])&owns&has_pp
    disable=(last>0)&((~owns)|has_pp)
    allowed=jnp.where(kind==VOLATILE_IDS['encore'],encore,jnp.where(kind==VOLATILE_IDS['disable'],disable,True))
    def create(c):
        c=e.effects.add_volatile(c,target,kind,actor,move,e)
        s,trace=c;row=s.volatile_state[target,mon,kind]
        turns=s.action_state[target,mon,e.tracking.ACTIVE_TURNS]
        duration=jnp.where(kind==VOLATILE_IDS['taunt'],3+((turns>0)&(~will)).astype(jnp.int32),
                 jnp.where(kind==VOLATILE_IDS['encore'],3+(~will).astype(jnp.int32),
                 jnp.where(kind==VOLATILE_IDS['disable'],5-will.astype(jnp.int32),0)))
        row=row.at[5].set(duration).at[7].set(jnp.where((kind==16)|(kind==17),last,0))
        row=jnp.where(allowed,row,0)
        return s._replace(volatile_state=s.volatile_state.at[target,mon,kind].set(row)),trace
    c=e._when(~present,create,c)
    def success(c):return e._event(c,e.VOL_START,target,mon,move=last,value=kind)
    def failed(c):
        s,trace=c;index=jnp.max(jnp.where((jnp.arange(e.EVENTS)<trace.ecount)&(trace.events[:,0]==e.EV_MOVE),jnp.arange(e.EVENTS),0))
        trace=trace._replace(events=trace.events.at[index,7].set(2))
        return e._event((s,trace),e.FAIL,actor,s.active[actor])
    return e.jax.lax.cond((~present)&allowed,success,failed,c)
