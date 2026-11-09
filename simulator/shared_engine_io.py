"""Host protocol adapter for shared_engine's numeric, recipient-scoped events.

No native oracle is imported. Only own roster information enters a request;
opposing HP is rounded exactly as HP Percentage Mod. Trusted state and queued
choices must never be passed to a policy.
"""
import jax
import jax.numpy as jnp
import numpy as np
from . import shared_engine as e
from . import gen5_engine_tables as tables
from . import gen5_tables as generic


def condition(hp,maxhp,status=0,public=False):
    if hp==0:return '0 fnt'
    shown=(hp*100+maxhp-1)//maxhp if public else hp
    if public and shown==100 and hp<maxhp:shown=99
    base=100 if public else maxhp
    return f'{shown}/{base}'+(' '+e.STATUS_NAMES[status] if status else '')


def details(s,side,mon):
    species=e.SPECIES[int(s.species[side,mon])];level=int(s.level[side,mon]);gender=int(s.gender[side,mon])
    return species+(f', L{level}' if level!=100 else '')+(', '+('','M','F')[gender] if gender else '')+(', shiny' if bool(s.shiny[side,mon]) else '')


def project_request(state,side):
    s=jax.device_get(state);kind=int(s.request[side])
    if kind==e.END:return None
    order=s.order[side];roster=[]
    for mon in order:
        sp=int(s.species[side,mon])
        if not sp:continue
        def move_key(mid):
            if mid==e.HIDDEN_POWER:return 'hiddenpower'+generic.DATA['types'][int(s.hp_type[side,mon])-1]['id']
            if mid in (e.RETURN,e.FRUSTRATION):
                happy=int(s.happiness[side,mon]);power=max(1,(happy if mid==e.RETURN else 255-happy)*10//25)
                return e.MOVE_KEYS[mid]+str(power)
            return e.MOVE_KEYS[mid]
        roster.append({'ident':f'p{side+1}: '+e.SPECIES[int(s.name[side,mon])],'details':details(s,side,mon),
         'condition':condition(int(s.hp[side,mon]),int(s.stats[side,mon,0]),int(s.status[side,mon])),
         'active':bool(mon==s.active[side]),
         'stats':dict(zip(('atk','def','spa','spd','spe'),map(int,s.stats[side,mon,1:]))),
         'moves':[move_key(int(m)) for m in s.moves[side,mon] if m],
         'baseAbility':e.ABILITIES[int(s.ability[side,mon])],'item':e.ITEMS[int(s.item[side,mon])],'pokeball':'pokeball'})
    req={'side':{'name':f'p{side+1}','id':f'p{side+1}','pokemon':roster}}
    if kind==e.WAIT:req['wait']=True
    elif kind==e.FORCED:
        req['forceSwitch']=[True]
        if np.sum(s.request==e.FORCED)==1:req['noCancel']=True
    else:
        mon=int(s.active[side]);rows=[]
        for slot,mid in enumerate(s.moves[side,mon]):
            mid=int(mid)
            if not mid:continue
            name=e.MOVE_NAMES[mid]
            if mid==e.HIDDEN_POWER:name+=' '+generic.DATA['types'][int(s.hp_type[side,mon])-1]['name']+' '+str(int(s.hp_power[side,mon]))
            if mid in (e.RETURN,e.FRUSTRATION):
                happy=int(s.happiness[side,mon]);name+=' '+str(max(1,(happy if mid==e.RETURN else 255-happy)*10//25))
            rows.append({'move':name,'id':e.MOVE_KEYS[mid],'pp':int(s.pp[side,mon,slot]),
                         'maxpp':int(e.PP[mid]),'target':tables.TARGETS[mid],
                         'disabled':bool(e.hook_ops.disabled(s)[side,mon,slot])})
        if all(row['disabled'] for row in rows):rows=[{'move':'Struggle','id':'struggle','target':'randomNormal','disabled':False}]
        req['active']=[{'moves':rows}]
    return req


def project_events(state,trace,audience):
    s=jax.device_get(state);t=jax.device_get(trace);out=[]
    for row in t.events[:int(t.ecount)]:
        kind,side,mon,move,target,tm,value,aux,status,_=map(int,row)
        ident=lambda side,mon:f'p{side+1}a: '+e.SPECIES[int(s.name[side,mon])]
        actor=ident(side,mon);foe=ident(target,tm);name=e.MOVE_NAMES[move] if 0<=move<len(e.MOVE_NAMES) else ''
        if kind==e.BLANK:line='|'
        elif kind==e.EV_MOVE:
            shown='' if aux==2 else foe.replace('a: ',': ') if aux in (3,4) else foe
            annotation={0:'',1:'|[miss]',2:'|[still]',3:'|[notarget]',4:''}[aux]
            line=f'|move|{actor}|{name}|'+shown+annotation
        elif kind in (e.DAMAGE,e.HEAL):
            line=f'|-'+('damage' if kind==e.DAMAGE else 'heal')+f'|{actor}|'+condition(value,aux,status,side!=audience)
            if kind==e.DAMAGE and move:line+='|[from] '+{1:'brn',2:'psn',3:'recoil',4:'Stealth Rock',5:'Spikes',6:'confusion',7:'Sandstorm',8:'Hail',9:'item: Life Orb'}[move]
            if kind==e.HEAL and move:line+='|[silent]' if move==-1 else '|[from] ability: Ice Body' if move==2 else '|[from] item: Leftovers'
        elif kind==e.CRIT:line=f'|-crit|{actor}'
        elif kind==e.IMMUNE:line=f'|-immune|{actor}'+({0:'',1:'|[from] ability: Immunity',2:'|[from] ability: Levitate'}[aux])
        elif kind==e.MISS:line=f'|-miss|{actor}|{foe}'
        elif kind==e.FAINT:line=f'|faint|{actor}'
        elif kind==e.WIN:line=f'|win|p{value}' if value else '|tie'
        elif kind==e.TURN:line=f'|turn|{value}'
        elif kind==e.UPKEEP:line='|upkeep'
        elif kind==e.SWITCH:line=f'|switch|{actor}|{details(s,side,mon)}|'+condition(value,aux,status,side!=audience)+('|[from] U-turn' if move else '')
        elif kind==e.STATUS:line=f'|-status|{actor}|'+e.STATUS_NAMES[status]+(f'|[from] move: {name}' if status==e.SLP and move else '')
        elif kind==e.ABILITY:line=f'|-activate|{actor}|ability: Synchronize'
        elif kind==e.END_ITEM:line=f'|-enditem|{actor}|Lum Berry|[eat]'
        elif kind==e.CURE:line=f'|-curestatus|{actor}|'+e.STATUS_NAMES[status]+'|[msg]'
        elif kind==e.FAIL:line=f'|-fail|{actor}'+('|heal' if aux==1 else '|'+e.STATUS_NAMES[status] if status else '')
        elif kind==e.CANT:line=f'|cant|{actor}|'+e.STATUS_NAMES[status]
        elif kind==e.ACTIVATE:
            line=(f'|-supereffective|{actor}' if aux>0 else f'|-resisted|{actor}') if move==-1 else f'|-activate|{actor}|move: {name}'
        elif kind==e.SINGLE:line=f'|-singleturn|{actor}|Protect'
        elif kind==e.SIDE_START:line=f'|-sidestart|p{side+1}: p{side+1}|'+('move: ' if move in (e.STEALTH_ROCK,e.field_tables.SCREEN_MOVES[1]) else '')+name
        elif kind==e.SIDE_END:line=f'|-sideend|p{side+1}: p{side+1}|'+('move: ' if move==e.field_tables.SCREEN_MOVES[1] else '')+name+(f'|[from] move: Rapid Spin|[of] {actor}' if aux==0 else '')
        elif kind==e.CONF_START:line=f'|-start|{actor}|confusion'
        elif kind==e.CONF_END:line=f'|-end|{actor}|confusion'
        elif kind==e.CONF_ACTIVATE:line=f'|-activate|{actor}|confusion'
        elif kind==e.SLEEP_CLAUSE:
            out.append({'type':'event','message':'|-message|Sleep Clause Mod activated.'})
            if value:out.append({'type':'event','message':"|-hint|Sleep Clause Mod prevents players from putting more than one of their opponent's Pokemon to sleep at a time"})
            continue
        elif kind==e.BOOST:line=f'|-'+('unboost' if status else 'boost')+f'|{actor}|{e.field_tables.STATS[aux]}|{abs(value)}'
        elif kind==e.WEATHER:line='|-weather|'+e.field_tables.WEATHER_NAMES[value]+(f'|[from] ability: {e.hook_tables.ABILITY_NAMES[move]}|[of] {actor}' if aux==2 else '|[upkeep]' if aux==1 else '')
        elif kind==e.REVEAL:line=f'|-ability|{actor}|'+e.hook_tables.ABILITY_NAMES[value]+('|boost' if aux else '')
        else:raise ValueError('unknown event')
        out.append({'type':'event','message':line})
    return out


def draw_rows(trace):
    t=jax.device_get(trace);out=[]
    for k in range(int(t.dcount)):
        lo,hi,form=map(int,t.args[k]);args=[2**32] if form==3 else [hi] if form==1 else [lo,hi]
        out.append({'args':args,'value':int(t.values[k]),'before':t.before[k].tolist(),'after':t.after[k].tolist(),'context':e.DRAW_CONTEXTS[int(t.context[k])]})
    return out


def player_view(s,side):
    """Only a private request and recipient-rendered history leave the adapter."""
    trace=e.empty_trace()._replace(events=s.history,ecount=s.hcount)
    prefix=['|gametype|singles','|player|p1|p1||','|player|p2|p2||','|gen|5','|tier|[Gen 5] Random Battle',
            '|rule|Sleep Clause Mod: Limit one foe put to sleep','|rule|HP Percentage Mod: HP is shown in percentages',
            "|rule|Illusion Level Mod: Illusion disguises the Pokemon's true level",'|',
            f'|teamsize|p1|{int(np.sum(np.asarray(s.species[0])>0))}',f'|teamsize|p2|{int(np.sum(np.asarray(s.species[1])>0))}','|start']
    return {'request':project_request(s,side),'events':[{'type':'event','message':line} for line in prefix]+project_events(s,trace,side)}


def save_state(path,state):
    """Explicit schema/table identity, no pickle or positional tuple loading."""
    e._validate_structure(state)
    np.savez(path,schema=np.int32(e.SCHEMA),data_sha256=np.asarray(generic.DATA_SHA256),
             **dict(zip(e.State._fields,jax.device_get(state),strict=True)))


def load_state(path,*,schema=None):
    """Load current states with explicit provenance-preserving legacy migrations."""
    with np.load(path,allow_pickle=False) as saved:
        stored=int(saved['schema']) if 'schema' in saved else None
        if schema is not None and stored is not None and stored!=schema:raise ValueError('schema mismatch')
        version=stored if schema is None else schema
        if version not in (1,2,3,4,5,6):raise ValueError('explicit supported state schema required')
        fields={k:jnp.asarray(saved[k]) for k in saved.files if k not in ('schema','data_sha256')}
        if version in (3,4,5,6):
            if 'data_sha256' not in saved or str(saved['data_sha256'])!=generic.DATA_SHA256:
                raise ValueError('state data table identity mismatch')
        else:
            legacy=set(e.State._fields[:e.State._fields.index('ivs')])
            if version==1:legacy.remove('hazards')
            if set(fields)!=legacy:raise ValueError('invalid legacy state fields')
            if version==1:fields['hazards']=jnp.zeros((2,3),jnp.int32)
            species=fields['species']
            if np.any(np.asarray(species)>3) or np.any(np.asarray(species)<0):raise ValueError('invalid legacy species IDs')
            ivs=jnp.full((2,6,6),31,jnp.int32);evs=jnp.zeros((2,6,6),jnp.int32)
            # Frozen schemas permit only Speed EVs 0 or 252. Stored stats
            # retain unmodified Speed even when status alters cached speed.
            # If low-level rounding makes both spreads equal, choose canonical
            # zero EVs: the old schema cannot recover that unrecorded genotype.
            baseline=((2*e.BASE_STATS[species,5]+31)*fields['level'])//100+5
            boosted=(species>0)&(fields['stats'][:,:,5]!=baseline)
            evs=evs.at[:,:,5].set(jnp.where(boosted,252,0))
            hp_type,power=generic.hidden_power(ivs)
            fields.update(ivs=ivs,evs=evs,nature=jnp.full((2,6),generic.NATURES['hardy'],jnp.int32),
                ability=tables.SPECIES_ABILITY[species],types=tables.SPECIES_TYPES[species],
                hp_type=jnp.where(species>0,generic.HP_TYPE_IDS[hp_type],0),hp_power=jnp.where(species>0,power,0),
                shiny=jnp.zeros((2,6),jnp.bool_),happiness=jnp.full((2,6),255,jnp.int32),
                name=tables.SPECIES_NAME[species],boosts=jnp.zeros((2,6,7),jnp.int32))
        if version<5:
            # Schema 4 could not record hazard effect creation/source metadata.
            history=np.asarray(fields['history'])[:int(fields['hcount'])]
            if np.any(np.asarray(fields.get('hazards',np.zeros((2,3),np.int32)))) or np.any(np.isin(history[:,0],[e.SIDE_START,e.SIDE_END])):
                raise ValueError('legacy hazard state requires replay/provenance for schema 5 effect order')
            fields.update(weather_state=e.field_ops.empty_weather(),screens=jnp.zeros((2,2,5),jnp.int32),
                          hazard_meta=jnp.zeros((2,2,3),jnp.int32),
                          sleep_clause_target=jnp.asarray(np.any(np.isin(history[:,0],[e.STATUS,e.SLEEP_CLAUSE]))))
        if version<4:
            # Earlier schemas omitted source/timer/effectOrder state. Only fresh
            # starts/private initial commitments can be reconstructed exactly.
            if int(fields['hcount'])!=3 or np.any(np.asarray(fields['status'])) or np.any(np.asarray(fields['stall_duration'])):
                raise ValueError('advanced legacy state requires replay/provenance to reconstruct schema 4 effect state')
            fields.update(effect_order=e.i32(0),status_state=e.effects.empty_status(),
                volatile_state=jnp.zeros((2,6,3,6),jnp.int32),ability_order=jnp.zeros((2,6),jnp.int32),
                item_order=jnp.zeros((2,6),jnp.int32))
            migrated=e.State(*(fields[k] for k in e.State._fields))
            for side in range(2):migrated,_=e.effects.activate((migrated,e.empty_trace()),e.i32(side),e)
            fields=dict(zip(e.State._fields,migrated,strict=True))
        if version<6:
            # Earlier engines explicitly rejected these hooks. Promoting an old
            # unsupported initializer would fabricate its startup/lock history.
            old_abilities=np.asarray(fields['ability']);old_items=np.asarray(fields['item'])
            if np.any(old_abilities>3) or np.any(old_abilities<0) or np.any(old_items<0) or np.any(old_items>=len(e.ITEMS)) or not np.all(np.asarray(e.field_tables.ITEM_IMPLEMENTED)[np.clip(old_items,0,len(e.ITEMS)-1)]):
                raise ValueError('legacy hook state requires replay/provenance')
            if fields['volatile_state'].shape!=(2,6,3,6):raise ValueError('invalid legacy volatile shape')
            fields['volatile_state']=jnp.pad(fields['volatile_state'],((0,0),(0,0),(0,1),(0,0)))
        if set(fields)!=set(e.State._fields):raise ValueError('invalid state field set')
        state=e.State(*(fields[k] for k in e.State._fields))
        e._validate_structure(state)
        return state


def effect_states(state,side,mon):
    """Trusted oracle/serialization projection; never an actor observation."""
    s=jax.device_get(state)
    target={'$pokemon':{'side':side,'mon':mon}}
    mid,order,ss,sm,present,time,start=map(int,s.status_state[side,mon])
    status={'id':e.STATUS_NAMES[mid],'effectOrder':order}
    if present:status['target']=target
    if ss>=0:status['source']={'$pokemon':{'side':ss,'mon':sm}}
    if time>=0:status['time']=time
    if start>=0:status['startTime']=start
    if mid==e.TOX:status['stage']=int(s.toxic[side,mon])
    volatiles={}
    for kind,key in enumerate(('protect','stall','confusion','choicelock')):
        exists,order,ss,sm,move,time=map(int,s.volatile_state[side,mon,kind])
        if not exists:continue
        row={'id':key,'name':key,'effectOrder':order,'target':target,
             'source':{'$pokemon':{'side':ss,'mon':sm}},'sourceSlot':f'p{ss+1}a'}
        if move>0:row['sourceEffect']={'$effect':e.MOVE_KEYS[move],'type':'Move'}
        if move<0:row['sourceEffect']={'$effect':e.ITEMS[-move],'type':'Item'}
        if kind==0:row['duration']=1
        if kind==1:row.update(counter=int(s.stall[side,mon]),duration=int(s.stall_duration[side,mon]))
        if kind==2:row['time']=time
        if kind==3:row['move']=e.MOVE_KEYS[time]
        volatiles[key]=row
    return {'statusState':status,'volatiles':volatiles,
        'abilityState':{'id':e.ABILITIES[int(s.ability[side,mon])],'effectOrder':int(s.ability_order[side,mon]),'target':target},
        'itemState':{'id':e.ITEMS[int(s.item[side,mon])],'effectOrder':int(s.item_order[side,mon]),'target':target},
        'speciesState':{'id':e.generic.to_id(e.SPECIES[int(s.species[side,mon])]),'effectOrder':0}}


def field_state(state):
    s=jax.device_get(state);weather,side,mon,move,duration,target=map(int,s.weather_state)
    current={'id':e.field_tables.WEATHER_KEYS[weather],'effectOrder':0}
    if weather:
        current.update(duration=duration,source={'$pokemon':{'side':side,'mon':mon}},sourceSlot=f'p{side+1}a')
    if target:current['target']={'$field':True}
    clause={'effectOrder':0,'id':'sleepclausemod','source':None}
    if bool(s.sleep_clause_target):clause['target']={'$field':True}
    return {'weather':e.field_tables.WEATHER_KEYS[weather],'weatherState':current,
            'pseudoWeather':{'sleepclausemod':clause},
            'terrain':'','terrainState':{'effectOrder':0,'id':''}}


def side_conditions(state,side):
    s=jax.device_get(state);out={}
    for index,key in enumerate(('stealthrock','spikes')):
        layers=int(s.hazards[side,index])
        if not layers:continue
        order,source,mon=map(int,s.hazard_meta[side,index])
        out[key]={'id':key,'effectOrder':order,'source':{'$pokemon':{'side':source,'mon':mon}},
                  'sourceSlot':f'p{source+1}a','target':{'$side':side}}
        if index==1:out[key]['layers']=layers
    for index,key in enumerate(e.field_tables.SCREEN_KEYS):
        duration,source,mon,move,order=map(int,s.screens[side,index])
        if not duration:continue
        out[key]={'id':key,'effectOrder':order,'duration':duration,
                  'source':{'$pokemon':{'side':source,'mon':mon}},'sourceSlot':f'p{source+1}a','target':{'$side':side}}
    return out
