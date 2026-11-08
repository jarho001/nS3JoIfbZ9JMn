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
    return f'{shown}/{base}'+(' '+('','brn','par','tox','fnt')[status] if status else '')


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
                         'disabled':bool(s.pp[side,mon,slot]==0)})
        if all(row['disabled'] for row in rows):rows=[{'move':'Struggle','id':'struggle','target':'randomNormal','disabled':False}]
        req['active']=[{'moves':rows}]
    return req


def project_events(state,trace,audience):
    s=jax.device_get(state);t=jax.device_get(trace);out=[]
    for row in t.events[:int(t.ecount)]:
        kind,side,mon,move,target,tm,value,aux,status,_=map(int,row)
        ident=lambda side,mon:f'p{side+1}a: '+e.SPECIES[int(s.name[side,mon])]
        actor=ident(side,mon);foe=ident(target,tm);name=e.MOVE_NAMES[move] if move>=0 else ''
        if kind==e.BLANK:line='|'
        elif kind==e.EV_MOVE:
            shown='' if aux==2 else foe.replace('a: ',': ') if aux in (3,4) else foe
            annotation={0:'',1:'|[miss]',2:'|[still]',3:'|[notarget]',4:''}[aux]
            line=f'|move|{actor}|{name}|'+shown+annotation
        elif kind in (e.DAMAGE,e.HEAL):
            line=f'|-'+('damage' if kind==e.DAMAGE else 'heal')+f'|{actor}|'+condition(value,aux,status,side!=audience)
            if kind==e.DAMAGE and move:line+='|[from] '+{1:'brn',2:'psn',3:'recoil',4:'Stealth Rock',5:'Spikes'}[move]
            if kind==e.HEAL and move:line+='|[from] item: Leftovers'
        elif kind==e.CRIT:line=f'|-crit|{actor}'
        elif kind==e.IMMUNE:line=f'|-immune|{actor}'+({0:'',1:'|[from] ability: Immunity',2:'|[from] ability: Levitate'}[aux])
        elif kind==e.MISS:line=f'|-miss|{actor}|{foe}'
        elif kind==e.FAINT:line=f'|faint|{actor}'
        elif kind==e.WIN:line=f'|win|p{value}' if value else '|tie'
        elif kind==e.TURN:line=f'|turn|{value}'
        elif kind==e.UPKEEP:line='|upkeep'
        elif kind==e.SWITCH:line=f'|switch|{actor}|{details(s,side,mon)}|'+condition(value,aux,status,side!=audience)+('|[from] U-turn' if move else '')
        elif kind==e.STATUS:line=f'|-status|{actor}|'+('','brn','par','tox')[status]
        elif kind==e.ABILITY:line=f'|-activate|{actor}|ability: Synchronize'
        elif kind==e.END_ITEM:line=f'|-enditem|{actor}|Lum Berry|[eat]'
        elif kind==e.CURE:line=f'|-curestatus|{actor}|'+('','brn','par','tox')[status]+'|[msg]'
        elif kind==e.FAIL:line=f'|-fail|{actor}'+('|heal' if aux==1 else '|'+('','brn','par','tox')[status] if status else '')
        elif kind==e.CANT:line=f'|cant|{actor}|par'
        elif kind==e.ACTIVATE:
            line=(f'|-supereffective|{actor}' if aux>0 else f'|-resisted|{actor}') if move==-1 else f'|-activate|{actor}|move: {name}'
        elif kind==e.SINGLE:line=f'|-singleturn|{actor}|Protect'
        elif kind==e.SIDE_START:line=f'|-sidestart|p{side+1}: p{side+1}|'+('move: ' if move==e.STEALTH_ROCK else '')+name
        elif kind==e.SIDE_END:line=f'|-sideend|p{side+1}: p{side+1}|{name}|[from] move: Rapid Spin|[of] {actor}'
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
    """Load schema 3 or explicitly migrate frozen schema 1/2 keyed states."""
    with np.load(path,allow_pickle=False) as saved:
        stored=int(saved['schema']) if 'schema' in saved else None
        if schema is not None and stored is not None and stored!=schema:raise ValueError('schema mismatch')
        version=stored if schema is None else schema
        if version not in (1,2,3):raise ValueError('explicit supported state schema required')
        fields={k:jnp.asarray(saved[k]) for k in saved.files if k not in ('schema','data_sha256')}
        if version==3:
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
        if set(fields)!=set(e.State._fields):raise ValueError('invalid state field set')
        state=e.State(*(fields[k] for k in e.State._fields))
        e._validate_structure(state)
        return state
