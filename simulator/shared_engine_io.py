"""Host protocol adapter for shared_engine's numeric, recipient-scoped events.

No native oracle is imported. Only own roster information enters a request;
opposing HP is rounded exactly as HP Percentage Mod. Trusted state and queued
choices must never be passed to a policy.
"""
import json
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
    species=e.SPECIES[int(s.base_species[side,mon])];level=int(s.level[side,mon]);gender=int(s.gender[side,mon])
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
         'stats':dict(zip(('atk','def','spa','spd','spe'),map(int,s.base_stats[side,mon,1:]))),
         'moves':[move_key(int(m)) for m in s.moves[side,mon] if m],
         'baseAbility':e.ABILITIES[int(s.base_ability[side,mon])],'item':e.ITEMS[int(s.item[side,mon])],'pokeball':'pokeball'})
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
                         'maxpp':min(5,int(e.identity_ops.BASE_PP[mid])) if bool(s.transformed[side,mon]) else int(e.PP[mid]),'target':tables.TARGETS[mid],
                         'disabled':bool((e.hook_ops.disabled(s)|e.phase_ops.disabled(s))[side,mon,slot])})
        if all(row['disabled'] for row in rows):rows=[{'move':'Struggle','id':'struggle','target':'randomNormal','disabled':False}]
        forced=int(e.phase_ops.forced_move(s)[side])
        req['active']=[{'moves':[{'move':'Recharge' if forced<0 else e.MOVE_NAMES[forced],'id':'recharge' if forced<0 else e.MOVE_KEYS[forced]}],'trapped':True} if forced else {'moves':rows}]
        if not forced and bool(e.trap_ops.has_bench(s,e)[side]):
            truth=int(s.action_state[side,mon,e.tracking.TRAPPED])
            if truth==1 or int(s.request_state[side,0]):req['active'][0]['trapped']=True
            elif bool(s.action_state[side,mon,e.tracking.MAYBE_TRAPPED]):req['active'][0]['maybeTrapped']=True
    return req


def project_events(state,trace,audience):
    s=jax.device_get(state);t=jax.device_get(trace);out=[]
    for row in t.events[:int(t.ecount)]:
        kind,side,mon,move,target,tm,value,aux,status,from_lock=map(int,row[:10])
        aname,tname,dspecies,dlevel,dgender,dshiny=map(int,row[10:])
        shown_details=e.SPECIES[dspecies]+(f", L{dlevel}" if dlevel!=100 else "")+(", "+("","M","F")[dgender]if dgender else "")+(", shiny"if dshiny else "")
        ident=lambda side,mon:f'p{side+1}a: '+e.SPECIES[aname if side==int(row[1]) and mon==int(row[2]) else tname]
        actor=ident(side,mon);foe=ident(target,tm);name=e.MOVE_NAMES[move] if 0<=move<len(e.MOVE_NAMES) else ''
        if kind==e.BLANK:line='|'
        elif kind in (e.EV_MOVE,e.ANIM):
            shown='' if aux==2 else foe.replace('a: ',': ') if aux in (3,4) else foe
            annotation={0:'',1:'|[miss]',2:'|[still]',3:'|[notarget]',4:''}[aux]
            line='|'+('move' if kind==e.EV_MOVE else '-anim')+f'|{actor}|{name}|'+shown+('|[from] move: '+e.MOVE_NAMES[-from_lock-1]if from_lock<0 else '|[from] lockedmove' if from_lock==1 else '|[from] move: Pursuit' if from_lock==2 else '|[from] '+e.MOVE_NAMES[tables.MOVE_IDS[e.phase_ops.VOLATILES[from_lock-e.called_ops.CONDITION_TAG]]]if from_lock>=e.called_ops.CONDITION_TAG else '|[from] ability: '+e.hook_tables.ABILITY_NAMES[from_lock-e.called_ops.ABILITY_TAG]if from_lock>=e.called_ops.ABILITY_TAG else '')+annotation
        elif kind in (e.DAMAGE,e.HEAL):
            line=f'|-'+('damage' if kind==e.DAMAGE else 'heal')+f'|{actor}|'+condition(value,aux,status,side!=audience)
            if kind==e.DAMAGE and move:line+='|[from] '+{1:'brn',2:'psn',3:'recoil',4:'Stealth Rock',5:'Spikes',6:'confusion',7:'Sandstorm',8:'Hail',9:'item: Life Orb',10:'Recoil',11:'ability: Rough Skin',12:'ability: Iron Barbs',13:'item: Rocky Helmet',14:'ability: Liquid Ooze',15:'ability: Dry Skin'}[move]
            if kind==e.DAMAGE and move in (11,12,13,14,15):line+='|[of] '+foe
            if kind==e.HEAL and move:line+='|[silent]' if move==-1 else '|[from] ability: Ice Body' if move==2 else '|[from] drain|[of] '+foe if move==3 else '|[from] ability: Rain Dish'if move==8 else '|[from] ability: Dry Skin'if move==7 else '|[from] ability: '+{4:'Water Absorb',5:'Volt Absorb',6:'Dry Skin'}[move]+'|[of] '+foe if move in (4,5,6) else '|[from] item: Leftovers'
        elif kind==e.CRIT:line=f'|-crit|{actor}'
        elif kind==e.IMMUNE:line=f'|-immune|{actor}'+('|confusion|[from] ability: Own Tempo'if aux==5 else '|[from] ability: '+e.hook_tables.ABILITY_NAMES[value]if aux==4 else {0:'',1:'|[from] ability: Immunity',2:'|[from] ability: Levitate',3:'|[from] ability: Soundproof'}[aux])
        elif kind==e.MISS:line=f'|-miss|{actor}|{foe}'
        elif kind==e.FAINT:line=f'|faint|{actor}'
        elif kind==e.TURN_WARNING:
            left=1000-value;duration='1 turn' if left==1 else f'{left} turns'
            line=f"|bigerror|You will auto-tie if the battle doesn't end in {duration} (on turn 1000)."
        elif kind==e.TURN_LIMIT:line='|message|It is turn 1000. You have hit the turn limit!'
        elif kind==e.WIN:line=f'|win|p{value}' if value else '|tie'
        elif kind==e.TURN:line=f'|turn|{value}'
        elif kind==e.UPKEEP:line='|upkeep'
        elif kind==e.YAWN_START:line=f'|-start|{actor}|move: Yawn|[of] {foe}'
        elif kind==e.YAWN_END:line=f'|-end|{actor}|move: Yawn|[silent]'
        elif kind==e.FLASH_START:line=f'|-start|{actor}|ability: Flash Fire'
        elif kind==e.FLASH_END:line=f'|-end|{actor}|ability: Flash Fire|[silent]'
        elif kind==e.SUPPRESS_ABILITY:line=f'|-endability|{actor}'
        elif kind==e.STATUS_ABILITY:line=f'|-activate|{actor}|ability: '+e.hook_tables.ABILITY_NAMES[value]
        elif kind==e.REST_ABILITY:line=f'|-fail|{actor}|[from] ability: '+e.hook_tables.ABILITY_NAMES[value]+f'|[of] {actor}'
        elif kind==e.ABILITY_CHANGE:line=(f'|-activate|{foe}|ability: Mummy|{actor}|[ability] '+e.hook_tables.ABILITY_NAMES[aux])if status==3 else (f'|-ability|{actor}|'+e.hook_tables.ABILITY_NAMES[value]+'|'+e.hook_tables.ABILITY_NAMES[aux]+f'|[from] ability: Trace|[of] {foe}')if status==2 else f'|-ability|{actor}|'+e.hook_tables.ABILITY_NAMES[value]+'|'+e.hook_tables.ABILITY_NAMES[aux]+f'|[from] move: {name}'+(''if status else f'|[of] {foe}')
        elif kind==e.ABILITY_SWAP:line=f'|-activate|{actor}|Skill Swap|'+e.hook_tables.ABILITY_NAMES[value]+'|'+e.hook_tables.ABILITY_NAMES[aux]+f'|[of] {foe}'
        elif kind==e.FORM_CHANGE:line=f'|-formechange|{actor}|'+e.SPECIES[value]+('|[msg]' if status in (0,1) else '|[silent]'if status==3 else '|')+('|[from] ability: '+e.hook_tables.ABILITY_NAMES[aux]if aux else '')
        elif kind==e.ILLUSION_END:
            out.extend({'type':'event','message':line}for line in (f'|replace|{actor}|{shown_details}',f'|-end|{actor}|Illusion',"|-hint|Illusion Level Mod is active, so this Pokémon's true level was hidden."));continue
        elif kind==e.TRANSFORM:line=f'|-transform|{actor}|{foe}'+('|[from] ability: Imposter' if aux else '')
        elif kind==e.ITEM_SWAP:line=f'|-activate|{actor}|move: Trick|[of] {foe}'
        elif kind==e.ITEM_GAIN:line=f'|-item|{actor}|'+e.hook_tables.ITEM_NAMES[value]+f'|[from] move: {name}'
        elif kind==e.ITEM_LOSS:line=f'|-enditem|{actor}|'+e.hook_tables.ITEM_NAMES[value]+('|[silent]' if aux else '')+f'|[from] move: {name}'+(f'|[of] {foe}' if move==e.item_ops.KNOCK else '')
        elif kind==e.PURSUIT_HINT:line="|-hint|A Pokemon can't switch between when it runs out of HP and when it faints"
        elif kind==e.PHAZE_ABILITY:line=f'|-activate|{actor}|ability: '+e.hook_tables.ABILITY_NAMES[value]
        elif kind==e.DRAG:line=f'|drag|{actor}|{shown_details}|'+condition(value,aux,status,side!=audience)
        elif kind==e.SWITCH:line=f'|switch|{actor}|{shown_details}|'+condition(value,aux,status,side!=audience)+('|[from] U-turn' if move else '')
        elif kind==e.STATUS:line=f'|-status|{actor}|'+e.STATUS_NAMES[status]+(f'|[from] move: {name}' if status==e.SLP and move else '')
        elif kind==e.ABILITY:line=f'|-activate|{actor}|ability: Synchronize'
        elif kind==e.END_ITEM:line=f'|-enditem|{actor}|Lum Berry|[eat]'
        elif kind==e.CURE:line=f'|-curestatus|{actor}|'+e.STATUS_NAMES[status]+('|[from] ability: '+e.hook_tables.ABILITY_NAMES[aux]+('|[silent]'if value else '') if aux else f'|[from] move: {name}' if move else '|[msg]')
        elif kind==e.FAIL:line=f'|-fail|{actor}'+('|move: Substitute'+('|[weak]' if aux==3 else '') if aux in (2,3) else '|heal' if aux==1 else '|'+e.STATUS_NAMES[status] if status else '')
        elif kind==e.CANT:line=f'|cant|{actor}|'+e.STATUS_NAMES[status]
        elif kind==e.ACTIVATE:
            line=(f'|-supereffective|{actor}' if aux>0 else f'|-resisted|{actor}') if move==-1 else f'|-activate|{actor}|move: {name}'
        elif kind==e.PARTIAL_START:line=f'|-activate|{actor}|move: {name}|[of] {foe}'
        elif kind==e.PARTIAL_END:line=f'|-end|{actor}|{name}|[partiallytrapped]'+('|[silent]' if aux else '')
        elif kind==e.PARTIAL_DAMAGE:line=f'|-damage|{actor}|'+condition(value,aux,status,side!=audience)+f'|[from] move: {name}|[partiallytrapped]'
        elif kind==e.SUB_DAMAGE:line=f'|-activate|{actor}|Substitute|[damage]'
        elif kind==e.SINGLE:line=f'|-singleturn|{actor}|'+('move: Magic Coat'if move==e.called_ops.MAGIC_COAT else 'Protect')
        elif kind==e.SIDE_START:line=f'|-sidestart|p{side+1}: p{side+1}|'+('move: ' if move in (e.STEALTH_ROCK,e.field_tables.SCREEN_MOVES[1]) else '')+name
        elif kind==e.SIDE_END:line=f'|-sideend|p{side+1}: p{side+1}|'+('move: ' if move==e.field_tables.SCREEN_MOVES[1] else '')+name+(f'|[from] move: Rapid Spin|[of] {actor}' if aux==0 else '')
        elif kind==e.CONF_START:line=f'|-start|{actor}|confusion'+('|[fatigue]' if aux else '')
        elif kind==e.CONF_END:line=f'|-end|{actor}|confusion'
        elif kind==e.CONF_ACTIVATE:line=f'|-activate|{actor}|confusion'
        elif kind==e.SLEEP_CLAUSE:
            out.append({'type':'event','message':'|-message|Sleep Clause Mod activated.'})
            if value:out.append({'type':'event','message':"|-hint|Sleep Clause Mod prevents players from putting more than one of their opponent's Pokemon to sleep at a time"})
            continue
        elif kind==e.BOOST:line=f'|-'+('unboost' if status else 'boost')+f'|{actor}|{e.field_tables.STATS[aux]}|{abs(value)}'
        elif kind==e.WEATHER:line='|-weather|'+e.field_tables.WEATHER_NAMES[value]+(f'|[from] ability: {e.hook_tables.ABILITY_NAMES[move]}|[of] {actor}' if aux==2 else '|[upkeep]' if aux==1 else '')
        elif kind==e.VOL_START:line=f'|-start|{actor}|'+('move: Taunt' if value==15 else e.MOVE_NAMES[tables.MOVE_IDS[e.phase_ops.VOLATILES[value]]])+('|'+name if value==17 else '')
        elif kind==e.VOL_END:line=f'|-end|{actor}|'+('move: Taunt' if value==15 else e.MOVE_NAMES[tables.MOVE_IDS[e.phase_ops.VOLATILES[value]]])
        elif kind==e.VOL_CANT:line=f'|cant|{actor}|'+('move: Taunt' if value==15 else 'Disable')+'|'+name
        elif kind==e.PREPARE:line=f'|-prepare|{actor}|{name}'
        elif kind==e.END_HELD_ITEM:line=f'|-enditem|{actor}|'+next(r['name'] for r in generic.DATA['items']if r['id']==e.ITEMS[value])
        elif kind==e.FLINCH_CANT:line=f'|cant|{actor}|flinch'
        elif kind==e.MUST_RECHARGE:line=f'|-mustrecharge|{actor}'
        elif kind==e.RECHARGE_CANT:line=f'|cant|{actor}|recharge'
        elif kind==e.HIT_COUNT:line=f'|-hitcount|'+(actor.replace('a: ',': ') if aux else actor)+f'|{value}'
        elif kind==e.FRISK_ITEM:line='|-item||'+e.hook_tables.ITEM_NAMES[value]+f'|[from] ability: Frisk|[of] {actor}'
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
    if not isinstance(s,e.State):raise ValueError('CoreState history requires the journal adapter')
    trace=e.empty_trace()._replace(events=s.history,ecount=s.hcount)
    prefix=['|gametype|singles','|player|p1|p1||','|player|p2|p2||','|gen|5','|tier|[Gen 5] Random Battle',
            '|rule|Sleep Clause Mod: Limit one foe put to sleep','|rule|HP Percentage Mod: HP is shown in percentages',
            "|rule|Illusion Level Mod: Illusion disguises the Pokemon's true level",'|',
            f'|teamsize|p1|{int(np.sum(np.asarray(s.species[0])>0))}',f'|teamsize|p2|{int(np.sum(np.asarray(s.species[1])>0))}','|start']
    return {'request':project_request(s,side),'events':[{'type':'event','message':line} for line in prefix]+project_events(s,trace,side)}


def save_state(path,state):
    """Explicit schema/table identity, no pickle or positional tuple loading."""
    if not isinstance(state,e.State):raise ValueError('CoreState persistence requires journal.save')
    e._validate_structure(state)
    np.savez(path,schema=np.int32(e.SCHEMA),data_sha256=np.asarray(generic.DATA_SHA256),
             **dict(zip(e.State._fields,jax.device_get(state),strict=True)))


_CAUSAL_FIELDS=('move_history','action_state','attacked_by','side_faints','battle_move','battle_success','battle_damage')


def _initial_causal(fields):
    occupied=fields['species']>0;active=fields['active'];side=jnp.arange(2)
    a=e.tracking.empty().at[:,:,e.tracking.NEWLY_SWITCHED].set(occupied.astype(jnp.int32))
    a=a.at[side,active,e.tracking.ACTIVE_TURNS].set(1).at[side,active,e.tracking.NEWLY_SWITCHED].set(0)
    a=a.at[side,active,e.tracking.SWITCH_COUNT].set(1)
    return dict(move_history=jnp.zeros((2,6,3,4),jnp.int32),action_state=a,
                attacked_by=jnp.zeros((2,6,e.tracking.ATTACK_RECORDS,7),jnp.int32),
                side_faints=jnp.full((2,2),-1,jnp.int32),battle_move=jnp.zeros(4,jnp.int32),battle_success=e.i32(0),battle_damage=e.i32(0))


def load_state(path,*,schema=None,replayed_state=None):
    """Load keyed states; advanced pre-causal states need an independent replay.

    replayed_state must be initialized/replayed from the original inputs and
    commands. Every field the old schema recorded must agree exactly. Missing
    causal fields are taken from that replay, never inferred from visible logs.
    """
    with np.load(path,allow_pickle=False) as saved:
        stored=int(saved['schema']) if 'schema' in saved else None
        if schema is not None and stored is not None and stored!=schema:raise ValueError('schema mismatch')
        version=stored if schema is None else schema
        if version not in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43):raise ValueError('explicit supported state schema required')
        fields={k:jnp.asarray(saved[k]) for k in saved.files if k not in ('schema','data_sha256')}
        if version in (3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43):
            if 'data_sha256' not in saved or str(saved['data_sha256'])!=generic.DATA_SHA256:
                raise ValueError('state data table identity mismatch')
        if version<39 and replayed_state is not None:
            e._validate_structure(replayed_state)
            end='ivs' if version<=2 else 'effect_order' if version==3 else 'weather_state' if version==4 else 'move_history' if version<9 else 'request_state'
            expected=set(e.State._fields if version>=32 else e.State._fields[:e.State._fields.index('ability_seek')] if version>=26 else e.State._fields[:e.State._fields.index('call_frames')] if version>=21 else e.State._fields[:e.State._fields.index('ability_staleness')] if version>=18 else e.State._fields[:e.State._fields.index('illusion')] if version==17 else e.State._fields[:e.State._fields.index('base_species')] if version==16 else e.State._fields[:e.State._fields.index('item_provenance')] if version>=11 else e.State._fields[:e.State._fields.index(end)])
            if version==1:expected.remove('hazards')
            if set(fields)!=expected:raise ValueError('invalid legacy field set for replay')
            for key,value in fields.items():
                replay=np.asarray(getattr(replayed_state,key));old=np.asarray(value)
                if key=='volatile_state' and version<9:
                    slots=3 if version<6 else 4 if version<8 else 5
                    if old.shape!=(2,6,slots,6):raise ValueError('invalid legacy volatile shape for replay')
                    replay=replay[:,:,:slots,:6]
                elif key=='volatile_state':
                    replay=replay[:,:,:27 if version>=30 else 26 if version>=25 else 25 if version>=24 else 24 if version>=20 else 23 if version>=15 else 22 if version>=13 else 21 if version==12 else 20,:]
                if key=='call_frames' and version<28:replay=replay[:,:16]
                if key=='history' and version<18:replay=replay[:,:10]
                if key=='action_state' and version<15:replay=replay[:,:,:21]
                if old.shape!=replay.shape or old.dtype!=replay.dtype or not np.array_equal(old,replay):
                    raise ValueError('legacy replay/provenance mismatch: '+key)
            return replayed_state
        if version<3:
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
            fields.update(_initial_causal(fields))
            fields['request_state']=jnp.zeros((2,2),jnp.int32)
            fields['item_provenance']=jnp.stack((fields['item'],jnp.zeros((2,6),jnp.int32)),axis=-1)
            fields.update(e.identity_ops.infer(fields,e));fields['illusion']=jnp.full((2,6),-1,jnp.int32);fields['ability_staleness']=jnp.zeros((2,6),jnp.int32);fields['call_frames']=e.called_ops.empty(e);fields['call_depth']=e.i32(0)
            fields['volatile_state']=jnp.pad(fields['volatile_state'],((0,0),(0,0),(0,len(e.phase_ops.VOLATILES)-3),(0,e.phase_ops.VOLATILE_WIDTH-6)))
            migrated=e.State(*(fields[k] for k in e.State._fields))
            for side in range(2):migrated,_=e.effects.activate((migrated,e.empty_trace()),e.i32(side),e)
            fields=dict(zip(e.State._fields,migrated,strict=True))
            fields['volatile_state']=fields['volatile_state'][:,:,:3,:6]
            for key in _CAUSAL_FIELDS:fields.pop(key)
            fields.pop('request_state');fields.pop('item_provenance')
            for key in (*e.identity_ops.FIELDS,'illusion','ability_staleness',*e.called_ops.FIELDS):fields.pop(key)
        if version<6:
            # Earlier engines explicitly rejected these hooks. Promoting an old
            # unsupported initializer would fabricate its startup/lock history.
            old_abilities=np.asarray(fields['ability']);old_items=np.asarray(fields['item'])
            if np.any(old_abilities>3) or np.any(old_abilities<0) or np.any(old_items<0) or np.any(old_items>=len(e.ITEMS)) or not np.all(np.asarray(e.field_tables.ITEM_IMPLEMENTED)[np.clip(old_items,0,len(e.ITEMS)-1)]):
                raise ValueError('legacy hook state requires replay/provenance')
            if fields['volatile_state'].shape!=(2,6,3,6):raise ValueError('invalid legacy volatile shape')
            fields['volatile_state']=jnp.pad(fields['volatile_state'],((0,0),(0,0),(0,1),(0,0)))
        if version<8:
            if fields['volatile_state'].shape!=(2,6,4,6):raise ValueError('invalid legacy volatile shape')
            # Earlier capability masks reject recharge moves, so no accepted
            # old continuation can contain the new forced phase.
            old_moves=np.asarray(fields['moves'])
            safe=np.clip(old_moves,0,len(e.MOVE_KEYS)-1)
            if np.any(np.asarray(e.phase_ops.RECHARGE)[safe]):raise ValueError('legacy recharge requires replay/provenance')
            fields['volatile_state']=jnp.pad(fields['volatile_state'],((0,0),(0,0),(0,1),(0,0)))
        if version<9:
            expected=set(e.State._fields[:e.State._fields.index('move_history')])
            if set(fields)!=expected:raise ValueError('invalid legacy state field set')
            history=np.asarray(fields['history'])[:int(fields['hcount'])]
            quiet=(int(fields['turn'])==1 and int(fields['hcount'])==3 and int(fields['qcount'])==0 and not bool(fields['midturn'])
                   and not np.any(np.asarray(fields['status'])) and not np.any(np.asarray(fields['boosts']))
                   and np.array_equal(history[:,0],[e.SWITCH,e.SWITCH,e.TURN]))
            if not quiet:raise ValueError('advanced legacy state requires replay/provenance for causal counters')
            fields.update(_initial_causal(fields))
            v=fields['volatile_state']
            if v.shape!=(2,6,5,6):raise ValueError('invalid legacy volatile shape')
            v=jnp.pad(v,((0,0),(0,0),(0,len(e.phase_ops.VOLATILES)-5),(0,e.phase_ops.VOLATILE_WIDTH-6)))
            kind=jnp.where(v[:,:,:,4]>0,e.phase_ops.SOURCE_MOVE,jnp.where(v[:,:,:,4]<0,e.phase_ops.SOURCE_ITEM,e.phase_ops.SOURCE_EMPTY))
            fields['volatile_state']=v.at[:,:,:,6].set(jnp.where(v[:,:,:,0]>0,kind,0))
        if version<11:
            species=np.asarray(fields['species']);ability=np.asarray(fields['ability'])
            if np.any(np.asarray(e.trap_ops.QUERY)[species]) or np.any(np.asarray(e.trap_ops.ABILITY_IMPLEMENTED)[ability]):
                raise ValueError('legacy trapping/disclosure requires replay/provenance')
            fields['request_state']=jnp.zeros((2,2),jnp.int32)
        if 9<=version<39:
            v=fields['volatile_state']
            slots=27 if version>=30 else 26 if version>=25 else 25 if version>=24 else 24 if version>=20 else 23 if version>=15 else 22 if version>=13 else 21 if version==12 else 20
            if v.shape!=(2,6,slots,e.phase_ops.VOLATILE_WIDTH):raise ValueError('invalid legacy volatile shape')
            fields['volatile_state']=jnp.pad(v,((0,0),(0,0),(0,len(e.phase_ops.VOLATILES)-slots),(0,0)))
        if 9<=version<15:
            a=fields['action_state']
            if a.shape!=(2,6,21):raise ValueError('invalid legacy action-state shape')
            fields['action_state']=jnp.pad(a,((0,0),(0,0),(0,3)))
        if version<16:
            initial=jnp.where(fields['item']>0,fields['item'],fields['action_state'][:,:,e.tracking.LAST_ITEM])
            fields['item_provenance']=jnp.stack((initial,jnp.zeros((2,6),jnp.int32)),axis=-1)
        if version<17:fields.update(e.identity_ops.infer(fields,e))
        if version<18:
            fields['illusion']=jnp.full((2,6),-1,jnp.int32)
            old=fields['history'];rows=[]
            # Earlier masks exclude disguise/forms; names and switch details are immutable.
            for row in old[:int(fields['hcount'])]:
                side,mon,target,tm=map(int,(row[1],row[2],row[4],row[5]))
                metadata=e.illusion_ops.event_identity(type('OldState',(),fields)(),side,mon,target,tm)
                rows.append(jnp.concatenate((row,jnp.asarray(metadata,jnp.int32))))
            history=jnp.zeros((e.HISTORY,e.EVENT_COLUMNS),jnp.int32)
            if rows:history=history.at[:len(rows)].set(jnp.stack(rows))
            fields['history']=history
        if version<21:fields['ability_staleness']=jnp.zeros((2,6),jnp.int32)
        if version<26:fields.update(call_frames=e.called_ops.empty(e),call_depth=e.i32(0))
        elif version<28:
            if int(fields['call_depth']) or np.any(np.asarray(fields['call_frames'])):raise ValueError('legacy active call frames require replay/provenance')
            if fields['call_frames'].shape!=(e.called_ops.FRAMES,16):raise ValueError('invalid legacy call frame shape')
            fields['call_frames']=jnp.pad(fields['call_frames'],((0,0),(0,4)))
        if version<32:fields['ability_seek']=jnp.zeros((2,6),jnp.int32)
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
    for kind,key in enumerate(e.phase_ops.VOLATILES):
        exists,order,ss,sm,move,time=map(int,s.volatile_state[side,mon,kind,:6])
        if not exists:continue
        row={'id':key,'name':e.MOVE_NAMES[e.PROTECT] if kind==0 else key,'effectOrder':order,'target':target,
             'source':{'$pokemon':{'side':ss,'mon':sm}},'sourceSlot':f'p{ss+1}a'}
        source_kind=int(s.volatile_state[side,mon,kind,6])
        if source_kind==e.phase_ops.SOURCE_EMPTY:row['sourceEffect']={'id':''}
        if source_kind==e.phase_ops.SOURCE_ABILITY:row['sourceEffect']={'$effect':e.ABILITIES[move],'type':'Ability'}
        if kind==e.phase_ops.VOLATILE_IDS['zenmode']:row['name']='Zen Mode'
        if kind==e.phase_ops.VOLATILE_IDS['gastroacid']:row['name']='Gastro Acid'
        if kind==e.flash_ops.SLOT:row['name']='Flash Fire'
        if kind==e.called_ops.COAT_SLOT:
            row.update(name='Magic Coat',duration=time)
            if int(s.volatile_state[side,mon,kind,10]):row['pranksterBoosted']=True
        if kind==e.phase_ops.VOLATILE_IDS['yawn']:row.update(name='Yawn',duration=time)
        if source_kind==e.phase_ops.SOURCE_CONDITION:row['sourceEffect']={'$effect':e.phase_ops.VOLATILES[move],'type':'Condition'}
        if move>0 and source_kind==e.phase_ops.SOURCE_MOVE:row['sourceEffect']={'$effect':e.MOVE_KEYS[move],'type':'Move'}
        if move<0:row['sourceEffect']={'$effect':e.ITEMS[-move],'type':'Item'}
        if kind==0:row['duration']=1
        if kind==1:row.update(counter=int(s.stall[side,mon]),duration=int(s.stall_duration[side,mon]))
        if kind==2:row['time']=time
        if kind==3:row['move']=e.MOVE_KEYS[time]
        if kind==4:row['duration']=time
        if kind==5:row.update(duration=time,move=e.MOVE_KEYS[int(s.volatile_state[side,mon,kind,7])],trueDuration=int(s.volatile_state[side,mon,kind,8]))
        if kind==6:row.update(duration=time,move=e.MOVE_KEYS[int(s.volatile_state[side,mon,kind,7])])
        if 7<=kind<=14:
            row['targetLoc']=int(s.volatile_state[side,mon,kind,9])
            if key in ('fly','dig','dive','bounce'):row.update(name=e.MOVE_NAMES[tables.MOVE_IDS[key]],duration=time)
        if key=='pursuit':row.update(name='Pursuit',duration=time)
        if key=='partiallytrapped':row.update(duration=time,boundDivisor=int(s.volatile_state[side,mon,kind,7]))
        if key=='substitute':row.update(name='Substitute',hp=int(s.volatile_state[side,mon,kind,7]))
        if key=='flinch':row['duration']=time
        if 15<=kind<=18:
            row['name']=e.MOVE_NAMES[tables.MOVE_IDS[key]]
            if kind!=18:row['duration']=time
            if kind in (16,17):row['move']=e.MOVE_KEYS[int(s.volatile_state[side,mon,kind,7])]
        volatiles[key]=row
    return {'statusState':status,'volatiles':volatiles,
        'abilityState':{'id':e.ABILITIES[int(s.ability_effect_id[side,mon])],'effectOrder':int(s.ability_order[side,mon]),'target':target,**({'seek':True}if int(s.ability_seek[side,mon])else{})},
        'itemState':{'id':e.ITEMS[int(s.item[side,mon])],'effectOrder':int(s.item_order[side,mon]),'target':target},
        'speciesState':{'id':e.generic.to_id(e.SPECIES[int(s.base_species[side,mon])]),'effectOrder':0}}


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


def move_snapshot(row):
    mid,kind,category,power=map(int,row)
    if mid==0:return None
    return {'id':e.MOVE_KEYS[mid],'type':'???' if kind==0 else generic.DATA['types'][kind-1]['name'],
            'category':('Status','Physical','Special')[category],'basePower':power}


def causal_state(state,side,mon,*,include_pursuit=False,include_items=False):
    """Trusted mechanics projection; deliberately absent from player_view."""
    s=jax.device_get(state);tr=e.tracking;a=s.action_state[side,mon]
    value=lambda k:int(a[k])
    typed=lambda x:({'$undefined':True},None,False,True)[int(x)]
    snapshots=[move_snapshot(row) for row in s.move_history[side,mon]]
    result={'lastMove':snapshots[0],'lastMoveEncore':snapshots[1],'lastMoveUsed':snapshots[2],
      'lastMoveTargetLoc':{'$undefined':True} if value(tr.TARGET_LOC)==-2 else value(tr.TARGET_LOC),
      'lastDamage':value(tr.LAST_DAMAGE),'activeTurns':value(tr.ACTIVE_TURNS),'activeMoveActions':value(tr.ACTIONS),
      'moveThisTurn':e.MOVE_KEYS[value(tr.MOVE_THIS_TURN)],'moveThisTurnResult':typed(a[tr.RESULT]),
      'moveLastTurnResult':typed(a[tr.PREV_RESULT]),'hurtThisTurn':None if value(tr.HURT)==-1 else value(tr.HURT),
      'timesAttacked':value(tr.TIMES_ATTACKED),'lastItem':e.ITEMS[value(tr.LAST_ITEM)],'attackedBy':[]}
    for key,column in [('trapped',tr.TRAPPED),('maybeTrapped',tr.MAYBE_TRAPPED),('maybeDisabled',tr.MAYBE_DISABLED),
         ('newlySwitched',tr.NEWLY_SWITCHED),('usedItemThisTurn',tr.USED_ITEM),('statsRaisedThisTurn',tr.RAISED),('statsLoweredThisTurn',tr.LOWERED),('ateBerry',tr.ATE_BERRY)]:result[key]=bool(a[column])
    if value(tr.TRAPPED)==2:result['trapped']='hidden'
    result['moveSlotsUsed']=[bool(value(tr.USED_SLOTS)&(1<<slot))for slot,move in enumerate(s.moves[side,mon])if move]
    if include_items:result.update(initialItem=e.ITEMS[int(s.item_provenance[side,mon,0])],pendingStaleness=({'$undefined':True},'internal','external')[int(s.item_provenance[side,mon,1])])
    if include_pursuit:result.update(beingCalledBack=bool(value(tr.CALLED_BACK)),pursuitActivated=bool(value(tr.PURSUIT_ACTIVE)),skipPursuit=bool(value(tr.SKIP_PURSUIT)))
    result['previouslySwitchedIn']=value(tr.SWITCH_COUNT)
    for row in s.attacked_by[side,mon,:value(tr.ATTACK_COUNT)]:
        source,sm,move,damage,this_turn,tag,number=map(int,row)
        result['attackedBy'].append({'source':{'side':source,'mon':sm},'move':e.MOVE_KEYS[move],'damage':damage,
            'thisTurn':bool(this_turn),'slot':f'p{source+1}a','damageValue':number if tag==tr.NUMBER else typed(tag)})
    return result


def identity_state(state,side,mon):
    s=jax.device_get(state);used=int(s.base_used[side,mon]);rows=[]
    for slot,move in enumerate(s.base_moves[side,mon]):
        move=int(move)
        if move:rows.append({'id':e.MOVE_KEYS[move],'pp':int(s.base_pp[side,mon,slot]),'maxpp':int(e.PP[move]),'used':bool(used&(1<<slot))})
    return {'baseSpecies':e.generic.to_id(e.SPECIES[int(s.base_species[side,mon])]),'baseAbility':e.ABILITIES[int(s.base_ability[side,mon])],
            'transformed':bool(s.transformed[side,mon]),'knownType':bool(s.known_type[side,mon]),
            'apparentType':'/'.join(e.generic.DATA['types'][int(v)-1]['name']for v in s.apparent_types[side,mon]if v),
            'baseMoves':rows,'virtualMoves':[bool(s.transformed[side,mon])for move in s.moves[side,mon]if move]}


def causal_side(state,side):
    s=jax.device_get(state)
    ref=lambda mon:None if int(mon)==-1 else {'side':side,'mon':int(mon)}
    return {'faintedThisTurn':ref(s.side_faints[side,0]),'faintedLastTurn':ref(s.side_faints[side,1])}


def causal_battle(state):
    s=jax.device_get(state)
    return {'lastMove':move_snapshot(s.battle_move),'lastSuccessfulMoveThisTurn':e.MOVE_KEYS[int(s.battle_success)] or None,'lastDamage':int(s.battle_damage)}


def private_packets(state,trace,side):
    """Recipient-only command feedback; never append to public battle events."""
    if side not in (0,1):raise ValueError('side must be 0 or 1')
    error=int(trace.private_errors[side])
    if not error:return []
    message="Can't undo: A trapping/disabling effect would cause undo to leak information" if error==3 else "Can't switch: The active Pokémon is trapped"
    packets=[f"p{side+1}\n|error|[{'Unavailable' if error==1 else 'Invalid'} choice] "+message]
    if error==1:
        request=project_request(state,side)
        request={'active':request['active'],**{k:v for k,v in request.items()if k!='active'},'update':True}
        packets.append(f"p{side+1}\n|request|"+json.dumps(request,ensure_ascii=False,separators=(',',':')))
    return packets
