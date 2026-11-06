"""Host protocol adapter for shared_engine's numeric, recipient-scoped events.

No native oracle is imported. Only own roster information enters a request;
opposing HP is rounded exactly as HP Percentage Mod. Trusted state and queued
choices must never be passed to a policy.
"""
import numpy as np
from . import shared_engine as e


def condition(hp,maxhp,status=0,public=False):
    if hp==0:return '0 fnt'
    shown=(hp*100+maxhp-1)//maxhp if public else hp
    base=100 if public else maxhp
    return f'{shown}/{base}'+(' '+('','brn','par','tox','fnt')[status] if status else '')


def details(s,side,mon):
    species=e.SPECIES[int(s.species[side,mon])];level=int(s.level[side,mon]);gender=int(s.gender[side,mon])
    return species+(f', L{level}' if level!=100 else '')+(', '+('','M','F')[gender] if gender else '')


def project_request(state,side):
    s=__import__('jax').device_get(state);kind=int(s.request[side])
    if kind==e.END:return None
    order=s.order[side];roster=[]
    for mon in order:
        sp=int(s.species[side,mon])
        if not sp:continue
        roster.append({'ident':f'p{side+1}: '+e.SPECIES[sp],'details':details(s,side,mon),
         'condition':condition(int(s.hp[side,mon]),int(s.stats[side,mon,0]),int(s.status[side,mon])),
         'active':bool(mon==s.active[side]),
         'stats':dict(zip(('atk','def','spa','spd','spe'),map(int,s.stats[side,mon,1:]))),
         'moves':[e.MOVE_KEYS[int(m)] for m in s.moves[side,mon] if m],
         'baseAbility':('','synchronize','immunity','levitate')[sp],'item':e.ITEMS[int(s.item[side,mon])],'pokeball':'pokeball'})
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
            rows.append({'move':e.MOVE_NAMES[mid],'id':e.MOVE_KEYS[mid],'pp':int(s.pp[side,mon,slot]),
                         'maxpp':int(e.PP[mid]),'target':'self' if mid in (e.PROTECT,e.RECOVER) else 'allAdjacent' if mid==e.EARTHQUAKE else 'normal',
                         'disabled':bool(s.pp[side,mon,slot]==0)})
        if all(row['disabled'] for row in rows):rows=[{'move':'Struggle','id':'struggle','target':'randomNormal','disabled':False}]
        req['active']=[{'moves':rows}]
    return req


def project_events(state,trace,audience):
    s=__import__('jax').device_get(state);t=__import__('jax').device_get(trace);out=[]
    for row in t.events[:int(t.ecount)]:
        kind,side,mon,move,target,tm,value,aux,status,_=map(int,row)
        ident=lambda side,mon:f'p{side+1}a: '+e.SPECIES[int(s.species[side,mon])]
        actor=ident(side,mon);foe=ident(target,tm);name=e.MOVE_NAMES[move] if move>=0 else ''
        if kind==e.BLANK:line='|'
        elif kind==e.EV_MOVE:
            shown='' if aux==2 else foe.replace('a: ',': ') if aux==3 else foe
            annotation={0:'',1:'|[miss]',2:'|[still]',3:'|[notarget]'}[aux]
            line=f'|move|{actor}|{name}|'+shown+annotation
        elif kind in (e.DAMAGE,e.HEAL):
            line=f'|-'+('damage' if kind==e.DAMAGE else 'heal')+f'|{actor}|'+condition(value,aux,status,side!=audience)
            if kind==e.DAMAGE and move:line+='|[from] '+{1:'brn',2:'psn',3:'recoil'}[move]
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
        else:raise ValueError('unknown event')
        out.append({'type':'event','message':line})
    return out


def draw_rows(trace):
    t=__import__('jax').device_get(trace);out=[]
    for k in range(int(t.dcount)):
        lo,hi,form=map(int,t.args[k]);args=[2**32] if form==3 else [hi] if form==1 else [lo,hi]
        out.append({'args':args,'value':int(t.values[k]),'before':t.before[k].tolist(),'after':t.after[k].tolist()})
    return out


def player_view(s,side):
    """Only a private request and recipient-rendered history leave the adapter."""
    trace=e.empty_trace()._replace(events=s.history,ecount=s.hcount)
    return {'request':project_request(s,side),'events':project_events(s,trace,side)}
