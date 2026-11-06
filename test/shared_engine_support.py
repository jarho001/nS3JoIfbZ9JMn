"""Independent native fixture reader and first-divergence assertions."""
import json
import os
from pathlib import Path
import subprocess
import jax
import numpy as np
from simulator import shared_engine as e
from simulator.shared_engine_io import project_events,project_request,draw_rows

ROOT=Path(__file__).resolve().parents[1]


def native_cases():
    path=os.environ.get('G3_NATIVE_CASES')
    if path:return json.loads(Path(path).read_text())['cases']
    completed=subprocess.run(['node','reference/shared-engine-cases.cjs','--batch','all'],cwd=ROOT,capture_output=True,text=True,check=True,timeout=120)
    return json.loads(completed.stdout)['cases']


def initialize_case(case):
    teams=[]
    for key in ('p1','p2'):
        teams.append([{'species':d['species'],'moves':d['moves'],'level':d['level'],'speed_evs':d['evs']['spe'],
                       'item':{'':'','Leftovers':'leftovers','Lum Berry':'lumberry'}[d['item']]} for d in case['setup'][key]])
    init=e.make_state(teams,[case['setup']['seed'],2,3,4]);s=init.state
    # Synthetic state installation is a trusted fixture, not ordinary reset.
    for setup in case['setup'].get('setup',[]):
        side=int(setup['side'][1])-1;mon=setup.get('mon',0)
        if 'status' in setup:
            status={'brn':e.BRN,'par':e.PAR,'tox':e.TOX}[setup['status']]
            # Native setStatus immediately consumes Lum through an Update.
            lum=int(s.item[side,mon])==2
            s=s._replace(status=s.status.at[side,mon].set(0 if lum else status),item=s.item.at[side,mon].set(0 if lum else s.item[side,mon]))
        if 'hp' in setup:s=s._replace(hp=s.hp.at[side,mon].set(setup['hp']))
        if 'pp' in setup:s=s._replace(pp=s.pp.at[side,mon].set(np.where(np.asarray(s.moves[side,mon])>0,setup['pp'],0)))
        if 'toxicStage' in setup:s=s._replace(toxic=s.toxic.at[side,mon].set(setup['toxicStage']))
        if 'stall' in setup:s=s._replace(stall=s.stall.at[side,mon].set(setup['stall']),stall_duration=s.stall_duration.at[side,mon].set(2))
    return s,init.trace


def choices_array(s,choices):
    result=[-1,-1]
    for side,command in choices.items():
        i=int(side[1])-1
        if command=='undo':result[i]=-2
        else:
            verb,n=command.split();slot=int(n)-1
            if verb=='switch':slot+=4
            elif not np.any(np.asarray(s.pp[i,int(s.active[i])])>0):slot=10
            result[i]=slot
    return e.i32(result)


def assert_state(test,s,native,label):
    s=jax.device_get(s)
    test.assertEqual(s.rng.tolist(),native['rng'],label+' RNG')
    test.assertEqual(int(s.turn),native['turn'],label+' turn')
    test.assertEqual(int(s.winner),native['winner'],label+' winner')
    test.assertEqual(bool(s.midturn),native['midTurn'],label+' midTurn')
    for side,n in enumerate(native['sides']):
        prefix=label+f' p{side+1}'
        test.assertEqual(int(s.active[side]),n['active'],prefix+' active')
        test.assertEqual(s.order[side,:len(n['order'])].tolist(),n['order'],prefix+' command roster')
        for p in n['pokemon']:
            m=p['id'];tag=prefix+f' mon{m}'
            for ours,theirs in [('hp','hp'),('status','status'),('toxic','toxicStage'),('item','item'),('speed','speed'),('fainted','fainted')]:
                v={'status':('','brn','par','tox','fnt')[int(s.status[side,m])],'item':e.ITEMS[int(s.item[side,m])]}.get(ours,np.asarray(getattr(s,ours)[side,m]).item())
                test.assertEqual(v,p[theirs],tag+' '+ours)
            test.assertEqual(int(s.stats[side,m,0]),p['maxhp'],tag+' maxHP')
            test.assertEqual(s.stats[side,m,1:].tolist(),[p['stats'][k] for k in ('atk','def','spa','spd','spe')],tag+' stats')
            test.assertEqual(s.pp[side,m,:len(p['moves'])].tolist(),[x['pp'] for x in p['moves']],tag+' PP')
            stall=None if not int(s.stall_duration[side,m]) else {'counter':int(s.stall[side,m]),'duration':int(s.stall_duration[side,m])}
            test.assertEqual(stall,p['stall'],tag+' stall')
            test.assertEqual(bool(s.protected[side,m]),p['protect'],tag+' protect')
        expected=None if native['terminal'] else n['request']
        test.assertEqual(project_request(s,side),expected,prefix+' private request')
        if not native['terminal']:test.assertEqual(np.asarray(e.legal_actions(s)[side]).tolist(),n['actions']['mask'],prefix+' mask')
    q=[]
    for row in s.queue[:int(s.qcount)]:
        kind,side,mon,action,priority,speed,source=map(int,row)
        if kind==e.ATTACK:
            mid=e.STRUGGLE if action==10 else int(s.moves[side,mon,action]);move=e.MOVE_KEYS[mid]
        else:move=None
        q.append({'choice':{e.BEFORE:'beforeTurn',e.ATTACK:'move',e.SWAP:'switch',e.INSTANT:'instaswitch',e.RUN_SWITCH:'runSwitch',e.RESIDUAL:'residual'}[kind],
                  'order':int(e.QORDER[kind]),'priority':priority,'speed':speed,
                  'actor':None if kind in (e.BEFORE,e.RESIDUAL) else {'side':side,'mon':mon},
                  'target':None if kind not in (e.SWAP,e.INSTANT) else {'side':side,'mon':int(s.order[side,action-4])},
                  'move':move,'source':'uturn' if source else None})
    test.assertEqual(q,native['queue'],label+' suspended queue')


def assert_step(test,before,result,expected,label):
    test.assertIn(int(result.code),(e.OK,e.PENDING),label+' code')
    native_draws=[{'args':[d['bound'],d['to']] if 'to' in d else [d['bound']],
                   'value':d['value'],'before':d['before'],'after':d['after']} for d in expected['draws']]
    test.assertEqual(draw_rows(result.trace),native_draws,label+' first divergent draw')
    assert_state(test,result.state,expected['after'],label)
    for side in range(2):test.assertEqual(project_events(result.state,result.trace,side),expected['events'][f'p{side+1}'],label+f' p{side+1} event stream')
