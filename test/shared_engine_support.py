"""Independent native fixture reader and first-divergence assertions."""
import json
import os
from pathlib import Path
import subprocess
import jax
import numpy as np
from simulator import shared_engine as e
from simulator.shared_engine_io import project_events,project_request,draw_rows
from simulator import gen5_tables as generic

ROOT=Path(__file__).resolve().parents[1]


def assert_placement(tree):
    requested=os.environ.get('JAX_PLATFORMS','').split(',')[0]
    expected={'cuda':'gpu','cpu':'cpu'}.get(requested)
    if expected is None:return
    assert jax.default_backend()==expected,(requested,jax.default_backend())
    for leaf in jax.tree.leaves(tree):
        assert isinstance(leaf,jax.Array),type(leaf)
        assert all(d.platform==expected for d in leaf.devices()),leaf.devices()


def native_cases():
    assert_placement(())
    path=os.environ.get('G3_NATIVE_CASES')
    if path:return json.loads(Path(path).read_text())['cases']
    completed=subprocess.run(['node','reference/shared-engine-cases.cjs','--batch','all'],cwd=ROOT,capture_output=True,text=True,check=True,timeout=120)
    return json.loads(completed.stdout)['cases']


def initialize_case(case):
    if case['setup'].get('generic'):
        init=e.make_gen5_state([case['setup'][key] for key in ('p1','p2')],
                              case['setup'].get('seedWords',[case['setup']['seed'],2,3,4]))
        if int(init.code)!=e.OK:raise AssertionError(case['id']+': initialization unsupported')
    else:
        teams=[]
        for key in ('p1','p2'):
            teams.append([{'species':d['species'],'moves':d['moves'],'level':d['level'],'speed_evs':d['evs']['spe'],
                           'item':{'':'','Leftovers':'leftovers','Lum Berry':'lumberry'}[d['item']]} for d in case['setup'][key]])
        init=e.make_state(teams,case['setup'].get('seedWords',[case['setup']['seed'],2,3,4]))
    s=init.state
    # Synthetic state installation is a trusted fixture, not ordinary reset.
    for setup in case['setup'].get('setup',[]):
        side=int(setup['side'][1])-1;mon=setup.get('mon',0)
        if 'status' in setup:
            status=e.STATUS_NAMES.index(setup['status'])
            # Native Pokemon.setStatus silently rejects an immune synthetic
            # installation; it is not an attempted status move disclosure.
            immune=status==e.TOX and int(s.species[side,mon]) in (e.SNORLAX,e.GENGAR)
            if not immune:
                s,trace=e._status((s,e.empty_trace()),side,side,e.i32(status))
                count=int(trace.ecount)
                s=s._replace(history=s.history.at[s.hcount:s.hcount+count].set(trace.events[:count]),hcount=s.hcount+count)
        if 'hp' in setup:s=s._replace(hp=s.hp.at[side,mon].set(setup['hp']))
        if 'pp' in setup:s=s._replace(pp=s.pp.at[side,mon].set(np.where(np.asarray(s.moves[side,mon])>0,setup['pp'],0)))
        if 'ppByMove' in setup:
            for slot,move in enumerate(s.moves[side,mon]):
                key=e.MOVE_KEYS[int(move)]
                if key in setup['ppByMove']:s=s._replace(pp=s.pp.at[side,mon,slot].set(setup['ppByMove'][key]))
        if 'toxicStage' in setup:s=s._replace(toxic=s.toxic.at[side,mon].set(setup['toxicStage']))
        if 'stall' in setup:
            s,_=e.effects.add_volatile((s,e.empty_trace()),e.i32(side),1,e.i32(side),e.i32(0),e)
            s=s._replace(stall=s.stall.at[side,mon].set(setup['stall']),stall_duration=s.stall_duration.at[side,mon].set(2))
    assert_placement((s,init.trace))
    return s,init.trace


def choices_array(s,choices):
    result=[-1,-1]
    for side,command in choices.items():
        i=int(side[1])-1
        if command=='undo':result[i]=-2
        else:
            verb,n=command.split();slot=int(n)-1
            if verb=='switch':slot+=4
            elif bool(e.legal_actions(s)[i,10]):slot=10
            result[i]=slot
    return e.i32(result)


def assert_state(test,s,native,label):
    assert_placement(s)
    s=jax.device_get(s)
    test.assertEqual(s.rng.tolist(),native['rng'],label+' RNG')
    test.assertEqual(int(s.turn),native['turn'],label+' turn')
    test.assertEqual(int(s.winner),native['winner'],label+' winner')
    test.assertEqual(bool(s.midturn),native['midTurn'],label+' midTurn')
    if 'phaseState' in native:
        from simulator.shared_engine_io import causal_battle
        test.assertEqual(json.dumps(causal_battle(s),sort_keys=True),json.dumps(native['phaseState'],sort_keys=True),label+' causal battle')
    if 'effectOrder' in native:test.assertEqual(int(s.effect_order),native['effectOrder'],label+' global effectOrder')
    if 'field' in native:
        from simulator.shared_engine_io import field_state
        test.assertEqual(field_state(s),native['field'],label+' field state')
    for side,n in enumerate(native['sides']):
        prefix=label+f' p{side+1}'
        if 'cantUndo' in n:test.assertEqual(bool(s.request_state[side,1]),n['cantUndo'],prefix+' cannot undo choice')
        if 'phaseState' in n:
            from simulator.shared_engine_io import causal_side
            test.assertEqual(json.dumps(causal_side(s,side),sort_keys=True),json.dumps(n['phaseState'],sort_keys=True),prefix+' causal side')
        if 'sideConditions' in n:
            from simulator.shared_engine_io import side_conditions
            test.assertEqual(side_conditions(s,side),n['sideConditions'],prefix+' side conditions')
        test.assertEqual(s.hazards[side].tolist(),n.get('hazards',[0,0,0]),prefix+' side hazards/order')
        test.assertEqual(int(s.active[side]),n['active'],prefix+' active')
        test.assertEqual(s.order[side,:len(n['order'])].tolist(),n['order'],prefix+' command roster')
        for p in n['pokemon']:
            m=p['id'];tag=prefix+f' mon{m}'
            for ours,theirs in [('hp','hp'),('status','status'),('toxic','toxicStage'),('item','item'),('speed','speed'),('fainted','fainted')]:
                v={'status':e.STATUS_NAMES[int(s.status[side,m])],'item':e.ITEMS[int(s.item[side,m])]}.get(ours,np.asarray(getattr(s,ours)[side,m]).item())
                test.assertEqual(v,p[theirs],tag+' '+ours)
            test.assertEqual(int(s.stats[side,m,0]),p['maxhp'],tag+' maxHP')
            test.assertEqual(s.stats[side,m,1:].tolist(),[p['stats'][k] for k in ('atk','def','spa','spd','spe')],tag+' stats')
            if 'types' in p:
                types=[generic.DATA['types'][int(t)-1]['name'] for t in s.types[side,m] if t]
                test.assertEqual(types,p['types'],tag+' types')
                test.assertEqual(e.SPECIES[int(s.name[side,m])],p['name'],tag+' identity')
                test.assertEqual(e.ABILITIES[int(s.ability[side,m])],p['ability'],tag+' ability')
                test.assertEqual(s.ivs[side,m].tolist(),[p['ivs'][k] for k in generic.STATS],tag+' IVs')
                test.assertEqual(s.evs[side,m].tolist(),[p['evs'][k] for k in generic.STATS],tag+' EVs')
                test.assertEqual(generic.DATA['natures'][int(s.nature[side,m])-1]['name'],p['nature'],tag+' nature')
                test.assertEqual(generic.DATA['types'][int(s.hp_type[side,m])-1]['name'],p['hpType'],tag+' Hidden Power type')
                test.assertEqual(int(s.hp_power[side,m]),p['hpPower'],tag+' Hidden Power power')
                test.assertEqual(bool(s.shiny[side,m]),p['shiny'],tag+' shiny')
                test.assertEqual(('','M','F')[int(s.gender[side,m])],p['gender'],tag+' gender')
            test.assertEqual(s.pp[side,m,:len(p['moves'])].tolist(),[x['pp'] for x in p['moves']],tag+' PP')
            stall=None if not int(s.stall_duration[side,m]) else {'counter':int(s.stall[side,m]),'duration':int(s.stall_duration[side,m])}
            test.assertEqual(stall,p['stall'],tag+' stall')
            test.assertEqual(bool(s.protected[side,m]),p['protect'],tag+' protect')
            if 'phaseState' in p:
                from simulator.shared_engine_io import causal_state
                test.assertEqual(json.dumps(causal_state(s,side,m,include_pursuit='beingCalledBack'in p['phaseState'],include_items='initialItem'in p['phaseState']),sort_keys=True),json.dumps(p['phaseState'],sort_keys=True),tag+' causal Pokemon')
            if 'statusState' in p:
                from simulator.shared_engine_io import effect_states
                for key,value in effect_states(s,side,m).items():test.assertEqual(value,p[key],tag+' '+key)
            test.assertEqual(bool(s.active[side]==m and not s.fainted[side,m]),p['active'],tag+' active flag')
            test.assertEqual(s.boosts[side,m].tolist(),[p['boosts'][k] for k in ('atk','def','spa','spd','spe','accuracy','evasion')],tag+' boosts')
            test.assertFalse(p['forceSwitchFlag'],tag+' no excluded phazing')
        expected=None if native['terminal'] else n['request']
        test.assertEqual(project_request(s,side),expected,prefix+' private request')
        if not native['terminal']:
            test.assertEqual(np.asarray(e.legal_actions(s)[side]).tolist(),n['actions']['mask'],prefix+' mask')
            action=int(s.pending[side]);mon=int(s.active[side]);pending=[]
            if action>=0:
                switch=4<=action<10
                pending=[{'choice':'switch' if switch else 'move','actor':{'side':side,'mon':mon},
                          'move':None if switch else 'recharge' if int(e.phase_ops.forced_move(s)[side])<0 else e.MOVE_KEYS[int(e.phase_ops.forced_move(s)[side])] if int(e.phase_ops.forced_move(s)[side])>0 else e.MOVE_KEYS[e.STRUGGLE if action==10 else int(s.moves[side,mon,action])],
                          'target':{'side':side,'mon':int(s.order[side,action-4])} if switch else None}]
            test.assertEqual(pending,n['pending'],prefix+' private pending choice')
            test.assertEqual(bool(s.request[side]>=e.WAIT or action>=0),n['choiceDone'],prefix+' ready')
    q=[]
    for row in s.queue[:int(s.qcount)]:
        kind,side,mon,action,priority,speed,source=map(int,row)
        if kind==e.ATTACK:
            move='recharge' if action==e.phase_ops.RECHARGE_ACTION else e.MOVE_KEYS[e.STRUGGLE if action==10 else int(s.moves[side,mon,action])]
        else:move=None
        q.append({'choice':{e.BEFORE:'beforeTurn',e.ATTACK:'move',e.SWAP:'switch',e.INSTANT:'instaswitch',e.RUN_SWITCH:'runSwitch',e.RESIDUAL:'residual',e.BEFORE_MOVE:'beforeTurnMove'}[kind],
                  'order':int(e.QORDER[kind]),'priority':priority,'speed':speed,
                  'actor':None if kind in (e.BEFORE,e.RESIDUAL) else {'side':side,'mon':mon},
                  'target':None if kind not in (e.SWAP,e.INSTANT) else {'side':side,'mon':int(s.order[side,action-4])},
                  'move':move,'source':'uturn' if source else None})
    test.assertEqual(q,native['queue'],label+' suspended queue')


def assert_step(test,before,result,expected,label):
    assert_placement((before,result))
    test.assertIn(int(result.code),(e.OK,e.PENDING),label+' code')
    native_draws=[{'args':[d['bound'],d['to']] if 'to' in d else [d['bound']],
                   'value':d['value'],'before':d['before'],'after':d['after'],'context':d['semanticContext']} for d in expected['draws']]
    test.assertEqual(draw_rows(result.trace),native_draws,label+' first divergent draw')
    assert_state(test,result.state,expected['after'],label)
    for side in range(2):test.assertEqual(project_events(result.state,result.trace,side),expected['events'][f'p{side+1}'],label+f' p{side+1} event stream')
