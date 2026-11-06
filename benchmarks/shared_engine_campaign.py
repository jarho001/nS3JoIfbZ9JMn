"""Stream independent native battles, compare exact boundaries, retain failures.

Run with an explicit JAX_PLATFORMS and a predeclared --config. Native snapshots
are consumed one battle at a time; successful raw samples are fixed by index.
"""
import argparse
from collections import Counter
import hashlib
import gzip
import itertools
import json
from pathlib import Path
import subprocess
import time
import tempfile
import unittest

import jax
import numpy as np

from simulator import shared_engine as e
from simulator.shared_engine_io import draw_rows, player_view
from test.shared_engine_support import initialize_case, choices_array, assert_state, assert_step


def placement(tree, platform):
    for leaf in jax.tree.leaves(tree):
        assert isinstance(leaf, jax.Array), type(leaf)
        assert all(d.platform == platform for d in leaf.devices()), leaf.devices()


def invariants(before, result):
    s, t = jax.device_get((result.state, result.trace))
    assert np.all((s.hp >= 0) & (s.hp <= s.stats[..., 0]))
    assert np.all((s.pp >= 0) & (s.pp <= np.asarray(e.PP)[s.moves]))
    assert 0 <= s.qcount <= e.QUEUE and 0 <= s.hcount <= e.HISTORY
    assert not np.any(t.events[int(t.ecount):])
    for field in ('args', 'values', 'context', 'before', 'after'):
        assert not np.any(getattr(t, field)[int(t.dcount):]), field+' padding'
    draws=draw_rows(t)
    previous=np.asarray(before.rng).tolist()
    for draw in draws:
        assert draw['before'] == previous
        previous=draw['after']
    assert previous == s.rng.tolist()
    if int(s.winner) >= 0:
        assert not np.any(e.legal_actions(s))
        assert np.sum(e.rewards(s)) == 0


def identity():
    files=['simulator/shared_engine.py','simulator/shared_engine_io.py',
           'reference/shared-engine-cases.cjs','test/shared_engine_support.py',
           'benchmarks/shared_engine_campaign.py','benchmarks/g3-campaign.json']
    return {'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--platform',choices=('cpu','gpu'),required=True)
    p.add_argument('--start-index',type=int,default=0)
    p.add_argument('--indices',help='Explicit fixed config indices for representative device replay')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path(args.config).read_text())
    source=identity();(out/'source.json').write_text(json.dumps(source,indent=2)+'\n')
    assert jax.default_backend()==args.platform,(jax.default_backend(),args.platform)
    check=unittest.TestCase();compiled=jax.jit(e.advance);batched=jax.jit(jax.vmap(e.advance))
    indices=list(map(int,args.indices.split(','))) if args.indices else None
    replay_indices=set(indices if indices is not None else (0,1,19,99,499,999))
    summary={'schemaVersion':1,'source':'source.json','config':args.config,'platform':args.platform,'start_index':args.start_index,
             'starts':0,'completed':0,'truncated':0,'capacity':0,'failing':0,'boundaries':0,
             'decisions':0,'ordinary':0,'synthetic':0,'effects':{},'pairs':{},'per_start':[]}
    effects=Counter();pairs=Counter();started=time.time()
    command=['node','reference/shared-engine-cases.cjs','--campaign',args.config,'--campaign-start',str(args.start_index)]
    if args.indices:command+=['--campaign-indices',args.indices]
    native=subprocess.Popen(command,stdout=subprocess.PIPE,text=True)
    current=None;target_reached=False
    try:
        for index,line in enumerate(native.stdout,start=args.start_index):
            if indices is not None:index=indices[summary['starts']]
            current=json.loads(line);summary['starts']+=1
            summary['synthetic' if current['classification']=='synthetic' else 'ordinary']+=1
            state,trace=initialize_case(current);placement((state,trace),args.platform)
            check.assertEqual(draw_rows(trace),current['initializationDraws'])
            assert_state(check,state,current['initial'],current['id']+' initial')
            histories={side:list(current['initialEvents'][f'p{side+1}']) for side in range(2)}
            for side in range(2):check.assertEqual(player_view(state,side)['events'],histories[side])
            capacity=False;count=0;replay=[]
            for n,step in enumerate(current['steps']):
                action=choices_array(state,step['choices']);result=compiled(state,action)
                placement((action,result),args.platform)
                if int(result.code)==e.CAPACITY:
                    for a,b in zip(jax.tree.leaves(state),jax.tree.leaves(result.state),strict=True):np.testing.assert_array_equal(a,b)
                    assert int(result.trace.ecount)==int(result.trace.dcount)==0
                    capacity=True;break
                assert_step(check,state,result,step,f"{current['id']}/{n+1}")
                invariants(state,result)
                # Fixed native samples receive eager and heterogeneous replay;
                # the ordinary campaign keeps the complete compiled contract.
                if index in replay_indices and n in (0,len(current['steps'])//2,len(current['steps'])-1):
                    with jax.disable_jit():eager=e.advance(state,action)
                    placement(eager,args.platform)
                    for a,b in zip(jax.tree.leaves(result),jax.tree.leaves(eager),strict=True):np.testing.assert_array_equal(a,b)
                    replay.append((state,action,result))
                state=result.state;count+=1;summary['boundaries']+=1
                summary['decisions']+=sum(choice!='undo' for choice in step['choices'].values())
                names=set()
                for side in range(2):
                    histories[side].extend(step['events'][f'p{side+1}'])
                    check.assertEqual(player_view(state,side)['events'],histories[side])
                    names.update(event['message'].split('|')[1] for event in step['events'][f'p{side+1}'] if event['message'].startswith('|'))
                effects.update(names);pairs.update('+'.join(x) for x in itertools.combinations(sorted(names),2))
            status='capacity' if capacity else 'completed' if int(state.winner)>=0 else 'truncated'
            if replay:
                with tempfile.TemporaryDirectory() as directory:
                    for slot,(original,action,expected_result) in enumerate(replay):
                        saved=Path(directory)/f'continuation-{slot}.npz'
                        np.savez(saved,**dict(zip(e.State._fields,jax.device_get(original),strict=True)))
                        with np.load(saved,allow_pickle=False) as archive:restored=e.State(*(jax.numpy.asarray(archive[name]) for name in e.State._fields))
                        continued=compiled(restored,action);placement(continued,args.platform)
                        for a,b in zip(jax.tree.leaves(continued),jax.tree.leaves(expected_result),strict=True):np.testing.assert_array_equal(a,b)
                states=jax.tree.map(lambda *x:jax.numpy.stack(x),*(x[0] for x in replay))
                actions=jax.numpy.stack([x[1] for x in replay]);result=batched(states,actions);placement(result,args.platform)
                for lane,(_,_,expected) in enumerate(replay):
                    for a,b in zip(jax.tree.leaves(result),jax.tree.leaves(expected),strict=True):np.testing.assert_array_equal(a[lane],b)
                if int(state.winner)>=0:
                    replacement,_=initialize_case(current)
                    reset=jax.jit(e.reset)(state,replacement,jax.numpy.stack((state.generation,state.episode)))
                    placement(reset,args.platform);assert int(reset.code)==e.OK
                    for a,b in zip(jax.tree.leaves(reset.state),jax.tree.leaves(replacement._replace(generation=state.generation+1)),strict=True):np.testing.assert_array_equal(a,b)
                summary.setdefault('replayed_indices',[]).append(index)
            summary[status]+=1
            summary['per_start'].append({'id':current['id'],'status':status,'boundaries':count,'native_boundaries':len(current['steps'])})
            if index in replay_indices:
                with gzip.open(out/f"sample-{index}.json.gz",'wt') as saved:json.dump(current,saved,separators=(',',':'))
            summary.update(effects=dict(effects),pairs=dict(pairs),elapsed_seconds=time.time()-started)
            (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            print(json.dumps({k:summary[k] for k in ('starts','completed','truncated','capacity','boundaries')}),flush=True)
            if summary['boundaries']>=config['targetBoundaries']:
                target_reached=True;break
        if not target_reached:
            code=native.wait(timeout=30)
            summary['native_returncode']=code
            expected=len(indices) if indices is not None else len(config['cases'])-args.start_index
            if code or summary['starts']!=expected:
                summary['status']='incomplete_native_runner'
                raise RuntimeError(f'native producer exit {code}; received {summary["starts"]}/{expected} starts')
        summary['status']='complete_target'
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    except Exception as error:
        summary['failing']+=1;summary['error']=str(error)
        summary['failing_start']=current['id'] if current is not None else None
        if current is not None:
            with gzip.open(out/'failure-native.json.gz','wt') as saved:json.dump(current,saved,separators=(',',':'))
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        raise
    finally:
        native.stdout.close()
        if native.poll() is None:native.terminate()
        native.wait(timeout=30)


if __name__=='__main__':main()
