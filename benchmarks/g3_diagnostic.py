"""Bounded fixed-pack diagnostic; reuses saved native fixtures and benchmark helpers."""
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
import unittest
from unittest.mock import patch

START=time.monotonic()
OUT=Path('test-logs/g3-testing-profiling/diagnostic')
OUT.mkdir(exist_ok=False)
META={'status':'running','budget_seconds':300,'rows':[],'gates':{},'checkpoint':'imports'}
def save():
    META['elapsed_seconds']=time.monotonic()-START
    (OUT/'results.json').write_text(json.dumps(META,indent=2)+'\n')
def checkpoint(text):
    META['checkpoint']=text;save();print(text,flush=True)
def expired(signum,frame):
    META['status']='budget_exhausted';save();os._exit(124)
signal.signal(signal.SIGALRM,expired);signal.alarm(299)
save()
import jax
import jax.numpy as jnp
import numpy as np
from simulator import shared_engine as e
from simulator.shared_engine_io import player_view
from test import shared_engine_support as support
from benchmarks.shared_engine import ready, stack, digest

def placement(tree,platform):
    for leaf in jax.tree.leaves(tree):
        assert isinstance(leaf,jax.Array)
        assert all(d.platform==platform for d in leaf.devices()),leaf.devices()

def run(initial,actions,starts):
    def step(state,inputs):
        action,start=inputs
        state=jax.lax.cond(start>=0,lambda _:jax.tree.map(lambda x:x[start],initial),lambda _:state,None)
        result=jax.vmap(e.advance)(state,action)
        return result.state,(result,jax.vmap(e.legal_actions)(result.state))
    return jax.lax.scan(step,jax.tree.map(lambda x:x[0],initial),(actions,starts))[1]

def main():
    global VALIDATION_ACTUAL, VALIDATION_EXPECTED
    checkpoint('freeze existing native expectations and exact choices')
    native=json.loads(Path('/tmp/g3-profile-status-family-native.json').read_text())
    by_id={c['id']:c for c in native['cases']}
    cases=[by_id[k] for k in ['F01-p1','F02-p1','F03-p1']]
    frozen={k:v for k,v in native.items() if k!='cases'};frozen['cases']=cases
    fixture=OUT/'native-pack.json';fixture.write_text(json.dumps(frozen,separators=(',',':'))+'\n')
    assert native['pin']==json.loads(Path('reference-lock.json').read_text())['commit']
    META.update(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),native_pin=native['pin'],fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
      source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('simulator/shared_engine.py'),Path('simulator/shared_engine_io.py'),Path('test/shared_engine_support.py'),Path('benchmarks/shared_engine.py'),Path(__file__),Path('benchmarks/g3_diagnostic_native.cjs')]},
      affinity=sorted(os.sched_getaffinity(0)),jax=jax.__version__,devices={p:list(map(str,jax.devices(p))) for p in ['cpu','gpu']},lengths={c['id']:len(c['steps']) for c in cases},
      output_contract='JAX retains every full Result/state/trace plus numeric legal masks on device. Showdown retains normal Battle state, string log and normal generated requests; no per-boundary oracle snapshots or per-draw observer. Host projection/transfer excluded. Different output representations and retention costs remain.')
    actions=[];starts=[]
    for i,c in enumerate(cases):
        for n,step in enumerate(c['steps']):
            a=[-1,-1]
            for side,command in step['choices'].items():
                side_index=int(side[1])-1;verb,slot=command.split();slot=int(slot)-1
                if verb=='switch':slot+=4
                else:
                    s=step['before']['sides'][side_index];mon=next(p for p in s['pokemon'] if p['id']==s['active'])
                    if not any(m['pp'] for m in mon['moves']):slot=10
                a[side_index]=slot
            actions.append(a);starts.append(i if n==0 else -1)
    host_actions=np.asarray(actions,np.int32);host_starts=np.asarray(starts,np.int32)
    compiled={};initials={};test=unittest.TestCase()
    # The first warm execution is also the one trajectory correctness gate.
    for platform in ['cpu','gpu']:
        device=jax.devices(platform)[0]
        with jax.default_device(device):
            checkpoint(platform+' initialize fixed pack')
            t=time.perf_counter();states=[support.initialize_case(c)[0] for c in cases];ready(states)
            META.setdefault('initialization_seconds',{})[platform]=time.perf_counter()-t
            initials[platform]=jax.device_get(states)
            for c,s in zip(cases,states):support.assert_state(test,s,c['initial'],c['id']+' initial')
            args=(jax.tree.map(lambda *x:np.stack(x)[:,None],*jax.device_get(states)),host_actions[:,None,:],host_starts)
            t=time.perf_counter();args=jax.device_put(args,device);ready(args);transfer=time.perf_counter()-t
            checkpoint(platform+' compile one pack')
            t=time.perf_counter();fn=jax.jit(run).lower(*args).compile();compile_seconds=time.perf_counter()-t
            checkpoint(platform+' warm and validate one pack')
            ready(args);t=time.perf_counter();result=ready(fn(*args));warm=time.perf_counter()-t
            placement((args,result),platform)
            host=jax.device_get(result);offset=0
            for c in cases:
                histories=[list(c['initialEvents'][f'p{i+1}']) for i in range(2)]
                before=jax.device_get(states[cases.index(c)])
                for n,step in enumerate(c['steps']):
                    r=jax.tree.map(lambda x:x[offset,0],host[0]);mask=host[1][offset,0];offset+=1
                    META['validation_position']={'platform':platform,'case':c['id'],'boundary':n+1}
                    VALIDATION_ACTUAL=r;VALIDATION_EXPECTED=step
                    # Placement was checked on every device leaf above. Compare the
                    # bulk-fetched outputs, reusing the actual compiled legal masks.
                    with patch.object(e,'legal_actions',lambda _:mask):
                        support.assert_step(test,before,r,step,f"{c['id']}/{n+1}")
                    for side in range(2):
                        histories[side].extend(step['events'][f'p{side+1}'])
                        test.assertEqual(player_view(r.state,side)['events'],histories[side])
                    before=r.state
                assert int(before.winner)>=0
            assert np.all(host[0].code==e.OK)
            META['gates'][platform]={'status':'passed','boundaries':len(actions),'actual_placement':True,'output_digest':digest(result),'compile_seconds':compile_seconds,'input_transfer_seconds':transfer,'warmup_seconds':warm}
            compiled[platform]=fn;save()
    assert META['gates']['cpu']['output_digest']==META['gates']['gpu']['output_digest']
    META['validation_position']=None;checkpoint('both correctness gates passed')
    native_process=subprocess.Popen(['node','benchmarks/g3_diagnostic_native.cjs',str(fixture)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    try:
        for copies in [1,32]:
            checkpoint(f'Showdown {copies} pack(s), one warm plus three repetitions')
            native_process.stdin.write(json.dumps({'copies':copies})+'\n');native_process.stdin.flush()
            row=json.loads(native_process.stdout.readline());assert 'error' not in row,row
            META['rows'].append(row);save()
            for platform in ['cpu','gpu']:
                with jax.default_device(jax.devices(platform)[0]):
                    checkpoint(f'{platform} {copies} pack(s) prepare inputs')
                    # Fresh independent state buffers for warm/each measured call.
                    t=time.perf_counter();base=jax.tree.map(lambda *x:np.stack(x),*initials[platform])
                    hostargs=(jax.tree.map(lambda x:np.repeat(x[:,None],copies,axis=1),base),np.repeat(host_actions[:,None,:],copies,axis=1),host_starts)
                    prep=time.perf_counter()-t
                    def fresh():
                        return ready(jax.device_put(jax.tree.map(lambda x:x.copy(),hostargs),jax.devices(platform)[0]))
                    t=time.perf_counter();args=fresh();transfer=time.perf_counter()-t
                    compile_seconds=0
                    if copies==1:fn=compiled[platform];warm=META['gates'][platform]['warmup_seconds']
                    else:
                        checkpoint(f'{platform} {copies} pack(s) compile')
                        t=time.perf_counter();fn=jax.jit(run).lower(*args).compile();compile_seconds=time.perf_counter()-t
                        t=time.perf_counter();warm_result=ready(fn(*args));warm=time.perf_counter()-t
                        del warm_result
                    samples=[];transfers=[transfer]
                    for repetition in range(3):
                        t=time.perf_counter();args=fresh();transfers.append(time.perf_counter()-t)
                        ready(args);t=time.perf_counter();result=ready(fn(*args));samples.append(time.perf_counter()-t)
                        placement(result,platform)
                        # Counter check and digest outside timing; no padded steps exist.
                        codes=np.asarray(result[0].code);assert np.all(codes==e.OK)
                        assert codes.size==len(actions)*copies
                        for end in np.cumsum([len(c['steps']) for c in cases])-1:assert np.all(np.asarray(result[0].state.winner[end])>=0)
                        del result
                    META['rows'].append({'backend':platform,'copies':copies,'samples_seconds':samples,'compile_seconds':compile_seconds,'warmup_seconds':warm,'host_preparation_seconds':prep,'input_transfer_seconds':transfers,'threads':len(list(Path('/proc/self/task').iterdir()))})
                    save()
    finally:
        native_process.terminate();native_process.wait(timeout=2)
    META['status']='complete';checkpoint('complete')
try:main()
except BaseException as error:
    META['status']='failed';META['first_divergence_or_error']=repr(error)
    if META.get('validation_position') and 'VALIDATION_ACTUAL' in globals():
        np.savez_compressed(OUT/'first-divergence.npz',**{'leaf_'+str(i):x for i,x in enumerate(jax.tree.leaves(VALIDATION_ACTUAL))})
        (OUT/'first-divergence-expected.json').write_text(json.dumps(VALIDATION_EXPECTED,indent=2)+'\n')
    save();raise
