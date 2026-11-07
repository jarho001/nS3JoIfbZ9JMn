"""One bounded GPU capture of a recorded, previously validated 96-battle loop."""
import argparse
from functools import partial
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

START=time.monotonic()
OUT=None
meta={'status':'running','budget_seconds':300,'checkpoint':'imports'}
def save():
    if OUT is None:return
    meta['elapsed_seconds']=time.monotonic()-START
    (OUT/'results.json').write_text(json.dumps(meta,indent=2)+'\n')
def checkpoint(text):
    meta['checkpoint']=text;save();print(text,flush=True)
def expired(signum,frame):
    meta['status']='budget_exhausted';save();os._exit(124)

import jax
import numpy as np
from simulator import shared_engine as e
from test.shared_engine_support import initialize_case
from benchmarks.shared_engine import ready, digest, nbytes
from benchmarks.g3_fixtures import ROOT, BASELINE, PROFILE, function_sha256, source_path, verify_source
from benchmarks.g3_diagnostic import run, run_lean


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=BASELINE)
    parser.add_argument('--output',type=Path,required=True,help='Fresh evidence directory')
    options=parser.parse_args()
    requested=options.output.resolve();requested.mkdir(parents=True,exist_ok=False);OUT=requested
    signal.signal(signal.SIGALRM,expired)
    remaining=299-int(time.monotonic()-START)
    if remaining<=0:expired(None,None)
    signal.alarm(remaining)
    baseline_path=options.baseline.resolve()
    baseline=json.loads(baseline_path.read_text())
    if baseline.get('status')!='complete':raise ValueError('Baseline diagnostic is incomplete')
    retention=baseline.get('retention','full')
    if retention not in ('full','lean'):raise ValueError('Unknown baseline output contract')
    loop_name='run_lean' if retention=='lean' else 'run'
    expected_loop=baseline.get('loop_sha256')
    if expected_loop is None:
        # The historical diagnostic predates a direct loop hash. Its recorded
        # follow-up supplies that hash without requiring the old CLI file bytes.
        legacy=json.loads(PROFILE.read_text())
        if legacy['native_fixture_sha256']!=baseline['fixture_sha256']:
            raise ValueError('Historical loop record belongs to another fixture')
        if legacy['source_sha256']!=baseline['source_sha256']:
            raise ValueError('Historical loop record belongs to another source')
        expected_loop=legacy['loop_sha256']
    diagnostic=ROOT/'benchmarks/g3_diagnostic.py'
    if function_sha256(diagnostic,loop_name)!=expected_loop:
        raise ValueError('The recorded measured loop has changed')
    verified=[]
    for path,expected in baseline['source_sha256'].items():
        resolved=source_path(path)
        if resolved==diagnostic:
            verified.append({'path':resolved.relative_to(ROOT).as_posix(),
                'scope':'measured function only; CLI changes are outside the capture',
                'recorded_file_sha256':expected,
                'current_file_sha256':hashlib.sha256(resolved.read_bytes()).hexdigest(),
                'function_sha256':expected_loop})
        else:verified.append(verify_source(path,expected))
    fixture=baseline_path.parent/'native-pack.json'
    if hashlib.sha256(fixture.read_bytes()).hexdigest()!=baseline['fixture_sha256']:
        raise ValueError('Frozen native fixture changed')
    cases=json.loads(fixture.read_text())['cases']
    lengths=tuple(len(c['steps']) for c in cases)
    selected=partial(run_lean,lengths=lengths) if retention=='lean' else run
    gate=baseline['gates']['gpu'].get('lean') if retention=='lean' else baseline['gates']['gpu']
    if not gate or gate.get('status')!='passed':raise ValueError('Recorded GPU correctness gate is missing')
    expected_digest=gate['output_digest']
    meta.update(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
      baseline=str(baseline_path),source_sha256=baseline['source_sha256'],source_verification=verified,
      profile_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      loop_sha256=expected_loop,retention=retention,native_fixture_sha256=baseline['fixture_sha256'],
      affinity=sorted(os.sched_getaffinity(0)),jax=jax.__version__,device=str(jax.devices()[0]),
      copies=32,complete_battles=len(cases)*32,useful_boundaries=sum(lengths)*32,
      contract='Recorded loop and output contract; no capture-time output elision or engine change.')
    if jax.default_backend()!='gpu':raise RuntimeError('An actual GPU backend is required')
    checkpoint('prepare frozen 96-battle inputs')
    states=[initialize_case(c)[0] for c in cases]
    actions=[];starts=[]
    for i,c in enumerate(cases):
        for n,step in enumerate(c['steps']):
            a=[-1,-1]
            for side,command in step['choices'].items():
                k=int(side[1])-1;verb,slot=command.split();slot=int(slot)-1
                if verb=='switch':slot+=4
                else:
                    s=step['before']['sides'][k];mon=next(p for p in s['pokemon'] if p['id']==s['active'])
                    if not any(m['pp'] for m in mon['moves']):slot=10
                a[k]=slot
            actions.append(a);starts.append(i if n==0 else -1)
    hoststates=jax.device_get(states)
    hostargs=(jax.tree.map(lambda *x:np.repeat(np.stack(x)[:,None],32,axis=1),*hoststates),
              np.repeat(np.asarray(actions,np.int32)[:,None,:],32,axis=1),np.asarray(starts,np.int32))
    args=ready(jax.device_put(hostargs))
    checkpoint('compile recorded 96-battle loop')
    t=time.perf_counter();compiled=jax.jit(selected).lower(*args).compile();meta['compile_seconds']=time.perf_counter()-t
    memory=compiled.memory_analysis()
    meta['compiler_memory']={k:getattr(memory,k) for k in ['argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes']}
    with gzip.open(OUT/'optimized-hlo.txt.gz','wt') as f:f.write(compiled.as_text())
    checkpoint('one warm execution and exact recorded-digest validation')
    t=time.perf_counter();warm=ready(compiled(*args));meta['warmup_seconds']=time.perf_counter()-t
    for x in jax.tree.leaves((args,warm)):
        if not all(d.platform=='gpu' for d in x.devices()):raise RuntimeError('Non-GPU input/output leaf')
    host=jax.device_get(warm)
    first=lambda tree:jax.tree.map(lambda x:x[0:1] if retention=='lean' else x[:,0:1],tree)
    if digest(first(host))!=expected_digest:raise ValueError('Recorded output digest differs')
    for x in jax.tree.leaves(host):
        first_leaf=x[0:1] if retention=='lean' else x[:,0:1]
        if not np.array_equal(x,np.broadcast_to(first_leaf,x.shape)):raise ValueError('Replica outputs differ')
    if retention=='lean':
        if any(np.any(s.failed)|np.any(~s.terminal) for s in host):raise ValueError('A lean lane did not complete')
        sizes={'total':nbytes(warm),'states':sum(nbytes(s.state) for s in warm),
               'state_history':sum(nbytes(s.state.history) for s in warm),'traces':0,'legal_masks':0}
    else:
        if not np.all(host[0].code==e.OK):raise ValueError('A full-output transition failed')
        sizes={'total':nbytes(warm),'states':nbytes(warm[0].state),'state_history':nbytes(warm[0].state.history),
               'traces':nbytes(warm[0].trace),'legal_masks':nbytes(warm[1])}
    meta['validation']='All output leaves of all 32 replicas match the recorded GPU/native-validated digest; GPU placement checked.'
    meta['output_bytes']=sizes
    del warm,host
    args=ready(jax.device_put(jax.tree.map(lambda x:x.copy(),hostargs)))
    profile_options=jax.profiler.ProfileOptions();profile_options.host_tracer_level=1;profile_options.python_tracer_level=0
    checkpoint('single short warmed GPU profiler capture')
    with jax.profiler.trace(str(OUT/'jax-trace'),create_perfetto_link=False,create_perfetto_trace=False,profiler_options=profile_options):
        with jax.profiler.TraceAnnotation('g3_96_complete_battles'):
            t=time.perf_counter();result=ready(compiled(*args));meta['profiled_call_wall_seconds']=time.perf_counter()-t
    if digest(first(jax.device_get(result)))!=expected_digest:raise ValueError('Captured output digest differs')
    meta['status']='complete';checkpoint('capture complete; no additional measurements')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        meta['status']='failed';meta['error']=repr(error);save();raise
    finally:signal.alarm(0)
