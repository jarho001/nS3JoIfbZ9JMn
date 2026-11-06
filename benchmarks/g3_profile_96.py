"""One bounded profile of the unchanged, parity-tested 96-battle diagnostic loop."""
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

START=time.monotonic()
OUT=Path('test-logs/g3-testing-profiling/diagnostic-profile-96')
OUT.mkdir(exist_ok=False)
meta={'status':'running','budget_seconds':270,'prior_setup_seconds':20.274,'checkpoint':'imports'}
def save():
    meta['elapsed_seconds']=time.monotonic()-START
    (OUT/'results.json').write_text(json.dumps(meta,indent=2)+'\n')
def checkpoint(text):
    meta['checkpoint']=text;save();print(text,flush=True)
def expired(signum,frame):
    meta['status']='budget_exhausted';save();os._exit(124)
signal.signal(signal.SIGALRM,expired);signal.alarm(269);save()
import jax
import numpy as np
from simulator import shared_engine as e
from test.shared_engine_support import initialize_case
from benchmarks.shared_engine import ready, digest, nbytes

def main():
    baseline=json.loads(Path('test-logs/g3-testing-profiling/diagnostic/results.json').read_text())
    for path,expected in baseline['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
    fixture=Path('test-logs/g3-testing-profiling/diagnostic/native-pack.json')
    assert hashlib.sha256(fixture.read_bytes()).hexdigest()==baseline['fixture_sha256']
    cases=json.loads(fixture.read_text())['cases']
    # Import only the exact recorded pure function, without executing the original
    # diagnostic's module-level mkdir, alarm or broad comparison entry point.
    source=Path('benchmarks/g3_diagnostic.py').read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    namespace={'jax':jax,'e':e}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'benchmarks/g3_diagnostic.py','exec'),namespace)
    run=namespace['run']
    meta.update(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      source_sha256=baseline['source_sha256'],profile_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      loop_sha256=hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest(),
      native_fixture_sha256=baseline['fixture_sha256'],affinity=sorted(os.sched_getaffinity(0)),
      jax=jax.__version__,device=str(jax.devices()[0]),copies=32,complete_battles=96,useful_boundaries=4096,
      contract='Exact diagnostic run function and complete retained outputs; no elision or engine change.')
    assert jax.default_backend()=='gpu'
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
    hostargs=(jax.tree.map(lambda *x:np.repeat(np.stack(x)[:,None],32,axis=1),*hoststates),np.repeat(np.asarray(actions,np.int32)[:,None,:],32,axis=1),np.asarray(starts,np.int32))
    args=ready(jax.device_put(hostargs))
    checkpoint('compile unchanged 96-battle loop')
    t=time.perf_counter();compiled=jax.jit(run).lower(*args).compile();meta['compile_seconds']=time.perf_counter()-t
    memory=compiled.memory_analysis()
    meta['compiler_memory']={k:getattr(memory,k) for k in ['argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes']}
    with gzip.open(OUT/'optimized-hlo.txt.gz','wt') as f:f.write(compiled.as_text())
    checkpoint('one warm execution and exact saved-digest validation')
    t=time.perf_counter();warm=ready(compiled(*args));meta['warmup_seconds']=time.perf_counter()-t
    for x in jax.tree.leaves((args,warm)):assert all(d.platform=='gpu' for d in x.devices())
    host=jax.device_get(warm)
    assert digest(jax.tree.map(lambda x:x[:,0:1],host))==baseline['gates']['gpu']['output_digest']
    for x in jax.tree.leaves(host):assert np.array_equal(x,np.broadcast_to(x[:,0:1],x.shape))
    assert np.all(host[0].code==e.OK)
    meta['validation']='Every output leaf of all 32 identical packs equals the previously CPU/GPU/native-validated complete trajectory digest; actual GPU placement checked.'
    meta['output_bytes']={'total':nbytes(warm),'states':nbytes(warm[0].state),'state_history':nbytes(warm[0].state.history),'traces':nbytes(warm[0].trace),'legal_masks':nbytes(warm[1])}
    del warm,host
    # Fresh inputs and all host work are outside the captured region.
    args=ready(jax.device_put(jax.tree.map(lambda x:x.copy(),hostargs)))
    options=jax.profiler.ProfileOptions();options.host_tracer_level=1;options.python_tracer_level=0
    checkpoint('single short warmed GPU profiler capture')
    with jax.profiler.trace(str(OUT/'jax-trace'),create_perfetto_link=False,create_perfetto_trace=False,profiler_options=options):
        with jax.profiler.TraceAnnotation('g3_96_complete_battles'):
            t=time.perf_counter();result=ready(compiled(*args));meta['profiled_call_wall_seconds']=time.perf_counter()-t
    assert digest(jax.tree.map(lambda x:x[:,0:1],jax.device_get(result)))==baseline['gates']['gpu']['output_digest']
    meta['status']='complete';checkpoint('capture complete; no additional measurements')
try:main()
except BaseException as error:
    meta['status']='failed';meta['error']=repr(error);save();raise
