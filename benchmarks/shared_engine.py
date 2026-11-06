"""Synchronized kernel, sustained scan, and host integration measurements.

Example: JAX_PLATFORMS=cpu python -m benchmarks.shared_engine --config
benchmarks/g3-profile.json --native /tmp/g3-profile-baseline-native.json
--platform cpu --output /tmp/profile-cpu-new

Native-supplied choice plans repeat only after real terminal/reset boundaries.
Full state and trace outputs are materialized; no checksum-only shortcut.
"""
import argparse
import cProfile
import datetime
import gzip
import hashlib
import json
import os
import platform
from pathlib import Path
import resource
import statistics
import subprocess
import time
from unittest.mock import patch

import jax
import jaxlib
import jax.numpy as jnp
import numpy as np

from simulator import shared_engine as e
from simulator.shared_engine_io import player_view
from test.shared_engine_support import initialize_case, choices_array
from benchmarks.shared_engine_campaign import placement


def nbytes(tree):return sum(x.size*x.dtype.itemsize for x in jax.tree.leaves(tree))
def ready(tree):return jax.block_until_ready(tree)
def stack(xs):
    # Preparation is outside timers. Avoid compiling thousands of tiny device
    # concatenations when a benchmark lane repeats a prepared native state.
    host={}
    for tree in xs:
        if id(tree) not in host:host[id(tree)]=jax.device_get(tree)
    return jax.tree.map(lambda *x:jax.device_put(np.stack(x)),*(host[id(tree)] for tree in xs))


def digest(tree):
    h=hashlib.sha256()
    for leaf in jax.tree.leaves(jax.device_get(tree)):
        h.update(str((leaf.shape,str(leaf.dtype))).encode());h.update(np.asarray(leaf).tobytes())
    return h.hexdigest()


def memory_available(platform):
    host=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    if platform=='cpu':return host
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','--id=0'],text=True).strip())*1024**2
    return min(host,free)


def measure(fn,args,repetitions,target,sustained=False,warm_calls=2):
    warm_start=time.perf_counter();warm_args=args
    for _ in range(warm_calls):
        warmed=ready(fn(*warm_args))
        if sustained:warm_args=(*warmed[0],*args[3:])
    warm_seconds=time.perf_counter()-warm_start
    t=time.perf_counter();ready(fn(*args));latency=time.perf_counter()-t
    inner=max(1,int(np.ceil(target/max(latency,1e-6))))
    samples=[];last=None;active_args=args
    for _ in range(repetitions):
        ready(args);start=time.perf_counter()
        for _ in range(inner):
            last=fn(*active_args);ready(last)
            if sustained:active_args=(*last[0],*args[3:])
        samples.append((time.perf_counter()-start)/inner)
    return {'median_seconds':statistics.median(samples),'min_seconds':min(samples),
            'max_seconds':max(samples),'p10_seconds':float(np.quantile(samples,.1)),
            'p90_seconds':float(np.quantile(samples,.9)),'samples_seconds':samples,
            'calls_per_repetition':inner,'total_measured_calls':inner*repetitions,
            'warm_calls':warm_calls,'warmup_seconds':warm_seconds},last


def plans(cases,cohort,size):
    by_id={c['id']:c for c in cases}
    names={'ordinary':['M03-duel-accuracy'],'mixed':['F01-p1','F02-p1','F03-p1','F04-p1'],
           'mixed-aligned':['F01-p1','F02-p1','F03-p1','F04-p1'],
           'tied':['M01-duel-tied'],'status':['F02-p1'],'pivot':['F03-p1'],
           'exhaustion':['F04-p1'],'terminal':['M03-duel-accuracy'],'rejected':['M03-duel-accuracy']}[cohort]
    selected=[by_id[names[k%len(names)]] for k in range(size)]
    initials=[];terminals=[];actions=[];lengths=[];prefixes=[];scalar=jax.jit(e.advance)
    for name in names:
        c=by_id[name];s,_=initialize_case(c);initial=s;plan=[];prefix=[]
        for step in c['steps']:
            prefix.append(s)
            a=choices_array(s,step['choices']);plan.append(np.asarray(a));s=scalar(s,a).state
        assert int(s.winner)>=0,'sustained plans require complete native battles'
        initials.append(initial);terminals.append(s);actions.append(plan);lengths.append(len(plan));prefixes.append(prefix)
    maxlen=max(lengths)
    padded=np.full((size,maxlen,2),-1,np.int32)
    states=[];resets=[];ends=[];lens=[];indices=[]
    for lane,c in enumerate(selected):
        i=names.index(c['id']);padded[lane,:lengths[i]]=actions[i];lens.append(lengths[i]);resets.append(initials[i]);ends.append(terminals[i])
        offset=(lane*7)%lengths[i] if cohort=='mixed' else 0
        indices.append(offset);states.append(prefixes[i][offset])
    initial=stack(states)
    if cohort=='terminal':
        # Advance each native plan separately, then stack absorbing states.
        initial=stack(ends)
    if cohort=='rejected':padded[:]=11
    return initial,jnp.asarray(padded),e.i32(lens),stack(resets),e.i32(indices),stack(ends)


def rollout(chunk,absorbing=False):
    def one(s,index,initial,plan,length):
        terminal=s.winner>=0
        if not absorbing:
            s=jax.lax.cond(terminal,lambda s:e.reset(s,initial,jnp.stack((s.generation,s.episode))).state,lambda s:s,s)
        action=plan[index]
        r=e.advance(s,action)
        counts=jnp.stack(((r.code==e.OK)&(r.trace.ecount>0),
                          jnp.sum(action>=0),
                          (s.winner<0)&(r.state.winner>=0),
                          terminal & (not absorbing),
                          r.code>=e.INVALID,r.code==e.PENDING,
                          r.trace.ecount,r.trace.dcount)).astype(jnp.int32)
        return r.state,(index+1)%length,r,counts
    batch_one=jax.vmap(one)
    def run(states,indices,counts,initial,plan,length):
        def body(carry,_):
            s,index,counts=carry;s,index,r,added=batch_one(s,index,initial,plan,length)
            return (s,index,counts+jnp.sum(added,axis=0)),r
        return jax.lax.scan(body,(states,indices,counts),None,length=chunk)
    return run


def diagnostic_ablations(states,actions,baseline,repetitions,target):
    """Experimental valid-input probes; neither changes the primary API timer."""
    results={}
    with patch.object(e,'_supported',lambda state:jnp.array(True)):
        fn=jax.jit(jax.vmap(lambda state,action:e.advance(state,action)))
        started=time.perf_counter();compiled=fn.lower(states,actions).compile()
        compilation=time.perf_counter()-started
    timing,result=measure(compiled,(states,actions),repetitions,target)
    assert digest(result)==digest(baseline)
    results['validation_removed']={'experimental':True,'contract':'Only valid prepared inputs; invalid/unsupported validation is bypassed. Not a production speedup.','measurement':timing,'lower_and_compile_seconds':compilation,'same_valid_output_digest':True}
    def state_and_code(state,action):
        result=e.advance(state,action)
        return result.state,result.code
    fn=jax.jit(jax.vmap(state_and_code))
    started=time.perf_counter();compiled=fn.lower(states,actions).compile()
    compilation=time.perf_counter()-started
    timing,result=measure(compiled,(states,actions),repetitions,target)
    assert digest(result)==digest((baseline.state,baseline.code))
    results['trace_outputs_elided']={'experimental':True,'contract':'State and diagnostic code only; required draw/event trace API outputs discarded. Not a production speedup.','measurement':timing,'lower_and_compile_seconds':compilation,'same_retained_output_digest':True}
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--native',required=True)
    p.add_argument('--platform',choices=('cpu','gpu'),required=True);p.add_argument('--output',required=True)
    p.add_argument('--cohorts',nargs='+');p.add_argument('--batches',nargs='+',type=int);p.add_argument('--chunks',nargs='+',type=int)
    args=p.parse_args();config=json.loads(Path(args.config).read_text());assert config['schemaVersion']==1
    assert config['repetitions']>=10
    if args.cohorts:assert set(args.cohorts)<=set(config['primary']+config['diagnostic'])
    if args.batches:assert set(args.batches)<=set(config['batches'])
    if args.chunks:assert set(args.chunks)<=set(config['chunks'])
    assert jax.default_backend()==args.platform
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    cases=json.loads(Path(args.native).read_text())['cases']
    files=['simulator/shared_engine.py','simulator/shared_engine_io.py','benchmarks/shared_engine.py',
           'benchmarks/shared_engine_campaign.py','test/shared_engine_support.py','rules/shared-engine-v1.json',args.config]
    meta={'schemaVersion':1,'platform':args.platform,'devices':list(map(str,jax.devices())),
          'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'python':platform.python_version(),'jaxlib':jaxlib.__version__,'numpy':np.__version__,
          'native_fixture_sha256':hashlib.sha256(Path(args.native).read_bytes()).hexdigest(),
          'native_pin':json.loads(Path(args.native).read_text())['pin'],
          'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
          'affinity':sorted(os.sched_getaffinity(0)),'config':config,'persistent_cache':str(jax.config.jax_compilation_cache_dir),
          'persistent_cache_enabled':bool(jax.config.jax_enable_compilation_cache),
          'selection':{'cohorts':args.cohorts,'batches':args.batches,'chunks':args.chunks},
          'jax':jax.__version__,'rows':[],'kernels':[],'skipped':[]}
    def save():
        (out/'results.json').write_text(json.dumps(meta,indent=2)+'\n')
    save()
    for cohort in config['primary']+config['diagnostic']:
        if args.cohorts and cohort not in args.cohorts:continue
        batches=config['batches'] if cohort in config['primary'] else config['diagnosticBatches']
        chunks=config['chunks'] if cohort in config['primary'] else config['diagnosticChunks']
        if args.batches:batches=[x for x in batches if x in args.batches]
        if args.chunks:chunks=[x for x in chunks if x in args.chunks]
        for size in batches:
            meta['checkpoint']={'cohort':cohort,'batch':size,'phase':'prepare and compile kernel'};save()
            single,_=initialize_case(cases[0]);bytes_lane=nbytes(single)+nbytes(e.empty_trace())+4
            free=memory_available(args.platform)
            if bytes_lane*size*config['allocationSafetyFactor']+config['compilerReserveBytes']>free*config['memoryFraction']:
                meta['skipped'].append({'cohort':cohort,'batch':size,'reason':'input/kernel allocation estimate','free_bytes':free});continue
            placement_start=time.perf_counter();states,plan,length,initials,indices,terminals=plans(cases,cohort,size)
            ready((states,plan,length,initials,indices));placement_seconds=time.perf_counter()-placement_start
            placement((states,plan,length),args.platform)
            kernel_actions=plan[jnp.arange(size),indices]
            kernel=jax.jit(jax.vmap(e.advance));t=time.perf_counter();lower=kernel.lower(states,kernel_actions);lower_seconds=time.perf_counter()-t
            t=time.perf_counter();compiled=lower.compile();compile_seconds=time.perf_counter()-t
            kernel_memory=compiled.memory_analysis()
            kernel_memory_dict={field:getattr(kernel_memory,field) for field in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}
            if sum(kernel_memory_dict[k] for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes'))>memory_available(args.platform)*config['memoryFraction']:
                meta['skipped'].append({'cohort':cohort,'batch':size,'reason':'kernel compiler memory analysis','memory':kernel_memory_dict});save();continue
            km,kr=measure(compiled,(states,kernel_actions),config['repetitions'],config['targetSeconds'])
            placement(kr,args.platform)
            meta['kernels'].append({'cohort':cohort,'batch':size,'measurement':km,'lower_seconds':lower_seconds,'compile_seconds':compile_seconds,'compiler_memory':kernel_memory_dict,'output_digest':digest(kr)})
            save()
            for chunk in chunks:
                meta['checkpoint']={'cohort':cohort,'batch':size,'chunk':chunk,'phase':'compile and measure full scan'};save()
                estimate=(nbytes(states)*2+nbytes(kr)*chunk)*config['allocationSafetyFactor']+config['compilerReserveBytes']
                free=memory_available(args.platform)
                if estimate>free*config['memoryFraction']:
                    meta['skipped'].append({'cohort':cohort,'batch':size,'chunk':chunk,'reason':'retained output allocation estimate','estimated_bytes':estimate,'free_bytes':free});continue
                run=jax.jit(rollout(chunk,cohort in ('terminal','rejected')))
                callargs=(states,indices,jnp.zeros(8,jnp.int32),initials,plan,length)
                t=time.perf_counter();lower=run.lower(*callargs);tracing=time.perf_counter()-t
                t=time.perf_counter();compiled_run=lower.compile();compiling=time.perf_counter()-t
                memory=compiled_run.memory_analysis()
                memory_dict={field:getattr(memory,field) for field in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}
                if sum(memory_dict[k] for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes'))>free*config['memoryFraction']:
                    meta['skipped'].append({'cohort':cohort,'batch':size,'chunk':chunk,'reason':'compiler memory analysis','memory':memory_dict});continue
                warm_calls=max(2,int(np.ceil(np.max(np.asarray(length))/chunk))+1)
                sm,result=measure(compiled_run,callargs,config['repetitions'],config['targetSeconds'],sustained=True,warm_calls=warm_calls)
                placement(result,args.platform)
                code=np.asarray(result[1].code);htrace=jax.device_get(result[1].trace)
                diagnostic=cohort in ('terminal','rejected')
                assert diagnostic or np.all(np.isin(code,[e.OK,e.PENDING])),np.unique(code)
                totals=np.asarray(result[0][2]);calls=sm['total_measured_calls']
                useful=float(totals[0])/calls
                row={'cohort':cohort,'batch':size,'chunk':chunk,'kernel':km,'sustained':sm,
                     'kernel_output_digest':digest(kr),'single_chunk_output_digest':digest(ready(compiled_run(*callargs))),
                     'kernel_lower_seconds':lower_seconds,'kernel_compile_seconds':compile_seconds,
                     'scan_lower_seconds':tracing,'scan_compile_seconds':compiling,'preparation_seconds':placement_seconds,
                     'useful_boundaries_per_call':useful,'cumulative_counts':dict(zip(('boundaries','decisions','completed_battles','resets','absorbing_or_rejected','pending','events','draws'),map(int,totals))),
                     'draws_per_call':float(totals[7])/calls,'events_per_call':float(totals[6])/calls,
                     'decisions_per_second':float(totals[1])/calls/sm['median_seconds'],
                     'completed_battles_per_second':float(totals[2])/calls/sm['median_seconds'],
                     'prepared_request_pairs':{','.join(map(str,pair)):int(count) for pair,count in zip(*np.unique(np.asarray(states.request),axis=0,return_counts=True))},
                     'last_chunk_output_request_pairs':{','.join(map(str,pair)):int(count) for pair,count in zip(*np.unique(np.asarray(result[1].state.request).reshape(-1,2),axis=0,return_counts=True))},
                     'boundaries_per_second':useful/sm['median_seconds'],'compiler_memory':memory_dict,
                     'host_maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     'device_memory_stats':jax.devices()[0].memory_stats(),
                     'output_bytes':nbytes(result),'full_output_contract':config['outputContract']}
                # Host conversion and projection are outside all device timers.
                if size<=32 and chunk==chunks[0]:
                    gets=[];puts=[]
                    for _ in range(config['repetitions']):
                        # Fresh outputs avoid timing an already cached host
                        # value. Kernel preparation is outside transfer timers.
                        fresh=ready(compiled(states,kernel_actions))
                        start=time.perf_counter();host=jax.device_get(fresh);gets.append(time.perf_counter()-start)
                        start=time.perf_counter();ready(jax.device_put(host));puts.append(time.perf_counter()-start)
                    row['transfer_bytes']=nbytes(fresh)
                    row['device_get_samples_seconds']=gets;row['device_put_samples_seconds']=puts
                    def project():
                        return json.dumps([player_view(jax.tree.map(lambda x:x[k],kr.state),side) for k in range(size) for side in range(2)])
                    hm,text=measure(project,(),config['repetitions'],config['targetSeconds']);row['projection_serialization']=hm
                    row['serialized_bytes']=len(text.encode())
                    scalar=jax.jit(e.advance)
                    host_actions=np.asarray(kernel_actions)
                    def host_submission():
                        views=[]
                        for k in range(size):
                            s=jax.tree.map(lambda x:x[k],states)
                            public=[player_view(s,side) for side in range(2)]
                            mask=np.asarray(e.legal_actions(s));action=e.i32(host_actions[k])
                            r=scalar(s,action);ready(r)
                            views.append({'before':public,'legal':mask.tolist(),'after':[player_view(r.state,side) for side in range(2)]})
                        return json.dumps(views)
                    row['host_submission'],_=measure(host_submission,(),config['repetitions'],config['targetSeconds'])
                    reset_fn=jax.jit(jax.vmap(e.reset));tokens=jnp.stack((terminals.generation,terminals.episode),axis=1)
                    row['reset'],reset_result=measure(reset_fn,(terminals,initials,tokens),config['repetitions'],config['targetSeconds'])
                    assert np.all(np.asarray(reset_result.code)==e.OK)
                    cProfile.runctx('project()',globals(),locals(),str(out/f'host-projection-{cohort}-{size}.prof'))
                    cProfile.runctx('host_submission()',globals(),locals(),str(out/f'host-submission-{cohort}-{size}.prof'))
                point=config['profilerPoint']
                if cohort==point['cohort'] and size==point['batch'] and chunk==point['chunk']:
                    with gzip.open(out/'profile.stablehlo.txt.gz','wt') as f:f.write(lower.as_text())
                    with gzip.open(out/'profile.optimized-hlo.txt.gz','wt') as f:f.write(compiled_run.as_text())
                    with jax.profiler.trace(str(out/'jax-trace'),create_perfetto_link=False,create_perfetto_trace=True):ready(compiled_run(*callargs))
                    row['profiler_trace']='jax-trace';row['hlo']='profile.stablehlo.txt.gz';row['optimized_hlo']='profile.optimized-hlo.txt.gz'
                    row['experimental_diagnostics']=diagnostic_ablations(states,kernel_actions,kr,config['repetitions'],config['targetSeconds'])
                meta['rows'].append(row);save()
                print(json.dumps({k:row[k] for k in ('cohort','batch','chunk','boundaries_per_second','compiler_memory')}),flush=True)
                del compiled_run,run,result,callargs
            del compiled,kernel,states,plan,length,kr,initials,indices,terminals,kernel_actions
            jax.clear_caches()
    meta['checkpoint']={'phase':'complete requested selection'};save()


if __name__=='__main__':main()
