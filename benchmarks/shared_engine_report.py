"""Summarize matched CPU/GPU measurements, exact output digests and profiles."""
import argparse
from collections import Counter,defaultdict
import csv
import gzip
import json
from pathlib import Path
import pstats
import re
import statistics


def profile_summary(directory):
    root=Path(directory);result={'host_profiles':{},'traces':[]}
    for path in sorted(root.glob('host-*.prof')):
        stats=pstats.Stats(str(path));groups=defaultdict(float)
        for (filename,line,name),(primitive,total,self_seconds,cumulative,callers) in stats.stats.items():
            groups[Path(filename).name]+=self_seconds
        result['host_profiles'][path.name]={'self_seconds':stats.total_tt,
            'self_time_by_file':dict(sorted(groups.items(),key=lambda x:-x[1])),
            'top_self_functions':[{'file':file,'line':line,'function':name,'calls':value[1],'self_seconds':value[2],'cumulative_seconds':value[3]} for (file,line,name),value in sorted(stats.stats.items(),key=lambda x:-x[1][2])[:12]]}
    for path in sorted(root.glob('jax-trace/**/*.trace.json.gz')):
        with gzip.open(path,'rt') as handle:trace=json.load(handle)
        events=trace['traceEvents'];processes={event['pid']:event.get('args',{}).get('name','') for event in events if event.get('ph')=='M' and event.get('name')=='process_name'}
        gpu_pids={pid for pid,name in processes.items() if 'gpu' in name.lower() or '/device:' in name.lower() and 'cpu' not in name.lower()}
        spans=[event for event in events if event.get('ph')=='X' and event.get('dur',0)>0]
        gpu=[event for event in spans if event.get('pid') in gpu_pids]
        totals=Counter();intervals=[]
        for event in gpu:
            totals[event['name']]+=event['dur'];intervals.append((event['ts'],event['ts']+event['dur']))
        busy=0;first=None;last=None
        for start,end in sorted(intervals):
            if first is None:first=start;last=end
            elif start>last:busy+=last-first;first=start;last=end
            else:last=max(last,end)
        if first is not None:busy+=last-first
        extent=max((end for start,end in intervals),default=0)-min((start for start,end in intervals),default=0)
        result['traces'].append({'path':str(path),'processes':processes,'complete_span_count':len(spans),
            'gpu_span_count':len(gpu),'gpu_summed_span_microseconds':sum(totals.values()),
            'gpu_union_busy_microseconds':busy,'gpu_first_to_last_microseconds':extent,
            'gpu_busy_fraction_within_first_last':busy/extent if extent else None,
            'gpu_top_names_by_summed_duration':dict(totals.most_common(15)),
            'note':'GPU process spans can include transfers; summed durations may overlap. Busy fraction uses their interval union. Host nested spans are not summed as wall time.'})
    hlo=root/'profile.optimized-hlo.txt.gz'
    if hlo.exists():
        with gzip.open(hlo,'rt') as handle:text=handle.read()
        result['optimized_hlo']={'path':str(hlo),'instruction_mentions':{name:len(re.findall(r'\b'+re.escape(name)+r'\(',text)) for name in ('while','conditional','copy','sort','scatter','dynamic-update-slice','fusion')},'note':'Static optimized HLO instruction mentions, not runtime shares.'}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--cpu',required=True);p.add_argument('--gpu',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();cpu=json.loads(Path(args.cpu).read_text());gpu=json.loads(Path(args.gpu).read_text())
    assert cpu['sha256']==gpu['sha256'],'Measured relevant source/config hashes differ'
    assert cpu['native_fixture_sha256']==gpu['native_fixture_sha256'],'Native workloads differ'
    result={'schemaVersion':1,'source_sha256':cpu['sha256'],'native_fixture_sha256':cpu['native_fixture_sha256'],
            'retention_caveat':'All per-boundary states/traces are materialized. The last warmup result remains live during measurement, and consecutive calls can temporarily overlap output lifetimes. Observed peaks describe this harness lifetime, not the minimum standalone advance footprint; compiler buffer analysis is per invocation. Conservative headroom guards also see allocator-cached memory as unavailable physical free memory.',
            'counter_semantics':'Raw decisions count submitted nonnegative choices: accepted player decisions for live primary cohorts, attempted submissions for terminal/rejected diagnostics. Diagnostics have zero useful boundaries and accepted player decisions. Timed periodic-plan repeat counts are calibrated per backend and disclosed in raw measurements.',
            'cpu':{'checkpoint':cpu['checkpoint'],'kernel_points':len(cpu['kernels']),'scan_points':len(cpu['rows']),'skipped':cpu['skipped']},
            'gpu':{'checkpoint':gpu['checkpoint'],'kernel_points':len(gpu['kernels']),'scan_points':len(gpu['rows']),'skipped':gpu['skipped']},
            'kernels':[],'scans':[],'kernel_crossovers':{},'scan_crossovers':{}}
    ck={(r['cohort'],r['batch']):r for r in cpu['kernels']};gk={(r['cohort'],r['batch']):r for r in gpu['kernels']}
    for key in sorted(ck.keys()&gk.keys()):
        c,g=ck[key],gk[key];assert c['output_digest']==g['output_digest'],f'Kernel output mismatch {key}'
        cm,gm=c['measurement']['median_seconds'],g['measurement']['median_seconds']
        result['kernels'].append({'cohort':key[0],'batch':key[1],'cpu_seconds':cm,'gpu_seconds':gm,'cpu_over_gpu':cm/gm,'exact_output_digest_equal':True})
    cr={(r['cohort'],r['batch'],r['chunk']):r for r in cpu['rows']};gr={(r['cohort'],r['batch'],r['chunk']):r for r in gpu['rows']}
    for key in sorted(cr.keys()&gr.keys()):
        c,g=cr[key],gr[key];assert c['single_chunk_output_digest']==g['single_chunk_output_digest'],f'Scan output mismatch {key}'
        result['scans'].append({'cohort':key[0],'batch':key[1],'chunk':key[2],
            'cpu_boundaries_per_second':c['boundaries_per_second'],'gpu_boundaries_per_second':g['boundaries_per_second'],
            'cpu_seconds':c['sustained']['median_seconds'],'gpu_seconds':g['sustained']['median_seconds'],
            'cpu_over_gpu':c['sustained']['median_seconds']/g['sustained']['median_seconds'],'exact_single_chunk_output_digest_equal':True})
    for cohort in cpu['config']['primary']:
        wins=[r['batch'] for r in result['kernels'] if r['cohort']==cohort and r['cpu_over_gpu']>1]
        result['kernel_crossovers'][cohort]=min(wins) if wins else None
        for chunk in cpu['config']['chunks']:
            wins=[r['batch'] for r in result['scans'] if r['cohort']==cohort and r['chunk']==chunk and r['cpu_over_gpu']>1]
            result['scan_crossovers'][f'{cohort}/chunk{chunk}']=min(wins) if wins else None
    for platform,data,path in [('cpu',cpu,args.cpu),('gpu',gpu,args.gpu)]:
        rows=data['rows'];data_summary=result[platform]
        target_kernels=0;target_scans=0;selection=data.get('selection',{})
        for cohort in data['config']['primary']+data['config']['diagnostic']:
            if selection.get('cohorts') and cohort not in selection['cohorts']:continue
            primary=cohort in data['config']['primary']
            batches=data['config']['batches'] if primary else data['config']['diagnosticBatches']
            chunks=data['config']['chunks'] if primary else data['config']['diagnosticChunks']
            if selection.get('batches'):batches=[x for x in batches if x in selection['batches']]
            if selection.get('chunks'):chunks=[x for x in chunks if x in selection['chunks']]
            target_kernels+=len(batches);target_scans+=len(batches)*len(chunks)
        data_summary.update(target_kernel_points=target_kernels,target_scan_points=target_scans,
                            skipped_kernel_points=target_kernels-len(data['kernels']),
                            skipped_scan_points=target_scans-len(rows))
        for row in rows:
            if row['cohort'] in data['config']['primary']:
                assert row['cumulative_counts']['absorbing_or_rejected']==0,row
                assert row['cumulative_counts']['pending']==0,row
                assert row['useful_boundaries_per_call']==row['batch']*row['chunk'],row
        data_summary['scan_compile_seconds']={'median':statistics.median(r['scan_compile_seconds'] for r in rows),'max':max(r['scan_compile_seconds'] for r in rows)}
        data_summary['host_process_peak_rss_bytes']=max(r['host_maxrss_kib'] for r in rows)*1024
        data_summary['max_compiler_scan_buffer_bytes']=max(sum(r['compiler_memory'][k] for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes'))-r['compiler_memory']['alias_size_in_bytes'] for r in rows)
        allocators=[r['device_memory_stats'] for r in rows if r.get('device_memory_stats')]
        if allocators:
            data_summary['jax_allocator']={key:max(value.get(key,0) for value in allocators) for key in ('peak_bytes_in_use','peak_pool_bytes','bytes_limit')}
        data_summary['host_points']=[{k:r[k] for k in ('cohort','batch','projection_serialization','host_submission','reset','device_get_samples_seconds','device_put_samples_seconds','transfer_bytes')} for r in rows if 'host_submission' in r]
        data_summary['experimental_diagnostics']=[{'cohort':r['cohort'],'batch':r['batch'],'chunk':r['chunk'],'baseline_kernel_seconds':r['kernel']['median_seconds'],'diagnostics':r['experimental_diagnostics']} for r in rows if 'experimental_diagnostics' in r]
        data_summary['profiles']=profile_summary(Path(path).parent)
    monitor=Path(args.gpu).parent.parent/'profile-gpu-device.csv'
    if monitor.exists():
        with monitor.open() as handle:observations=list(csv.reader(handle))
        def values(column):
            result=[]
            for row in observations:
                try:result.append(float(row[column].strip()))
                except (ValueError,IndexError):pass
            return result
        result['gpu']['device_monitor']={'path':str(monitor),'samples':len(observations),
            'peak_total_device_used_mib':max(values(1),default=None),
            'min_reported_device_free_mib':min(values(2),default=None),
            'peak_reported_gpu_utilization_percent':max(values(3),default=None),
            'peak_power_watts':max(values(5),default=None),
            'sm_clock_mhz_range':[min(values(6),default=None),max(values(6),default=None)],
            'note':'One-second device-wide samples include display/other activity and can miss short peaks. JAX process allocator high-water data and compiler buffer analysis are separately retained in raw measurements.'}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'matched_kernel_points':len(result['kernels']),'matched_scan_points':len(result['scans']),'kernel_crossovers':result['kernel_crossovers'],'scan_crossovers':result['scan_crossovers']},indent=2))


if __name__=='__main__':main()
