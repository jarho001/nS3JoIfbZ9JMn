"""Offline hardware-stream accounting for the single fixed-pack GPU capture."""
from collections import Counter
import gzip
import json
import re
from pathlib import Path

root=Path('test-logs/g3-testing-profiling/diagnostic-profile-96')
path=next(root.glob('jax-trace/**/*.trace.json.gz'))
with gzip.open(path,'rt') as f:events=json.load(f)['traceEvents']
processes={e['pid']:e['args']['name'] for e in events if e.get('ph')=='M' and e.get('name')=='process_name'}
threads={(e['pid'],e['tid']):e['args']['name'] for e in events if e.get('ph')=='M' and e.get('name')=='thread_name'}
annotation=next(e for e in events if e.get('name')=='g3_96_complete_battles' and e.get('ph')=='X')
lo=annotation['ts'];hi=lo+annotation['dur']
streams=[e for e in events if e.get('ph')=='X' and e.get('dur',0)>0 and 'GPU' in processes.get(e.get('pid'),'') and threads.get((e.get('pid'),e.get('tid')),'').startswith('Stream #') and lo<=e['ts'] and e['ts']+e['dur']<=hi]
kernels=[e for e in streams if 'kernel_details' in e.get('args',{})]
nonkernels=[e for e in streams if 'kernel_details' not in e.get('args',{})]
counts=Counter(e['name'] for e in kernels);durations=Counter()
for e in kernels:durations[e['name']]+=e['dur']
intervals=sorted((e['ts'],e['ts']+e['dur']) for e in streams)
merged=[]
for begin,end in intervals:
    if merged and begin<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
    else:merged.append([begin,end])
busy=sum(end-begin for begin,end in merged)
extent=max(end for begin,end in intervals)-intervals[0][0]
copy=[e for e in kernels if e['name']=='memcpy32_post']
transfers={}
for name in sorted({e['name'] for e in nonkernels}):
    selected=[e for e in nonkernels if e['name']==name]
    sizes=Counter(int(re.search(r'\bsize:(\d+)',e.get('args',{}).get('memcpy_details','')).group(1)) for e in selected)
    transfers[name]={'calls':len(selected),'sizes_bytes':dict(sizes),'total_bytes':sum(size*count for size,count in sizes.items()),'summed_microseconds':sum(e['dur'] for e in selected)}
result={'transfer_summary':transfers,'host_command_buffer_executions':sum(e.get('name')=='command_buffer::execute' and e.get('ph')=='X' and lo<=e['ts'] and e['ts']+e.get('dur',0)<=hi for e in events),'trace':str(path),'capture_annotation_microseconds':annotation['dur'],'hardware_streams':{str(k):v for k,v in threads.items() if 'GPU' in processes.get(k[0],'')},
 'hardware_stream_event_count':len(streams),'kernel_count':len(kernels),'nonkernel_count':len(nonkernels),'kernel_summed_microseconds':sum(durations.values()),'stream_interval_union_microseconds':busy,'first_to_last_stream_microseconds':extent,'union_fraction_within_first_last':busy/extent,'union_fraction_within_annotation':busy/annotation['dur'],
 'top_kernels_by_count':[{'name':name,'calls':count,'summed_microseconds':durations[name]} for name,count in counts.most_common(12)],
 'top_kernels_by_summed_duration':[{'name':name,'calls':counts[name],'summed_microseconds':duration} for name,duration in durations.most_common(12)],
 'memcpy32_post':{'calls':len(copy),'summed_microseconds':sum(e['dur'] for e in copy),'example_args':copy[0]['args'] if copy else None},
 'nonkernel_events':dict(Counter(e['name'] for e in nonkernels)),
 'notes':['Only GPU hardware Stream threads inside the named captured call are included; enclosing XLA annotations are excluded.','Summed kernel durations can overlap; interval-union busy time is not SM utilization.','Profiler overhead changes timings; use the preceding unprofiled diagnostic for performance comparisons.','Kernel names and counts do not by themselves attribute copies to retained snapshots rather than internal control flow.']}
(root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
