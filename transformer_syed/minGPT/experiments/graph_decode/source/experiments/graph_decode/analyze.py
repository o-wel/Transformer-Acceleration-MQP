"""Clustered paired analysis; discovery is never pooled with confirmation."""
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from collections import Counter
import numpy as np

HERE=Path(__file__).resolve().parent
MODES=['original','previous_fusion','eager_fused','graph_plain','graph_fused','merged_graph']
LABELS={'original':'Original LoRA + cache + FlashAttention',
    'previous_fusion':'Previous joint fusion', 'eager_fused':'Same runtime without graph replay',
    'graph_plain':'Graph replay, ordinary LoRA expansion', 'graph_fused':'Graph replay + joint fusion',
    'merged_graph':'Merged LoRA + graph replay'}
CONTRASTS=[('beyond_previous_fusion','previous_fusion','graph_fused'),
    ('versus_original','original','graph_fused'),('graph_replay_isolated','eager_fused','graph_fused'),
    ('fusion_after_graphing','graph_plain','graph_fused'),('versus_merged_graph','merged_graph','graph_fused'),
    ('merged_graph_versus_original','original','merged_graph')]


def main():
    runs=[json.loads(p.read_text()) for p in sorted((HERE/'results').glob('confirmation-*.json'))]
    assert len(runs)==3 and all(r['complete'] for r in runs),'Wait for three completed allocations'
    assert sorted(r['replicate'] for r in runs)==[0,1,2]
    assert all(r['sources']==runs[0]['sources'] for r in runs)
    assert all(r['adapter_sha256']==runs[0]['adapter_sha256'] for r in runs)
    assert all(r['weights_before']==r['weights_after']==runs[0]['weights_before'] for r in runs)
    workloads=sorted({(s['batch'],s['prompt'],s['new_tokens']) for s in runs[0]['samples']})
    assert len(workloads)==8
    rng=np.random.default_rng(20261108);result=dict(contrasts={},workloads=[],setup={},checks={},
        sources=runs[0]['sources'],adapter_sha256=runs[0]['adapter_sha256'],
        allocations=[dict(job=r['job'],node=r['node'],gpu=r['gpu'],telemetry=r['gpu_telemetry'],
                          resident_allocated_mib=r['resident_allocated_mib']) for r in runs])
    uuids={r['gpu_telemetry'].splitlines()[1].split(',')[0] for r in runs};result['physical_gpus']=len(uuids)
    for label,num,den in CONTRASTS:
        result['contrasts'][label]={}
        for metric in ('wall_ms','event_ms'):
            z=np.zeros((3,8,10))
            for a,r in enumerate(runs):
                for w,work in enumerate(workloads):
                    for block in range(10):
                        subset={s['mode']:s for s in r['samples'] if
                            (s['batch'],s['prompt'],s['new_tokens'])==work and s['block']==block}
                        assert set(subset)==set(MODES)
                        assert subset[num]['prompt_sha256']==subset[den]['prompt_sha256']
                        z[a,w,block]=np.log(subset[num][metric]/subset[den][metric])
            boots=[]
            for _ in range(5000):
                alloc=rng.integers(0,3,3);blocks=rng.integers(0,10,(3,8,10))
                boot=z[alloc[:,None,None],np.arange(8)[None,:,None],blocks]
                boots.append(np.exp(boot.mean()))
            result['contrasts'][label][metric]=dict(ratio=float(np.exp(z.mean())),
                ci95=[float(v) for v in np.quantile(boots,[.025,.975])],
                per_workload=[float(v) for v in np.exp(z.mean(axis=(0,2)))])
    csvrows=[];amortization=[]
    for work in workloads:
        entries={}
        for name in MODES:
            vals=[s for r in runs for s in r['samples'] if
                (s['batch'],s['prompt'],s['new_tokens'])==work and s['mode']==name]
            row=dict(batch=work[0],prompt=work[1],new_tokens=work[2],mode=name,
                median_wall_ms=float(np.median([v['wall_ms'] for v in vals])),
                geometric_wall_ms=float(np.exp(np.mean(np.log([v['wall_ms'] for v in vals])))),
                median_event_ms=float(np.median([v['event_ms'] for v in vals])),
                median_incremental_peak_mib=float(np.median([v['incremental_peak_mib'] for v in vals])))
            entries[name]=row;csvrows.append(row)
        result['workloads'].append(dict(shape=work,modes=entries))
        for name in ('graph_fused','graph_plain','merged_graph'):
            setup=float(np.median([r['setup'][f'{work[0]}-{name}']['total_ms'] for r in runs]))
            wall=entries[name]['median_wall_ms'];prev=entries['previous_fusion']['median_wall_ms']
            eager=entries['eager_fused']['median_wall_ms']
            row=dict(batch=work[0],prompt=work[1],new_tokens=work[2],mode=name,observed_setup_ms=setup,
                per_request_ms_at_1=wall+setup,per_request_ms_at_10=wall+setup/10,
                per_request_ms_at_100=wall+setup/100,
                break_even_vs_previous=math.ceil(setup/(prev-wall)) if prev>wall else None,
                break_even_vs_eager_fused=math.ceil(setup/(eager-wall)) if eager>wall else None)
            amortization.append(row)
    result['amortization']=amortization
    for b in (1,4):
        result['setup'][str(b)]={}
        for name in ('eager_plain','graph_plain','eager_fused','graph_fused','merged_graph'):
            result['setup'][str(b)][name]={k:float(np.median([r['setup'][f'{b}-{name}'][k] for r in runs]))
                for k in ('total_ms','warmup_capture_ms','incremental_resident_mib')}
    for name in ('eager_plain','graph_plain','eager_fused','graph_fused','merged_graph'):
        vals=[x for r in runs for x in r['checks'] if x['mode']==name]
        result['checks'][name]=dict(cases=len(vals),max_error=max(v['max_error'] for v in vals),
            argmax_disagreements=sum(v['argmax_disagreements'] for v in vals))
    result['raw_sequence_agreement']={}
    for name in MODES:
        equal=total=0
        for r in runs:
            for work in workloads:
                for block in range(10):
                    by={s['mode']:s for s in r['samples'] if
                        (s['batch'],s['prompt'],s['new_tokens'])==work and s['block']==block}
                    a,b=by[name]['output_ids'],by['original']['output_ids']
                    equal+=sum(x==y for x,y in zip(a,b));total+=len(a)
        result['raw_sequence_agreement'][name]=dict(equal=equal,total=total)
    quality_files=list((HERE/'results').glob('quality-[0-9]*.json'))
    assert len(quality_files)==1,'Need the supplementary quality run'
    quality=json.loads(quality_files[0].read_text());assert quality['complete'],'Wait for quality completion'
    assert quality['weights_before']==quality['weights_after']==runs[0]['weights_before'][0]
    assert quality['adapter_sha256']==runs[0]['adapter_sha256']
    for p in ('experiments/graph_decode/runtime.py','experiments/graph_decode/CONFIRMATION_PROTOCOL.md',
              'experiments/joint_decode/kernel.py','mingpt/model.py'):
        assert quality['sources'][p]==runs[0]['sources'][p]
    result['quality']=quality['summary']
    result['profiler']={}
    for path in (HERE/'results').glob('profile-*.json'):
        events=json.loads(path.read_text())['traceEvents']
        intervals=sorted((e['ts'],e['ts']+e['dur']) for e in events
                         if e.get('cat')=='kernel' and 'dur' in e)
        end=-1;busy=0
        for a,b in intervals:
            busy+=max(0,b-max(a,end));end=max(end,b)
        span=max(b for a,b in intervals)-min(a for a,b in intervals)
        calls=Counter(e.get('name') for e in events if e.get('cat')=='cuda_runtime')
        result['profiler'][path.name]=dict(kernel_count=len(intervals),kernel_span_ms=span/1000,
            kernel_busy_ms=busy/1000,instrumented_busy_fraction=busy/span,cuda_runtime_counts=dict(calls))
    (HERE/'PROFILER_EVIDENCE.json').write_text(json.dumps(result['profiler'],indent=2))
    jobs=[r['job'] for r in runs]+[quality['job']]
    jobs += [json.loads(p.read_text())['job'] for p in (HERE/'results').glob('pilot-*.json')]
    result['accounting']=subprocess.check_output(['sacct','-j',','.join(jobs),'-X','--format=JobID,State,ElapsedRaw,AllocTRES','-n','-P'],text=True)
    (HERE/'ACCOUNTING.txt').write_text(result['accounting'])
    gpu_seconds=sum(int(line.split('|')[2]) for line in result['accounting'].splitlines() if line.strip())
    result['gpu_allocation_seconds']=gpu_seconds
    (HERE/'SUMMARY.json').write_text(json.dumps(result,indent=2))
    for path,rows in [(HERE/'TIMINGS.csv',csvrows),(HERE/'AMORTIZATION.csv',amortization)]:
        with path.open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    primary=result['contrasts']['beyond_previous_fusion']['wall_ms']
    headline=result['contrasts']['versus_original']['wall_ms']
    text=['# Larger minGPT speedup: graph replay + joint LoRA/KV/attention decoding','',
        f"The combined runtime is **{headline['ratio']:.3f}x faster than the original all-three path** and "
        f"**{primary['ratio']:.3f}x faster than the previous fusion candidate**, on the declared held-out workloads. "
        'The main additional gain comes from established CUDA graph replay; publication novelty is not established.','',
        'Three fresh A30 allocations, eight held-out shapes, ten paired measured blocks per shape; '
        'three warmup blocks excluded. Ratio = comparison latency / candidate latency. '
        'Hierarchical 95% intervals resample allocations and paired blocks.','',
        '| Comparison | Wall speedup | 95% interval | CUDA-event ratio |','|---|---:|---:|---:|']
    for label,_,_ in CONTRASTS:
        x=result['contrasts'][label];v=x['wall_ms'];lo,hi=v['ci95']
        text.append(f"| {label} | {v['ratio']:.4f}x | {lo:.4f}–{hi:.4f}x | {x['event_ms']['ratio']:.4f}x |")
    text+=['',f"The declared larger-gain criterion (primary interval wholly above 2x) "
        f"{'passed' if primary['ci95'][0]>2 else 'did not pass'}.",'',
        '## Every held-out workload','',
        '| Batch / prompt / output | Original ms | Previous fusion ms | Eager fixed-buffer ms | Graph ordinary LoRA ms | Graph joint fusion ms | Merged graph ms |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for w in result['workloads']:
        text.append('| '+' / '.join(map(str,w['shape']))+' | '+' | '.join(f"{w['modes'][n]['median_wall_ms']:.2f}" for n in MODES)+' |')
    text+=['','## Setup and memory','',
        'Request times above include prefill, cache reset/copy, decode and output accumulation. '
        'They exclude graph construction and model load. Observed setup includes buffer allocation, '
        'warmup and capture with an already populated Triton disk cache; it is not clean-machine cold start.','',
        '| Batch | Joint graph setup ms | Joint runtime incremental resident MiB | Total comparison resident MiB |',
        '|---|---:|---:|---:|']
    for b in (1,4):
        s=result['setup'][str(b)]['graph_fused'];total=np.median([r['resident_allocated_mib'] for r in runs])
        text.append(f"| {b} | {s['total_ms']:.2f} | {s['incremental_resident_mib']:.2f} | {total:.2f} |")
    text+=['','The total comparison footprint includes both model variants and every control runner. '
        'It is not the footprint of deploying only one runner. Incremental request peaks are in TIMINGS.csv; '
        'persistent cache and graph allocations are not zero-cost.','',
        '| Batch / prompt / output | Joint graph: first request including setup ms | At 10 requests ms/request | At 100 requests ms/request | Break-even vs previous fusion |',
        '|---|---:|---:|---:|---:|']
    for x in amortization:
        if x['mode']=='graph_fused':
            text.append(f"| {x['batch']} / {x['prompt']} / {x['new_tokens']} | {x['per_request_ms_at_1']:.2f} | "
                f"{x['per_request_ms_at_10']:.2f} | {x['per_request_ms_at_100']:.2f} | {x['break_even_vs_previous']} |")
    text+=['','## Numerical and fresh-task checks','',
        'All teacher-forced, first/last-position and stale-cache checks passed. '
        'Eager and graph logits for the same kernel were bitwise equal in the declared checks. '
        'Weights were unchanged. Unsupported batch/context shapes were rejected.','',
        '| Runtime | Teacher-forced cases | Max error vs original Flash | Argmax disagreements |',
        '|---|---:|---:|---:|']
    for n,x in result['checks'].items():text.append(f"| {n} | {x['cases']} | {x['max_error']:.6g} | {x['argmax_disagreements']} |")
    text+=['','The four-token instrumented traces retain approximately the same GPU kernel count. '
        'Eager fixed-buffer execution issued 1,517 CUDA kernel launch API calls; graph execution issued '
        '371 plus three graph launches. Those remaining individual launches include prefill. '
        'This supports launch consolidation rather than reduced model depth. Profiler timing itself '
        'perturbs execution and is not used in request-latency estimates.']
    text+=['','Fresh supplementary data: 360 questions, 120 independent generated matches. '
        'Familiar names/templates with new score quadruples; one trained adapter.','',
        '| Runtime | Exact reference accuracy | Field accuracy | Answer-token NLL |',
        '|---|---:|---:|---:|']
    for n in ('original','previous_fusion','graph_fused'):
        q=result['quality'][n]
        text.append(f"| {n} | {q['reference_exact']:.2%} | {q['fields_correct']:.2%} | {q['nll']:.6f} |")
    for n,x in result['quality']['comparisons'].items():
        text+=['',f"{n}: decoded answer agreement {x['exact_answer_agreement']}/360; raw sequence agreement "
            f"{x['exact_raw_sequence_agreement']}/360; absolute NLL difference {x['absolute_nll_difference']:.6g}; "
            f"<=0.01 gate passed: {x['nll_gate_passed']}. Accuracy difference {x['accuracy_difference']:+.6f}, "
            f"match-cluster 95% interval {x['accuracy_difference_cluster_ci95']}."]
        if 'zero_discordance_probability_upper95' in x:
            text+=['',f"For {n}, no match showed decoded-answer discordance. The one-sided 95% upper bound on "
                f"match discordance probability is {x['zero_discordance_probability_upper95']:.2%}. "
                'A zero-width accuracy bootstrap is not proof of population equality.']
    text+=['','## Interpretation and limits','',
        'Graph replay accounts for the large improvement; the graph_plain/graph_fused contrast separately '
        'quantifies the smaller LoRA-fusion increment after replay. Merged graph serving is a strong control '
        'and should be preferred when its deployment tradeoffs fit. Its task-level quality was not assessed '
        'in the fresh three-mode task check, though its numerical gates passed.','',
        f"Three timing allocations used {len(uuids)} physical A30 GPUs; their intervals do not establish a GPU-family "
        'population effect. Native FP16, one model/rank, no fixed clocks, no concurrent serving or adapter changes. '
        'Official PyTorch FlashAttention is used for prefill; custom online-softmax decoding is not official FA1/FA2. '
        'Graph execution and LoRA graph serving are prior art. This is a measured improvement in this repository, '
        'not a state-of-the-art or established-novelty claim.','',
        f"Total discovery/confirmation/quality allocation: {gpu_seconds/60:.2f} GPU-minutes across five jobs. "
        'Model loading, warmups, graph capture and validation are included in allocation time.','',
        '![Speedup comparisons](speedups.svg)','',
        'See RESEARCH.md, CONFIRMATION_PROTOCOL.md, SUMMARY.json, TIMINGS.csv, AMORTIZATION.csv '
        'and results/ for the complete design, source hashes, raw timings, outputs and traces.']
    (HERE/'RESULTS.md').write_text('\n'.join(text)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4.8))
    labels=['Beyond previous joint fusion (primary)','Versus original all-three runtime',
        'Graph replay on the same runtime','LoRA fusion after graph replay','Versus merged graph runtime',
        'Merged graph versus original runtime']
    for i,(key,_,_) in enumerate(CONTRASTS):
        v=result['contrasts'][key]['wall_ms'];lo,hi=v['ci95'];ratio=v['ratio']
        ax.errorbar(ratio,5-i,xerr=[[ratio-lo],[hi-ratio]],fmt='o',capsize=4,
            color='#156082' if ratio>1 else '#b24c24')
    ax.set_yticks(range(6),labels[::-1]);ax.axvline(1,color='gray',linestyle='--')
    ax.set_xscale('log');ax.set_xlabel('Latency ratio (higher is faster)')
    ax.set_title('GPT-2 / rank-4 LoRA / A30: larger decode improvement\nThree allocations, eight held-out shapes; paired 95% intervals')
    ax.grid(axis='x',alpha=.2);fig.tight_layout()
    fig.savefig(HERE/'speedups.svg',bbox_inches='tight');fig.savefig(HERE/'speedups.png',dpi=180,bbox_inches='tight');plt.close(fig)
    print(json.dumps(result['contrasts'],indent=2))


if __name__=='__main__':main()
