"""Selection and reporting use declared objectives and independent replication units."""
import csv
import datetime
import hashlib
import html
import itertools
import json
import math
from pathlib import Path
import random
import statistics
from experiments.paper_hyperparameters.common import read,save,sha,verify
from experiments.paper_hyperparameters.protocol import (DESIGN,GRID,BASE,SEARCH_SEEDS,CONFIRM_SEEDS,SEARCH_WORKLOADS,WORKLOADS,CHUNKS,TILES,config_key,workload_key)


def gmean(values):return math.exp(statistics.mean(math.log(x) for x in values))


def choose_lora(rows,margin=.01):
    best=min(rows,key=lambda r:(r['mean_nll'],r['config']['rank'],r['config']['alpha'],r['config']['lr']))
    eligible=[r for r in rows if r['mean_nll']<=best['mean_nll']+margin]
    winner=min(eligible,key=lambda r:(r['mean_step_ms'],r['config']['rank'],r['config']['alpha'],r['config']['lr']))
    return best,winner


def select(study):
    verify(study)
    if (study/'selection.json').exists():return read(study/'selection.json')
    rows=[];inputs={};schedules={s:set() for s in SEARCH_SEEDS}
    for c in GRID:
        seeds=[]
        for seed in SEARCH_SEEDS:
            path=study/'search-training'/config_key(c)/f'seed-{seed}'/'training.json'
            r=read(path);assert r['complete'] and r['successful_updates']==DESIGN['train']['steps']
            assert r['frozen_base_before']==r['frozen_base_after']
            assert len(r['validation'])==DESIGN['train']['steps']//DESIGN['train']['validate_every']
            schedules[seed].add(r['schedule_sha256']);seeds.append(r);inputs[str(path.relative_to(study))]=sha(path)
        rows.append(dict(config=c,mean_nll=statistics.mean(r['validation_nll'] for r in seeds),
                         sd_nll=statistics.stdev(r['validation_nll'] for r in seeds),
                         mean_step_ms=statistics.mean(r['warm_median_step_ms'] for r in seeds),
                         seeds=[dict(seed=r['seed'],nll=r['validation_nll'],step_ms=r['warm_median_step_ms'],selected_step=r['selected_step']) for r in seeds]))
    assert all(len(v)==1 for v in schedules.values()),'Sampling order differs across candidates'
    best,winner=choose_lora(rows,DESIGN['quality']['lora_nll_margin'])
    runtime=[read(study/'runtime-search'/f'session-{i}.json') for i in range(3)]
    gates=[read(study/'runtime-search'/f'gates-{i}.json') for i in range(3)]
    for i,r in enumerate(runtime):
        assert r['complete'] and len(r['workloads'])==len(SEARCH_WORKLOADS)
        for stem in ('session','gates'):
            path=study/'runtime-search'/f'{stem}-{i}.json';inputs[str(path.relative_to(study))]=sha(path)
    scores={}
    for kind,configs in [('cache',CHUNKS),('fa',TILES)]:
        scores[kind]=[]
        for config in configs:
            name=f'cache-{config}' if kind=='cache' else 'fa-'+'-'.join(map(str,config))
            if not all(g[kind][name]['valid'] for g in gates):continue
            values=[statistics.median(s['wall_ms'] for s in r['workloads'][workload_key(w)]['samples'][name])
                    for r in runtime for w in SEARCH_WORKLOADS]
            scores[kind].append(dict(config=config,geometric_mean_ms=gmean(values)))
    chunk=min(scores['cache'],key=lambda r:r['geometric_mean_ms'])['config']
    tiles=min(scores['fa'],key=lambda r:r['geometric_mean_ms'])['config']
    pareto=[r for r in rows if not any(q['mean_nll']<=r['mean_nll'] and q['mean_step_ms']<=r['mean_step_ms'] and
                                       (q['mean_nll']<r['mean_nll'] or q['mean_step_ms']<r['mean_step_ms']) for q in rows)]
    selected=dict(lora=winner['config'],quality_best_lora=best['config'],chunk=chunk,tiles=list(tiles),
                  lora_nll_cutoff=best['mean_nll']+DESIGN['quality']['lora_nll_margin'],
                  lora_search=rows,runtime_search=scores,pareto_configs=[r['config'] for r in pareto],
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),input_sha256=inputs,
                  protocol_sha256=sha(study/'protocol.json'),frozen_before_confirmation_and_test=True)
    save(study/'selection.json',selected)
    (study/'selection.sha256').write_text(sha(study/'selection.json')+'\n')
    print('FROZEN SELECTION',json.dumps({k:selected[k] for k in ('lora','quality_best_lora','chunk','tiles')}),flush=True)
    return selected


def hierarchical_interval(cube,draws=5000,seed=20261105):
    """cube[training seed][allocation][fixed workload][paired repeat] = log ratio."""
    if not cube or any(not sessions for sessions in cube):raise ValueError('Missing independent units')
    n=len(cube);rng=random.Random(seed)
    point=math.exp(statistics.mean(statistics.mean(statistics.mean(statistics.mean(repeats) for repeats in workloads)
                                                for workloads in sessions) for sessions in cube))
    values=[]
    for _ in range(draws):
        seed_means=[]
        for s in rng.choices(cube,k=n):
            session_means=[]
            for workloads in rng.choices(s,k=len(s)):
                session_means.append(statistics.mean(statistics.mean(rng.choices(rep,k=len(rep))) for rep in workloads))
            seed_means.append(statistics.mean(session_means))
        values.append(math.exp(statistics.mean(seed_means)))
    values.sort();return dict(speedup=point,ci95=[values[int(.025*draws)],values[min(draws-1,int(.975*draws))]],
                              training_seeds=n,allocations_per_seed=len(cube[0]))


def comparison_cube(records,baseline,tuned,workloads):
    cube=[]
    for seed in CONFIRM_SEEDS:
        sessions=[]
        for session in range(3):
            r=records[(seed,session)];rows=[]
            for w in workloads:
                samples=r['workloads'][workload_key(w)]['samples']
                b=samples[baseline];t=samples[tuned]
                assert len(b)==len(t)==DESIGN['timing']['confirmation_repeats']
                assert all(x['repeat']==y['repeat'] and x['prompt_seed']==y['prompt_seed'] for x,y in zip(b,t))
                rows.append([math.log(x['wall_ms']/y['wall_ms']) for x,y in zip(b,t)])
            sessions.append(rows)
        cube.append(sessions)
    return cube


def quality_summary(quality,draws):
    groups=sorted({r['group_id'] for r in quality[0]['models']['baseline']['outputs']['combined']['answers']})
    assert len(groups)==300
    metrics={}
    for metric in ('reference_exact','fields_correct'):
        matrices={label:[] for label in ('baseline','tuned')}
        for q in quality:
            for label in matrices:
                rows=[r for r in q['models'][label]['outputs']['combined']['answers'] if '-counter-' not in r['match_id']]
                assert len(rows)==900
                matrices[label].append([statistics.mean(r['scores'][metric] for r in rows if r['group_id']==g) for g in groups])
        delta=[[t-b for b,t in zip(bb,tt)] for bb,tt in zip(matrices['baseline'],matrices['tuned'])]
        rng=random.Random(DESIGN['stats']['seed']);samples=[]
        for _ in range(draws):
            seeds=rng.choices(range(5),k=5);matches=rng.choices(range(len(groups)),k=len(groups))
            samples.append(statistics.mean(delta[s][m] for s in seeds for m in matches))
        samples.sort()
        metrics[metric]=dict(baseline=statistics.mean(map(statistics.mean,matrices['baseline'])),
                             tuned=statistics.mean(map(statistics.mean,matrices['tuned'])),
                             paired_difference=statistics.mean(map(statistics.mean,delta)),
                             difference_ci95=[samples[int(.025*draws)],samples[min(draws-1,int(.975*draws))]])
    return metrics


def report(study):
    verify(study);selected=read(study/'selection.json')
    records={(seed,i):read(study/'confirmation-timing'/f'seed-{seed}-session-{i}.json') for seed in CONFIRM_SEEDS for i in range(3)}
    quality=[read(study/'confirmation-quality'/f'seed-{seed}.json') for seed in CONFIRM_SEEDS]
    assert all(r['complete'] for r in [*records.values(),*quality])
    comparisons=dict(combined=('factorial-000','factorial-111'),lora=('lora_baseline','lora_tuned'),
                     cache=('cache_baseline','cache_tuned'),fa=('fa_baseline','fa_tuned'))
    stats={};draws=DESIGN['stats']['bootstrap_draws']
    for name,(b,t) in comparisons.items():
        print('BOOTSTRAP',name,flush=True)
        stats[name]={split:hierarchical_interval(comparison_cube(records,b,t,ws),draws)
                     for split,ws in [('heldout',WORKLOADS[len(SEARCH_WORKLOADS):]),('search',SEARCH_WORKLOADS),('all',WORKLOADS)]}
    per_workload={workload_key(w):hierarchical_interval(comparison_cube(records,'factorial-000','factorial-111',[w]),draws) for w in WORKLOADS}
    training=[]
    for seed in CONFIRM_SEEDS:
        pair=[read(study/'confirmation-training'/config_key(c)/f'seed-{seed}'/'training.json') for c in (BASE,selected['lora'])]
        assert all(r['complete'] for r in pair) and pair[0]['schedule_sha256']==pair[1]['schedule_sha256']
        training.append(dict(seed=seed,baseline_ms=pair[0]['warm_median_step_ms'],tuned_ms=pair[1]['warm_median_step_ms'],
                             baseline_nll=pair[0]['validation_nll'],tuned_nll=pair[1]['validation_nll']))
    train_cube=[[[[math.log(r['baseline_ms']/r['tuned_ms'])]] ] for r in training]
    train_stats=hierarchical_interval(train_cube,draws)
    qstats=quality_summary(quality,draws)
    passed=all(m['runtime_nll_gate_passed'] for q in quality for m in q['models'].values())
    result=dict(complete=True,runtime_quality_gates_passed=passed,selected=selected,comparisons=stats,
                combined_per_workload=per_workload,training=training,training_speedup=train_stats,
                quality=qstats,protocol_sha256=sha(study/'protocol.json'))
    save(study/'results.json',result)
    with (study/'TIMINGS.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['seed','allocation','workload','mode','repeat','order','prompt_seed','wall_ms','cuda_ms','peak_allocated_bytes','incremental_peak_bytes','output_sha256'])
        for (seed,session),r in records.items():
            for wk,w in r['workloads'].items():
                for mode,rows in w['samples'].items():
                    for x in rows:writer.writerow([seed,session,wk,mode,*[x[k] for k in ('repeat','order','prompt_seed','wall_ms','cuda_ms','peak_allocated_bytes','incremental_peak_bytes','output_sha256')]])
    with (study/'QUALITY_BY_SEED.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['seed','model','runtime','split','reference_exact','fields_correct','paired_fields_correct'])
        for q in quality:
            for name,m in q['models'].items():
                for variant,v in m['outputs'].items():
                    rows=v['answers']
                    for split,counter in [('original',False),('counterfactual',True)]:
                        subset=[r for r in rows if ('-counter-' in r['match_id'])==counter]
                        originals={(r['group_id'],r['kind']):r for r in rows if '-counter-' not in r['match_id']}
                        counters={(r['group_id'],r['kind']):r for r in rows if '-counter-' in r['match_id']}
                        paired=statistics.mean(originals[k]['scores']['fields_correct'] and counters[k]['scores']['fields_correct'] for k in originals)
                        writer.writerow([q['seed'],name,variant,split,statistics.mean(r['scores']['reference_exact'] for r in subset),
                                         statistics.mean(r['scores']['fields_correct'] for r in subset),paired])
    import subprocess
    if (study/'jobs.json').exists():
        ids=','.join(j['id'] for j in read(study/'jobs.json')['jobs'].values())
        try:
            accounting=subprocess.run(['sacct','-j',ids,'--parsable2','--format=JobID,JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocTRES'],capture_output=True,text=True,timeout=45)
            (study/'ACCOUNTING.csv').write_text(accounting.stdout)
        except (OSError,subprocess.TimeoutExpired) as e:
            (study/'ACCOUNTING_ERROR.txt').write_text(repr(e))
    compute={}
    for phase in ('search-training','confirmation-training'):
        runs=[read(p) for p in (study/phase).glob('*/seed-*/training.json')]
        compute[phase]=dict(runs=len(runs),successful_updates=sum(r['successful_updates'] for r in runs),
                            measured_update_seconds=sum(r['measured_update_seconds'] for r in runs),
                            training_elapsed_seconds=sum(r['training_elapsed_seconds'] for r in runs))
    save(study/'COMPUTE.json',compute)
    with (study/'FACTORIAL.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['mode','geometric_mean_latency_ms','speedup_vs_000'])
        latency={mode:gmean([x['wall_ms'] for r in records.values() for w in r['workloads'].values() for x in w['samples'][mode]])
                 for mode in (f'factorial-{l}{k}{a}' for l,k,a in itertools.product([0,1],repeat=3))}
        for name,value in latency.items():writer.writerow([name,value,latency['factorial-000']/value])
    # Fixed, explicit factorial contrasts in log latency: sum(sign*log(latency))/4.
    interactions={}
    for mask in [(0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2)]:
        contrast=sum(math.prod(1 if bits[i] else -1 for i in mask)*math.log(latency['factorial-'+''.join(map(str,bits))])
                     for bits in itertools.product([0,1],repeat=3))/4
        interactions[''.join('LCF'[i] for i in mask)]=dict(log_latency_contrast=contrast,exp_contrast=math.exp(contrast))
    save(study/'FACTORIAL_CONTRASTS.json',interactions)
    rows=['# Independent confirmation results','',
          f"Runtime quality gates passed: **{passed}**. Five new training seeds, three GPU allocations per seed, twenty paired repetitions per workload.",'',
          f"Selected LoRA: `{selected['lora']}`. Cache chunk: `{selected['chunk']}`. Prototype tiles/warps: `{selected['tiles']}`.",'',
          'Primary endpoint: combined tuned/untuned comparison on eight held-out workload shapes. Ratios are untuned latency / tuned latency; values above one indicate speedup. All three methods are enabled on both sides.','',
          '| Comparison | Held-out speedup | Hierarchical 95% CI |','|---|---:|---:|']
    for name,r in stats.items():
        x=r['heldout'];rows.append(f"| {name} | {x['speedup']:.4f}× | {x['ci95'][0]:.4f}–{x['ci95'][1]:.4f}× |")
    rows+=['',f"LoRA training-step speedup: {train_stats['speedup']:.4f}× (95% CI {train_stats['ci95'][0]:.4f}–{train_stats['ci95'][1]:.4f}×).",'',
           '| Combined test quality | Untuned | Tuned | Paired difference 95% CI |','|---|---:|---:|---:|']
    for name,q in qstats.items():rows.append(f"| {name} | {q['baseline']:.4f} | {q['tuned']:.4f} | {q['difference_ci95'][0]:.4f}–{q['difference_ci95'][1]:.4f} |")
    rows+=['','All secondary intervals are descriptive. The primary combined interval is the prespecified confirmatory endpoint.','',
           'Claims are limited to this GPT-2 implementation, the A30, synthetic basketball task, declared grid and workload family. FA results describe an FA1-derived prototype. No production backend is included.','',
           'See METHODS.md in the frozen snapshot for the design, formulas, precision policy, exclusions and reference papers. All raw answers and per-allocation records are retained.']
    (study/'RESULTS.md').write_text('\n'.join(rows)+'\n')
    body='<!doctype html><meta charset="utf-8"><title>Paper experiment results</title><style>body{max-width:1100px;margin:40px auto;font:16px/1.5 system-ui}pre{white-space:pre-wrap}</style><h1>Paper experiment results</h1><pre>'+html.escape('\n'.join(rows))+'</pre>'
    (study/'RESULTS.html').write_text(body)
    save(study/'combined-configs.json',dict(untuned=dict(lora=BASE,chunk=128,tiles=[16,32,4]),tuned={k:selected[k] for k in ('lora','chunk','tiles')},precision=DESIGN['precision'],attention=DESIGN['implementation']))
    artifact_hashes={str(p.relative_to(study)):sha(p) for p in study.rglob('*.json') if p.name!='ARTIFACT_SHA256.json'}
    save(study/'ARTIFACT_SHA256.json',artifact_hashes)
    print('REPORT COMPLETE',study,flush=True)
