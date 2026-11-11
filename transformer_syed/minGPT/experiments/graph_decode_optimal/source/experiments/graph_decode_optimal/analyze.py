"""Paired confirmation analysis, preserving discovery/test separation."""
import csv
import json
import math
from pathlib import Path
import subprocess
import numpy as np

HERE=Path(__file__).resolve().parent
MODES=['original','previous_fusion','eager_fused','plain_graph','default_graph','selected_graph','merged_graph']
CONTRASTS=[('primary','original','selected_graph'),('beyond_previous_fusion','previous_fusion','selected_graph'),
    ('graph_replay','eager_fused','selected_graph'),('lora_fusion','plain_graph','selected_graph'),
    ('runtime_tuning','default_graph','selected_graph'),('versus_merged','merged_graph','selected_graph'),
    ('merged_versus_original','original','merged_graph')]


def main():
    paths=sorted((HERE/'results').glob('confirmation-*.json'));runs=[json.loads(p.read_text()) for p in paths]
    assert len(runs)==5 and all(r['complete'] for r in runs),'Need all five complete confirmation units'
    assert sorted(r['seed'] for r in runs)==[101,202,303,404,505]
    selection=json.loads((HERE/'selection.json').read_text())
    assert all(r['sources']==selection['sources'] for r in runs)
    assert all(r['selection_sha256']==runs[0]['selection_sha256'] for r in runs)
    assert all(r['weights_before']==r['weights_after'] for r in runs)
    assert all(r['quality']['data_sha256']==runs[0]['quality']['data_sha256'] for r in runs)
    assert all(all(c['nll_gate_passed'] for c in r['quality']['summary']['comparisons'].values()) for r in runs)
    shapes=sorted({(s['batch'],s['prompt'],s['new_tokens']) for s in runs[0]['samples']})
    assert len(shapes)==8
    rng=np.random.default_rng(20261110)
    result=dict(selected=selection['selected'],contrasts={},workloads=[],quality={},setup={},amortization=[],
        units=[dict(seed=r['seed'],job=r['job'],node=r['node'],gpu=r['gpu'],gpu_telemetry=r['gpu_telemetry'],
            checkpoint=r['checkpoint'],resident_allocated_mib=r['resident_allocated_mib']) for r in runs],
        selection_sha256=runs[0]['selection_sha256'],sources=runs[0]['sources'])
    result['physical_gpu_uuids']=sorted({r['gpu_telemetry'].splitlines()[1].split(',')[0] for r in runs})
    for label,num,den in CONTRASTS:
        result['contrasts'][label]={}
        for metric in ('wall_ms','event_ms'):
            z=np.empty((5,8,10))
            for a,r in enumerate(runs):
                for w,shape in enumerate(shapes):
                    for block in range(10):
                        subset={s['mode']:s for s in r['samples'] if (s['batch'],s['prompt'],s['new_tokens'])==shape and s['block']==block}
                        assert set(subset)==set(MODES) and subset[num]['prompt_sha256']==subset[den]['prompt_sha256']
                        z[a,w,block]=np.log(subset[num][metric]/subset[den][metric])
            boots=[]
            for _ in range(5000):
                units=rng.integers(0,5,5);blocks=rng.integers(0,10,(5,8,10))
                boots.append(np.exp(z[units[:,None,None],np.arange(8)[None,:,None],blocks].mean()))
            result['contrasts'][label][metric]=dict(ratio=float(np.exp(z.mean())),
                ci95=[float(v) for v in np.quantile(boots,[.025,.975])],
                by_shape=[float(v) for v in np.exp(z.mean(axis=(0,2)))],
                by_unit=[float(v) for v in np.exp(z.mean(axis=(1,2)))])
    csvrows=[]
    for shape in shapes:
        modes={}
        for name in MODES:
            vals=[s for r in runs for s in r['samples'] if (s['batch'],s['prompt'],s['new_tokens'])==shape and s['mode']==name]
            assert len(vals)==50
            row=dict(batch=shape[0],prompt=shape[1],new_tokens=shape[2],mode=name,
                median_ms=float(np.median([v['wall_ms'] for v in vals])),
                geometric_ms=float(np.exp(np.mean(np.log([v['wall_ms'] for v in vals])))),
                median_event_ms=float(np.median([v['event_ms'] for v in vals])),
                median_incremental_peak_mib=float(np.median([v['incremental_peak_mib'] for v in vals])))
            modes[name]=row;csvrows.append(row)
        result['workloads'].append(dict(shape=shape,modes=modes))
        setup=float(np.median([r['setup'][f'{shape[0]}-selected_graph']['total_ms'] for r in runs]))
        original=modes['original']['median_ms'];candidate=modes['selected_graph']['median_ms']
        result['amortization'].append(dict(batch=shape[0],prompt=shape[1],new_tokens=shape[2],setup_ms=setup,
            request_1_ms=candidate+setup,request_10_ms=candidate+setup/10,request_100_ms=candidate+setup/100,
            break_even_requests=math.ceil(setup/(original-candidate)) if original>candidate else None))
    for b in (1,4):
        result['setup'][str(b)]={n:{k:float(np.median([r['setup'][f'{b}-{n}'][k] for r in runs]))
            for k in ('total_ms','incremental_resident_mib')} for n in ('selected_graph','default_graph','merged_graph')}
    groups=sorted({r['match_id'] for r in runs[0]['quality']['outputs']})
    assert len(groups)==120
    quality_arrays={n:np.empty((5,120)) for n in ('original','selected_graph','merged_graph')}
    for a,r in enumerate(runs):
        for g,group in enumerate(groups):
            vals=[x for x in r['quality']['outputs'] if x['match_id']==group]
            assert len(vals)==3
            for n in quality_arrays:quality_arrays[n][a,g]=np.mean([x['methods'][n]['scores']['reference_exact'] for x in vals])
    seeds=rng.integers(0,5,(5000,5));matches=rng.integers(0,120,(5000,120))
    for n,array in quality_arrays.items():
        boots=np.array([array[s[:,None],m[None,:]].mean() for s,m in zip(seeds,matches)])
        summaries=[r['quality']['summary'][n] for r in runs]
        result['quality'][n]=dict(mean_exact_accuracy=float(array.mean()),ci95=[float(v) for v in np.quantile(boots,[.025,.975])],
            mean_field_accuracy=float(np.mean([q['fields_correct'] for q in summaries])),
            mean_nll=float(np.mean([q['nll'] for q in summaries])),
            by_seed=[dict(seed=r['seed'],**r['quality']['summary'][n]) for r in runs])
        if np.all(array==1):
            result['quality'][n]['all_observed_correct']=True
            result['quality'][n]['any_error_match_probability_upper95']=1-.05**(1/120)
    result['quality']['comparisons']={}
    for name in ('selected_graph','merged_graph'):
        differences=quality_arrays[name]-quality_arrays['original']
        boots=np.array([differences[s[:,None],m[None,:]].mean() for s,m in zip(seeds,matches)])
        pair=[r['quality']['summary']['comparisons'][name] for r in runs]
        result['quality']['comparisons'][name]=dict(mean_accuracy_difference=float(differences.mean()),
            difference_ci95=[float(v) for v in np.quantile(boots,[.025,.975])],
            exact_answer_agreement=sum(x['exact_answer_agreement'] for x in pair),
            raw_sequence_agreement=sum(x['raw_sequence_agreement'] for x in pair),cases=1800,
            max_seed_absolute_nll_difference=max(x['absolute_nll_difference'] for x in pair),
            all_nll_gates_passed=all(x['nll_gate_passed'] for x in pair))
        if result['quality']['comparisons'][name]['exact_answer_agreement']==1800:
            result['quality']['comparisons'][name]['any_discordance_match_probability_upper95']=1-.05**(1/120)
    result['checks']={}
    for name in ('eager_fused','plain_graph','default_graph','selected_graph','merged_graph'):
        vals=[c for r in runs for c in r['checks'] if c['mode']==name]
        result['checks'][name]=dict(cases=len(vals),max_error=max(c['max_error'] for c in vals),argmax_disagreements=sum(c['argmax_disagreements'] for c in vals))
    jobs=[r['job'] for r in runs]+[json.loads(p.read_text())['job'] for p in (HERE/'results').glob('search-*.json')]
    accounting=subprocess.check_output(['sacct','-j',','.join(jobs),'-X','--format=JobID,State,ElapsedRaw,AllocTRES','-n','-P'],text=True)
    result['accounting']=accounting;result['gpu_seconds']=sum(int(l.split('|')[2]) for l in accounting.splitlines() if l.strip())
    (HERE/'ACCOUNTING.txt').write_text(accounting)
    (HERE/'SUMMARY.json').write_text(json.dumps(result,indent=2))
    for path,rows in [(HERE/'TIMINGS.csv',csvrows),(HERE/'AMORTIZATION.csv',result['amortization'])]:
        with path.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Selected:',result['selected'])
    print('Primary:',result['contrasts']['primary']['wall_ms'])
    print('Quality:',{n:x['mean_exact_accuracy'] for n,x in result['quality'].items() if n!='comparisons'})


if __name__=='__main__':main()
