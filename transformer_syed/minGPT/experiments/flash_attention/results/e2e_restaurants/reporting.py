"""Measurement-driven reports: isolated LR sensitivity, tradeoffs and joint choices."""
from collections import defaultdict,Counter
import csv,hashlib,html,json,random,statistics as st
from pathlib import Path
import matrix_report as base
HERE=Path(__file__).resolve().parent

def collect(out,pattern):return [r for p in sorted(out.glob(pattern)) for r in json.loads(p.read_text())]
def dump(path,rows):base.csv_write(path,rows)
def pareto(rows):
    keys=['validation_nll','training_update_seconds','peak_training_bytes']
    return [r for r in rows if not any(all(q[k]<=r[k] for k in keys) and any(q[k]<r[k] for k in keys) for q in rows)]
def build(out):
    if (out/'plan.json').exists():base.build(out)
    else:(out/'REPORT_STATUS.json').write_text(json.dumps(dict(status='PARTIAL')))
    status=json.loads((out/'REPORT_STATUS.json').read_text())
    trials=[json.loads(p.read_text()) for p in (out/'trials').glob('*/result.json')]
    done=[r for r in trials if r.get('complete')]
    flat=[dict(id=hashlib.sha256(json.dumps(dict(config=r['config'],examples=r['examples'],seed=r['seed']),sort_keys=True).encode()).hexdigest()[:16],
        **r['config'],examples=r['examples'],seed=r['seed'],gpu=r['gpu'],sample_sha256=r['sample_sha256'],
        validation_nll=r['validation_nll'],median_step_ms=r['median_step_ms'],training_update_seconds=sum(r['step_ms'])/1000,
        trajectory_wall_seconds=r['elapsed_seconds'],peak_training_bytes=r['peak_training_bytes'],selected_step=r['selected_step']) for r in done]
    dump(out/'TRIALS.csv',flat)
    lines=['# E2E restaurant generation: '+out.name,'Status: '+status['status'],'',
        'Contribution: measured integration of LoRA, a local FA1-derived prototype and KV caching; no new algorithm is claimed.',
        'Best/poor means observed within a staged finite search. Equal example budgets do not imply equal updates.',
        'FP32 precedes FP16; FP16 starts only after successful completion of the FP32 report gate.',
        'FP16 means autocast with FP32 master parameters/gradients/AdamW state and stable reductions. TF32 is disabled in both policies.',
        'Search selects by validation token NLL. Semantic slot accuracy is a separate measured generation objective.',
        'Fact grading uses the pinned upstream E2E heuristic scorer; it allows paraphrases but is not a complete factuality judge.',
        'Semantic error includes missing, added, wrong and repeated slot realizations. No exact reference-string requirement.',
        'Human audits are exported but remain UNREVIEWED until independently annotated. Fluency is not established by slot accuracy.',
        'Full fine-tuning has fewer search candidates; method-only superiority is not established.',
        '', '## Training hyperparameter sensitivity']
    for budget in [1200,9600]:
        rr=[r for r in flat if r['method']=='lora' and r['seed']==42 and r['examples']==budget]
        if rr:
            best=min(rr,key=lambda r:r['validation_nll']);worst=max(rr,key=lambda r:r['validation_nll'])
            lines += [f"Budget {budget}: {len(rr)} observed candidates; best/worst validation NLL {best['validation_nll']:.6g}/{worst['validation_nll']:.6g}; ratio {worst['validation_nll']/best['validation_nll']:.3f}. This is NLL sensitivity, not an accuracy multiplier.",
                      'Best configuration: '+json.dumps({k:best[k] for k in ['rank','scale','lr','targets','dropout','weight_decay','batch','microbatch','length']}),
                      'Poor configuration: '+json.dumps({k:worst[k] for k in ['rank','scale','lr','targets','dropout','weight_decay','batch','microbatch','length']})]
    # Isolate LR while holding all other core-grid factors and sample stream fixed.
    groups=defaultdict(list)
    for r in flat:
        if r['method']=='lora' and r['examples']==1200 and r['seed']==42:
            key=(r['rank'],r['scale'],r['targets'],r['gpu'],r['sample_sha256'])
            groups[key].append(r)
    lr=[]
    for key,rr in sorted(groups.items()):
        if {r['lr'] for r in rr}!={1e-4,3e-4,1e-3}:continue
        b=min(rr,key=lambda r:r['validation_nll']);w=max(rr,key=lambda r:r['validation_nll'])
        lr.append(dict(rank=key[0],scale=key[1],targets=key[2],best_lr=b['lr'],worst_lr=w['lr'],
            best_nll=b['validation_nll'],worst_nll=w['validation_nll'],nll_difference=w['validation_nll']-b['validation_nll'],nll_ratio=w['validation_nll']/b['validation_nll']))
    dump(out/'LR_SENSITIVITY.csv',lr)
    lines += ['', '## Does AdamW learning rate matter?']
    if lr:lines += [f"{len(lr)} matched three-rate groups; median worst/best NLL ratio {st.median(r['nll_ratio'] for r in lr):.3f}; median absolute NLL spread {st.median(r['nll_difference'] for r in lr):.6g}.",
        'Best-rate frequencies: '+json.dumps(dict(Counter(str(r['best_lr']) for r in lr))),
        'These screening contrasts fix rank, scale, target projections, seed, GPU class, example budget and sample stream. One seed and checkpoint selection limit causal/generalization claims. AdamW itself is held fixed.']
    else:lines+=['Awaiting complete matched three-rate groups. No learning-rate conclusion yet.']
    frontier=[]
    for method in ['lora','full']:
        rr=[r for r in flat if r['method']==method and r['seed']==42 and r['examples']==9600]
        frontier+=pareto(rr)
    dump(out/'TRAINING_PARETO.csv',frontier)
    lines += ['', '## Different user priorities',
        'TRAINING_PARETO.csv separates validation NLL, total successful-update compute and peak allocated training memory; frontiers are computed within each adaptation method.',
        'Training compute excludes evaluation/checkpoint overhead; trajectory wall time is also retained. Median update latency alone is not total training time when batch changes update count.',
        'Serving latency/memory are separate from training cost. Inferior quality cannot be exchanged for speed without explicitly stating the quality requirement.']
    profiles=[]
    for method in ['lora','full']:
        rr=[r for r in flat if r['method']==method and r['seed']==42 and r['examples']==9600]
        if not rr:continue
        minimum=min(r['validation_nll'] for r in rr)
        for relative in [0.,.1,.25]:
            candidates=[r for r in rr if r['validation_nll']<=minimum*(1+relative)+1e-12]
            for objective in ['training_update_seconds','peak_training_bytes']:
                b=min(candidates,key=lambda r:r[objective]);profiles.append(dict(method=method,relative_nll_allowance=relative,objective=objective,**b))
    dump(out/'PRIORITY_PROFILES.csv',profiles)
    lines += ['PRIORITY_PROFILES.csv gives fastest-compute and lowest-memory candidates under 0%, 10%, or 25% relative validation-NLL allowances. These are illustrative NLL constraints, not established acceptable semantic-quality losses.']
    if (out/'plan.json').exists():
        lines += ['', '## Individual versus combined settings',
            'See CALIBRATION.csv, CONFIRMATION.csv, COMPARISONS.csv and the full factorial section below. Independent attention/cache choices and joint choices use the same checkpoint/workload, separate calibration/confirmation prompts, randomized case order and seven confirmation timings.',
            'Only strict parity-passing cells can substantiate output-preserving speed claims. Direct combined measurements are used; isolated speedups are never multiplied.']
        conf=collect(out,'confirmation-*.json');interactions=[]
        for r in conf:
            if r['state']['attention']!='joint_best':continue
            target=dict(lora=r['state']['lora'],attention='best',cache='best')
            independent=next((q for q in conf if q['seed']==r['seed'] and q['workload']==r['workload'] and q['state']==target),None)
            if independent:
                valid=r['status']=='ok' and independent['status']=='ok'
                interactions.append(dict(seed=r['seed'],lora=r['state']['lora'],**r['workload'],passing=valid,
                    independently_selected_ms=independent['wall_ms'],jointly_selected_ms=r['wall_ms'],
                    independent_over_joint=(independent['wall_ms']/r['wall_ms']) if valid else None,
                    independent_config=json.dumps(independent['config']),joint_config=json.dumps(r['config'])))
        dump(out/'JOINT_VS_INDIVIDUAL.csv',interactions)
        valid=[r for r in interactions if r['passing']]
        if valid:lines += [f"{len(valid)} passing paired seed/workload comparisons; median independent/joint latency ratio {st.median(r['independent_over_joint'] for r in valid):.3f}. This may show a gain, no change or a reversal; no win is assumed."]
        qs=collect(out,'quality-*.json');audit=[]
        for q in qs:
            if q['state']['attention']!='off' or q['state']['cache']!='off':continue
            for row in sorted(q.get('outputs',[]),key=lambda r:r['id'])[:20]:
                audit.append(dict(blind_id=hashlib.sha256((str(q['seed'])+json.dumps(q['state'])+row['id']).encode()).hexdigest()[:16],mr=row['mr'],text=row['text'],human_facts_correct='',human_fluent='',notes=''))
        random.Random(20261116).shuffle(audit);dump(out/'HUMAN_AUDIT_BLINDED.csv',audit)
        existing=(out/'PERFORMANCE_REVIEW.md').read_text()
        lines += ['', '## Full integration matrix',existing]
    text='\n'.join(lines)+'\n'
    (out/'PERFORMANCE_REVIEW.md').write_text(text)
    (out/'PERFORMANCE_REVIEW.html').write_text('<!doctype html><meta charset="utf-8"><title>E2E performance review</title><pre>'+html.escape(text)+'</pre>')
    base.pdf(out/'PERFORMANCE_REVIEW.pdf',lines)
