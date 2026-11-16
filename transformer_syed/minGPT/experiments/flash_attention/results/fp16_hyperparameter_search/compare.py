"""Compare matched precision results; separate fixed configurations from retuned winners."""
import argparse
import json
from pathlib import Path
import statistics
import csv


def trials(path):
    return [r for p in (path/'trials').glob('*/result.json') if (r:=json.loads(p.read_text())).get('complete')]


def key(r):
    return json.dumps([r['config'],r['examples'],r['seed'],r['gpu']],sort_keys=True)


def build(root):
    text=['# FP16 versus FP32 matched performance review','',
        'The contribution is the integration of LoRA, an FA1-derived attention prototype and KV caching, with explicit measurement of hyperparameter sensitivity. Improvements are empirical outcomes, not assumptions.','',
        'FP16 is mixed precision: FP16 projection/attention operands and KV caches; FP32 master weights, optimizer state and stable reductions. A completed GPU preflight is required before each sweep.','',
        'Fixed-configuration comparisons below match exact config, example budget, seed and GPU model. Independently retuned finalists can differ and are reported separately. Repeated timings are descriptive; they are not independent allocations.','']
    rows=[]
    all_complete=True
    for task in ['basketball','spam']:
        half=root/f'{task}_fp16'
        full=root/('fp32_basketball_reference' if task=='basketball' else 'spam_fp32')
        a=trials(half);b=trials(full)
        lookup={key(r):r for r in b}
        paired=[]
        for r in a:
            if key(r) not in lookup:continue
            ref=lookup[key(r)]
            pair=dict(task=task,config=r['config'],examples=r['examples'],seed=r['seed'],gpu=r['gpu'],fp16_nll=r['validation_nll'],fp32_nll=ref['validation_nll'],nll_difference=r['validation_nll']-ref['validation_nll'],fp32_over_fp16_step_latency=ref['median_step_ms']/r['median_step_ms'],fp32_over_fp16_peak_memory=ref['peak_training_bytes']/r['peak_training_bytes'])
            paired.append(pair);rows.append(pair)
        status_path=half/'REPORT_STATUS.json'
        status=json.loads(status_path.read_text()) if status_path.exists() else {}
        complete=status.get('complete',False)
        if task=='spam':
            control=full/'REPORT_STATUS.json'
            complete &= control.exists() and json.loads(control.read_text()).get('complete',False)
        all_complete &= complete
        text += [f'## {task}','',f"Status: {'complete' if complete else 'running / partial'}. Completed FP16 trials: {len(a)}; matched precision pairs: {len(paired)}.",'']
        if paired:
            for budget in [1200,9600]:
                group=[r for r in paired if r['examples']==budget and r['seed']==42 and r['config']['method']=='lora']
                if group:text += [f"{budget}-example LoRA comparison ({len(group)} matched configs): median FP32/FP16 update-latency ratio {statistics.median(r['fp32_over_fp16_step_latency'] for r in group):.3f}; median validation-NLL change (FP16−FP32) {statistics.median(r['nll_difference'] for r in group):.6g}.",'']
        for precision,directory in [('FP16',half),('FP32',full)]:
            selection=directory/'selection.json'
            if selection.exists():
                s=json.loads(selection.read_text())
                text += [f'{precision} independently tuned winner: `{json.dumps(s["winner"],sort_keys=True)}`.','']
        text += [f'Per-task best/worst quality and joint-system sensitivity, dtype audits and held-out quality: [{task} FP16 report]({task}_fp16/PERFORMANCE_REVIEW.md).','']
    text.insert(2,'Status: '+('COMPLETE.' if all_complete else 'PARTIAL — experiments are still running; do not cite as final findings.'))
    text += ['## Scope','',
        'Basketball reuses the completed FP32 study as requested. SMS receives a new matched FP32 control. Both tasks retain their existing data splits; this does not establish fresh unseen-data generalization after earlier studies. Training compute excludes selection/evaluation overhead; inference does not amortize tuning cost. Compare GPU model and software metadata before attributing differences solely to precision.','',
        'Best-versus-worst ratios are conditional on a finite grid. Quality-rejected configurations have diagnostic timings but cannot substantiate quality-preserving speedups. LoRA tuning and serving-tile/cache tuning have separate costs. See task reports for direct combined measurements; isolated gains are never multiplied.']
    (root/'PERFORMANCE_REVIEW.md').write_text('\n'.join(text)+'\n')
    (root/'PRECISION_PAIRS.json').write_text(json.dumps(rows,indent=2))
    if rows:
        flat=[dict(r,config=json.dumps(r['config'],sort_keys=True)) for r in rows]
        with (root/'PRECISION_PAIRS.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);build(p.parse_args().root.resolve())
