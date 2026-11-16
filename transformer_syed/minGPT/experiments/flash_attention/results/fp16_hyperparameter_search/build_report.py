"""Report measured tuning spread without presuming improvement or novelty of algorithms."""
import argparse
import csv
import html
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent


def build(out):
    trials=[json.loads(p.read_text()) for p in (out/'trials').glob('*/result.json')]
    done=[r for r in trials if r.get('complete')]
    plan=json.loads((out/'plan.json').read_text())
    cells=sum([json.loads(p.read_text()) for p in sorted(out.glob('systems-*.json'))],[])
    test={m:json.loads((out/f'{m}-test.json').read_text()) for m in ['lora','full'] if (out/f'{m}-test.json').exists()}
    complete=(all((out/f'{stage}.json').exists() for stage in ['screen-0','screen-1','screen-2','refine','regularize','batch','confirm','selection']) and len(cells)==1440 and all(len(test.get(m,[]))==3 for m in ['lora','full']))
    # A complete artifact set must also satisfy execution/audit requirements.
    required_dtype='torch.float16' if plan['dtype']=='fp16' else 'torch.float32'
    audit_ok=all(r.get('frozen_base_unchanged') and r.get('dtype_audit',{}).get('compute_dtype')==required_dtype
                 and r.get('dtype_audit',{}).get('projection_outputs')
                 and set(r['dtype_audit']['projection_outputs'].values())=={required_dtype} for r in done)
    screens=[r for r in done if r['examples']==1200 and r['seed']==42 and r['config']['method']=='lora']
    repeat_ok=all(r['status']=='oom' or len(r.get('samples',[]))==7 for r in cells)
    complete=bool(complete and len(screens)==90 and audit_ok and repeat_ok)
    smoke=out/'smoke.json'
    audited=json.loads(smoke.read_text()) if smoke.exists() else None
    lines=['# Integrated optimization and hyperparameter performance review','',
        f"Status: {'COMPLETE' if complete else 'PARTIAL — no final conclusion yet'}. Task: {plan['task']}; compute: {plan['dtype']}; TF32 disabled.",'',
        'The research contribution is the integration of LoRA adaptation, an FA1-derived attention prototype, and KV caching, and the measured sensitivity of their joint configuration. No new attention or adaptation algorithm is claimed. Large gains or severe degradation are hypotheses to test, not assumed outcomes.','',
        f'Completed training trials: {len(done)}; serving cells: {len(cells)}/1440.','',
        '## Precision verification','',
        ('GPU preflight passed. Actual attention operands/outputs, projection outputs, dynamic and static KV caches were asserted against the requested compute dtype. Full and LoRA optimizer updates completed with finite gradients.' if audited else 'GPU preflight has not yet passed; FP16 execution is not yet confirmed.'),'',
        'FP16 means CUDA FP16 autocast for linear/attention computation and FP16 KV storage. Master weights, gradients, AdamW state, stable loss/normalization reductions, and Triton accumulators remain FP32. This is mixed precision, not an all-FP16 arithmetic or storage claim. Training uses GradScaler, initial scale 128; a skipped/nonfinite update fails the run. Inference uses the same precision policy.','',
        '## Matched methodology','',
        'The basketball protocol reproduces the completed FP32 staged search: 90 core configurations (r=2/4/8/16/32, alpha/r=1/2/4, LR=1e-4/3e-4/1e-3, attention/all projections), 1,200 screening examples, best nine retrained for 9,600 examples, dropout×decay 3×3, batch/accumulation/length sweeps, top-three confirmation with seeds 42/43/44, and the same full fine-tuning controls. Checkpoint and configuration selection use validation NLL only. Fixed sample schedules, token-weighted gradient accumulation, clipping, checkpoint opportunities, warmups (2), measured repeats (7), and systems workload grid match FP32.','',
        'SMS uses the existing group-disjoint SMS spam split, 96-token message cap, ham/spam next-token supervision and fixed label-logit classifier. It uses the identical staged hyperparameter search and example budgets as basketball. Its matched FP32 companion is run because the earlier SMS experiment only searched two learning rates and cannot serve as a matched 90-configuration precision baseline. Spam F1, precision, recall, balanced accuracy and raw predictions accompany NLL. Task NLLs are not directly comparable across basketball and spam.','',
        'The custom FA1-derived algorithm and tile grid are shared across precisions; PyTorch FlashAttention is not substituted. Systems retain the original strict logits tolerance (atol=rtol=3e-4) and full greedy-token parity gate. FP16 rounding failures are recorded as parity_failed and excluded from quality-preserving speed claims; the acceptance gate is not relaxed to manufacture a win. Separate preflight checks compare FP16 kernels to an FP64 reference with atol=.012, rtol=.04, and cache logits with atol=.1, rtol=.02; these diagnostic tolerances do not change the serving acceptance gate.','',
        '## Best versus worst observed hyperparameters','',
        'Comparisons below hold task, precision, training budget, seed, method and GPU class fixed. Full-budget staged candidates are a selected subset; screening and full-budget results are kept separate. NLL spread describes validation sensitivity, not a held-out accuracy improvement. Only the frozen winner and full control receive test evaluation.','',
        '| Budget | Candidates | Best validation NLL | Worst validation NLL | Worst/best NLL | Worst−best NLL |','|---|---|---|---|---|---|']
    spreads=[]
    for budget in [1200,9600]:
        rr=[r for r in done if r['examples']==budget and r['seed']==42 and r['config']['method']=='lora']
        for gpu in sorted({r['gpu'] for r in rr}):
            group=[r for r in rr if r['gpu']==gpu]
            if not group:continue
            best=min(group,key=lambda r:r['validation_nll']);worst=max(group,key=lambda r:r['validation_nll'])
            b,w=best['validation_nll'],worst['validation_nll']
            lines.append(f'| {budget}, {gpu} | {len(group)} | {b:.6g} | {w:.6g} | {w/b:.3g} | {w-b:.6g} |')
            spreads.append(dict(examples=budget,gpu=gpu,best=best,worst=worst))
    lines+=['','Exact extreme configurations are retained in TUNING_SPREAD.json; full trial metrics in TRIALS.csv.','',
        '## Integration: joint configuration sensitivity','',
        'Within each fixed method, GPU, batch, prompt and output workload, compare the fastest and slowest passing measured systems configurations. Tile/cache interaction contrasts in INTERACTIONS.csv fix tile geometry. Direct joint measurements establish integration effects; isolated speedups are not multiplied. LoRA and full training costs are separate from inference costs.','',
        '| Method | Batch | Prompt | Output | GPU | Passing cells | Fastest ms | Slowest ms | Slowest/fastest |','|---|---|---|---|---|---|---|---|---|']
    systems_spread=[]
    keys=sorted({(r['method'],r['batch'],r['prompt_tokens'],r['new_tokens'],r.get('gpu','unknown')) for r in cells})
    for key in keys:
        group=[r for r in cells if (r['method'],r['batch'],r['prompt_tokens'],r['new_tokens'],r.get('gpu','unknown'))==key and r['status']=='ok']
        if not group:continue
        best=min(group,key=lambda r:r['wall_ms']);worst=max(group,key=lambda r:r['wall_ms'])
        ratio=worst['wall_ms']/best['wall_ms']
        lines.append('| '+' | '.join(map(str,key))+f" | {len(group)} | {best['wall_ms']:.3f} | {worst['wall_ms']:.3f} | {ratio:.3f} |")
        systems_spread.append(dict(workload=key,best=best,worst=worst,ratio=ratio))
    diagnostic=[r for r in cells if r['status']=='parity_failed' and len(r.get('samples',[]))==7]
    if diagnostic:
        lines += ['', '### Diagnostic timings with failed quality gates', '',
            f'These {len(diagnostic)} cells were timed with the same warmups/repeats but failed original FP32-equivalent parity criteria. The table shows one worst-latency representative per method/backend/cache. They describe cost only and cannot support a quality-preserving acceleration claim.', '',
            '| Method | Backend | Cache | Batch | Prompt | Output | Latency ms | Logits pass | Tokens equal |',
            '|---|---|---|---|---|---|---|---|---|']
        # Summarize one worst-latency failure per method/backend/cache; raw cells remain in JSON/CSV.
        summarized={}
        for cell in diagnostic:
            key=(cell['method'],cell['backend'],cell['cache_mode'])
            if key not in summarized or cell['wall_ms']>summarized[key]['wall_ms']:
                summarized[key]=cell

        for r in summarized.values():
            lines.append(f"| {r['method']} | {r['backend']} | {r['cache_mode']} | {r['batch']} | {r['prompt_tokens']} | {r['new_tokens']} | {r['wall_ms']:.3f} | {r['logits_pass']} | {r['tokens_equal']} |")
    failed=[r for r in cells if r['status']!='ok']
    lines += ['',f'Excluded serving cells: {len(failed)}. Failures and errors remain in systems-*.json.','',
        '## Held-out task quality','']
    for method,rows in test.items():
        lines += [f'### {method}','',f'Completed seeds: {len(rows)}/3.','', '| Metric | Mean | Seed SD |','|---|---|---|']
        for name in rows[0]['metrics']:
            values=[r['metrics'][name] for r in rows]
            lines.append(f'| {name} | {statistics.mean(values):.6g} | {statistics.stdev(values) if len(values)>1 else 0:.6g} |')
    lines+=['','## Interpretation and limits','',
        'The key message is the importance of tuning adaptation and serving hyperparameters together while tracking quality, latency and memory. The observed best/worst ratios quantify this only within the tested grid and workload. A small spread, a slowdown, or failure of parity gates must be reported equally. No universal optimum or guaranteed dramatic improvement is claimed.','',
        'This preserves the FP32 study’s limitations: staged selection, unequal full/LoRA search budgets, one model, fixed task splits, synthetic timing prompts, fixed case order, and repeated timing within one allocation per batch shard. Repetitions are not independent GPU replications; seed standard deviations are not dataset confidence intervals. Padded length changes computation rather than real task context. Hardware classes must match before precision ratios are claimed. FP16 master-weight storage and reductions are disclosed above.','',
        'Owen-style sensitivity plots are recreated from this study’s measured cells; the external Owen source graphs are not present in this checkout and have not been represented as reviewed evidence. Existing FP32 results are preserved at ../hyperparameter_search/run_fp32/.','']
    if spreads or systems_spread:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for sp in spreads:
            rows=[r for r in done if r['examples']==sp['examples'] and r['seed']==42 and r['config']['method']=='lora' and r['gpu']==sp['gpu']]
            values=sorted(r['validation_nll'] for r in rows)
            axes[0].plot(range(1,len(values)+1),values,label=f"{sp['examples']} examples")
        axes[0].set(xlabel='Configuration rank within budget',ylabel='Validation NLL',title='Hyperparameter quality sensitivity')
        if spreads:axes[0].legend()
        for method in ['lora','full']:
            values=sorted(r['ratio'] for r in systems_spread if r['workload'][0]==method)
            if values:axes[1].plot(range(1,len(values)+1),values,label=method)
        axes[1].set(xlabel='Workload rank',ylabel='Slowest / fastest passing latency',title='Joint attention/cache sensitivity')
        if systems_spread:axes[1].legend()
        fig.tight_layout();fig.savefig(out/'sensitivity.png',dpi=180);plt.close(fig)
        lines+=['![Measured quality and latency sensitivity](sensitivity.png)','']
    text='\n'.join(lines)
    (out/'PERFORMANCE_REVIEW.md').write_text(text)
    for name,value in [('TUNING_SPREAD.json',spreads),('SYSTEMS_SPREAD.json',systems_spread),('REPORT_STATUS.json',dict(complete=complete,completed_trials=len(done),systems_cells=len(cells),excluded_cells=len(failed),precision_verified=bool(audited),expected_compute=plan['dtype']))]:
        (out/name).write_text(json.dumps(value,indent=2))
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import landscape, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    styles=getSampleStyleSheet()
    story=[]
    for line in lines:
        if line.startswith('!['):
            story.append(Image(str(out/'sensitivity.png'),width=700,height=233))
        elif line.strip():
            style=styles['Heading1'] if line.startswith('# ') else styles['Heading2'] if line.startswith('## ') else styles['BodyText']
            story.append(Paragraph(html.escape(line.lstrip('# ')),style))
        else:story.append(Spacer(1,7))
    SimpleDocTemplate(str(out/'PERFORMANCE_REVIEW.pdf'),pagesize=landscape(A4)).build(story)
    # A standalone HTML report does not require external libraries or assets.
    import base64
    img=out/'sensitivity.png'
    visual='<img style="max-width:100%" src="data:image/png;base64,'+base64.b64encode(img.read_bytes()).decode()+'">' if img.exists() else ''
    (out/'PERFORMANCE_REVIEW.html').write_text('<!doctype html><meta charset="utf-8"><title>FP16 performance review</title><style>body{max-width:1100px;margin:3em auto;font:16px system-ui}pre{white-space:pre-wrap}</style>'+visual+'<pre>'+html.escape(text)+'</pre>')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    build(p.parse_args().out.resolve())
