"""Three-way quality comparison; full fine-tuning versus LoRA efficiency."""
import base64
import csv
import html
import json
import statistics as stats
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from experiments.sports.build_report import page

OUT=ROOT/'experiments/sports/matched/results'


def main():
    selection=json.loads((OUT.parent/'selection.json').read_text())
    assert selection.get('complete'), 'Runs have not all completed'
    runs={m:[json.loads((ROOT/info['runs'][str(s)]/'report.json').read_text()) for s in [42,43,44]] for m,info in selection['methods'].items()}
    for l,f in zip(runs['lora'],runs['full']):
        for key in ['initial_base_sha256','sample_order_sha256','data_sha256','processed_training_tokens','supervised_training_tokens','gpu','torch_version','source_sha256']:
            assert l[key]==f[key],key
        assert l['selected_changes']['frozen_changed_elements']==0
        assert all(x['frozen_changed_elements']==0 for x in l['history'])
        assert l['config']['steps']==f['config']['steps']==1200
        for r in [l,f]:
            assert r['selected_validation_nll']==min(x['nll'] for x in r['validation_history'])
            assert r['batch_generation_checked']
    OUT.mkdir(exist_ok=True)
    l,f=runs['lora'][0],runs['full'][0]
    vals=lambda m,k:[r[m]['metrics'][k]*100 for r in runs[m]]
    avg=lambda m,k:stats.mean(r[k] for r in runs[m])
    fmt=lambda x:f'{stats.mean(x):.1f} ± {stats.stdev(x):.1f}'
    base=l['base']['metrics']
    mem={m:avg(m,'training_peak_allocated_bytes')/2**20 for m in runs}
    times={m:avg(m,'training_step_seconds') for m in runs}
    paragraphs=[
        'Correct comparison: pretrained GPT-2 (reference), full fine-tuning on the new data, and LoRA fine-tuning on the same data. Full fine-tuning is now trained and evaluated for the same budget as LoRA; the previous 40-step full-training run was a memory-only baseline.',
        f"Controls: same GPT-2 starting weights, training examples and sample order, FP32 with TF32 off, batch 8, padded length 128, 1,200 steps, seeds 42/43/44, and {l['gpu']}. Starting-weight and batch-order SHA256 fingerprints match between methods for every seed. Each run processes {l['processed_training_tokens']:,} padded token positions; supervised answer-token budgets also match by seed.",
        f"Equal tuning budget: both methods tried learning rates 0.001 and 0.0001 for 1,200 steps on seed 42. Validation NLL at steps 300/600/900/1200 selected the rate and checkpoint; both rate choices were saved before new test evaluation. Selected rates: full {selection['methods']['full']['selected_lr']:g}, LoRA {selection['methods']['lora']['selected_lr']:g}. Seeds 43/44 then used the chosen rate. Four training runs per method including the unselected candidate; six selected runs in the final comparison.",
        'Data: 480 train matches, 40 validation matches, 40 test matches, three questions per match, plus 120 score-swapped test questions. Test team names and score combinations are absent from training. This is the existing disclosed synthetic benchmark, not a newly blinded test set; no new test-based tuning was used.',
        f"Primary-answer score: pretrained {base['primary_correct']:.1%}; full {fmt(vals('full','primary_correct'))}%; LoRA {fmt(vals('lora','primary_correct'))}%. Answer + score fields: pretrained {base['fields_correct']:.1%}; full {fmt(vals('full','fields_correct'))}%; LoRA {fmt(vals('lora','fields_correct'))}%. ± is sample SD across three training seeds, not a confidence interval.",
        f"Trainable scalar parameters: full {f['parameters']['trainable']:,} (100%); LoRA {l['parameters']['trainable']:,} ({l['parameters']['trainable_percent']:.4f}% of the LoRA model). LoRA total {l['parameters']['total']:,}; full total {f['parameters']['total']:,}. LoRA updates {f['parameters']['trainable']/l['parameters']['trainable']:.1f} times fewer trainable parameters. All LoRA base-weight audits passed.",
        f"Mean full-budget peak allocated GPU memory: full {mem['full']:.1f} MiB; LoRA {mem['lora']:.1f} MiB ({1-mem['lora']/mem['full']:.1%} lower). Mean synchronized training-step time: full {times['full']:.1f} s; LoRA {times['lora']:.1f} s. Timing excludes transfer, validation, audits, checkpoint I/O and test generation. End-to-end training wall times are reported separately in JSON.",
        'Limits: metrics are transparent but format-sensitive field checks, not broad conversational quality. Three questions share each match. Generated templates can be reused, so score-swap consistency and failures matter. Both methods are trained for the full budget but evaluated at their validation-selected checkpoint. This does not establish quality equivalence or time-to-quality without further experiments.'
    ]
    table=['| Metric | Pretrained reference | Full fine-tuning | LoRA fine-tuning |','|---|---:|---:|---:|']
    for key,label in [('primary_correct','Primary answer (%)'),('fields_correct','Answer + scores (%)'),('reference_exact','Exact reference (%)')]:
        table.append(f'| {label} | {base[key]*100:.1f} | {fmt(vals("full",key))} | {fmt(vals("lora",key))} |')
    for kind in ['winner','halftime','comeback']:
        scores={m:[r[m]['metrics']['by_kind'][kind]['primary_correct']*100 for r in runs[m]] for m in runs}
        table.append(f"| {kind} primary (%) | {l['base']['metrics']['by_kind'][kind]['primary_correct']*100:.1f} | {fmt(scores['full'])} | {fmt(scores['lora'])} |")
    for m in runs:
        for r in runs[m]:
            assert r['reference_answers_seen_in_training']==0
    paired={m:[r[m]['paired_primary_accuracy']*100 for r in runs[m]] for m in runs}
    table.append(f"| Both score-swap answers correct (%) | {l['base']['paired_primary_accuracy']*100:.1f} | {fmt(paired['full'])} | {fmt(paired['lora'])} |")
    table += [f"| Total stored parameters | {f['parameters']['total']:,} | {f['parameters']['total']:,} | {l['parameters']['total']:,} |",
              f"| Trainable parameters during adaptation | 0 | {f['parameters']['trainable']:,} | {l['parameters']['trainable']:,} |",
              f"| Actually changed scalars (selected seed 42) | 0 | {f['selected_changes']['changed_elements']:,} | {l['selected_changes']['changed_elements']:,} |",
              f"| Parameter storage (MiB) | {f['parameters']['parameter_bytes']/2**20:.2f} | {f['parameters']['parameter_bytes']/2**20:.2f} | {l['parameters']['parameter_bytes']/2**20:.2f} |",
              f"| Mean peak allocated training memory (MiB) | N/A | {mem['full']:.1f} | {mem['lora']:.1f} |",
              f"| Mean peak reserved training memory (MiB) | N/A | {avg('full','training_peak_reserved_bytes')/2**20:.1f} | {avg('lora','training_peak_reserved_bytes')/2**20:.1f} |",
              f"| Mean training-step time (s) | N/A | {times['full']:.1f} | {times['lora']:.1f} |"]
    lines=['# Matched full fine-tuning versus LoRA','']+['\n'+p+'\n' for p in paragraphs]+table
    lines += ['','## Validation selection','| Method | Seed | Learning rate | Selected step | Validation NLL | Test NLL | Changed scalars |','|---|---:|---:|---:|---:|---:|---:|']
    for m in ['full','lora']:
        for r in runs[m]:
            lines.append(f"| {m} | {r['config']['seed']} | {r['config']['lr']:g} | {r['selected_step']} | {r['selected_validation_nll']:.5f} | {r[m]['test_nll']:.5f} | {r['selected_changes']['changed_elements']:,} |")
    lines += ['','## All learning-rate candidates (validation only)','| Method | Rate | Best validation NLL | Step |','|---|---:|---:|---:|']
    for m,info in selection['methods'].items():
        for candidate in info['candidates']:
            lines.append(f"| {m} | {candidate['lr']:g} | {candidate['validation_nll']:.5f} | {candidate['step']} |")
    diagnostic_path=OUT.parent/'familiar-control.json'
    diagnostic=[]
    if diagnostic_path.exists():
        control=json.loads(diagnostic_path.read_text())
        diagnostic.append('Exploratory familiar-name control: added after observing the seed-42 unseen-name failures. Only team names were replaced; scores, question types and checkpoints were unchanged. No retraining or tuning. This is a diagnostic, not a replacement for the original test.')
        diagnostic.append(f"Reference answers overlapping training: {control['reference_answers_seen_in_training']} / 120.")
        for key in ['primary_correct','fields_correct','reference_exact']:
            values={m:[v['metrics'][key]*100 for v in control['methods'][m].values()] for m in ['full','lora']}
            diagnostic.append(f"{key}: pretrained {control['base']['metrics'][key]*100:.1f}%; full {fmt(values['full'])}%; LoRA {fmt(values['lora'])}%.")
        lines += ['','## Exploratory familiar-name control']+diagnostic
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    keys=['primary_correct','fields_correct','reference_exact']
    for offset,m,color,label in [(-.25,'base','#9ca3af','Pretrained'),(0,'full','#cb7a27','Full fine-tuning'),(.25,'lora','#256aa3','LoRA')]:
        values=[base[k]*100 if m=='base' else stats.mean(vals(m,k)) for k in keys]
        errors=[0 if m=='base' else stats.stdev(vals(m,k)) for k in keys]
        ax.bar([i+offset for i in range(3)],values,.25,yerr=errors,capsize=4,label=label,color=color)
    ax.set(xticks=range(3),xticklabels=['Primary answer','Answer + scores','Exact reference'],ylim=(0,105),ylabel='Held-out questions (%)',title='Same training data and budget; validation-selected checkpoints')
    ax.legend(); fig.savefig(OUT/'quality.png',dpi=180); quality=fig
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    values=[[f['parameters']['trainable']/1e6,l['parameters']['trainable']/1e6],[mem['full'],mem['lora']],[times['full'],times['lora']]]
    labels=['Trainable parameters (millions)','Peak allocated GPU memory (MiB)','Training-step time (seconds)']
    for ax,values,ylabel in zip(axes,values,labels):
        ax.bar(['Full','LoRA'],values,color=['#cb7a27','#256aa3']); ax.set(ylabel=ylabel)
        for i,v in enumerate(values): ax.text(i,v,f'{v:,.2f}',ha='center',va='bottom',fontsize=9)
    axes[0].set_yscale('log'); axes[0].set_ylim(.1,1000)
    fig.savefig(OUT/'efficiency.png',dpi=180); efficiency=fig
    examples=[]
    baseout=l['base']['outputs']; fullout=f['full']['outputs']; loraout=l['lora']['outputs']
    first_shared=next((i for i,(a,b) in enumerate(zip(fullout,loraout)) if a['scores']['reference_exact'] and b['scores']['reference_exact']),0)
    disagreement=next((i for i,(a,b) in enumerate(zip(fullout,loraout)) if a['scores']['reference_exact']!=b['scores']['reference_exact']),None)
    for title,i in [('First shared exact-reference match',first_shared),('First exact-reference disagreement',disagreement)]:
        if i is not None:
            examples.append((title,[baseout[i]['question'],'Reference: '+baseout[i]['reference'],'PRETRAINED: '+baseout[i]['output'],'FULL: '+fullout[i]['output'],'LORA: '+loraout[i]['output']]))
    with PdfPages(OUT/'PROFESSOR_REPORT.pdf') as pdf:
        fig=page('Full fine-tuning versus LoRA',paragraphs[:5]); fig.savefig(OUT/'cover.png',dpi=120); pdf.savefig(fig); plt.close(fig)
        fig=page('Parameter and runtime efficiency',paragraphs[5:]+['FP32 parameter storage is 4 bytes per scalar. Allocated and reserved GPU memory are separate, non-additive allocator measurements; peaks include activations, gradients and optimizer states. They exclude some driver/context memory. minGPT stores separate input/output embedding matrices, so its physical parameter count exceeds tied GPT-2.','The older reports compared LoRA quality to an untouched reference and used a 40-step full-training memory run. Use this matched report for full-versus-LoRA claims.']); pdf.savefig(fig); plt.close(fig)
        pdf.savefig(quality); pdf.savefig(efficiency)
        for title,paras in examples:
            fig=page(title,paras); pdf.savefig(fig); plt.close(fig)
        if diagnostic:
            fig=page('Exploratory control: familiar team names',diagnostic); pdf.savefig(fig); plt.close(fig)
        details=[]
        for kind in ['winner','halftime','comeback']:
            details.append(kind+': '+', '.join(f"{m} {fmt([r[m]['metrics']['by_kind'][kind]['primary_correct']*100 for r in runs[m]])}%" for m in ['full','lora']))
        details += ['Seed 42 is the demo model for each method, not the seed selected for best test score.', 'All raw outputs, including every failure, are available in RESULTS.html and ALL_OUTPUTS.csv. See README.md and PRESENTATION.md for the protocol and live-demo commands.']
        fig=page('Task breakdown and interpretation',details); pdf.savefig(fig); plt.close(fig)
    rows=[]; blocks=[]
    for idx,seed in enumerate([42,43,44]):
        for split,key in [('test','outputs'),('counterfactual','counterfactual_outputs')]:
            b=runs['lora'][idx]['base'][key]; fs=runs['full'][idx]['full'][key]; ls=runs['lora'][idx]['lora'][key]
            for br,fr,lr in zip(b,fs,ls):
                assert br['question']==fr['question']==lr['question']
                row=dict(seed=seed,split=split,kind=br['kind'],match_id=br['match_id'],question=br['question'],reference=br['reference'],pretrained=br['output'],full=fr['output'],lora=lr['output'],pretrained_fields=br['scores']['fields_correct'],full_fields=fr['scores']['fields_correct'],lora_fields=lr['scores']['fields_correct'])
                rows.append(row)
                title=f"Seed {seed} · {split} · {br['match_id']} · {br['kind']} · full {'PASS' if row['full_fields'] else 'FAIL'} · LoRA {'PASS' if row['lora_fields'] else 'FAIL'}"
                blocks.append('<details><summary>'+html.escape(title)+'</summary>'+''.join('<h4>'+label+'</h4><pre>'+html.escape(row[k])+'</pre>' for label,k in [('Question','question'),('Reference','reference'),('Pretrained','pretrained'),('Full fine-tuning','full'),('LoRA','lora')])+'</details>')
    with (OUT/'ALL_OUTPUTS.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    images=''.join('<img src="data:image/png;base64,'+base64.b64encode((OUT/p).read_bytes()).decode()+'">' for p in ['quality.png','efficiency.png'])
    (OUT/'RESULTS.html').write_text('<!doctype html><meta charset="utf-8"><title>Full fine-tuning versus LoRA</title><style>body{max-width:1100px;margin:40px auto;padding:20px;font:17px/1.5 system-ui}img{width:100%}pre{white-space:pre-wrap}details{padding:12px;border:1px solid #ccc;margin:8px 0}input{padding:12px;width:90%}</style><h1>Full fine-tuning versus LoRA</h1>'+''.join('<p>'+html.escape(p)+'</p>' for p in paragraphs)+images+('<h2>Exploratory familiar-name control</h2>'+''.join('<p>'+html.escape(p)+'</p>' for p in diagnostic) if diagnostic else '')+'<input placeholder="Filter tasks, teams, seeds or PASS/FAIL" oninput="document.querySelectorAll(\'details\').forEach(d=>d.hidden=!d.textContent.toLowerCase().includes(this.value.toLowerCase()))">'+''.join(blocks))
    print('\n'.join(table)); print(OUT/'PROFESSOR_REPORT.pdf')


if __name__=='__main__': main()
