"""Build PDF, plots, Markdown, CSV and a self-contained HTML evidence viewer."""
import base64
import csv
import html
import json
import statistics
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/sports/results'
SEEDS=[42,43,44]


def read(name):
    return json.loads((ROOT/'out'/name/'report.json').read_text())


def pct(values):
    return f'{statistics.mean(values)*100:.1f}% ± {statistics.stdev(values)*100:.1f} pp'


def page(title,paragraphs):
    fig=plt.figure(figsize=(8.27,11.69))
    fig.text(.08,.95,title,fontsize=18,weight='bold',va='top')
    y=.90
    for text in paragraphs:
        wrapped='\n'.join(textwrap.fill(line,92) for line in text.split('\n'))
        fig.text(.08,y,wrapped,fontsize=10,va='top',linespacing=1.45)
        y-=.021*(wrapped.count('\n')+1)+.018
    return fig


def main():
    runs=[read(f'sports-seed{s}') for s in SEEDS]
    full=read('sports-full-memory')
    OUT.mkdir(exist_ok=True)
    r=runs[0]
    c=r['parameters']
    mean=lambda metric: [x['lora']['metrics'][metric] for x in runs]
    base=r['base']['metrics']
    mb=lambda n:n/2**20
    lm=r['training_memory_through_step_40']
    fm=full['training_memory_through_step_40']
    reduction=1-lm['peak_allocated_bytes']/fm['peak_allocated_bytes']
    summary=[
        'Research question: Can LoRA adapt pretrained minGPT/GPT-2 to answer questions about new basketball matches, while leaving the base weights unchanged?',
        'Dataset: original synthetic match scores. Train: 480 matches / 1,440 questions. Validation: 40 matches / 120 questions. Test: 40 matches / 120 questions, plus 120 paired score-swapped questions. Test team names and score quadruples are absent from training. None of the test reference answers occurs in training.',
        f"Protocol: FP32, TF32 off, {r['gpu']}, rank 4 / alpha 8, batch 8, padded sequence length 128, AdamW lr 0.001, 1,200 steps per seed. Seeds 42, 43, 44 were fixed before test evaluation. Select checkpoints by validation NLL every 300 steps; evaluate test after selection. Seed 42 is the predeclared live-demo model.",
        f"Held-out primary-answer score: base {base['primary_correct']:.1%}; LoRA {pct(mean('primary_correct'))}. Correct primary plus score numbers: base {base['fields_correct']:.1%}; LoRA {pct(mean('fields_correct'))}. Exact reference match: base {base['reference_exact']:.1%}; LoRA {pct(mean('reference_exact'))}.",
        f"Both members of counterfactual pairs correct (primary answer): base {r['base']['paired_primary_accuracy']:.1%}; LoRA {pct([x['lora']['paired_primary_accuracy'] for x in runs])}. A pair counts only when both original and score-swapped answers are correct.",
        f"LoRA model: {c['total']:,} total scalar parameters; {c['trainable']:,} trainable ({c['trainable_percent']:.4f}%); {c['frozen']:,} frozen. Selected seed-42 checkpoint: {r['selected_changes']['changed_elements']:,} changed scalars; {r['selected_changes']['frozen_changed_elements']} frozen scalars changed. Every audit of every LoRA seed verified the frozen base.",
        f"Storage: all LoRA model parameters {mb(c['parameter_bytes']):.2f} MiB; adapters only {mb(c['trainable_parameter_bytes']):.2f} MiB. Matched first-40-step peak GPU allocated: LoRA {mb(lm['peak_allocated_bytes']):.2f} MiB, full fine-tuning {mb(fm['peak_allocated_bytes']):.2f} MiB ({reduction:.1%} lower). Reserved: {mb(lm['peak_reserved_bytes']):.2f} vs {mb(fm['peak_reserved_bytes']):.2f} MiB.",
        'Interpretation: This tests conditional task learning on unseen inputs, not general sports conversation or proof of abstract reasoning. Outputs can reuse learned sentence patterns. Scoring is deliberately transparent but format-sensitive; inspect raw outputs and failures. Three questions share each match; 120 questions are not 120 independent matches. ± denotes sample SD across three training seeds, not a confidence interval.'
    ]
    summary[3] += ' LoRA primary scores by task (means): ' + ', '.join(f"{kind} {statistics.mean(x['lora']['metrics']['by_kind'][kind]['primary_correct'] for x in runs):.1%}" for kind in ['winner','halftime','comeback']) + '.'
    summary[7] += ' LoRA outputs exactly matching a full training reference, by seed: ' + ', '.join(str(x['lora']['outputs_exactly_seen_in_training']) for x in runs) + ' out of 120 each. Novel wording alone is not proof of correctness.'
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    metrics=['primary_correct','fields_correct','reference_exact']
    labels=['Primary answer','Answer + scores','Exact reference']
    positions=list(range(3))
    axs[0].bar([x-.18 for x in positions],[base[m]*100 for m in metrics],.36,label='Pretrained base',color='#a5acb8')
    axs[0].bar([x+.18 for x in positions],[statistics.mean(mean(m))*100 for m in metrics],.36,
               yerr=[statistics.stdev(mean(m))*100 for m in metrics],capsize=4,label='LoRA: mean ± SD',color='#2867a1')
    axs[0].set(xticks=positions,xticklabels=labels,ylim=(0,105),ylabel='Held-out questions (%)',title='Unseen matches and team names')
    axs[0].tick_params(axis='x',labelsize=9)
    axs[0].legend(fontsize=9)
    axs[1].bar(['Base','LoRA'],[r['base']['paired_primary_accuracy']*100,statistics.mean([x['lora']['paired_primary_accuracy'] for x in runs])*100],color=['#a5acb8','#2867a1'])
    axs[1].set(ylim=(0,105),ylabel='Pairs with both answers correct (%)',title='Does the answer follow swapped scores?')
    fig.savefig(OUT/'quality.png',dpi=180)
    quality=fig
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    axs[0].bar(['Full training','LoRA'],[full['parameters']['trainable']/1e6,c['trainable']/1e6],color=['#a5acb8','#2867a1'])
    axs[0].set(yscale='log',ylabel='Trainable parameters (millions, log scale)',title='Trainable count; base storage remains')
    for i,v in enumerate([full['parameters']['trainable'],c['trainable']]): axs[0].text(i,v/1e6*1.15,f'{v:,}',ha='center',fontsize=9)
    axs[0].set_ylim(.1,1000)
    axs[1].bar(['Full training','LoRA'],[mb(fm['peak_allocated_bytes']),mb(lm['peak_allocated_bytes'])],color=['#a5acb8','#2867a1'])
    axs[1].set(ylabel='Peak allocated GPU memory (MiB)',title='Same first 40 steps, batch 8, length 128')
    for i,v in enumerate([fm['peak_allocated_bytes'],lm['peak_allocated_bytes']]): axs[1].text(i,mb(v)+25,f'{mb(v):.0f} MiB',ha='center')
    fig.savefig(OUT/'parameters-memory.png',dpi=180)
    memory=fig
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    for run,seed in zip(runs,SEEDS):
        ax.plot([h['step'] for h in run['history']],[h['changed_elements'] for h in run['history']],marker='o',label=f'Adapters: seed {seed}')
    ax.plot([1,1200],[0,0],'k--',label='Frozen base: all seeds')
    ax.set(xlabel='Training step',ylabel='Scalars different from initialization',title='Trainable does not mean already changed')
    ax.legend(fontsize=9)
    fig.savefig(OUT/'changes.png',dpi=180)
    changes_plot=fig
    pairs=list(zip(r['base']['outputs'],r['lora']['outputs']))
    improved=next(((b,l) for b,l in pairs if l['scores']['fields_correct'] and not b['scores']['fields_correct']),pairs[0])
    failure=next(((b,l) for b,l in pairs if not l['scores']['fields_correct']),None)
    cf_index=next((i for i,(l,cf) in enumerate(zip(r['lora']['outputs'],r['lora']['counterfactual_outputs'])) if l['scores']['fields_correct'] and cf['scores']['fields_correct']),0)
    with PdfPages(OUT/'PROFESSOR_REPORT.pdf') as pdf:
        f=page('LoRA: unseen-match questions',summary[:5]); f.savefig(OUT/'report-cover.png',dpi=120); pdf.savefig(f); plt.close(f)
        f=page('Parameters, memory and limits',summary[5:]+[
            'Allocated is peak live PyTorch tensor memory; reserved is the caching allocator footprint. Neither is whole-device/driver memory, and they must not be added. Peak includes gradients, optimizer states, activations and workspaces. CPU audit snapshots are outside GPU storage.',
            'Memory reference: separate full-training process, same GPU, data, seed, batch and length; 40 steps only. It is not an equal-duration quality comparison. No FlashAttention, KV cache, AMP or TF32. Timing excludes transfers and audit/evaluation; see JSON for scope.',
            'Reproduce and demonstrate: see README.md and PRESENTATION.md. Raw responses for every seed, question and paired score edit are in ALL_OUTPUTS.csv and RESULTS.html.']); f.savefig(OUT/'report-methods.png',dpi=120); pdf.savefig(f); plt.close(f)
        pdf.savefig(quality); pdf.savefig(memory); pdf.savefig(changes_plot)
        b,l=improved
        f=page('Example: first measured improvement',['Selection rule: first test question where seed-42 LoRA passes the answer+scores check and base fails.',l['question'],'Reference: '+l['reference'],'BASE: '+b['output'],'LORA: '+l['output']]); pdf.savefig(f); plt.close(f)
        orig=r['lora']['outputs'][cf_index]; cf=r['lora']['counterfactual_outputs'][cf_index]
        f=page('Example: answer changes with the scores',['Selection rule: first seed-42 pair with both answer+scores checks correct.',orig['question'],'LORA: '+orig['output'],cf['question'],'LORA: '+cf['output']]); pdf.savefig(f); plt.close(f)
        if failure:
            b,l=failure
            f=page('Failure to discuss',['Selection rule: first seed-42 test question failing the answer+scores check. Automated checks can also misclassify valid paraphrases; inspect this output.',l['question'],'Reference: '+l['reference'],'BASE: '+b['output'],'LORA: '+l['output']]); pdf.savefig(f); plt.close(f)
        bad_pair=next(((o,c) for o,c in zip(r['lora']['outputs'],r['lora']['counterfactual_outputs']) if o['kind']=='comeback' and not (o['scores']['fields_correct'] and c['scores']['fields_correct'])),None)
        if bad_pair:
            o,counter=bad_pair
            f=page('Limit: comeback score-swap test',['First seed-42 comeback pair where at least one answer+scores check fails.',o['question'],'Reference: '+o['reference'],'LORA: '+o['output'],counter['question'],'Reference: '+counter['reference'],'LORA: '+counter['output']]); pdf.savefig(f); plt.close(f)
    lines=['# LoRA unseen-match experiment','']+['\n'+x+'\n' for x in summary]
    lines+=['\n| Seed | Selected step | Validation NLL | Test NLL | Primary correct | Answer + scores | Exact reference | Both pair members correct |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for run,seed in zip(runs,SEEDS):
        m=run['lora']['metrics']
        lines.append(f"| {seed} | {run['selected_step']} | {run['selected_validation_nll']:.4f} | {run['lora']['test_nll']:.4f} | {m['primary_correct']:.1%} | {m['fields_correct']:.1%} | {m['reference_exact']:.1%} | {run['lora']['paired_primary_accuracy']:.1%} |")
    lines += ['', '| Question type | Base primary correct | LoRA primary correct (mean ± SD) |', '|---|---:|---:|']
    for kind in ['winner','halftime','comeback']:
        vals=[run['lora']['metrics']['by_kind'][kind]['primary_correct'] for run in runs]
        lines.append(f"| {kind} | {r['base']['metrics']['by_kind'][kind]['primary_correct']:.1%} | {pct(vals)} |")
    lines += ['','![Quality](quality.png)','![Parameters and memory](parameters-memory.png)','![Changed scalars](changes.png)']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    csvrows=[]
    details=[]
    for run,seed in zip(runs,SEEDS):
        for split,key in [('test','outputs'),('counterfactual','counterfactual_outputs')]:
            for b,l in zip(run['base'][key],run['lora'][key]):
                csvrows.append(dict(seed=seed,split=split,match_id=l['match_id'],kind=l['kind'],question=l['question'],reference=l['reference'],base=b['output'],lora=l['output'],base_primary=b['scores']['primary_correct'],lora_primary=l['scores']['primary_correct'],base_fields=b['scores']['fields_correct'],lora_fields=l['scores']['fields_correct'],lora_exact=l['scores']['reference_exact']))
                title=f"Seed {seed} · {split} · {l['match_id']} · {l['kind']} · LoRA fields {'PASS' if l['scores']['fields_correct'] else 'FAIL'}"
                details.append('<details><summary>'+html.escape(title)+'</summary>'+''.join('<h4>'+label+'</h4><pre>'+html.escape(text)+'</pre>' for label,text in [('Question',l['question']),('Reference',l['reference']),('Base',b['output']),('LoRA',l['output'])])+'</details>')
    with (OUT/'ALL_OUTPUTS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0])); w.writeheader(); w.writerows(csvrows)
    plots=''.join('<img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()+'">' for name in ['quality.png','parameters-memory.png','changes.png'])
    (OUT/'RESULTS.html').write_text('<!doctype html><meta charset="utf-8"><title>LoRA unseen-match results</title><style>body{max-width:1050px;margin:40px auto;padding:20px;font:17px/1.55 system-ui;color:#182c3b}img{width:100%}details{border:1px solid #ccd5de;padding:12px;margin:8px 0}pre{white-space:pre-wrap;font:15px/1.5 monospace}input{padding:12px;width:90%;font:inherit}</style><h1>LoRA: unseen-match questions</h1>'+''.join('<p>'+html.escape(s)+'</p>' for s in summary)+plots+'<h2>All outputs</h2><input id="filter" placeholder="Filter by seed, task, PASS/FAIL, team or text" oninput="document.querySelectorAll(\'details\').forEach(d=>d.hidden=!d.textContent.toLowerCase().includes(this.value.toLowerCase()))">'+''.join(details))
    print(OUT/'PROFESSOR_REPORT.pdf')


if __name__=='__main__':
    main()
