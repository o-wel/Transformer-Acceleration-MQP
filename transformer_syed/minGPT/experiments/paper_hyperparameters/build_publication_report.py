"""Build a readable PDF/HTML/Markdown report from completed, frozen study artifacts."""
import argparse
import base64
import collections
import csv
import datetime
import hashlib
import html
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import markdown
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak

ROOT=Path(__file__).resolve().parents[2]
TEAL='#147d86';NAVY='#17334b';ORANGE='#c3762a';GRAY='#7a8794'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','savefig.bbox':'tight'})


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def gm(values):return math.exp(statistics.mean(math.log(x) for x in values))


def pct(x):return f'{100*x:.2f}%'


def config(c):return f"r={c['rank']}, alpha={c['alpha']}, LR={c['lr']:g}"


class Document:
    def __init__(self,out):
        self.out=out;self.md=[];self.story=[];self.width=A4[0]-84
        self.styles=getSampleStyleSheet()
        self.styles.add(ParagraphStyle(name='ReportTitle',fontName='Helvetica-Bold',fontSize=25,leading=29,textColor=colors.HexColor(NAVY),spaceAfter=15))
        self.styles.add(ParagraphStyle(name='ReportSubtitle',fontName='Helvetica',fontSize=12,leading=17,textColor=colors.HexColor(TEAL),spaceAfter=10))
        self.styles.add(ParagraphStyle(name='ReportBody',fontName='Helvetica',fontSize=9.5,leading=13.3,spaceAfter=8))
        self.styles.add(ParagraphStyle(name='ReportCaption',fontName='Helvetica',fontSize=8.2,leading=11,textColor=colors.HexColor('#47596a'),spaceAfter=10))
        self.styles.add(ParagraphStyle(name='ReportCell',fontName='Helvetica',fontSize=8.1,leading=10.5))
        self.styles['Heading1'].keepWithNext=1;self.styles['Heading2'].keepWithNext=1
        self.styles['ReportBody'].allowWidows=0;self.styles['ReportBody'].allowOrphans=0
        self.styles['Heading1'].fontSize=16;self.styles['Heading1'].leading=20;self.styles['Heading1'].textColor=colors.HexColor(NAVY)
        self.styles['Heading2'].fontSize=11.5;self.styles['Heading2'].leading=15;self.styles['Heading2'].textColor=colors.HexColor(TEAL)
    def inline(self,s):
        s=html.escape(str(s))
        s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
        s=re.sub(r'`(.+?)`',r'<font name="Courier">\1</font>',s)
        return s
    def title(self,title,subtitle):
        self.md += ['# '+title,'',subtitle,'']
        self.story += [Paragraph(self.inline(title),self.styles['ReportTitle']),Paragraph(self.inline(subtitle),self.styles['ReportSubtitle'])]
    def h(self,text,level=1):
        self.md += ['#'*(level+1)+' '+text,'']
        self.story += [Paragraph(self.inline(text),self.styles['Heading1' if level==1 else 'Heading2'])]
    def p(self,text):
        self.md += [text,''];self.story += [Paragraph(self.inline(text),self.styles['ReportBody'])]
    def table(self,headers,rows,widths=None):
        clean=lambda x:str(x).replace('|',' / ').replace('\n',' ')
        self.md += ['| '+' | '.join(map(clean,headers))+' |','|'+'|'.join(['---']*len(headers))+'|']
        self.md += ['| '+' | '.join(map(clean,row))+' |' for row in rows]+['']
        values=[[Paragraph(self.inline(x),self.styles['ReportCell']) for x in row] for row in [headers,*rows]]
        widths=widths or [1]*len(headers);widths=[self.width*w/sum(widths) for w in widths]
        t=Table(values,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dfecef')),
                              ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8fa')]),
                              ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),
                              ('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),
                              ('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor(TEAL))]))
        self.story += [t,Spacer(1,12)]
    def figure(self,name,caption):
        self.md += [f'![{caption}](figures/{name}.png)','',caption,'']
        import PIL.Image
        with PIL.Image.open(self.out/'figures'/f'{name}.png') as im:w,h=im.size
        width=self.width;height=width*h/w
        self.story.append(KeepTogether([Image(str(self.out/'figures'/f'{name}.png'),width=width,height=height),Spacer(1,5),Paragraph(self.inline(caption),self.styles['ReportCaption'])]))
    def page(self):self.story.append(PageBreak())
    def finish(self):
        md='\n'.join(self.md);(self.out/'HYPERPARAMETER_STUDY_REPORT.md').write_text(md)
        body=markdown.markdown(md,extensions=['tables'])
        for p in (self.out/'figures').glob('*.png'):
            data=base64.b64encode(p.read_bytes()).decode()
            body=body.replace(f'src="figures/{p.name}"',f'src="data:image/png;base64,{data}"')
        css='body{font:16px/1.55 system-ui;color:#17334b;max-width:1100px;margin:40px auto;padding:20px}h1,h2,h3{line-height:1.2}h2{margin-top:40px}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}td,th{padding:8px;text-align:left;border-bottom:1px solid #dbe2e8;vertical-align:top}th{background:#dfecef}img{max-width:100%}code{font-size:.87em}a{color:#147d86}@media print{body{font-size:11px;margin:0}h2{break-after:avoid}table,img{break-inside:avoid}}'
        (self.out/'HYPERPARAMETER_STUDY_REPORT.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Hyperparameter Study Report</title><style>'+css+'</style><body>'+body+'</body></html>')
        def footer(canvas,doc):
            canvas.saveState();canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor(GRAY))
            canvas.drawString(42,23,'MQP | GPT-2 hyperparameter study | Prototype only')
            canvas.drawRightString(A4[0]-42,23,f'{doc.page}')
            canvas.restoreState()
        pdf=SimpleDocTemplate(str(self.out/'HYPERPARAMETER_STUDY_REPORT.pdf'),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=38,bottomMargin=40,
                              title='Hyperparameter Selection and Independent Confirmation: LoRA, KV Cache and FA1-Derived Attention',author='MQP experiment report')
        pdf.build(self.story,onFirstPage=footer,onLaterPages=footer)


def figure_save(out,name,fig):
    fig.savefig(out/'figures'/f'{name}.png',dpi=200)
    fig.savefig(out/'figures'/f'{name}.svg');plt.close(fig)


def build(study,out):
    out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    r=read(study/'results.json');s=r['selected'];protocol=read(study/'protocol.json');pre=read(study/'preflight.json')
    assert r['complete'] and r['runtime_quality_gates_passed']
    manifest=read(study/'manifest.json');assert sha(study/'protocol.json')==manifest['protocol_sha256']
    for rel,expected in manifest['snapshot_sha256'].items():assert sha(study/'snapshot/repo'/rel)==expected,rel
    assert sha(study/'selection.json')==(study/'selection.sha256').read_text().strip()
    for rel,expected in s['input_sha256'].items():assert sha(study/rel)==expected,rel
    original_hashes=read(study/'ARTIFACT_SHA256.json')
    for rel,expected in original_hashes.items():assert sha(study/rel)==expected,rel
    timing=list(csv.DictReader((study/'TIMINGS.csv').open()))
    quality=list(csv.DictReader((study/'QUALITY_BY_SEED.csv').open()))
    workloads=protocol['search_workloads']+protocol['heldout_workloads'];wk=lambda w:f"b{w['batch']}-p{w['prompt']}-n{w['new_tokens']}"
    heldout={wk(w) for w in protocol['heldout_workloads']}
    records=[read(p) for p in sorted((study/'confirmation-timing').glob('seed-*-session-*.json'))]
    assert len(records)==15 and all(v['complete'] and v['weights_before']==v['weights_after'] for v in records)
    for record in records:
        assert len(record['workloads'])==14
        for block in record['workloads'].values():
            samples=block['samples'];ref=next(iter(samples.values()))
            assert len(samples)==14
            for rows in samples.values():
                assert len(rows)==20
                assert [(x['repeat'],x['prompt_seed']) for x in rows]==[(x['repeat'],x['prompt_seed']) for x in ref]
    assert len(timing)==58800
    logs=collections.defaultdict(list)
    for row in timing:
        scope='heldout' if row['workload'] in heldout else 'search'
        logs[(scope,row['mode'])].append(math.log(float(row['wall_ms'])))
    absolute={scope:{mode:math.exp(statistics.mean(logs[(scope,mode)])) for mode in records[0]['modes']} for scope in ['heldout','search']}
    assert abs(absolute['heldout']['factorial-000']/absolute['heldout']['factorial-111']-r['comparisons']['combined']['heldout']['speedup'])<1e-10
    training=list(study.glob('search-training/*/seed-*/training.json'))+list(study.glob('confirmation-training/*/seed-*/training.json'))
    assert len(training)==145
    for p in training:
        v=read(p);assert v['complete'] and v['successful_updates']==2400 and len(v['validation'])==8
        assert v['frozen_base_before']==v['frozen_base_after']
    jobs=read(study/'jobs.json')['jobs'];ids=','.join(v['id'] for v in jobs.values())
    try:
        acc=subprocess.run(['sacct','-j',ids,'--parsable2','--format=JobID,JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocTRES'],capture_output=True,text=True,timeout=30,check=True)
        (out/'FINAL_ACCOUNTING.csv').write_text(acc.stdout)
    except (OSError,subprocess.SubprocessError):shutil.copy2(study/'ACCOUNTING.csv',out/'FINAL_ACCOUNTING.csv')
    account=list(csv.DictReader((out/'FINAL_ACCOUNTING.csv').open(),delimiter='|'))
    account=[x for x in account if '.' not in x['JobID']]
    gpu_hours=sum(int(x['ElapsedRaw']) for x in account if 'gres/gpu=1' in x['AllocTRES'])/3600
    compute=read(study/'COMPUTE.json')
    envs=[v['environment'] for v in records];guids={e['gpu_uuid'] for e in envs};nodes={e['node'] for e in envs}
    summary=dict(complete=True,study=str(study),report_date='2026-11-10',verification=dict(source_files=len(manifest['snapshot_sha256']),
                 original_artifact_hashes=len(original_hashes),selection_inputs=len(s['input_sha256']),training_runs=145,
                 confirmation_allocations=15,timing_rows=len(timing),timing_pairing=True,weights_unchanged=True),
                 gpu_hours=gpu_hours,physical_gpus=len(guids),nodes=len(nodes),heldout_geometric_ms=absolute['heldout'])
    save(out/'REPORT_VERIFICATION.json',summary)
    # 1. Main inference results: dimensionless ratios, zero-effect reference at 1.
    names=['lora','cache','fa','combined'];labels=['LoRA settings','KV-cache prefill','FA prototype tiles','All three together']
    vals=np.array([r['comparisons'][n]['heldout']['speedup'] for n in names]);lo=np.array([r['comparisons'][n]['heldout']['ci95'][0] for n in names]);hi=np.array([r['comparisons'][n]['heldout']['ci95'][1] for n in names])
    fig,ax=plt.subplots(figsize=(8.8,3.7));y=np.arange(4)
    for i in range(4):ax.errorbar(vals[i],y[i],xerr=[[vals[i]-lo[i]],[hi[i]-vals[i]]],fmt='o',color=TEAL if i==3 else GRAY,capsize=4,markersize=7)
    ax.axvline(1,color='#59636e',ls='--',lw=1);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(.998,1.052)
    ax.set_xlabel('Untuned / tuned generation latency (speedup ratio)');ax.set_title('Independent confirmation on eight held-out workload shapes',loc='left')
    for i,v in enumerate(vals):ax.text(v+.0024,i,f'{v:.4f}x',va='center',fontsize=9)
    ax.grid(axis='x',alpha=.2);fig.tight_layout();figure_save(out,'confirmation_speedups',fig)
    # 2. Fixed workload results, preserving small slowdowns rather than hiding them.
    fig,ax=plt.subplots(figsize=(9,6.4));ys=np.arange(14)
    for i,w in enumerate(workloads):
        v=r['combined_per_workload'][wk(w)];a,b=v['ci95'];c=TEAL if w['split']=='heldout' else GRAY
        ax.errorbar(v['speedup'],i,xerr=[[v['speedup']-a],[b-v['speedup']]],fmt='o',color=c,capsize=3,markersize=5)
    ax.set_yticks(ys,[f"B{w['batch']} / P{w['prompt']} / N{w['new_tokens']}" for w in workloads]);ax.invert_yaxis();ax.axvline(1,color='#59636e',ls='--',lw=1)
    ax.axhline(5.5,color='#d7dfe5',lw=1);ax.set_xlabel('Untuned / tuned latency');ax.set_title('Combined settings: workload dependence',loc='left');ax.grid(axis='x',alpha=.2)
    ax.text(.98,1.02,'Gray: tuning shapes remeasured | Teal: held-out shapes',transform=ax.transAxes,ha='right',fontsize=8)
    fig.tight_layout();figure_save(out,'workload_speedups',fig)
    # 3. Paired quality and training measurements; five seeds shown explicitly.
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    seeds=protocol['confirmation_seeds']
    for i,seed in enumerate(seeds):
        q=[next(x for x in quality if int(x['seed'])==seed and x['model']==label and x['runtime']=='combined' and x['split']=='original') for label in ['baseline','tuned']]
        axes[0].plot([0,1],[100*float(x['reference_exact']) for x in q],marker='o',lw=1.2,label=str(seed),alpha=.85)
        t=r['training'][i];axes[1].plot([0,1],[t['baseline_ms'],t['tuned_ms']],marker='o',lw=1.2,alpha=.85)
    axes[0].set_ylabel('Exact reference match (%)');axes[0].set_ylim(80,101);axes[0].set_title('Original held-out questions')
    axes[1].set_ylabel('Median warm optimizer step (ms)');axes[1].set_title('LoRA training cost')
    for ax in axes:ax.set_xticks([0,1],['Untuned','Tuned']);ax.set_xlim(-.15,1.15);ax.grid(axis='y',alpha=.2)
    axes[0].legend(title='Seed',fontsize=7,loc='lower right',ncol=2,frameon=False);fig.tight_layout();figure_save(out,'quality_training_pairs',fig)
    # 4. Search tradeoff; the quality-best and cost-best are separate selections.
    fig,ax=plt.subplots(figsize=(8.8,4.7));rows=s['lora_search']
    sc=ax.scatter([v['mean_step_ms'] for v in rows],[v['mean_nll'] for v in rows],c=[math.log2(v['config']['rank']) for v in rows],cmap='viridis',s=40,alpha=.8)
    for selected,label,marker,c in [(s['lora'],'Operational choice','*',TEAL),(s['quality_best_lora'],'Quality-best','D',ORANGE),(protocol['baseline_lora'],'Untuned control','s',NAVY)]:
        v=next(x for x in rows if x['config']==selected);ax.scatter(v['mean_step_ms'],v['mean_nll'],marker=marker,s=150 if marker=='*' else 70,color=c,edgecolor='white',label=label,zorder=4)
    ax.axhline(s['lora_nll_cutoff'],ls='--',color=GRAY,label='Eligibility cutoff');ax.set_yscale('log');ax.set_xlabel('Mean warm training-step median across three search seeds (ms)');ax.set_ylabel('Mean validation-selected answer NLL');ax.set_title('LoRA discovery: quality and cost',loc='left');ax.legend(fontsize=8,frameon=False)
    bar=fig.colorbar(sc,ax=ax,pad=.02);bar.set_ticks([1,2,3,4,5],labels=['2','4','8','16','32']);bar.set_label('LoRA rank');ax.grid(alpha=.15);fig.tight_layout();figure_save(out,'lora_search_tradeoff',fig)
    # 5. Runtime discovery (not confirmation evidence).
    fig=plt.figure(figsize=(9,6.6));grid=fig.add_gridspec(2,2,height_ratios=[1,1.1]);axes=[fig.add_subplot(grid[0,:]),fig.add_subplot(grid[1,0]),fig.add_subplot(grid[1,1])]
    cache=s['runtime_search']['cache'];axes[0].plot(range(6),[x['geometric_mean_ms'] for x in cache],'-o',color=TEAL);axes[0].set_xticks(range(6),[str(x['config']) for x in cache],rotation=35);axes[0].set_xlabel('Prefill chunk tokens');axes[0].set_ylabel('Geometric mean latency (ms)');axes[0].set_title('Cache discovery');axes[0].grid(alpha=.2)
    tiles=s['runtime_search']['fa'];mins=min(x['geometric_mean_ms'] for x in tiles);maxs=max(x['geometric_mean_ms'] for x in tiles)
    for ax,warps in zip(axes[1:],[4,8]):
        matrix=np.array([[next(x['geometric_mean_ms'] for x in tiles if x['config']==[q,k,warps]) for k in [16,32,64,128]] for q in [16,32,64]])
        ax.imshow(matrix,cmap='YlGnBu_r',vmin=mins,vmax=maxs);ax.set_xticks(range(4),[16,32,64,128]);ax.set_yticks(range(3),[16,32,64]);ax.set_xlabel('Key tile');ax.set_ylabel('Query tile');ax.set_title(f'Prototype: {warps} warps')
        for i in range(3):
            for j in range(4):ax.text(j,i,f'{matrix[i,j]:.0f}',ha='center',va='center',fontsize=10,color='white' if matrix[i,j]<1100 else NAVY)
    fig.tight_layout();figure_save(out,'runtime_search',fig)
    doc=Document(out)
    doc.title('Hyperparameter Selection and Independent Confirmation','LoRA, full-context KV caching, and FA1-derived attention in minGPT')
    doc.p('**Study:** study-20261105-v2. **Prepared:** November 10, 2026. **Scope:** prototype tuning only. Discovery and confirmation completed successfully; this document presents the completed experiment rather than the earlier pilot.')
    doc.h('Executive summary')
    primary=r['comparisons']['combined']['heldout'];a,b=primary['ci95'];sp=primary['speedup']
    doc.p(f"Combining the independently selected settings yielded a **{sp:.4f}x speedup** on eight held-out workload shapes (hierarchical 95% CI {a:.4f}-{b:.4f}x). This corresponds to {100*(sp-1):.2f}% higher fixed-workload throughput, or {100*(1-1/sp):.2f}% lower geometric-mean latency. The gain is modest and depends strongly on prompt length.")
    q=r['quality']['reference_exact']
    doc.p(f"Original-test exact reference accuracy increased from **{pct(q['baseline'])} to {pct(q['tuned'])}**, a {100*q['paired_difference']:.2f} percentage-point improvement (paired 95% CI {100*q['difference_ci95'][0]:.2f}-{100*q['difference_ci95'][1]:.2f} points). This compares independently trained LoRA configurations; it is not evidence that cache scheduling or attention tiles improve semantic quality.")
    doc.table(['Component','Untuned control','Selected setting'],[
        ['LoRA',config(protocol['baseline_lora']),config(s['lora'])],
        ['Cache prefill',str(protocol['baseline_chunk'])+' tokens','1024 tokens (full prompt prefill)'],
        ['Attention prototype','Query 16 / key 32 / 4 warps','Query 64 / key 128 / 4 warps']],widths=[1,2,2])
    doc.p('The phrase **best tested under the declared quality/cost objective** is supported. Global optimality, production FlashAttention performance, and gains on other models, hardware classes or tasks are not established. The independently selected combination is not a jointly optimized optimum.')
    doc.figure('confirmation_speedups','Figure 1. Independent inference confirmation on held-out workload shapes. Bars are hierarchical 95% intervals; the dashed line marks no speed change. LoRA training is evaluated separately.')
    doc.page();doc.h('1. Research questions and study design')
    doc.p('The study asks whether tuning each method improves its relevant cost, and whether independently chosen settings transfer to an all-three inference configuration. The control was declared before measurements and was not chosen as the slowest candidate. Every combined comparison keeps LoRA, cache and prototype attention enabled; only their parameter settings change.')
    doc.table(['Phase','Replication / budget','Purpose'],[
        ['LoRA discovery','45 settings x 3 seeds = 135 runs; 2400 successful updates each','Select quality-constrained training cost'],
        ['Runtime discovery','6 cache + 24 prototype settings; 3 allocations; 6 shapes; 10 measured repeats','Select independent inference settings'],
        ['LoRA confirmation','2 settings x 5 new seeds = 10 runs; 2400 updates each','Measure transfer to new training seeds'],
        ['Timing confirmation','5 seeds x 3 allocations = 15 jobs; 14 shapes; 20 measured repeats','Primary held-out latency and interactions'],
        ['Quality confirmation','5 seeds; 2 LoRA models; eager and combined paths','900 test + 900 related counterfactual questions per condition']],widths=[1.15,2.35,2])
    doc.p('Discovery seeds were 11, 22 and 33. Confirmation seeds were 101, 202, 303, 404 and 505. The original pilot was exploratory and was excluded from this study. The earlier v1 submission was cancelled in response to the prototype-only scope decision before discovery training began.')
    doc.h('Model, precision and hardware',2)
    doc.p(f"The model uses the GPT-2 small architecture (12 blocks, 12 heads, embedding width 768, head width 64, context window 1024). This minGPT implementation stores the input embedding and output head separately: 163,037,184 frozen base scalars, rather than tying their storage. Baseline and selected adapters contain 589,824 and 294,912 trainable scalars respectively. Adapters remain unmerged. All conditions use FP32 stored weights with FP16 autocast; TF32 is disabled.")
    e=pre['environment']
    doc.p(f"All allocations used NVIDIA A30 GPUs. The 15 confirmation timing jobs spanned {len(guids)} physical GPU UUIDs on {len(nodes)} nodes. They are separate allocation sessions, not 15 different physical GPUs. Recorded software: Python {e['python']}, PyTorch {e['torch']}, CUDA {e['cuda']}, Triton {e['triton']}, Transformers {e['transformers']}, NumPy {e['numpy']}. Build and package details are preserved in torch-config.txt and software-freeze.txt.")
    doc.h('Data and separation from the pilot',2)
    doc.table(['Split','Independent matches','Questions','Role'],[['Training',480,1440,'Existing authored task data'],['Validation',200,600,'Fresh matches; selection only'],['Test',300,900,'Fresh matches and new team names'],['Counterfactual', 'Same 300 test groups',900,'Related score-swapped questions']],widths=[1,1.2,1,2.2])
    doc.p('Matches are split before their three question types are generated. New validation/test score quadruples exclude previously used scores; test uses eight team names absent from training and the prior pilot test teams. Counterfactual questions remain grouped with their original match. The generator seed is 20261105. This evaluates compositional generalization within controlled basketball templates, not open-ended dialogue or unseen task instructions.')
    doc.h('2. Discovery and selection methodology')
    doc.h('LoRA: quality-constrained training cost',2)
    doc.p('The exhaustive grid crosses ranks {2,4,8,16,32}, alpha/rank {1,2,4}, and learning rates {0.0001,0.0003,0.001}. Adapters target fused QKV, attention output, and both MLP projections in every block. All candidates use AdamW, zero weight decay, constant learning rate, gradient norm clipping at 1, batch 8, padded length 128, dropout 0 and 2400 successful optimizer updates. Sampling order is identical across candidates within a seed and checked by schedule hashes.')
    doc.p('Each run selects its checkpoint by validation answer-token NLL at steps 300,600,...,2400, using the earliest checkpoint on an exact tie. Candidate quality is the mean selected NLL across the three search seeds. Let q* be the best such mean. A setting is eligible when its mean NLL is at most q* + 0.01. Among eligible settings, select the lowest mean of the three warm training-step medians, breaking ties by lower rank, alpha and LR. The first 50 updates of each process segment are excluded from warm timing. Skipped/nonfinite updates fail the task.')
    best=next(x for x in s['lora_search'] if x['config']==s['quality_best_lora']);chosen=next(x for x in s['lora_search'] if x['config']==s['lora']);base=next(x for x in s['lora_search'] if x['config']==protocol['baseline_lora'])
    eligible=sum(x['mean_nll']<=s['lora_nll_cutoff'] for x in s['lora_search'])
    doc.table(['Discovery role','Configuration','Mean validation NLL','Mean step ms'],[
        ['Untuned control',config(base['config']),f"{base['mean_nll']:.8g}",f"{base['mean_step_ms']:.3f}"],
        ['Operational winner',config(chosen['config']),f"{chosen['mean_nll']:.8g}",f"{chosen['mean_step_ms']:.3f}"],
        ['Quality-best only',config(best['config']),f"{best['mean_nll']:.8g}",f"{best['mean_step_ms']:.3f}"]],widths=[1.2,2.1,1.25,1])
    doc.p(f"The cutoff was {s['lora_nll_cutoff']:.8g}; {eligible}/45 candidates qualified. The additive 0.01 margin is permissive relative to near-zero validation losses on this task. The operational winner nevertheless had NLL close to the quality-best candidate. The quality-best rank-32 setting is a discovery result; independent confirmation compares the operational winner against the control.")
    doc.figure('lora_search_tradeoff','Figure 2. All 45 LoRA candidates, averaged over three discovery seeds. Lower is better on both axes. The dashed eligibility cutoff and separate cost-selected/quality-best choices make the multiobjective decision explicit.')
    doc.h('Cache and prototype attention',2)
    doc.p('Cache tuning tests chunks {32,64,128,256,512,1024}. The 1024 setting performs a single full prefill because all prompt lengths are smaller. Every setting retains the full K/V context; there is no eviction, quantization or reduced attention span. Prototype tuning crosses query tiles {16,32,64}, key tiles {16,32,64,128}, and warp counts {4,8}. Every tile choice runs the same FA1-derived Triton forward algorithm. No production FlashAttention backend is measured.')
    doc.p('Runtime discovery holds the untuned LoRA configuration fixed for seeds 11/22/33. Cache tuning uses eager attention; prototype tuning disables caching. Five warmup blocks precede ten measured blocks per workload, with shuffled modes and matched random-token prompts within each block. Each eligible setting is ranked by the equally weighted geometric mean of per-allocation/per-workload median wall latency. Settings must pass numerical and validation-NLL checks in every discovery allocation.')
    doc.figure('runtime_search','Figure 3. Runtime selection measurements only: cache settings and the 24 prototype launches. Values are geometric-mean generation latency in milliseconds over the six search workload shapes and three allocations. These timings select settings; they are not reused as confirmation evidence.')
    doc.h('3. Independent confirmation and statistical analysis')
    doc.p('After selection.json was frozen and hashed, selected and untuned LoRA configurations were retrained with five new seeds. Within each seed, training order was shuffled. Each model again received 2400 updates and the same eight validation checkpoint opportunities. Hyperparameters were not changed using confirmation timing or test answers.')
    doc.table(['Workload set','Batch','Prompt tokens','Generated tokens','Shapes'],[['Search shapes, remeasured','1 or 4','32, 256, 896','64','6'],['Held-out primary shapes','1 or 4','128 or 512','32 or 128','8']],widths=[2,1,1.5,1.3,.6])
    doc.p('Every confirmation timing allocation measures all 14 conditions after five warmup blocks, with twenty measured paired blocks and shuffled mode order. A fresh deterministic prompt is used for each block. Timing always generates exactly N tokens, including after EOS, and prompt plus output stays within the context window. Last-token-only head projection is used equally. Backend selection, telemetry, input construction/transfer, garbage collection and output hashing are outside synchronized wall timing. CUDA-event latency and allocator peaks are also recorded.')
    doc.p('For each paired observation z = log(T_untuned / T_tuned). The reported ratio is exp(mean(z)), with equal weighting of fixed workload shapes, allocation sessions and training seeds. Ratios above 1 denote speedup. The primary endpoint covers the eight held-out shapes. Search-shape and all-shape aggregates, isolated methods, per-shape results and interactions are secondary descriptive analyses.')
    doc.p('Intervals use 5000 hierarchical paired bootstrap draws: resample five training seeds, three allocation sessions within each seed, and twenty paired repeats within each allocation/workload. Workload shapes are fixed and retain equal weight. The 2400 held-out pairs per comparison are not treated as 2400 independent GPU experiments. Training-step intervals resample paired seed-level medians. Quality differences use a crossed paired bootstrap over five seeds and 300 match groups. Secondary intervals are not adjusted for multiple comparisons.')
    doc.h('4. Confirmation results')
    doc.table(['Comparison','Held-out ratio','95% interval','All 14 shapes'],[
        [label,f"{r['comparisons'][name]['heldout']['speedup']:.4f}x",f"{r['comparisons'][name]['heldout']['ci95'][0]:.4f}-{r['comparisons'][name]['heldout']['ci95'][1]:.4f}",f"{r['comparisons'][name]['all']['speedup']:.4f}x"] for name,label in zip(names,labels)],widths=[2,1,1.5,1])
    doc.p(f"The primary combined result excludes 1 on the speedup side and passes the numerical quality gates. Held-out geometric-mean latency was {absolute['heldout']['factorial-000']:.2f} ms for all-untuned settings and {absolute['heldout']['factorial-111']:.2f} ms for all-tuned settings. The all-14-shape aggregate is {r['comparisons']['combined']['all']['speedup']:.4f}x. These are workload-weighted geometric aggregates, not one representative request or a mean across a production workload distribution.")
    doc.figure('workload_speedups','Figure 4. Combined settings for all 14 shapes. B=batch, P=prompt tokens, N=generated tokens. Intervals crossing 1 indicate no resolved speed change under the bootstrap model. Short-prompt cases show nearly flat results or small slowdowns; the strongest gains occur with longer prefills.')
    doc.table(['Set','B','P','N','Combined ratio','95% interval'],[
        [w['split'],w['batch'],w['prompt'],w['new_tokens'],f"{r['combined_per_workload'][wk(w)]['speedup']:.4f}x",f"{r['combined_per_workload'][wk(w)]['ci95'][0]:.4f}-{r['combined_per_workload'][wk(w)]['ci95'][1]:.4f}"] for w in workloads],widths=[1.05,.4,.5,.5,1.1,1.55])
    doc.h('LoRA training and answer quality',2)
    tr=r['training_speedup']
    doc.p(f"Confirmed warm-step speedup was {tr['speedup']:.4f}x (95% CI {tr['ci95'][0]:.4f}-{tr['ci95'][1]:.4f}x), approximately {100*(tr['speedup']-1):.2f}% higher update throughput. Seed 303 was slightly slower with the selected configuration. LoRA inference improvement was also small ({100*(r['comparisons']['lora']['heldout']['speedup']-1):.2f}%). These results support a small cost change, not a large LoRA-only acceleration.")
    doc.figure('quality_training_pairs','Figure 5. Paired confirmation measurements across five training seeds. Left: exact-match accuracy on the original 900 held-out questions. Right: warm optimizer-step medians. Connecting lines pair the same seed; training speed does not improve for every seed.')
    doc.table(['Quality on original test questions','Untuned','Tuned','Difference, points','Paired 95% CI, points'],[
        [label,pct(v['baseline']),pct(v['tuned']),f"{100*v['paired_difference']:+.2f}",f"{100*v['difference_ci95'][0]:+.2f} to {100*v['difference_ci95'][1]:+.2f}"] for key,label in [('reference_exact','Exact reference'),('fields_correct','Primary + numeric fields')] for v in [r['quality'][key]]],widths=[1.75,.8,.8,1.1,1.4])
    quality_rows=[]
    for label in ['baseline','tuned']:
        vals=[x for x in quality if x['model']==label and x['runtime']=='combined']
        original=[x for x in vals if x['split']=='original'];counter=[x for x in vals if x['split']=='counterfactual']
        quality_rows.append([label,pct(statistics.mean(float(x['reference_exact']) for x in counter)),pct(statistics.mean(float(x['fields_correct']) for x in counter)),pct(statistics.mean(float(x['paired_fields_correct']) for x in original))])
    doc.table(['Combined configuration','Counterfactual exact','Counterfactual fields','Both-pair fields'],quality_rows,widths=[1.4,1.2,1.2,1.2])
    doc.p('Counterfactual and paired summaries above are descriptive means across seeds. A paired field success requires both the original and corresponding score-swapped question to pass field scoring. The 900 original and 900 counterfactual questions still represent 300 underlying match groups, not 1800 independent matches. Mean original-test exact/field accuracy is unchanged by the eager-to-combined runtime switch within each LoRA configuration; rare per-example differences remain in raw outputs.')
    doc.h('5. Transfer and parameter interactions')
    doc.p('The 2^3 factorial varies parameter levels while keeping all three methods enabled. In the bit code LCF, 0 denotes the untuned parameter and 1 the selected parameter for LoRA, cache and the attention prototype respectively. LoRA weights change with L; C and F hold those weights fixed. This is a parameter factorial, not a method on/off ablation.')
    factorial=list(csv.DictReader((study/'FACTORIAL.csv').open()))
    doc.table(['LCF','Held-out latency, ms','Held-out ratio vs 000','All-shape ratio vs 000'],[
        [x['mode'].split('-')[1],f"{absolute['heldout'][x['mode']]:.2f}",f"{absolute['heldout']['factorial-000']/absolute['heldout'][x['mode']]:.4f}x",f"{float(x['speedup_vs_000']):.4f}x"] for x in factorial],widths=[.6,1.6,1.6,1.6])
    doc.p(f"The combined gain is largely associated with the prefill chunk change. On held-out shapes, changing the cache parameter alone (010) measured {absolute['heldout']['factorial-010']:.2f} ms, compared with {absolute['heldout']['factorial-111']:.2f} ms for 111. Adding the selected LoRA and tile settings did not provide an additional latency benefit in this aggregate; the full combination remains faster than 000. Prototype-only uncached gains therefore do not transfer additively to cached inference. This descriptive comparison does not claim that 010 is a newly selected joint optimum.")
    doc.p('Isolated speedups must not be multiplied. The factorial table and saved log-latency contrasts directly measure interactions. No joint search or test-driven retuning followed the independent selections, and no alternative factorial row replaces the prespecified 111-vs-000 primary comparison.')
    doc.h('6. Numerical checks, task scoring and examples')
    doc.p('All 24 prototype launches passed six FP64-reference attention comparisons each (144 total), using FP16 Q/K/V and prespecified atol 0.012/rtol 0.04. All six cache settings passed twelve same-prefix logit comparisons each (72 total), at batches 1 and 4 with uneven chunks, long prefills and three teacher-forced decode steps. Cache logit tolerances were atol 0.1/rtol 0.02. Prototype and cache discovery required absolute validation-NLL change no greater than 0.01 versus eager inference of the same weights. Confirmation rechecked the selected runtime NLL on both newly trained models.')
    qfiles=[read(study/'confirmation-quality'/f'seed-{seed}.json') for seed in seeds]
    max_delta=max(abs(v['validation']['combined']-v['validation']['eager']) for qf in qfiles for v in qf['models'].values())
    doc.p(f"All ten confirmation model/runtime NLL gates passed; the largest absolute eager/combined validation-NLL change was {max_delta:.8g}. All 145 training runs completed 2400 successful updates and preserved frozen base weights. All 15 timing allocations preserved parameter hashes. A rank-32 optimizer smoke test, maximum-workload runtime checks, baseline PTX and a prototype profiler trace are preserved. These forward-inference checks do not establish correctness or speed for prototype backward/training kernels.")
    doc.p('Normalized exact reference match is the primary task metric. Field scoring checks the primary answer and the expected multiset of numbers. It is intentionally simpler and can miss semantic errors, especially a wrong team name in a comeback answer. The following illustrations were chosen after evaluation by a stated rule, not as a representative sample: first original seed-101 example corrected by tuning, then first original seed-202 tuned exact-match failure, in generator order.')
    for seed,condition in [(101,'corrected'),(202,'failure')]:
        qf=next(x for x in qfiles if x['seed']==seed)
        bq=qf['models']['baseline']['outputs']['combined']['answers'];tq=qf['models']['tuned']['outputs']['combined']['answers']
        for x,y in zip(bq,tq):
            if '-counter-' in x['match_id']:continue
            ok=(not x['scores']['reference_exact'] and y['scores']['reference_exact']) if condition=='corrected' else not y['scores']['reference_exact']
            if ok:
                doc.h(f'Illustrative example: seed {seed}',2);doc.p(x['question'].replace('\n',' '))
                doc.table(['Reference','Untuned answer','Tuned answer'],[[x['reference'],x['text'],y['text']]],widths=[1,1,1])
                if condition=='failure':doc.p('The tuned answer incorrectly names Hornets instead of Herons. It fails exact reference scoring but passes the coarse comeback field check because Yes and the numeric multiset are correct. This example is a reason to prefer the exact-reference metric here and to retain raw outputs.')
                break
    doc.h('7. Compute cost and artifact provenance')
    phase_hours=collections.defaultdict(float)
    for row in account:
        if 'gres/gpu=1' in row['AllocTRES']:phase_hours[row['JobName']]+=int(row['ElapsedRaw'])/3600
    doc.table(['GPU stage','Allocated A30-hours'],[[k.replace('paper-',''),f'{v:.3f}'] for k,v in phase_hours.items()]+[['Total',f'{gpu_hours:.3f}']],widths=[3,1.5])
    update_hours=sum(v['measured_update_seconds'] for v in compute.values())/3600
    doc.p(f"The study used {gpu_hours:.2f} allocated A30 GPU-hours, including model loading, evaluation, warmups, kernel work and quality decoding within GPU jobs. Measured training updates alone account for {update_hours:.2f} hours over 348,000 successful updates. There are 58,800 measured confirmation generations across 14 modes, 14 workloads, 15 allocations and 20 repeats; runtime discovery adds 5400 measured generations. Quality evaluation retains 36,000 generated answers across models, runtime paths, seeds and original/counterfactual questions. Deployment latency excludes this search/evaluation cost; no amortization claim is made.")
    states=collections.Counter(x['State'] for x in account)
    doc.p(f"Scheduler accounting records {len(account)} top-level jobs/array tasks. Final states in the report-time accounting query: {dict(states)}. The original ACCOUNTING.csv was captured inside the running CPU report job, so its report-job row can say RUNNING even though that job subsequently completed; FINAL_ACCOUNTING.csv in this report directory refreshes that status. No experiment was rerun to create this document.")
    doc.table(['Artifact relative to study directory','Contents'],[
        ['protocol.json; snapshot/repo/.../METHODS.md','Frozen design and full methods'],['manifest.json; selection.json; selection.sha256','Source/data hashes and frozen selection provenance'],
        ['results.json; TIMINGS.csv','Primary/secondary intervals and per-repeat timing data'],['QUALITY_BY_SEED.csv; confirmation-quality/','Seed summaries, all raw answers and scores'],
        ['FACTORIAL.csv; FACTORIAL_CONTRASTS.json','Measured parameter combinations and descriptive interactions'],['search-training/; confirmation-training/','Learning curves, update timings, audits and adapter checkpoints'],
        ['prototype-baseline.ptx; prototype.trace.json.gz','Prototype implementation evidence'],['report/REPORT_VERIFICATION.json','Report-time source/artifact checks and sample counts']],widths=[2.5,2.5])
    doc.p(f"Protocol SHA-256: {sha(study/'protocol.json')}. Selection SHA-256: {sha(study/'selection.json')}. This presentation was generated after completion from frozen artifacts. Source files, {len(original_hashes)} original JSON artifact hashes, {len(s['input_sha256'])} selection inputs, training completion and timing pairing/invariance were checked before export.")
    doc.h('8. Interpretation and limitations')
    limitations=[
        'Finite search: the selected query/key tiles and full-prefill setting lie at tested grid boundaries. The data establish the selected settings within this search, not an unconstrained global optimum. Full prefill has a natural context-length ceiling; larger prototype tiles were not tested.',
        'Objectives differ: LoRA was selected for training-step cost under a validation-NLL constraint; cache and prototype settings were selected for inference latency. Quality-best LoRA and cost-best eligible LoRA are distinct. Neither time-to-quality nor a jointly optimized serving configuration was the selection target.',
        'Workload dependence: the 2.88% primary speedup is an equal-weight aggregate of eight fixed shapes. Longer prefills gain more; some short-prompt batch-4 shapes are slightly slower. Different workload mixtures can give different aggregate results.',
        'Statistical scope: narrow intervals describe the recorded seeds, repeated A30 allocations, fixed shapes and bootstrap model. Four physical GPUs across three nodes do not represent multiple architectures or arbitrary deployment systems. Secondary intervals are descriptive and not multiplicity adjusted.',
        'Task scope: synthetic basketball facts, known instruction templates and a restricted score distribution can make validation NLL nearly saturate. New names/matches help test compositional transfer but do not establish broad reasoning or real-world sports competence. Exact matching can reject valid paraphrases, while field checks can accept semantic errors.',
        'Implementation scope: this is an FA1-derived Triton prototype, not the original optimized FA1 CUDA library or a production FlashAttention implementation. Only inference forward behavior is evaluated for prototype tile tuning. Results should not be described as production FA1/FA2/FA3 benchmarks.',
        'Precision and lifecycle: FP32 stored weights with FP16 autocast, unmerged adapters, full-context dynamic caches and a 1024-token window are fixed. Cache rollover, eviction, quantization, merged-LoRA serving and other precisions were not studied. Floating-point shape changes can alter close greedy decisions; exact token parity is not assumed.',
        'Attribution and cost: quality improvement compares different learned LoRA weights and is not an attention/cache quality effect. Most combined latency improvement tracks cache prefill scheduling, while LoRA-only speed changes are small. Search cost is separate from request latency and has not been shown to amortize.'
    ]
    for i,text in enumerate(limitations,1):doc.p(f'**{i}.** '+text)
    doc.h('Suggested paper wording',2)
    doc.p(f"Under a prespecified finite search on minGPT/GPT-2 with NVIDIA A30 GPUs, independently selected LoRA, full-context cache-prefill and FA1-derived prototype settings achieved a {sp:.4f}x geometric-mean inference speedup on eight workload shapes withheld from tuning (hierarchical paired-bootstrap 95% CI {a:.4f}-{b:.4f}x). Across five independent confirmation training seeds, exact reference accuracy on 300 newly generated test-match groups increased from {pct(q['baseline'])} to {pct(q['tuned'])}. Gains varied by workload and were largely associated with prefill scheduling; these findings do not imply global optimality or production FlashAttention performance.")
    doc.h('Appendix A. Complete LoRA discovery results')
    doc.p('Each row summarizes three search seeds. NLL is the mean of validation-selected checkpoints; SD is sample SD across seeds. Step cost is the mean of warm-step medians. Eligibility uses the frozen additive NLL margin. The complete per-seed data remain in selection.json and the training directories.')
    doc.table(['Rank','Alpha','LR','Mean NLL','NLL SD','Step ms','Eligible'],[
        [v['config']['rank'],v['config']['alpha'],f"{v['config']['lr']:g}",f"{v['mean_nll']:.5g}",f"{v['sd_nll']:.4g}",f"{v['mean_step_ms']:.3f}",'Yes' if v['mean_nll']<=s['lora_nll_cutoff'] else 'No'] for v in s['lora_search']],widths=[.45,.5,.8,1.25,1.05,.85,.7])
    doc.h('Appendix B. Confirmation seed details')
    seed_rows=[]
    for t in r['training']:
        qs=[next(x for x in quality if int(x['seed'])==t['seed'] and x['model']==m and x['runtime']=='combined' and x['split']=='original') for m in ['baseline','tuned']]
        seed_rows.append([t['seed'],f"{t['baseline_ms']:.3f}",f"{t['tuned_ms']:.3f}",f"{t['baseline_nll']:.5g}",f"{t['tuned_nll']:.5g}",pct(float(qs[0]['reference_exact'])),pct(float(qs[1]['reference_exact']))])
    doc.table(['Seed','Untuned step ms','Tuned step ms','Untuned val NLL','Tuned val NLL','Untuned exact','Tuned exact'],seed_rows,widths=[.5,1.1,1.1,1.15,1.15,.95,.95])
    doc.p('Seed 202 has worse selected validation NLL for the tuned configuration than for the control, despite higher exact test accuracy. This illustrates that teacher-forced validation likelihood and greedy-decoded test accuracy are distinct metrics; the report does not hide seed-level variation.')
    doc.h('Appendix C. Background references and reproducibility')
    doc.p('The following background references were recorded in the frozen methods. The exact search grids, numerical tolerances, workload design and selection objectives are choices of this study, not claims that those papers prescribe these values.')
    for text in ['Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. https://arxiv.org/abs/2106.09685',
                 'Dao et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. https://arxiv.org/abs/2205.14135',
                 'Cawley and Talbot. On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. https://jmlr.org/papers/v11/cawley10a.html']:
        doc.p(text)
    doc.p('The complete frozen methodology is included alongside this report as METHODS_FROZEN.md. JSON, CSV and standalone PNG/SVG figures are supplied for review and reuse. Run the report builder against the completed study to regenerate the presentation; it writes a separate report directory and leaves the frozen experiment and original results unchanged.')
    shutil.copy2(study/'snapshot/repo/experiments/paper_hyperparameters/METHODS.md',out/'METHODS_FROZEN.md')
    shutil.copy2(study/'combined-configs.json',out/'combined-configs.json')
    doc.finish()
    save(out/'REPORT_ARTIFACT_SHA256.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='REPORT_ARTIFACT_SHA256.json'})
    print(out/'HYPERPARAMETER_STUDY_REPORT.pdf')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,default=ROOT/'experiments/paper_hyperparameters/runs/study-20261105-v2');parser.add_argument('--out',type=Path);args=parser.parse_args()
    build(args.study.resolve(),args.out.resolve() if args.out else args.study.resolve()/'report')
