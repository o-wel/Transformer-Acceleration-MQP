"""Assemble verified final study artifacts into Markdown, standalone HTML and PDF."""
from pathlib import Path
import base64
import hashlib
import html
import json
import re
import statistics as st
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'.plot_deps'))
OUT=HERE/'run_fp32'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    selection=json.loads((OUT/'selection.json').read_text())
    trials=[json.loads(p.read_text()) for p in (OUT/'trials').glob('*/result.json')]
    cells=sum([json.loads((OUT/f'systems-{s}.json').read_text()) for s in range(3)],[])
    quality={m:json.loads((OUT/f'{m}-test.json').read_text()) for m in ['full','lora']}
    assert len(trials)==127 and all(r['complete'] and r['frozen_base_unchanged'] for r in trials)
    assert len(cells)==1440 and all(r['status']=='ok' and r['tokens_equal'] and len(r['samples'])==7 for r in cells)
    assert all(len(q)==3 and all(len(r['outputs'])==120 for r in q) for q in quality.values())
    assert selection['frozen_before_test']
    key=lambda r:(r['method'],r['batch'],r['prompt_tokens'],r['new_tokens'],r['backend'],r['cache_mode'],r['bm'],r['bn'],r['warps'])
    assert len({key(r) for r in cells})==1440
    lookup={key(r):r for r in cells}
    def cell(method,batch,length,new,backend,cache,bm=16,bn=32,w=4):
        return lookup[(method,batch,length,new,backend,cache,bm,bn,w)]
    # Printable framework figure accompanies the editable Mermaid source.
    fig,ax=plt.subplots(figsize=(11,4.8));ax.set(xlim=(0,11),ylim=(0,5));ax.axis('off')
    boxes=[(0.2,3.3,3.0,1.15,'Adaptation / training','Pretrained GPT-2 → LoRA A/B\nRank, alpha, LR, targets\nRegularization, batch and length'),
           (3.9,3.3,3.0,1.15,'Validation selection','Equal-example comparisons\nThree-seed confirmation\nFreeze configurations before test'),
           (7.6,3.3,3.0,1.15,'Prompt prefill','Many query tokens\nEager or exact tiled attention\nInitialize K/V storage'),
           (7.6,.65,3.0,1.55,'Autoregressive decode','One query; growing K/V history\nDynamic or preallocated cache\nEager / tiled / hybrid policy\nAppend K/V; produce next token'),
           (.2,.65,6.7,1.55,'Evaluation','Training: validation loss, memory, compute and parameters\nServing: prefill, decode, latency, throughput and memory\nCorrectness: logits and token parity; final held-out accuracy')]
    for x,y,w,h,title,body in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor='#eef4fc',edgecolor='#315780'))
        ax.text(x+w/2,y+h-.16,title,ha='center',va='top',fontsize=11,weight='bold')
        ax.text(x+w/2,y+h-.48,body,ha='center',va='top',fontsize=9,linespacing=1.5)
    for start,end in [((3.3,3.9),(3.8,3.9)),((7.,3.9),(7.5,3.9)),((9.1,3.2),(9.1,2.3)),((7.5,1.4),(7.,1.4))]:
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',lw=1.6,color='#315780'))
    fig.tight_layout();fig.savefig(OUT/'framework.png',dpi=190,bbox_inches='tight');fig.savefig(OUT/'framework.svg',bbox_inches='tight');plt.close(fig)
    winner=selection['winner']
    params=[r for r in trials if r['config']==winner and r['seed']==42 and r['examples']==9600][0]
    sections=['# Final report: tuning LoRA, tiled attention and KV caching','',
      'Completed November 3, 2026. GPT-2 / fictional sports question answering / NVIDIA A30 / FP32 with TF32 disabled.','',
      '## 1. Findings','',
      'The staged search completed 127 unique training trials and 1,440 combined serving configurations. Every serving configuration passed the required eager-reference logit tolerance and greedy-token parity checks; no measured cell ran out of memory. Each cell retains seven timing repetitions after two warmups.','',
      'The selected configuration is rank 32, alpha 64, learning rate 0.0003, all attention and MLP projections, LoRA input dropout 0.05, AdamW weight decay 0.01, effective batch 4, microbatch 4, and padded training length 128. It achieved mean validation answer-token NLL 0.016746 with seed standard deviation 0.009248 across seeds 42, 43 and 44. This is the best observed configuration within the staged search.','',
      'Held-out answer-field accuracy averaged 93.06% for selected LoRA and 24.17% for the selected full-fine-tuning control. This is a task-specific result with unequal search budgets: LoRA received a broad search while full fine-tuning received three learning rates. It does not establish general superiority of LoRA.','',
      'Caching produced large latency gains for long prompts. The preferred attention/cache configuration depended on workload. Unmerged adapters added inference work, so the selected LoRA model was slower than the full control at short prompts despite its training-memory and parameter advantages.','',
      '## 2. Methodology','',
      '![Combined framework](framework.png)','',
      '### 2.1 LoRA','',
      'Start every candidate from identical pretrained weights. Freeze the base and train W + (alpha/r)BA at the fused Q/K/V and attention output projections; the all-target variant also adapts both MLP projections. Adapter dropout acts before A only. The selected model trains '+f"{params['trainable_parameters']:,} of {params['total_parameters']:,} stored parameters ({100*params['trainable_parameters']/params['total_parameters']:.2f}%)."+' The minGPT implementation stores an untied output head, so its total parameter count differs from tied-weight GPT-2 counts.','',
      '### 2.2 Exact tiled attention','',
      'The implementation is a local FA1-derived Algorithm 1/4 prototype with K/V tiles outer and query tiles inner. Query tiles 16/32, key tiles 32/64 and warps 4/8 form eight settings. This is not the original optimized FA1 CUDA kernel or production FA2/3/4. No PyTorch FlashAttention backend is used in this study. Training hyperparameter selection uses eager attention throughout; attention settings are serving ablations on frozen selected checkpoints.','',
      '### 2.3 KV caching','',
      'Compare uncached prefix recomputation, dynamic concatenating KV storage, and preallocated full-context KV storage. Requests are serial within each model, use equal-length dense batches, and stay within 1,024 position tokens. Static capacity and logical used-cache bytes are distinct metrics. The preallocated variant still incurs any layout copies required by the attention prototype.','',
      '### 2.4 Combined design and fair comparison','',
      'The primary factorial comparison uses full/LoRA × eager/tiled × cache off/dynamic. Additional factors are a hybrid policy with tiled prefill and eager single-token decode, static caching, attention tile geometry, batch 1/4/8, prompt 32/128/512/896 and generation 32/64. The 1,440 cells include hybrid/off controls, which provide repeated timing controls because all uncached queries have length greater than one.','',
      'Each within-model comparison uses the same checkpoint and synthetic prompt. Timed generation emits exactly the requested token count; quality evaluation trims at EOS. CUDA events measure prefill/decode, host wall time measures total request latency, and allocated/reserved memory is retained in raw samples. Cases run in fixed order, so drift and thermal/order effects remain limitations. Timing warmups exclude kernel compilation. Speed comparisons are descriptive and do not include cross-run confidence intervals.','',
      '## 3. Search protocol','',
      table(['Stage','Search','Budget'],[
       ['Core','5 ranks × 3 scales × 3 LRs × 2 target sets = 90','1,200 examples each'],
       ['Refine','Top 9 core candidates','9,600 examples each'],
       ['Regularize','3 dropout × 3 decay settings','9,600 examples each'],
       ['Batch / memory / length','Batch 4/8/16/32; microbatch 1/2/4/8; length 128/256/512','9,600 examples each'],
       ['Confirm','Top 3 unique candidates × seeds 42/43/44','9,600 examples per seed'],
       ['Full control','LR 1e-4/3e-4/1e-3; selected LR repeated on seeds 43/44','9,600 examples per run']]),'',
      'Completed duplicate configurations are reused; stage counts therefore exceed the 127 unique trials when summed. Validation selects checkpoints and candidates. Only the frozen selected LoRA and full configurations receive held-out evaluation. Core screening is exhaustive over its four dimensions; later stages are conditional searches. Short screening can eliminate candidates that improve later.','',
      'The data contains 1,440 training, 120 validation and 120 test rows. Test team names are absent from training. Microbatch losses are weighted by supervised-token counts. Identical seeds yield the same flattened sampled-example stream. At equal 9,600 examples, batch 4 receives 2,400 optimizer updates and batch 32 receives 300: batch effects include a changed update budget. With dropout enabled, microbatch partitioning also changes RNG consumption, so different validation losses are not solely accumulation-rounding effects.','',
      '## 4. Hyperparameter findings','',
      '![Core grid](core_grid_heatmaps.png)','',
      '### 4.1 Rank, scaling, LR and targets','',
      'The table below contains the nine candidates promoted to full-budget refinement, all at batch/microbatch 8, length 128, dropout 0 and decay 0. These are seed-42 results, not independent three-seed confirmations of every rank.','',
      table(['Rank','Alpha','LR','Targets','Validation NLL','Update ms'],[
       [r['config']['rank'],f"{r['config']['rank']*r['config']['scale']:g}",f"{r['config']['lr']:g}",r['config']['targets'],f"{r['validation_nll']:.6f}",f"{r['median_step_ms']:.2f}"]
       for r in sorted(json.loads((OUT/'refine.json').read_text()),key=lambda r:r['validation_nll'])]),'',
      'Rank 4 with alpha 16 and LR 0.001 nearly matched the best unregularized rank-32 refinement loss (0.023254 versus 0.023162). Rank 32 is the winner of the subsequent conditional search; the study did not exhaustively retune regularization and batching around rank 4. Do not claim that high rank is necessary.','',
      '### 4.2 Regularization','',
      table(['Dropout','Decay','Validation NLL','Update ms'],[
       [r['config']['dropout'],r['config']['weight_decay'],f"{r['validation_nll']:.6f}",f"{r['median_step_ms']:.2f}"]
       for r in sorted(json.loads((OUT/'regularize.json').read_text()),key=lambda r:r['validation_nll'])]),'',
      'Dropout 0.05 with decay 0.01 was the strongest seed-42 regularization pair. Its small gain over dropout/decay zero at batch 8 should be treated as a selected observation rather than an isolated causal effect or statistically resolved improvement.','',
      '### 4.3 Batch, accumulation and padded sequence length','',
      table(['Batch','Micro','Length','Updates','Validation NLL','Compute s','Peak GiB'],[
       [r['config']['batch'],r['config']['microbatch'],r['config']['length'],r['steps'],f"{r['validation_nll']:.6f}",f"{sum(r['step_ms'])/1000:.1f}",f"{r['peak_training_bytes']/2**30:.2f}"]
       for r in sorted(json.loads((OUT/'batch.json').read_text()),key=lambda r:(r['config']['length'],r['config']['batch'],r['config']['microbatch']))]),'',
      'Smaller microbatches reduced memory but increased runtime. Increasing padded length from 128 to 256/512 increased update time from 90.11 ms to 187.88/393.78 ms without improving the selected validation loss. The task examples already fit within 128 tokens; this is a padding-cost experiment, not evidence about learning longer real contexts. Peak training allocation is sampled after updates; the CUDA high-water mark may include an earlier validation pass.','',
      '### 4.4 Three-seed confirmation','',
      table(['Batch','Microbatch','Mean validation NLL','Seed SD'],[
       [r['config']['batch'],r['config']['microbatch'],f"{r['mean_validation_nll']:.6f}",f"{r['stdev_validation_nll']:.6f}"] for r in selection['ranked']]),'',
      'The batch-4 candidate wins on mean validation loss; batch 8/microbatch 4 has lower seed variance. Three seeds provide limited precision and do not establish significance of the difference.','',
      '![Quality and update cost](quality_cost.png)','',
      '## 5. Held-out quality and training cost','']
    qr=[]
    for metric in ['primary_correct','score_numbers_correct','fields_correct','reference_exact']:
        row=[metric]
        for method in ['full','lora']:
            vals=[r['metrics'][metric]*100 for r in quality[method]]
            row += [f'{st.mean(vals):.2f}%',f'{st.stdev(vals):.2f} pp']
        qr.append(row)
    sections += [table(['Metric','Full mean','Full seed SD','LoRA mean','LoRA seed SD'],qr),'',
       'Answer-field correctness requires both the primary answer and expected score numbers. The evaluation is on 120 shared test questions per seed; the 360 scored answers per method are not 360 independent questions. Seed SD describes training-run variability, not a confidence interval over new datasets. LoRA answer-field results were 96.67%, 82.50% and 100.00%; full-control results were 18.33%, 31.67% and 22.50%.','']
    cost=[]
    for method in ['full','lora']:
        chosen=selection['winner'] if method=='lora' else selection['full_control']
        rr=[r for r in trials if r['examples']==9600 and r['config']==chosen]
        cost.append([method,f"{rr[0]['trainable_parameters']:,}",f"{st.mean(sum(r['step_ms'])/1000 for r in rr):.1f}",f"{st.mean(r['peak_training_bytes']/2**30 for r in rr):.2f}"])
    sections += [table(['Method','Trainable parameters','Mean compute seconds','Mean peak GiB'],cost),'',
      'These costs cover all 9,600-example training updates, even when an earlier checkpoint wins. They exclude model loading, validation and checkpoint I/O. The full control trains all parameters; LoRA-specific dropout/target/rank entries in its shared config schema do not apply to full fine-tuning. This comparison uses the selected LR and the winner-derived batch, length and weight decay; the full control did not receive a matching broad optimizer/batch search.','',
      '## 6. Combined LoRA, attention and cache results','',
      '### 6.1 Primary eight-cell comparison','',
      'Fixed workload: batch 8, prompt 896, output 64; fixed tiles 16/32, warps 4. Values are medians of seven repetitions. Logical cache occupancy is additional state, and static capacity is excluded here because the primary comparison uses the dynamic cache.','']
    factorial=[]
    for method in ['full','lora']:
        for backend in ['eager','fa1']:
            for cache in ['off','dynamic']:
                r=cell(method,8,896,64,backend,cache)
                factorial.append([method,backend,cache,f"{r['wall_ms']:.2f}",f"{st.median(s['prefill_ms'] for s in r['samples']):.2f}",f"{st.median(s['decode_ms_per_token_step'] for s in r['samples']):.2f}",f"{max(s['peak_allocated_bytes'] for s in r['samples'])/2**30:.2f}"])
    sections += [table(['Method','Attention','Cache','Request ms','Prefill ms','Decode ms/token','Peak GiB'],factorial),'',
      'For selected LoRA at this long workload, eager caching reduced request latency from 18,412.38 ms to 874.11 ms (21.06x). Tiled attention at default geometry with dynamic cache took 895.09 ms, slightly slower than eager cached. Hybrid dynamic took 817.22 ms. The fastest measured LoRA cell used tiled dynamic with tiles 16/64 and four warps at 800.17 ms (23.01x versus eager uncached, and 1.09x versus eager cached). Best-of-grid timings are selected performance observations and can contain selection noise.','',
      'At batch 1, prompt 512, output 64, default tiled attention slowed uncached LoRA generation (2,179.61 ms versus eager 1,385.28 ms), while tiled dynamic caching beat eager dynamic (594.70 ms versus 641.62 ms). Thus the attention/cache interaction is workload-dependent; caching does not universally diminish attention acceleration.','',
      '### 6.2 Mitigations and tile tuning','',
      'At batch 8/prompt 896/output 64, hybrid dynamic reduced LoRA request time from the default tiled/dynamic 895.09 ms to 817.22 ms. Static storage improved default tiled latency to 850.87 ms and reduced peak allocation from about 1.94 GiB to 1.69 GiB, but hybrid/static was slower than hybrid/dynamic (830.61 versus 817.22 ms). Preallocation reserves full context capacity and can be less attractive at short contexts. These measurements support tuning the combination rather than selecting each mechanism independently.','',
      '![Combined serving latency](combined_latency.png)','',
      'The attention/cache interaction ratio is (eager-off/tiled-off)/(eager-dynamic/tiled-dynamic), computed at fixed geometry for every workload in INTERACTIONS.csv. Ratios above one indicate that tiling benefit shrinks with caching; values below one indicate the opposite. Raw latency and variation must accompany this descriptive ratio. SYSTEMS.csv includes all eight tile/warp settings and all cache modes; ANALYSIS.md lists the fastest observed setting per workload.','',
      '## 7. Related work and scope','',
      (HERE/'LITERATURE.md').read_text().replace('# Related work and experimental implications','### Primary-source review',1),'',
      '## 8. Limitations and next experiments','',
      'The full precision study is complete. FP16 was optional and was not run. Results describe a custom FP32 FA1-derived prototype on A30, one small task and this minGPT implementation. They do not measure production FA2/3/4, lossy/quantized caches, paging, split-KV decode or merged LoRA. A merged-adapter serving ablation is a useful next step because unmerged A/B computation is visible in inference latency.','',
      'The top-three confirmation candidates all share rank 32 and differ in batch/microbatch. Lower-rank candidates remain competitive in seed-42 refinement, and full fine-tuning had less tuning compute. The broad conclusion is that tuning materially affects outcomes, with workload-specific combinations for inference; universal hyperparameters and method superiority remain unestablished.','',
      '## 9. Reproducibility and artifacts','',
      '127 completed unique trials; 1,440/1,440 serving cells passed; 10,080 measured serving repetitions plus warmups. The preliminary GPU gate passed 24 attention forward/backward cases and 18 full-model cache/backend cases. Data/source hashes, exact source snapshots, trial checkpoints, sample-stream hashes, raw timing samples, raw test outputs and Slurm job logs are retained. See source_hashes.json, snapshot_hashes.json, source/, trials/, systems-0/1/2.json and logs/.','',
      'The complete machine-readable tables are TRIALS.csv, PARETO.csv, SYSTEMS.csv and INTERACTIONS.csv. selection.json records the frozen validation choice. framework.svg is a standalone export; METHODS.md retains an editable Mermaid diagram. FINAL_REPORT.md is editable, FINAL_REPORT.html embeds all figures, and FINAL_REPORT.pdf is printable.']
    md='\n'.join(sections)+'\n';(OUT/'FINAL_REPORT.md').write_text(md)
    import markdown
    body=markdown.markdown(md,extensions=['tables','fenced_code'])
    def embed(match):
        p=OUT/match.group(1)
        return 'src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"'
    body=re.sub(r'src="([^\"]+\.png)"',embed,body)
    (OUT/'FINAL_REPORT.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Transformer Hyperparameter Study</title><style>body{font:16px/1.6 system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#17263a}h1,h2,h3{line-height:1.25}h2{margin-top:2em;color:#254f79}table{border-collapse:collapse;width:100%;font-size:13px;margin:1em 0}td,th{border:1px solid #ced7e3;padding:7px;text-align:left}th{background:#eef4fc}img{max-width:100%;height:auto}a{color:#235c9b}@media print{body{font-size:10pt}h2,h3{break-after:avoid}table,img{break-inside:avoid}}</style></head><body>'+body+'</body></html>')
    pdf(md)
    inventory={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['FINAL_REPORT.md','FINAL_REPORT.html','FINAL_REPORT.pdf','TRIALS.csv','SYSTEMS.csv','INTERACTIONS.csv','selection.json','framework.svg']}
    (OUT/'FINAL_REPORT_HASHES.json').write_text(json.dumps(dict(complete=True,training_trials=127,systems_cells=1440,measured_repetitions=10080,sha256=inventory),indent=2))
    print('Verified study complete; wrote FINAL_REPORT.md, FINAL_REPORT.html, FINAL_REPORT.pdf and artifact hashes')


def pdf(md):
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import matplotlib
    fonts=Path(matplotlib.get_data_path())/'fonts/ttf'
    pdfmetrics.registerFont(TTFont('StudySans',str(fonts/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('StudySans-Bold',str(fonts/'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFontFamily('StudySans',normal='StudySans',bold='StudySans-Bold',italic='StudySans',boldItalic='StudySans-Bold')
    styles=getSampleStyleSheet()
    for name in ['Normal','Heading1','Heading2','Heading3']:styles[name].fontName='StudySans-Bold' if name.startswith('Heading') else 'StudySans'
    styles['Normal'].fontSize=9;styles['Normal'].leading=13
    styles['Heading1'].fontSize=19;styles['Heading2'].fontSize=14;styles['Heading3'].fontSize=11
    styles['Normal'].spaceAfter=7
    def inline(s):
        s=html.escape(s)
        s=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',lambda m:f'<link href="{m[2]}" color="#235c9b">{m[1]}</link>',s)
        s=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',s)
        return s.replace('`','')
    story=[];lines=md.splitlines();i=0;width=A4[0]-76
    while i<len(lines):
        line=lines[i];i+=1
        if not line.strip():continue
        if line.startswith('|'):
            rows=[line]
            while i<len(lines) and lines[i].startswith('|'):rows.append(lines[i]);i+=1
            data=[]
            for row in rows:
                if re.fullmatch(r'[| :\-]+',row):continue
                data.append([Paragraph(inline(c.strip()),styles['Normal']) for c in row.strip('|').split('|')])
            n=len(data[0]);tbl=Table(data,colWidths=[width/n]*n,repeatRows=1,hAlign='LEFT')
            tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eef4fc')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#ced7e3')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4)]))
            story += [tbl,Spacer(1,9)];continue
        match=re.match(r'!\[[^\]]*\]\(([^)]+)\)',line)
        if match:
            from PIL import Image as PILImage
            path=OUT/match[1]
            with PILImage.open(path) as image:w,h=image.size
            story += [Image(str(path),width=width,height=width*h/w),Spacer(1,10)];continue
        match=re.match(r'^(#{1,3}) (.+)',line)
        if match:story.append(Paragraph(inline(match[2]),styles[f'Heading{len(match[1])}']));continue
        story.append(Paragraph(inline(line),styles['Normal']))
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#62758a'))
        canvas.drawString(38,24,'FP32 transformer hyperparameter study | November 3, 2026')
        canvas.drawRightString(A4[0]-38,24,str(doc.page))
    SimpleDocTemplate(str(OUT/'FINAL_REPORT.pdf'),pagesize=A4,rightMargin=38,leftMargin=38,topMargin=36,bottomMargin=38,
                      title='Transformer Hyperparameter Study: Final Report',author='minGPT experiments').build(story,onFirstPage=footer,onLaterPages=footer)


if __name__=='__main__':main()
