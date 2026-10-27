"""Render measured FlashAttention results as PDF, Markdown, HTML, plots, and CSV."""
import base64
import csv
import html
import json
import math
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

OUT = Path(__file__).resolve().parent/'results'
COLORS = {'eager':'#8c96a5', 'math':'#df9d36', 'flash':'#167b83'}


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def text_page(title, sections):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(.07, .95, title, size=19, weight='bold', va='top')
    y = .895
    for heading, content in sections:
        fig.text(.07, y, heading, size=11, weight='bold', va='top')
        y -= .027
        if isinstance(content, tuple):
            headers, rows = content
            height = .024*(len(rows)+1)
            ax = fig.add_axes([.06, y-height, .88, height])
            ax.axis('off')
            t = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='upper center', bbox=[0,0,1,1])
            t.auto_set_font_size(False)
            t.set_fontsize(8)
            for (r,c), cell in t.get_celld().items():
                cell.set_edgecolor('#dde2e8')
                if r == 0:
                    cell.set_facecolor('#e6eff4')
                    cell.set_text_props(weight='bold')
            y -= height+.035
        else:
            lines = '\n'.join(textwrap.fill(line, 96) for line in content.split('\n'))
            fig.text(.07, y, lines, size=9.5, va='top', linespacing=1.4)
            y -= .0185*(lines.count('\n')+1)+.025
        if y < .055:
            raise RuntimeError(f'Page overflow: {title} at {heading}')
    return fig


def main():
    report = json.loads((OUT/'report.json').read_text())
    assert report['complete']
    assert all(m['weights_unchanged'] for m in report['models'].values())
    assert not report['gpu_tests']['failures'] and not report['gpu_tests']['errors']
    assert all(math.isfinite(loss) for rows in report['training'].values() for r in rows for loss in r['final_loss'].values())
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    pages, md, csvrows, html_sections = [], [], [], []

    def add(title, sections):
        pages.append(text_page(title, sections))
        md.extend(['', '## '+title, ''])
        blocks = ['<h2>'+html.escape(title)+'</h2>']
        for heading, content in sections:
            md.extend(['**'+heading+'**', ''])
            blocks.append('<h3>'+html.escape(heading)+'</h3>')
            if isinstance(content, tuple):
                headers, rows = content
                md.extend([table(headers, rows), ''])
                blocks.append('<table><tr>'+''.join('<th>'+html.escape(str(c))+'</th>' for c in headers)+'</tr>'+''.join(
                    '<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</table>')
            else:
                md.extend([content, ''])
                blocks.append('<p>'+html.escape(content)+'</p>')
        html_sections.append('\n'.join(blocks))

    def chart(fig, stem, caption):
        fig.savefig(OUT/(stem+'.png'), dpi=180)
        fig.savefig(OUT/(stem+'.svg'))
        pages.append(fig)
        md.extend(['', f'![{caption}]({stem}.png)', ''])
        encoded = base64.b64encode((OUT/(stem+'.png')).read_bytes()).decode()
        html_sections.append('<figure><img src="data:image/png;base64,'+encoded+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>')

    def val(row, mode, key='wall_ms'):
        return row['summary'][mode][key]['median']

    p512 = next(r for r in report['models']['pretrained']['performance'] if r['batch']==1 and r['prompt_tokens']==512)
    l512 = next(r for r in report['models']['lora']['performance'] if r['batch']==1 and r['prompt_tokens']==512)
    ftrain = next(r for r in report['training']['full'] if r['length']==512)
    ltrain = next(r for r in report['training']['lora'] if r['length']==512)
    main_findings = (
        f"At batch 1 / prompt 512 / 64 generated tokens, forced FlashAttention changes pretrained uncached generation from "
        f"{val(p512,'eager-uncached'):.1f} to {val(p512,'flash-uncached'):.1f} ms "
        f"({val(p512,'eager-uncached')/val(p512,'flash-uncached'):.2f}x). Flash plus cache takes {val(p512,'flash-cached'):.1f} ms "
        f"({val(p512,'eager-uncached')/val(p512,'flash-cached'):.2f}x versus eager without cache).\n"
        f"For the trained unmerged LoRA model, the corresponding flash-only and combined speedups are "
        f"{val(l512,'eager-uncached')/val(l512,'flash-uncached'):.2f}x and {val(l512,'eager-uncached')/val(l512,'flash-cached'):.2f}x.\n"
        f"At training batch 2 / length 512, FlashAttention speeds up full updates by "
        f"{val(ftrain,'eager')/val(ftrain,'flash'):.2f}x and LoRA updates by {val(ltrain,'eager')/val(ltrain,'flash'):.2f}x. "
        f"LoRA + flash versus full eager changes step latency by {val(ftrain,'eager')/val(ltrain,'flash'):.2f}x and lowers peak allocation by "
        f"{1-val(ltrain,'flash','peak_allocated_bytes')/val(ftrain,'eager','peak_allocated_bytes'):.1%}. "
        f"These are execution measurements, not time-to-quality results.")
    config = (f"{report['gpu']}; PyTorch {report['torch']}; CUDA {report['cuda']}; TF32 off. Job {report['job']} "
              f"on {report['node']}; run started {report['date_utc']}. GPT-2: 12 layers, width 768, 12 heads, vocabulary 50,257, context 1,024. "
              f"Inference uses BF16 weights/activations for EVERY backend. Training uses FP32 master weights, BF16 autocast and AdamW. "
              f"{report['warmups']} warmups and {report['repeats']} measured repetitions per case.")
    add('FlashAttention in minGPT: measured results', [
        ('What was run', config), ('Measured highlights', main_findings),
        ('Controls', 'Within each checkpoint, weights, prompts, token counts, precision, output-head handling and cache implementation are held fixed. Eager is the original explicit attention; math forces SDPA math; flash forces SDPA FlashAttention with no fallback. Inference backend/cache order is shuffled with fixed seeds each repetition. LoRA adapters remain unmerged.'),
        ('Scope', 'Speed measurements use deterministic random-token prompts and real pretrained/trained checkpoints. Task quality uses the existing 120 held-out sports questions on pretrained, full-finetuned and LoRA seed-42 checkpoints. Full fine-tuning shares pretrained inference architecture, so its timing is not repeated. All inference weight hashes matched before and after.')])

    fig, axs = plt.subplots(1,3,figsize=(12,4.6),layout='constrained')
    nodes = [('Q/K/V projections','LoRA adds low-rank updates here'), ('Attention kernel','Flash fuses tiled attention'), ('Decode history','KV cache reuses saved K and V')]
    for ax, (title, subtitle) in zip(axs,nodes):
        ax.axis('off')
        ax.text(.5,.83,title,ha='center',fontsize=14,weight='bold')
        ax.text(.5,.67,subtitle,ha='center',fontsize=10,wrap=True)
    axs[0].text(.5,.38,'Frozen W x\n+ B(A x) × scale\n\nAdapters remain unmerged',ha='center',va='center',fontsize=12,bbox=dict(boxstyle='round',fc='#e6eff4',ec='#b9c8d5'))
    axs[1].text(.5,.38,'Eager: store score matrix\nFlash: tiled fused CUDA kernel\n\nSame attention equation',ha='center',va='center',fontsize=12,bbox=dict(boxstyle='round',fc='#ddf0ed',ec='#a9c8c3'))
    axs[2].text(.5,.38,'Prefill: full prompt\nDecode: one new query\n\nDynamic K/V concatenation',ha='center',va='center',fontsize=12,bbox=dict(boxstyle='round',fc='#fff0d9',ec='#e4c69c'))
    fig.suptitle('Three optimizations affect different parts of the model',fontsize=16)
    chart(fig,'mechanism','LoRA, FlashAttention and KV caching act at different stages; their measured benefits need not multiply.')

    fig, axs = plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax, name in zip(axs,['pretrained','lora']):
        rows = report['models'][name]['performance']
        x = np.arange(len(rows))
        for j, mode in enumerate(['eager-uncached','flash-uncached','eager-cached','flash-cached']):
            ax.bar(x+(j-1.5)*.19,[val(r,mode) for r in rows],.19,label=mode,
                   color=['#8c96a5','#71b4b3','#bac5d3','#167b83'][j],
                   yerr=[r['summary'][mode]['wall_ms']['stdev'] for r in rows],capsize=2)
        ax.set(xticks=x,xticklabels=[f"B{r['batch']} / P{r['prompt_tokens']}" for r in rows],ylabel='Request wall time (ms)',title=name+' GPT-2: 64 new tokens')
        ax.tick_params(axis='x',rotation=25)
        ax.legend(fontsize=8)
    chart(fig,'generation-speed','Median synchronized wall latency; error bars are sample SD over seven repetitions. Lower is faster.')

    for name in ['pretrained','lora']:
        rows = report['models'][name]['performance']
        t = []
        for r in rows:
            t.append([f"{r['batch']} / {r['prompt_tokens']}", f"{val(r,'eager-uncached'):.1f}",f"{val(r,'flash-uncached'):.1f}",
                      f"{val(r,'eager-cached'):.1f}",f"{val(r,'flash-cached'):.1f}",f"{val(r,'eager-uncached')/val(r,'flash-cached'):.2f}x"])
            for mode, values in r['summary'].items():
                csvrows.append(dict(section='inference',model=name,batch=r['batch'],length=r['prompt_tokens'],mode=mode,
                                    **{k:v['median'] for k,v in values.items()},wall_ms_stdev=values['wall_ms']['stdev']))
        detail = []
        for r in rows:
            for backend in ['eager','math','flash']:
                u,c = backend+'-uncached',backend+'-cached'
                detail.append([f"{r['batch']}/{r['prompt_tokens']}",backend,f"{val(r,u):.1f}",f"{val(r,c):.1f}",
                               f"{val(r,c,'aggregate_tokens_per_second'):.1f}",f"{val(r,u)/val(r,c):.2f}x"])
        add(name.title()+': generation and cache interaction',[
            ('Median request time, milliseconds', (['Batch/prompt','Eager off','Flash off','Eager on','Flash on','Combined'],t)),
            ('All kernels: median request time and cache gain', (['B/P','Backend','Off ms','On ms','On tok/s','Cache gain'], detail)),
            ('Read the comparisons', 'Off/on means KV cache disabled/enabled. Combined divides eager-uncached by flash-cached; cache gain compares off/on at one fixed backend. Token throughput counts all generated tokens across the batch. Request time includes prefill, greedy token selection, output concatenation and all 64 generation steps; excludes model loading, tokenization, input transfer, warmup and cleanup.')])

    fig, axs = plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for col, name in enumerate(['pretrained','lora']):
        rows = report['models'][name]['performance']
        x = np.arange(len(rows))
        for backend in ['eager','math','flash']:
            axs[0,col].plot(x,[val(r,backend+'-cached','prefill_ms') for r in rows],'-o',label=backend,color=COLORS[backend])
            axs[1,col].plot(x,[val(r,backend+'-cached','decode_ms_per_token_step') for r in rows],'-o',label=backend,color=COLORS[backend])
        for row in [0,1]:
            axs[row,col].set(xticks=x,xticklabels=[f"B{r['batch']}/P{r['prompt_tokens']}" for r in rows])
            axs[row,col].tick_params(axis='x',rotation=25)
            axs[row,col].legend()
        axs[0,col].set(title=name+': prefill + first token',ylabel='CUDA event time (ms)')
        axs[1,col].set(title=name+': cached decode',ylabel='CUDA event ms per batched step')
    chart(fig,'prefill-decode','Prefill includes first-token selection/append; decode averages the remaining 63 steps. CUDA events include stream idle gaps from host dispatch.')

    kernels=json.loads((OUT/'attention-kernels.json').read_text())
    kernel_table=[]
    for row in kernels['rows']:
        b=row['backends']
        kernel_table.append([f"{row['batch']}/{row['prompt_tokens']}",row['phase'],
                             f"{b['eager']['median_us']:.2f}",f"{b['math']['median_us']:.2f}",
                             f"{b['flash']['median_us']:.2f}",f"{row['speedup']:.2f}x"])
        for backend,stats in b.items():
            csvrows.append(dict(section='attention_only',model='one_attention_layer',batch=row['batch'],
                                length=row['prompt_tokens'],mode=backend+'-'+row['phase'],
                                median_us=stats['median_us'],mean_us=stats['mean_us'],stdev_us=stats['stdev_us'],
                                incremental_peak_bytes=stats['incremental_peak_bytes']))
    add('Attention-only GPU latency',[
        ('Median microseconds per attention operation',(['B/P','Phase','Eager us','Math us','Flash us','FA gain'],kernel_table)),
        ('What is isolated',f"A separate {kernels['gpu']} allocation, job {kernels['job']} on {kernels['node']}. Synthetic BF16 Q/K/V, 12 heads, head dimension 64, contiguous head-major layout. Prefill uses square causal attention; decode uses one query and prompt+63 cached keys. Projections, LoRA branches, cache concatenation, norms, MLP, logits and sampling are excluded."),
        ('Timing method','Capture 20 attention operations in one CUDA graph; warm up five replays, measure 15 replays with CUDA events, divide by 20. Graph replay reduces per-operation Python dispatch; this is a diagnostic benchmark and is not enabled in minGPT generation. Timings include the captured attention operation and graph scheduling, not a single individual CUDA kernel. Eager/math are also graphed. Do not substitute these ratios for end-to-end model speedups.')])
    fig,axs=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for ax,phase in zip(axs,['prefill','decode']):
        selected=[r for r in kernels['rows'] if r['phase']==phase]
        x=np.arange(len(selected))
        for j,backend in enumerate(['eager','math','flash']):
            ax.bar(x+(j-1)*.24,[r['backends'][backend]['median_us'] for r in selected],.24,
                   label=backend,color=COLORS[backend])
        ax.set(xticks=x,xticklabels=[f"B{r['batch']}/P{r['prompt_tokens']}" for r in selected],
               ylabel='GPU microseconds per attention operation',title='Attention only: '+phase,yscale='log')
        ax.tick_params(axis='x',rotation=25)
        ax.legend()
    chart(fig,'attention-kernels','Attention-only CUDA graph measurements; logarithmic latency axis. Separate from full-request timings.')

    fig, axs = plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,name in zip(axs,['full','lora']):
        rows=report['training'][name]
        x=np.arange(len(rows))
        for j,backend in enumerate(['eager','math','flash']):
            ax.bar(x+(j-1)*.24,[val(r,backend) for r in rows],.24,label=backend,color=COLORS[backend],
                   yerr=[r['summary'][backend]['wall_ms']['stdev'] for r in rows],capsize=3)
        ax.set(xticks=x,xticklabels=[str(r['length']) for r in rows],xlabel='Sequence length, batch 2',ylabel='Training step wall time (ms)',title=name+' adaptation: forward + backward + AdamW')
        ax.legend()
    chart(fig,'training-speed','Mixed BF16 training with FP32 master weights and AdamW states. Median and sample SD; no convergence or quality claim.')

    training_table=[]
    for name, rows in report['training'].items():
        for r in rows:
            for backend in ['eager','math','flash']:
                values=r['summary'][backend]
                training_table.append([name,str(r['length']),backend,f"{val(r,backend):.1f}",
                                       f"{val(r,backend,'peak_allocated_bytes')/2**20:.0f}",f"{val(r,'eager')/val(r,backend):.2f}x"])
                csvrows.append(dict(section='training',model=name,batch=r['batch'],length=r['length'],mode=backend,
                                    **{k:v['median'] for k,v in values.items()},wall_ms_stdev=values['wall_ms']['stdev']))
    add('Training: cost, memory and parameter counts',[
        ('All measured training configurations',(['Method','Length','Backend','Step ms','Peak MiB','FA gain*'],training_table)),
        ('Parameter counts',f"Full adaptation trains {ftrain['parameters']['trainable']:,} parameters. LoRA trains {ltrain['parameters']['trainable']:,} "
         f"({ltrain['parameters']['trainable_percent']:.4f}% of its total). This minGPT has an untied vocabulary head, so total parameters differ from tied-head Hugging Face GPT-2. FlashAttention changes neither count. *Gain is eager divided by the selected backend within each method/length."),
        ('Training protocol','Fresh checkpoint, same seed and synthetic batch for each kernel; full adaptation starts from pretrained GPT-2, LoRA from the trained adapter. Disable dropout; run 3 warmups plus 7 measured AdamW steps at lr 1e-5. Inputs match across backends within each method; full/LoRA batches can differ in values. Steps evolve weights independently; low-precision rounding may differ. Gradients are cleared before timing and AdamW states are materialized by warmup. No checkpoints from these diagnostic steps replace the trained models.')])

    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    categories=[(m,b) for m in ['full','lora'] for b in ['eager','math','flash']]
    for j,length in enumerate([128,512,896]):
        values=[val(next(r for r in report['training'][m] if r['length']==length),b,'peak_allocated_bytes')/2**20 for m,b in categories]
        axs[0].bar(np.arange(6)+(j-1)*.24,values,.24,label=f'L={length}')
    axs[0].set(xticks=np.arange(6),xticklabels=[m+'\n'+b for m,b in categories],ylabel='Peak allocated GPU memory (MiB)',title='Training peak: weights + states + activations')
    axs[0].legend()
    for j,mode in enumerate(['eager-uncached','flash-uncached','eager-cached','flash-cached']):
        axs[1].bar(np.arange(2)+(j-1.5)*.19,[val(r,mode,'peak_allocated_bytes')/2**20 for r in [p512,l512]],.19,label=mode)
    axs[1].set(xticks=[0,1],xticklabels=['Pretrained','LoRA'],ylabel='Peak allocated GPU memory (MiB)',title='Inference peak: B1 / P512 / 64 new tokens')
    axs[1].legend(fontsize=8)
    chart(fig,'gpu-memory','PyTorch allocated memory, not whole-device memory. KV caching uses additional storage; flash reduces attention intermediates.')

    memory=[]
    for name in ['pretrained','lora']:
        for r in report['models'][name]['performance']:
            memory.append([name,f"{r['batch']}/{r['prompt_tokens']}",
                           f"{val(r,'eager-uncached','peak_allocated_bytes')/2**20:.1f}",
                           f"{val(r,'flash-uncached','peak_allocated_bytes')/2**20:.1f}",
                           f"{val(r,'flash-cached','peak_allocated_bytes')/2**20:.1f}",
                           f"{val(r,'flash-cached','cache_bytes')/2**20:.1f}"])
    add('GPU memory: savings and cache storage',[
        ('Inference peak allocation, MiB',(['Model','B/P','Eager off','Flash off','Flash on','KV bytes*'],memory)),
        ('Memory definitions','*KV bytes are shown in MiB. Peak allocation includes weights, attention-mask buffers, activations, workspaces and live caches. Logical KV storage is 2 × layers × batch × cached tokens × hidden width × 2 bytes for BF16. Final cache length is prompt + 63; the final generated token has not yet been fed back. This implementation concatenates caches dynamically, temporarily keeping old/new tensors.'),
        ('What FlashAttention changes','FlashAttention avoids materializing the full attention score/probability matrix. It does not remove model weights, LoRA weights or the K/V history needed for cached decoding. Peak allocated minus baseline measures additional live tensors in that run; reserved memory also depends on allocator history and must not be added to allocated memory. Both values and all samples are retained in JSON/CSV.')])

    quality=[]
    changed=[]
    for name,info in report['models'].items():
        for mode,q in info['quality'].items():
            quality.append([name,mode,f"{q['metrics']['primary_correct']:.1%}",f"{q['metrics']['fields_correct']:.1%}",
                            str(q['identical_all_40_tokens']),str(q['identical_until_eos'])])
            if q['identical_until_eos'] < 120:
                changed.append(f"{name}/{mode}: {120-q['identical_until_eos']} changed continuations")
    add('Task quality: every backend and cache setting',[
        ('120 held-out questions per checkpoint',(['Checkpoint','Mode','Primary','Fields','Same 40','Same EOS'],quality)),
        ('How to read it','Same 40 counts identical 40-token sequences; Same EOS compares tokens up to EOS. Both compare against eager-uncached BF16 within the SAME checkpoint. Primary/Fields are the existing transparent sports-field checks. Full outputs, token IDs, references and per-question scores are preserved in the per-checkpoint quality JSON and ALL_OUTPUTS.csv. These are seed-42 checkpoints, not three-seed quality averages.'),
        ('Numerical differences', f"Fused kernels reorder floating-point operations. LoRA retained at least {min(q['identical_until_eos'] for q in report['models']['lora']['quality'].values())}/120 identical continuations through EOS across all modes; flash+cache primary accuracy is {report['models']['lora']['quality']['flash-cached']['metrics']['primary_correct']:.1%}. Pretrained and fully finetuned continuations changed in some BF16 modes; inspect the CSV. Post-EOS differences do not affect the scored answer.")])

    errors=[]
    for name, info in report['models'].items():
        for backend,p in info['parity'].items():
            errors.append([name,backend,f"{p['max_abs_error']:.4f}",f"{p['cached_max_abs_error']:.4f}",f"{p['argmax_agreement']:.1%}"])
    old = json.loads((OUT.parents[1]/'kv_cache/results/report.json').read_text())
    q_precision=[]
    for name,info in report['models'].items():
        historical=old['models'][name]['quality']['uncached']['primary_correct']
        current=info['quality']['eager-uncached']['metrics']['primary_correct']
        flash=info['quality']['flash-cached']['metrics']['primary_correct']
        q_precision.append([name,f'{historical:.1%}',f'{current:.1%}',f'{flash:.1%}'])
    add('Correctness, precision and CUDA evidence',[
        ('Real-checkpoint logit checks',(['Checkpoint','Backend','Max vs eager','Cache max','Argmax agree'],errors)),
        ('Precision context: primary-answer score',(['Checkpoint','Earlier FP32 eager','BF16 eager','BF16 flash+KV'],q_precision)),
        ('Tests and unchanged weights',f"GPU attention suite: {report['gpu_tests']['run']} tests, {report['gpu_tests']['failures']} failures, {report['gpu_tests']['errors']} errors, {report['gpu_tests']['skipped']} skipped. Includes forced CUDA flash forward/backward parity, nonzero LoRA gradients, multi-token offset cache masks and window rollover. All inference before/after SHA256 weight hashes match. Raw BF16 logit checks use rtol .05 / atol .25 to detect gross errors; measured max errors above are more informative than that loose threshold."),
        ('Kernel selection','Flash is forced with SDPBackend.FLASH_ATTENTION; fallback backends are disabled. Separate profiler traces cover prefill + single-token cached decode for pretrained/LoRA and forward/backward AdamW at training length 512. Operator summaries and CUDA kernel names are saved in KERNEL_EVIDENCE.json. Profiled executions are excluded from timing samples.')])

    evidence={}
    trace_paths=list(OUT.glob('*trace.json'))
    for path in trace_paths:
        trace=json.loads(path.read_text())
        kernels=sorted({e.get('name','') for e in trace.get('traceEvents',[]) if e.get('cat')=='kernel'})
        evidence[path.name]=dict(cuda_kernel_count=len(kernels),
                                attention_kernels=[k for k in kernels if any(w in k.lower() for w in ['flash','fmha','softmax','attention'])],
                                all_kernel_names=kernels)
    save_path=OUT/'KERNEL_EVIDENCE.json'
    save_path.write_text(json.dumps(evidence,indent=2))
    flash_evidence=[(name,entry) for name,entry in evidence.items() if '-flash-' in name]
    assert flash_evidence and all(any('flash' in k.lower() or 'fmha' in k.lower() for k in entry['attention_kernels']) for _,entry in flash_evidence), 'Missing actual flash CUDA kernel evidence'
    cached_gains=[]
    for name in ['pretrained','lora']:
        for row in report['models'][name]['performance']:
            cached_gains.append(val(row,'eager-cached')/val(row,'flash-cached'))
    interpretation=(f"After KV caching is already enabled, flash's measured request speedup ranges from {min(cached_gains):.2f}x to {max(cached_gains):.2f}x over the ten model/shape cases. "
                    'Do not multiply standalone LoRA, flash and cache speedups. Read the combined measured configurations. Small batch-one decode can be dominated by projection, Python dispatch, sampling or cache-management costs; this is an inference from workload scaling, not an occupancy/bandwidth diagnosis.')
    add('Interpretation, limits and reproduction',[
        ('What the measurements support',interpretation),
        ('Single-GPU limits','One A30 allocation and seven repetitions per case do not establish cross-device or serving performance. Inference mode order is shuffled; training kernels are measured sequentially on fresh models, so slow thermal/load drift could affect training ratios. Timings include CPU dispatch and GPU execution but exclude setup, cleanup, transfers and evaluation. CUDA-event intervals include idle stream gaps; summed kernel times are separate profiler evidence.'),
        ('Context and LoRA limits','All timing cases remain inside the 1,024-token context. At context overflow this learned-absolute-position model rebuilds its cache to preserve behavior, losing incremental reuse. LoRA adapters are active and unmerged, so inference pays their projection overhead. The training run measures ten diagnostic updates per configuration, not adaptation convergence, final quality or time to target loss.'),
        ('Precision and prior results','The earlier KV-cache report used FP32; this report uses BF16 inference. Compare kernels only within the same precision here. The historical quality table is context, not a controlled precision experiment. Training uses FP32 parameters/optimizer states and BF16 autocast, a different memory footprint from native-BF16 inference.'),
        ('Reproduce', 'From the minGPT root: sbatch experiments/flash_attention/run_gpu.sh. Queue run_kernels.sh with --dependency=afterok:JOB_ID, then run python experiments/flash_attention/build_report.py. The runner refuses to overwrite an existing report.json. Source hashes, job/node/software versions, raw samples, numerical checks, token outputs, profiler traces and parameter hashes are saved. Code snapshot files accompany the measured run; use them to reproduce the source hashes.'),
        ('Artifacts','FLASH_ATTENTION_REPORT.pdf is the presentation report. RESULTS.md and standalone RESULTS.html contain these tables and charts. METRICS.csv contains every summary metric; ALL_OUTPUTS.csv contains all task responses. report.json plus component JSON files retain raw measurements.')])

    all_outputs=[]
    for name in report['models']:
        q=json.loads((OUT/f'{name}-quality.json').read_text())
        questions=q['questions']
        ref=q['outputs']['eager-uncached']
        for mode, rows in q['outputs'].items():
            for r, baseline in zip(rows,ref):
                source=questions[r['index']]
                all_outputs.append(dict(model=name,mode=mode,index=r['index'],question=source['question'],reference=source['answer'],
                                        output=r['text'],same_all_tokens=r['tokens']==baseline['tokens'],same_until_eos=r['tokens_until_eos']==baseline['tokens_until_eos'],
                                        **r['scores']))
    for filename,rows in [('METRICS.csv',csvrows),('ALL_OUTPUTS.csv',all_outputs)]:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with (OUT/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (OUT/'RESULTS.md').write_text('\n'.join(md))
    (OUT/'PRESENTATION.md').write_text(
        '# FlashAttention: presentation notes\n\n'+main_findings+'\n\n'+interpretation+
        '\n\nShow generation-speed.png first, then attention-kernels.png to distinguish kernel and full-model gains. '
        'Use training-speed.png and gpu-memory.png for adaptation costs. The task-quality table discloses all BF16 output changes. '
        'Every comparison here uses the same precision within its phase; do not compare the older FP32 timing report directly. '
        'Flash was forced and verified in real CUDA traces. These are one-GPU measurements, not convergence or production-serving claims.\n')
    (OUT/'RESULTS.html').write_text('<!doctype html><html><meta charset="utf-8"><title>FlashAttention in minGPT</title><style>body{font:16px/1.6 system-ui;max-width:1120px;margin:40px auto;padding:0 24px;color:#17252d}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px;border:1px solid #d5dfe6;text-align:right}th{background:#e6eff4}img{width:100%}h2{margin-top:48px}figcaption{color:#52616c}</style><body><h1>FlashAttention in minGPT: measured results</h1>'+''.join(html_sections)+'</body></html>')
    with PdfPages(OUT/'FLASH_ATTENTION_REPORT.pdf') as pdf:
        for i, fig in enumerate(pages):
            fig.text(.98,.015,f'{i+1} / {len(pages)}',ha='right',size=8,color='#667780')
            pdf.savefig(fig)
            fig.savefig(OUT/f'page-{i+1}.png',dpi=110)
            plt.close(fig)
    print(main_findings)
    print(interpretation)
    print(f'Created {len(pages)} PDF pages and {len(all_outputs)} output rows in {OUT}')


if __name__=='__main__':
    main()
