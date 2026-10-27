"""Measured FP32 tiled attention, precision bridge, quality and kernel evidence."""
import base64
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from experiments.flash_attention.build_report import text_page,table

OUT=Path(__file__).resolve().parent/'results'
LABELS={'fp32_eager':'FP32 eager','fp32_tiled':'FP32 tiled (IEEE dots)',
        'fp32_efficient_fastf32':'FP32 efficient (FastF32)', 'bf16_eager':'BF16 eager','bf16_flash':'BF16 flash'}
COLORS=['#8793a5','#247c9a','#d39432','#9abcb6','#187c65']


def main():
    report=json.loads((OUT/'report.json').read_text())
    assert report['complete'] and report['tests']['failures']==report['tests']['errors']==0
    variants=list(report['variants'])
    same_fp32=0
    for name,models in report['models'].items():
        assert len({models[v]['weights_before'] for v in ['fp32_eager','fp32_tiled','fp32_efficient_fastf32']})==1
        assert models['bf16_eager']['weights_before']==models['bf16_flash']['weights_before']
        reference=models['fp32_eager']['quality']['outputs']['uncached']
        same_fp32+=sum(row['tokens_until_eos']==ref['tokens_until_eos']
                       for rows in models['fp32_tiled']['quality']['outputs'].values()
                       for row,ref in zip(rows,reference))
    figures=[];markdown=['# FP32 tiled attention in minGPT',''];htmlparts=[];metrics=[]
    def add(title,sections):
        figures.append(text_page(title,sections))
        markdown.extend(['','## '+title,''])
        blocks=['<h2>'+html.escape(title)+'</h2>']
        for heading,content in sections:
            markdown.extend(['**'+heading+'**',''])
            blocks.append('<h3>'+html.escape(heading)+'</h3>')
            if isinstance(content,tuple):
                headers,rows=content
                markdown.extend([table(headers,rows),''])
                blocks.append('<table><tr>'+''.join('<th>'+html.escape(str(c))+'</th>' for c in headers)+'</tr>'+''.join(
                    '<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</table>')
            else:
                markdown.extend([content,''])
                blocks.append('<p>'+html.escape(content)+'</p>')
        htmlparts.append(''.join(blocks))
    def chart(fig,name,caption):
        fig.savefig(OUT/(name+'.png'),dpi=180)
        fig.savefig(OUT/(name+'.svg'))
        figures.append(fig)
        markdown.extend(['',f'![{caption}]({name}.png)',''])
        encoded=base64.b64encode((OUT/(name+'.png')).read_bytes()).decode()
        htmlparts.append('<figure><img src="data:image/png;base64,'+encoded+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>')
    def case(name,variant,batch=1,length=896):
        return next(r for r in report['models'][name][variant]['performance'] if r['batch']==batch and r['prompt_tokens']==length)
    def value(name,variant,mode='uncached',key='wall_ms',batch=1,length=896):
        return case(name,variant,batch,length)['summary'][mode][key]['median']
    def gain(name,first,second,mode='uncached',batch=1,length=896):
        return value(name,first,mode,batch=batch,length=length)/value(name,second,mode,batch=batch,length=length)
    p=report['models']['pretrained']['fp32_tiled']['logits']
    findings=(f"At batch 1 / prompt 896 / 64 generated tokens, pretrained FP32 eager takes {value('pretrained','fp32_eager'):.1f} ms; "
              f"custom FP32 tiled attention takes {value('pretrained','fp32_tiled'):.1f} ms ({gain('pretrained','fp32_eager','fp32_tiled'):.2f}x). "
              f"With caching already enabled, that ratio is {gain('pretrained','fp32_eager','fp32_tiled','cached'):.2f}x.\n"
              f"The same-GPU precision bridge at this shape gives {gain('pretrained','fp32_eager','bf16_eager'):.2f}x for FP32 eager to BF16 eager, "
              f"{gain('pretrained','bf16_eager','bf16_flash'):.2f}x for BF16 eager to BF16 flash, and "
              f"{gain('pretrained','fp32_eager','bf16_flash'):.2f}x for the combined change.\n"
              f"The custom FP32 model's max logit difference from FP32 eager is {p['max_abs_error']:.6f}; its implementation avoids "
              f'the full score matrix but is an untuned inference prototype. It matched FP32 eager tokens through EOS on {same_fp32}/480 comparisons across both checkpoints and cache settings.')
    add('FP32 tiled attention: measured results',[
        ('What was built','A Triton tiled-attention inference kernel using online softmax, FP32 Q/K/V, FP32 state/output, and explicit input_precision="ieee" for BOTH matrix products. It supports causal prefill, one-token decoding, and offset multi-token KV-cache chunks. No BF16/FP16 conversions or TF32 dot products are used. Approximate GPU FP32 exponential and changed reduction order mean outputs need not be bitwise identical.'),
        ('Measured findings',findings),
        ('Hardware and protocol',f"A100 attempts produced no measurements: the first allocation raced cancellation; a follow-up remained pending with a roughly two-hour estimate. Main job {report['job']} ran on {report['gpu']} / {report['node']}. PyTorch {report['torch']}, CUDA {report['cuda']}, Triton {report['triton']}. Started {report['started_utc']}. {report['warmups']} warmups and {report['repeats']} measured repetitions. TF32 flags disabled and NVIDIA_TF32_OVERRIDE=0. One resident model gives direct allocation peaks. PyTorch uses one CPU intra-op thread within a four-CPU allocation."),
        ('Scope','Real pretrained GPT-2 and the trained seed-42 unmerged LoRA checkpoint; 120 held-out sports questions each. GPT-2 has 12 layers, width 768, 12 heads, head width 64, and a 1,024-token context. minGPT uses an untied output head. Older experiment artifacts are preserved separately.')])

    backendrows=[['FP32 eager','FP32','Original separate matmul/mask/softmax'],
                 ['FP32 tiled','FP32 IEEE dots','Custom online-softmax prototype'],
                 ['FP32 efficient','FP32 / FastF32','PyTorch forced efficient backend'],
                 ['BF16 eager','BF16','Precision-change reference'],
                 ['BF16 flash','BF16','PyTorch forced FlashAttention']]
    add('What FP32 means in this experiment',[
        ('Five measured variants',(['Variant','Arithmetic / inputs','Role'],backendrows)),
        ('Important finding about PyTorch efficient attention','The installed CUDA source selects CUTLASS OpMultiplyAddFastF32 and a Tensor Core OpClass for FP32 on SM80+. That is an FP32-input memory-efficient backend with its own arithmetic, not our IEEE-dot reference. Global cuBLAS TF32 flags do not establish how this separately compiled kernel multiplies values. We keep it clearly labeled and do not claim its speedup demonstrates pure IEEE FP32 execution.'),
        ('Custom-kernel evidence','The prototype explicitly requests IEEE dots. Compiled PTX, cubins and SASS disassembly are saved; validation requires fma.rn.f32 and rejects mma.sync and .tf32. Float64 attention references check head widths 8/64/128, odd sequence lengths, prefill, and cached chunks. Tests also cover the real fused-QKV strides, active LoRA updates, context-window rollover and unsupported backward.'),
        ('What this does not claim','This is not the official FlashAttention-2 CUDA backend operating in FP32. It is our FP32 implementation of the tiled online-softmax algorithm. It avoids the quadratic score matrix but uses FP32 CUDA-core multiplication rather than the BF16 Tensor Core path. Forward accuracy, memory reduction and speed must be evaluated separately. The custom kernel has no backward or dropout implementation.')])

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    krows=[]
    for ax,phase in zip(axs,['prefill','decode']):
        selected=[r for r in report['kernels'] if r['phase']==phase]
        x=np.arange(len(selected))
        for j,v in enumerate(variants):
            ax.bar(x+(j-2)*.16,[r['backends'][v]['median_us'] for r in selected],.16,label=LABELS[v],color=COLORS[j])
        ax.set(xticks=x,xticklabels=[str(r['length']) for r in selected],xlabel='Prompt length; batch 1',ylabel='GPU microseconds / attention operation (log)',yscale='log',title=phase+' attention only')
        ax.legend(fontsize=7)
        for r in selected:
            b=r['backends']
            krows.append([r['length'],phase]+[f"{b[v]['median_us']:.2f}" for v in variants])
            for v,s in b.items():
                metrics.append(dict(section='attention_only',model='one_layer',variant=v,batch=1,length=r['length'],mode=phase,median_us=s['median_us']))
    chart(fig,'attention-kernels','Attention-only GPU latency; lower is faster. The custom prototype is not uniformly faster, particularly for one-token decode.')
    add('Attention-only latency and interpretation',[
        ('Median microseconds per operation',(['Length','Phase','FP32 eager','FP32 tiled','Efficient','BF16 eager','BF16 flash'],krows)),
        ('Timing method','Synthetic identical FP32 Q/K/V values (cast only for BF16 variants), 12 heads, width 64. Decode has one query and prompt+63 keys. CUDA graph captures 20 operations; warm for at least 0.1 seconds and measure 15 replays with CUDA events, dividing by 20. Compilation, projections, MLP, output head, sampling, and cache concatenation are excluded. These graph speedups are not full-model generation speedups.'),
        ('Why decode can lose','The prototype uses a 16-query tile even for one query and processes K/V blocks serially inside each program. Short decode workloads can pay more launch/loop overhead than eager. This is an implementation-based explanation, not a measured occupancy diagnosis. Future work would specialize one-token decoding and tune tiles; these reported measurements use the original fixed 16x32 tile.')])

    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,name in zip(axs,['pretrained','lora']):
        rows=report['models'][name]['fp32_eager']['performance']
        x=np.arange(len(rows))
        for j,v in enumerate(variants):
            ys=[value(name,v,batch=r['batch'],length=r['prompt_tokens']) for r in rows]
            ax.bar(x+(j-2)*.16,ys,.16,label=LABELS[v],color=COLORS[j],
                   yerr=[case(name,v,r['batch'],r['prompt_tokens'])['summary']['uncached']['wall_ms']['stdev'] for r in rows],capsize=2)
        ax.set(xticks=x,xticklabels=[f"B{r['batch']}/P{r['prompt_tokens']}" for r in rows],ylabel='Median request wall time (ms)',title=('LoRA' if name=='lora' else 'Pretrained')+': 64 new tokens, no KV cache')
        ax.legend(fontsize=7)
    chart(fig,'generation-speed','Full-model generation: median and sample SD across seven repetitions. Five variants separate precision and backend effects.')
    for name in ['pretrained','lora']:
        rows=[]
        for v in variants:
            rows.append([LABELS[v],f"{value(name,v):.1f}",f"{value(name,v,'cached'):.1f}",
                         f"{value(name,v)/value(name,v,'cached'):.2f}x",f"{value(name,v,'cached','aggregate_tokens_per_second'):.1f}"])
        details=[]
        for ref in report['models'][name]['fp32_eager']['performance']:
            b,l=ref['batch'],ref['prompt_tokens']
            details.append([f'{b}/{l}',f"{gain(name,'fp32_eager','fp32_tiled',batch=b,length=l):.2f}x",
                            f"{gain(name,'fp32_eager','bf16_eager',batch=b,length=l):.2f}x",
                            f"{gain(name,'bf16_eager','bf16_flash',batch=b,length=l):.2f}x",
                            f"{gain(name,'fp32_eager','bf16_flash',batch=b,length=l):.2f}x"])
        add(('LoRA' if name=='lora' else 'Pretrained')+': model speed, precision and KV cache',[
            ('B1 / P896 / 64 generated tokens',(['Variant','No cache ms','Cache ms','Cache gain','Cache tok/s'],rows)),
            ('No-cache speedup attribution at every shape',(['Batch/prompt','FP32 tiled','BF16 only','FA at BF16','Combined'],details)),
            ('Timing definitions','FP32 tiled compares FP32 eager / custom tiled at fixed precision. BF16 only compares FP32 eager / BF16 eager. FA at BF16 compares BF16 eager / BF16 flash. Combined compares FP32 eager / BF16 flash. Cache gain is uncached / cached at fixed variant. Ratios below one mean a slowdown.'),
            ('Controls and limitations','Same checkpoints, deterministic prompts, output count, greedy selection and final-token-only vocabulary projection. Cache order alternates each repetition. Variants run in recorded sequential blocks so only one model is resident; backend order is not interleaved and slow clock/load drift can affect ratios. Request time includes prefill, sampling and all 64 steps, excludes cleanup, compilation, loading, tokenization, transfers and warmups. Seven repetitions characterize short-term variation, not cross-device uncertainty.')])

    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    memory=[]
    for ax,name in zip(axs,['pretrained','lora']):
        x=np.arange(len(variants))
        for j,mode in enumerate(['uncached','cached']):
            ys=[value(name,v,mode,'peak_allocated_bytes')/2**20 for v in variants]
            ax.bar(x+(j-.5)*.35,ys,.35,label=mode)
        ax.set(xticks=x,xticklabels=['FP32\neager','FP32\ntiled','FP32\nefficient','BF16\neager','BF16\nflash'],ylabel='Peak allocated GPU MiB',title=('LoRA' if name=='lora' else 'Pretrained')+': B1 / P896 / 64 new tokens')
        ax.legend()
        for v in variants:
            memory.append([name,v.replace('fp32_efficient_fastf32','efficient'),
                           f"{value(name,v,'uncached','peak_allocated_bytes')/2**20:.1f}",
                           f"{value(name,v,'cached','peak_allocated_bytes')/2**20:.1f}",
                           f"{value(name,v,'cached','cache_bytes')/2**20:.1f}"])
    chart(fig,'gpu-memory','Direct measured model peaks. Precision changes model/cache storage; tiling primarily changes attention intermediates.')
    add('Memory and cache accounting',[
        ('B1 / P896: peak allocated MiB and final KV MiB',(['Model','Variant','No cache','Cache','KV'],memory)),
        ('What is counted','Live PyTorch tensors including weights, legacy causal-mask buffers, inputs, logits, activations and caches. Reserved memory and baseline allocations are retained in raw JSON; reserved and allocated must not be added. Triton compiler CPU memory and driver/device overhead are outside this allocator metric. No other model is resident. Legacy causal-mask buffers remain allocated in every backend; tiled attention masks arithmetically and does not use them.'),
        ('Interpretation','BF16 changes weights, activations and KV element width. FP32 tiled versus FP32 eager isolates the algorithm/implementation change without halving tensor element size. FP32 cache bytes are twice BF16 for the same shape. Dynamic cache concatenation may keep old and new allocations alive temporarily. Final cache length is prompt+63; all cases stay within the 1,024-token context.')])

    training=[]
    for row in report['training']:
        s=row['summary']
        training.append([row['method'],row['length'],row['variant'].replace('fp32_efficient_fastf32','efficient'),
                         f"{s['wall_ms']['median']:.1f}",f"{s['peak_allocated_bytes']['median']/2**20:.0f}"])
        metrics.append(dict(section='training',model=row['method'],variant=row['variant'],batch=row['batch'],length=row['length'],mode='step',
                            **{k:v['median'] for k,v in s.items()},wall_ms_stdev=s['wall_ms']['stdev']))
    add('Training controls: prototype has no backward',[
        ('Forward + backward + AdamW, batch 2',(['Method','Length','Variant','Step ms','Peak MiB'],training)),
        ('What these rows mean','FP32 eager and efficient use FP32 parameters and FP32 inputs; efficient retains the FastF32 caveat. BF16 rows use FP32 master parameters/AdamW states and BF16 autocast. Full starts from pretrained GPT-2; LoRA starts from the selected adapter. Each kernel starts fresh with the same seed and the same synthetic batch, including across full/LoRA at a fixed length. Disable dropout; lr 1e-5, three warmups and seven measured updates.'),
        ('Custom FP32 limit','There is no tiled_fp32 training row. The prototype explicitly raises when gradients are requested; it does not silently substitute eager backward. These diagnostic updates measure cost, not convergence, time-to-quality, or newly trained task accuracy. The learned checkpoints used for inference are not overwritten.')])

    alloutputs=[]
    questions=[json.loads(s) for s in (ROOT/'experiments/sports/test.jsonl').read_text().splitlines()]
    for name in ['pretrained','lora']:
        qrows=[];numerics=[]
        ref=report['models'][name]['fp32_eager']['quality']['outputs']['uncached']
        for v in variants:
            info=report['models'][name][v]
            assert info['weights_before']==info['weights_after']
            numerics.append([v.replace('fp32_efficient_fastf32','efficient'),f"{info['logits']['max_abs_error']:.6f}",
                             f"{info['logits']['cache_max_abs_error']:.6f}",f"{info['logits']['argmax_agreement']:.1%}"])
            for mode,rows in info['quality']['outputs'].items():
                same=sum(r['tokens_until_eos']==b['tokens_until_eos'] for r,b in zip(rows,ref))
                m=info['quality']['metrics'][mode]
                qrows.append([v.replace('fp32_efficient_fastf32','efficient'),mode,f"{m['primary_correct']:.1%}",f"{m['fields_correct']:.1%}",f'{same}/120'])
                for r,b in zip(rows,ref):
                    q=questions[r['index']]
                    alloutputs.append(dict(model=name,variant=v,mode=mode,index=r['index'],question=q['question'],reference=q['answer'],output=r['text'],
                                           identical_until_eos=r['tokens_until_eos']==b['tokens_until_eos'],**r['scores']))
            for row in info['performance']:
                for mode,s in row['summary'].items():
                    metrics.append(dict(section='inference',model=name,variant=v,batch=row['batch'],length=row['prompt_tokens'],mode=mode,
                                        **{k:c['median'] for k,c in s.items()},wall_ms_stdev=s['wall_ms']['stdev']))
        add(('LoRA' if name=='lora' else 'Pretrained')+': correctness and held-out answers',[
            ('120 questions; compare tokens to FP32 eager uncached',(['Variant','Cache','Primary','Fields','Same EOS'],qrows)),
            ('Real-model logits versus FP32 eager',(['Variant','Max error','Cache error','Argmax agree'],numerics)),
            ('Meaning','FP32 logit and chunked-cache checks use combined atol=rtol=0.0003. Direct kernel tests also compare against float64. BF16 differences are measured, not asserted identical. Primary and Fields use the existing transparent sports scoring, not a broad language-quality metric. Small rounding differences can change greedy continuation. All 2,400 per-question outputs and comparisons are retained; inference parameter hashes match before/after for every variant.')])

    evidence={}
    for path in OUT.glob('*trace.json'):
        data=json.loads(path.read_text())
        evidence[path.name]=sorted({e.get('name','') for e in data.get('traceEvents',[]) if e.get('cat')=='kernel'})
    (OUT/'KERNEL_EVIDENCE.json').write_text(json.dumps(evidence,indent=2))
    for path in ['tiled-prefill.ptx','tiled-decode.ptx']:
        text=(OUT/path).read_text().lower()
        assert 'fma.rn.f32' in text and 'mma.sync' not in text and '.tf32' not in text
    audit=json.loads((OUT/'INSTRUCTION_AUDIT.json').read_text())
    assert all(x['tensor_mma_instructions']==0 and x['opcode_counts']['FFMA']>0 for x in audit.values())
    add('Evidence, limitations and reproduction',[
        ('GPU validation',f"{report['tests']['run']}-test attention validation suite passed on the GPU worker; no failures/errors, {report['tests']['skipped']} skipped. The custom PTX assertions and binary FFMA/Tensor-Core instruction audit passed for prefill/decode. Installed efficient-kernel sources document FastF32 selection. CUDA traces, PTX, cubins and disassembly are retained."),
        ('Performance limits','One A30 allocation, sequential backend blocks, fixed tiles and limited repetitions. Attention-only graphs reduce host dispatch but are not used in actual generation. A forward FP32 prototype can save attention intermediate storage and still lose performance, especially on decode. No A100 measurements were collected. An optimized implementation would need tile tuning, specialized decode and a tested backward kernel.'),
        ('Precision limits','IEEE here describes FP32 multiplication/accumulation in the custom matrix products, verified in PTX. Online softmax uses standard GPU FP32 approximate exponentials; reduction order differs from eager. This does not promise bitwise equality or perfectly rounded transcendental functions. PyTorch efficient remains labeled FastF32 and must not be grouped with the IEEE-dot result.'),
        ('Reproduction','From the minGPT root: sbatch experiments/fp32_attention/run_gpu.sh. Then run python experiments/fp32_attention/build_report.py in an environment with Matplotlib and NumPy. The runner refuses to overwrite existing report.json. Source snapshots, hashes and the five-variant order are retained. Older experiment results remain untouched.'),
        ('Artifacts','FP32_ATTENTION_REPORT.pdf; RESULTS.md; standalone RESULTS.html; METRICS.csv; ALL_OUTPUTS.csv; raw report.json plus per-variant samples; compiler PTX/cubins and CUDA traces. The selected kernel is model.set_attention_backend("tiled_fp32"); use eval() and inference_mode(). The built-in alternative is "efficient", with the FastF32 caveat.')])

    for filename,rows in [('METRICS.csv',metrics),('ALL_OUTPUTS.csv',alloutputs)]:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (OUT/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    (OUT/'RESULTS.md').write_text('\n'.join(markdown))
    (OUT/'PRESENTATION.md').write_text('# FP32 attention: measured findings\n\n'+findings+'\n\nKeep the custom FP32 prototype, PyTorch FastF32 efficient attention, and BF16 FlashAttention distinctly labeled. The custom kernel has no backward. See attention-kernels.png, generation-speed.png and gpu-memory.png, then the per-checkpoint answer tables.\n')
    (OUT/'RESULTS.html').write_text('<!doctype html><html><meta charset="utf-8"><title>FP32 tiled attention</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#17252d}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px;border:1px solid #d5dfe6}th{background:#e6eff4}img{width:100%}h2{margin-top:40px}</style><body><h1>FP32 tiled attention in minGPT</h1>'+''.join(htmlparts)+'</body></html>')
    with PdfPages(OUT/'FP32_ATTENTION_REPORT.pdf') as pdf:
        for i,fig in enumerate(figures):
            fig.text(.98,.015,f'{i+1} / {len(figures)}',ha='right',size=8,color='#667780')
            pdf.savefig(fig);fig.savefig(OUT/f'page-{i+1}.png',dpi=110);plt.close(fig)
    print(findings)
    print('Created',len(figures),'pages;',len(alloutputs),'quality output rows')


if __name__=='__main__':
    main()
