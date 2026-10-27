"""Render paired FP16/FP32 inference and validated FP32 backward measurements."""
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
VARIANTS=['fp32_eager','fp32_tiled','fp16_eager','fp16_flash']
LABELS={'fp32_eager':'FP32 eager','fp32_tiled':'FP32 tiled','fp16_eager':'FP16 AMP eager','fp16_flash':'FP16 AMP Flash'}
COLORS=['#8b96a8','#287e9b','#c9aa74','#238570']


def main():
    r=json.loads((OUT/'report.json').read_text())
    assert r['complete']
    assert r['tests']['failures']==r['tests']['errors']==r['tests']['skipped']==0
    for name,info in r['models'].items():
        assert info['weight_hash_before']==info['weight_hash_after_inference']==info['weight_hash_after_training_reset']
        for v in VARIANTS: assert info['gradient_check'][v]['gradients_finite']
    for rows in r['training'].values():
        for row in rows:
            for v,updates in row['updates'].items():
                assert len(updates)==7
                assert all(x['successful'] and x['gradients_finite'] for x in updates)
                if v.startswith('fp16'): assert all(x['loss_scale']==128 for x in updates)
    figures=[];md=['# Main FP16/FP32 attention comparison',''];parts=[];metrics=[]
    def add(title,sections):
        figures.append(text_page(title,sections));md.extend(['','## '+title,''])
        blocks=['<h2>'+html.escape(title)+'</h2>']
        for heading,content in sections:
            md.extend(['**'+heading+'**','']);blocks.append('<h3>'+html.escape(heading)+'</h3>')
            if isinstance(content,tuple):
                headers,rows=content;md.extend([table(headers,rows),''])
                blocks.append('<table><tr>'+''.join('<th>'+html.escape(str(c))+'</th>' for c in headers)+'</tr>'+''.join(
                    '<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</table>')
            else:
                md.extend([content,'']);blocks.append('<p>'+html.escape(content)+'</p>')
        parts.append(''.join(blocks))
    def chart(fig,name,caption):
        fig.savefig(OUT/(name+'.png'),dpi=180);fig.savefig(OUT/(name+'.svg'));figures.append(fig)
        md.extend(['',f'![{caption}]({name}.png)',''])
        encoded=base64.b64encode((OUT/(name+'.png')).read_bytes()).decode()
        parts.append('<figure><img src="data:image/png;base64,'+encoded+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>')
    def row(name,batch=1,length=896):
        return next(x for x in r['models'][name]['inference'] if x['batch']==batch and x['prompt_tokens']==length)
    def val(name,v,cache=False,key='wall_ms',batch=1,length=896):
        return row(name,batch,length)['summary'][v+('-cached' if cache else '-uncached')][key]['median']
    def gain(name,a,b,cache=False,batch=1,length=896):
        return val(name,a,cache,batch=batch,length=length)/val(name,b,cache,batch=batch,length=length)
    def tr(method,length=896): return next(x for x in r['training'][method] if x['length']==length)
    def tv(method,v,length=896,key='wall_ms'): return tr(method,length)['summary'][v][key]['median']

    add('First: what the backend names mean',[
        ('One PyTorch model; different attention implementations',([
            'Name','Execution','Role'],[
            ['Eager','Separate matmul/mask/softmax','Original minGPT baseline'],
            ['Tiled FP32','Custom Triton online-softmax tiles','FP32 forward AND backward'],
            ['Flash','Forced PyTorch FlashAttention','FP16 attention in this run'],
            ['Math','PyTorch SDPA math reference','Validation; not main speed baseline'],
            ['Efficient','Separate PyTorch SDPA backend','Prior FP32 FastF32 experiment']])),
        ('Same equation; different execution','All compute softmax(Q K^T / sqrt(head_width) + causal_mask) V. Eager materializes the score/probability matrices. Tiled attention processes blocks with running softmax state; its backward recomputes block probabilities. Flash uses a heavily optimized fused CUDA implementation of tiled attention. Eager also executes CUDA kernels: it does not mean CPU.'),
        ('Why these are distinct','Tiled FP32 is our custom implementation, not the official FlashAttention CUDA backend with its dtype check removed. Our dot products request IEEE FP32. PyTorch FlashAttention here receives FP16 Q/K/V. Internal accumulation and rounding can differ even with matched input precision.'),
        ('Why math and efficient are not the main comparison','Math is a useful reference, but FP16 math normally uses FP32 intermediates and can be slower/more memory-intensive than eager. The installed efficient FP32 kernel uses a FastF32 Tensor Core specialization; FP32 tensors alone do not establish IEEE FP32 multiplication. Neither backend is used as the main performance baseline here.')])
    add('Fair comparisons and precision policy',[
        ('The four comparisons',(['Ratio (baseline / candidate)','What it measures'],[
            ['FP32 eager / FP32 tiled','Implementation at fixed FP32'],
            ['FP16 eager / FP16 Flash','Flash backend at fixed FP16 policy'],
            ['FP32 eager / FP16 eager','Precision-policy change'],
            ['FP32 eager / FP16 Flash','Combined practical change']])),
        ('Identical master parameters','All four main variants use the SAME FP32 model and checkpoint. FP16 rows enable FP16 autocast for both eager and Flash. Inference therefore measures AMP FP16 with FP32 master weights, not native FP16 model storage. Weight-casting/cache costs are included. Holding master weights fixed avoids quantizing model storage between paired requests.'),
        ('Training policy','All rows use FP32 master parameters and AdamW states. Both FP16 variants use identical GradScaler settings: initial scale 128, large growth interval. Every recorded step must have finite gradients, scale 128 and exactly one optimizer update. FP32-to-FP16 ratios include required casting/scaling overhead; they are not pure hardware-bitwidth ratios.'),
        ('Historical results','The earlier BF16 report used native BF16 inference weights; the earlier FP32 tiled report had forward only and sequential backend blocks. Those results remain valid historical measurements, but must not be relabeled FP16 or pooled with this different paired AMP protocol. Native FP16 serving could have different latency and memory.')])

    full_e=tv('full','fp32_eager');full_t=tv('full','fp32_tiled')
    highlights=(f"At batch 1 / prompt 896 / 64 generated tokens, pretrained FP32 eager takes {val('pretrained','fp32_eager'):.1f} ms; "
                f"FP32 tiled takes {val('pretrained','fp32_tiled'):.1f} ms ({gain('pretrained','fp32_eager','fp32_tiled'):.2f}x). "
                f"FP16 AMP eager takes {val('pretrained','fp16_eager'):.1f} ms and FP16 Flash takes {val('pretrained','fp16_flash'):.1f} ms "
                f"({gain('pretrained','fp16_eager','fp16_flash'):.2f}x at fixed FP16 policy).\n"
                f"At training batch 2 / length 896, full FP32 eager takes {full_e:.1f} ms and tiled FP32 {full_t:.1f} ms "
                f"({full_e/full_t:.2f}x); peaks are {tv('full','fp32_eager',key='peak_allocated_bytes')/2**20:.0f} and "
                f"{tv('full','fp32_tiled',key='peak_allocated_bytes')/2**20:.0f} MiB. A ratio below one means slower, even if memory improves.")
    add('Measured findings and experiment scope',[
        ('Measured results',highlights),
        ('Hardware',f"{r['gpu']}; job {r['job']} on {r['node']}. PyTorch {r['torch']}, CUDA {r['cuda']}, Triton {r['triton']}. Started {r['started_utc']}. TF32 disabled. Same allocation, one resident model, one CPU intra-op thread. Three warmups and seven measured repetitions per mode/case."),
        ('Inference controls','Pretrained GPT-2 and trained seed-42 unmerged LoRA. B1/P128, B1/P512, B1/P896 and B4/P128; cache off/on; 64 new tokens with no early EOS stopping. All eight precision/backend/cache modes are shuffled each repetition. Timings include prefill, selection/append and decode; exclude loading, transfers, compilation, warmups and cleanup.'),
        ('Training controls','Batch 2; lengths 128/512/896. Every individual trial resets original weights, zero AdamW moments and step=0; same batch across methods and variants. State copies/reset and gradient clearing are outside timing; forward, backward, loss scaling and AdamW update are inside. Variants are shuffled every repetition. This measures reset-step cost, not convergence or time to target quality.')])

    correctness=[]
    for name,info in r['models'].items():
        for v in VARIANTS:
            c=info['gradient_check'][v]
            correctness.append([name,v,f"{c['loss_abs_error']:.2e}",f"{c['gradient_relative_l2']:.2e}",f"{c['gradient_max_abs']:.2e}"])
    successful=sum(len(u) for rows in r['training'].values() for rr in rows for u in rr['updates'].values())
    add('FP32 backward: implementation and validation',[
        ('What changed','Forward saves Q/K/V, output and one FP32 log-normalizer per query. Backward forms delta=sum(output*doutput), then recomputes probabilities in tiles to accumulate dQ, dK and dV. No globally materialized quadratic score/probability matrix is saved or recomputed. Key/value tiles own their gradients without atomics. Noncontiguous-input/upstream-gradient copies are linear-size and included in measured costs.'),
        ('Real-checkpoint gradients versus FP32 eager',(['Model','Variant','Loss error','Grad rel L2','Grad max abs'],correctness)),
        ('Validation evidence',f"{r['tests']['run']}-method attention suite passed on the GPU worker with no failures/errors/skips. It includes float64 forward/gradient references, odd/cached lengths, noncontiguous layouts, real model/LoRA gradients, four AdamW updates, linear saved-state checks and compiled forward/backward FP32 precision. All {successful} measured optimizer steps were successful and finite; no overflow-skipped steps were counted."),
        ('Current limits','First-order gradients only; no higher-order derivatives or attention dropout. Dropout is zero in every measured variant. FP32 GPU exp/log approximations and reduction order prevent a bitwise-equivalence promise. The kernel is a fixed-tile prototype: implementing backward enables a legitimate training comparison, not an assumption of a speed win.')])

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,name in zip(axs,['pretrained','lora']):
        cases=r['models'][name]['inference'];x=np.arange(len(cases))
        for j,v in enumerate(VARIANTS):
            key=v+'-uncached'
            ax.bar(x+(j-1.5)*.2,[z['summary'][key]['wall_ms']['median'] for z in cases],.2,label=LABELS[v],color=COLORS[j],
                   yerr=[z['summary'][key]['wall_ms']['stdev'] for z in cases],capsize=2)
        ax.set(xticks=x,xticklabels=[f"B{z['batch']}/P{z['prompt_tokens']}" for z in cases],ylabel='Request wall latency (ms)',title=('LoRA' if name=='lora' else 'Pretrained')+': no cache / 64 new tokens')
        ax.legend(fontsize=8)
    chart(fig,'inference-speed','Median synchronized wall request time; error bars are sample SD over seven repetitions. AMP FP16 includes its casting policy.')
    for name in ['pretrained','lora']:
        main=[]
        for v in VARIANTS:
            base='fp16_eager' if v.startswith('fp16') else 'fp32_eager'
            main.append([LABELS[v],f"{val(name,v):.1f}",f"{val(name,v,True):.1f}",f"{val(name,v)/val(name,v,True):.2f}x",f"{gain(name,base,v):.2f}x"])
        pairs=[]
        for z in r['models'][name]['inference']:
            b,l=z['batch'],z['prompt_tokens']
            pairs.append([f'{b}/{l}',f"{gain(name,'fp32_eager','fp32_tiled',batch=b,length=l):.2f}x",
                          f"{gain(name,'fp16_eager','fp16_flash',batch=b,length=l):.2f}x",
                          f"{gain(name,'fp32_eager','fp16_eager',batch=b,length=l):.2f}x",
                          f"{gain(name,'fp32_eager','fp16_flash',batch=b,length=l):.2f}x"])
        add(('LoRA' if name=='lora' else 'Pretrained')+': inference comparisons',[
            ('B1/P896: median request milliseconds',(['Variant','No cache','Cache','Cache gain','Backend gain*'],main)),
            ('No-cache speedup attribution at all shapes',(['B/P','FP32 tiled','FP16 Flash','Precision','Combined'],pairs)),
            ('Definitions','*Backend gain compares to eager at the SAME precision policy. Cache gain is no-cache/cache within a variant. Precision is FP32 eager/FP16 AMP eager; combined is FP32 eager/FP16 Flash. Ratios below one are slowdowns. Cached token throughput, prefill and decode CUDA-event metrics, memory and every raw sample are retained in CSV/JSON.'),
            ('Interpretation','Flash, cache and LoRA effects must be read from their measured combinations, not multiplied from unrelated workloads. Unmerged LoRA adds projection work. AMP behavior can differ for frozen versus trainable weights; these measurements do not establish optimal native-FP16 serving performance.')])

    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,method in zip(axs,['full','lora']):
        rows=r['training'][method];x=np.arange(len(rows))
        for j,v in enumerate(VARIANTS):
            ax.bar(x+(j-1.5)*.2,[z['summary'][v]['wall_ms']['median'] for z in rows],.2,label=LABELS[v],color=COLORS[j],
                   yerr=[z['summary'][v]['wall_ms']['stdev'] for z in rows],capsize=2)
        ax.set(xticks=x,xticklabels=[str(z['length']) for z in rows],xlabel='Sequence length; batch 2',ylabel='Forward + backward + AdamW (ms)',title=method+' reset-step training')
        ax.legend(fontsize=8)
    chart(fig,'training-speed','Paired reset-step training. Every trial starts from identical master weights, batch and zero AdamW moments. Copies/reset are outside timing.')
    for method in ['full','lora']:
        rows=[]
        for z in r['training'][method]:
            for v in VARIANTS:
                base='fp16_eager' if v.startswith('fp16') else 'fp32_eager'
                s=z['summary'][v]
                rows.append([z['length'],LABELS[v],f"{s['wall_ms']['median']:.1f}",f"{s['peak_allocated_bytes']['median']/2**20:.0f}",
                             f"{z['summary'][base]['wall_ms']['median']/s['wall_ms']['median']:.2f}x"])
                metrics.append(dict(section='training',model=method,batch=2,length=z['length'],variant=v,mode='reset_step',
                                    **{k:c['median'] for k,c in s.items()},wall_ms_stdev=s['wall_ms']['stdev'],successful_steps=len(z['updates'][v])))
        attribution=[]
        for z in r['training'][method]:
            times={v:z['summary'][v]['wall_ms']['median'] for v in VARIANTS}
            attribution.append([z['length'],f"{times['fp32_eager']/times['fp32_tiled']:.2f}x",
                                f"{times['fp16_eager']/times['fp16_flash']:.2f}x",
                                f"{times['fp32_eager']/times['fp16_eager']:.2f}x",
                                f"{times['fp32_eager']/times['fp16_flash']:.2f}x"])
        add(('LoRA' if method=='lora' else 'Full')+': training time and allocation',[
            ('Every measured training case',(['Length','Variant','Step ms','Peak MiB','Backend gain'],rows)),
            ('Training speedup attribution',(['Length','FP32 tiled','FP16 Flash','Precision','Combined'],attribution)),
            ('Numerical checks','All losses/gradients were finite and every optimizer advanced exactly one step. Both FP16 modes held scale 128 with no skips; FP32 used no scaling. Reset-step cost does not establish convergence or long-run quality.'),
            ('Meaning of a tradeoff','Tiling can reduce saved storage while adding recomputation, launches and layout copies. The prototype is not tuned like production FlashAttention. Report memory and time separately; backward support itself does not establish a speed win.')])

    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    x=np.arange(4)
    for j,name in enumerate(['pretrained','lora']):
        axs[0].bar(x+(j-.5)*.35,[val(name,v,key='peak_allocated_bytes')/2**20 for v in VARIANTS],.35,label=name)
    for j,method in enumerate(['full','lora']):
        axs[1].bar(x+(j-.5)*.35,[tv(method,v,key='peak_allocated_bytes')/2**20 for v in VARIANTS],.35,label=method)
    for ax in axs:
        ax.set(xticks=x,xticklabels=['FP32\neager','FP32\ntiled','FP16 AMP\neager','FP16 AMP\nFlash'],ylabel='Peak allocated GPU MiB');ax.legend()
    axs[0].set_title('Inference B1/P896, no cache');axs[1].set_title('Training B2/L896')
    chart(fig,'gpu-memory','PyTorch allocation peaks under fixed FP32 master-weight policy. AMP FP16 casting can add storage; this is not native-half model memory.')
    mem=[]
    for name in ['pretrained','lora']:
        for v in VARIANTS:
            mem.append([name,LABELS[v],f"{val(name,v,key='peak_allocated_bytes')/2**20:.1f}",f"{val(name,v,True,'peak_allocated_bytes')/2**20:.1f}",f"{val(name,v,True,'cache_bytes')/2**20:.1f}"])
    add('Memory policy and KV-cache interpretation',[
        ('B1/P896 inference, MiB',(['Model','Variant','No cache peak','Cache peak','KV MiB'],mem)),
        ('What allocated memory measures','Live PyTorch tensor allocations: master weights, original causal-mask buffers, inputs, activations, gradients, optimizer states, AMP casts and caches as applicable. Reserved allocator memory is recorded separately and must not be added. Compiler/driver overhead and kernel-private storage are outside this allocator metric; static kernel resource usage is saved.'),
        ('Precision policy matters','AMP FP16 preserves FP32 master weights and may retain cast copies. Do not claim these inference peaks are native-FP16 model storage. FP16 cache elements are half the width of FP32 cache elements; logical bytes use 2 * layers * batch * cached_tokens * hidden_width * element_bytes. Final cache length is prompt+63.'),
        ('Implementation limits','All backends retain legacy causal-mask buffers for checkpoint compatibility, even though tiled/Flash mask arithmetically. Dynamic KV concatenation can briefly retain old/new tensors. All timed cases stay inside the context window. At overflow learned absolute positions require rebuilding the cache.')])

    outputs=[]
    for name in ['pretrained','lora']:
        q=json.loads((OUT/f'{name}-quality.json').read_text());rows=[]
        for key,s in q['summary'].items():
            rows.append([key,f"{s['metrics']['primary_correct']:.1%}",f"{s['metrics']['fields_correct']:.1%}",f"{s['identical_until_eos']}/120"])
            for rr in q['outputs'][key]:
                original=q['questions'][rr['index']]
                outputs.append(dict(model=name,mode=key,index=rr['index'],question=original['question'],reference=original['answer'],output=rr['text'],**rr['scores']))
        # A second within-precision comparison is critical: FP16 Flash vs FP16 eager.
        within=[]
        for cache in ['uncached','cached']:
            for a,b,label in [('fp32_eager','fp32_tiled','FP32 tiled vs eager'),('fp16_eager','fp16_flash','FP16 Flash vs eager')]:
                same=sum(x['tokens_until_eos']==y['tokens_until_eos'] for x,y in zip(q['outputs'][a+'-'+cache],q['outputs'][b+'-'+cache]))
                within.append([label,cache,f'{same}/120'])
        add(('LoRA' if name=='lora' else 'Pretrained')+': held-out answer quality',[
            ('120 questions: reference is FP32 eager uncached',(['Variant/cache','Primary','Fields','Same EOS'],rows)),
            ('Agreement at fixed precision policy',(['Comparison','Cache','Same EOS'],within)),
            ('Meaning','Primary/Fields use the existing transparent sports-field scorer. All generated logits were checked finite. Small rounding differences can change greedy continuation even when forward/gradient tests pass. Post-EOS tokens are retained but do not affect scored answers. These are existing seed-42 checkpoints, not newly trained models or three-seed averages. Every one of the 1,920 main answers and token sequences is retained.')])
        for rr in r['models'][name]['inference']:
            for key,s in rr['summary'].items():
                metrics.append(dict(section='inference',model=name,batch=rr['batch'],length=rr['prompt_tokens'],variant=key.rsplit('-',1)[0],mode=key.rsplit('-',1)[1],
                                    **{k:c['median'] for k,c in s.items()},wall_ms_stdev=s['wall_ms']['stdev']))

    native_path=OUT/'native_fp16/report.json'
    if native_path.exists():
        native=json.loads(native_path.read_text())
        assert native['complete'], 'Native-FP16 supplement has not finished'
        native_table=[];native_quality=[];native_metrics=[];native_outputs=[]
        for name,info in native['models'].items():
            assert info['weight_hash_before']==info['weight_hash_after']
            representative=next(z for z in info['inference'] if z['batch']==1 and z['prompt_tokens']==896)
            for v in ['fp16_native_eager','fp16_native_flash']:
                off=representative['summary'][v+'-uncached'];on=representative['summary'][v+'-cached']
                baseline=representative['summary']['fp16_native_eager-uncached']['wall_ms']['median']
                native_table.append([name,v.replace('fp16_native_',''),f"{off['wall_ms']['median']:.1f}",f"{on['wall_ms']['median']:.1f}",
                                     f"{baseline/off['wall_ms']['median']:.2f}x",f"{off['peak_allocated_bytes']['median']/2**20:.1f}"])
                for cache,answers in info['quality'][v]['outputs'].items():
                    for answer in answers:
                        native_outputs.append(dict(model=name,variant=v,mode=cache,index=answer['index'],output=answer['text'],**answer['scores']))
            eager=info['quality']['fp16_native_eager'];flash=info['quality']['fp16_native_flash']
            same={mode:sum(a['tokens_until_eos']==b['tokens_until_eos'] for a,b in zip(eager['outputs'][mode],flash['outputs'][mode])) for mode in ['uncached','cached']}
            native_quality.append([name,f"{eager['metrics']['uncached']['primary_correct']:.1%} / {eager['metrics']['cached']['primary_correct']:.1%}",
                                   f"{flash['metrics']['uncached']['primary_correct']:.1%} / {flash['metrics']['cached']['primary_correct']:.1%}",
                                   f"{same['uncached']} / {same['cached']}"])
            for z in info['inference']:
                for key,stats in z['summary'].items():
                    native_metrics.append(dict(model=name,batch=z['batch'],length=z['prompt_tokens'],mode=key,
                                               **{k:c['median'] for k,c in stats.items()},wall_ms_stdev=stats['wall_ms']['stdev']))
        for filename,records in [('METRICS.csv',native_metrics),('ALL_OUTPUTS.csv',native_outputs)]:
            with (OUT/'native_fp16'/filename).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for rec in records for k in rec)))
                writer.writeheader();writer.writerows(records)
        add('Supplement: native FP16 inference',[
            ('Separate deployment policy and allocation',f"{native['gpu']}, job {native['job']} on {native['node']}. Weights, embeddings/masks and activations are natively FP16, with no autocast. Eager and Flash are paired within this allocation, using the same shapes, inputs, warmups and repetitions. This supplement is inference-only; training retains FP32 master weights in the main experiment."),
            ('B1/P896: request ms and no-cache peak MiB',(['Model','Backend','No cache','Cache','Flash gain','Peak MiB'],native_table)),
            ('Primary accuracy and fixed-policy agreement',(['Model','Eager off/on','Flash off/on','Same EOS off/on'],native_quality)),
            ('How to interpret it','The main AMP comparison holds master-weight storage fixed; this native policy changes storage and execution. These separately allocated samples are not pooled with main results. Differences across policies must not be attributed solely to casting. Read eager versus Flash within each policy for the clean backend comparison. Same EOS compares Flash/eager at the same cache setting, out of 120 questions.'),
            ('Evidence','The native_fp16 subdirectory retains all four shapes, both checkpoint hashes, raw timing samples, metrics CSV, 960 answer records and all token sequences. No native-FP16 training or convergence claim is made.')])

    evidence={}
    for path in OUT.glob('*trace.json'):
        data=json.loads(path.read_text());names=sorted({e.get('name','') for e in data.get('traceEvents',[]) if e.get('cat')=='kernel'})
        evidence[path.name]=names
    (OUT/'KERNEL_EVIDENCE.json').write_text(json.dumps(evidence,indent=2))
    for name in ['pretrained','lora']:
        names=evidence[f'{name}-fp16_flash-forward-backward-trace.json']
        assert any('flash' in n.lower() and 'bwd' in n.lower() for n in names)
        tiled=evidence[f'{name}-fp32_tiled-forward-backward-trace.json']
        assert any('backward_q' in n for n in tiled) and any('backward_kv' in n for n in tiled)
    audit=json.loads((OUT/'INSTRUCTION_AUDIT.json').read_text())
    assert all(s['tensor_mma_instructions']==0 for s in audit.values())
    add('CUDA evidence, limits and reproduction',[
        ('Actual kernels','Separate traces show PyTorch FP16 FlashAttention forward/backward and custom tiled forward, row-delta, dQ and dK/dV kernels on both architectures. Profiling is excluded from timing samples. Saved PTX/cubins/SASS verify FP32 custom dot products without Tensor Core matrix instructions or TF32. Static register/local/shared resource usage is preserved; no occupancy or bandwidth counters are inferred.'),
        ('What is established','The main FA comparison now uses FP16, with an FP16 eager baseline under the same policy. Custom tiled FP32 has validated first-order backward, making its training cost/memory comparison legitimate. Cross-precision ratios are labeled separately. A slower custom backward can still provide a memory tradeoff; no speed, convergence or accuracy equivalence is assumed.'),
        ('Limits','One A30 allocation and seven short repetitions per case. All timed inference trials follow complete warmups; compilation is excluded. AMP/native-half policies differ, and frozen LoRA can affect autocast-cache behavior. Reset-step training is a controlled cost experiment, not a realistic evolving optimizer trajectory. Dropout and higher-order custom gradients are unsupported.'),
        ('Reproduction','Run sbatch experiments/fp16_fp32_attention/run_gpu.sh, then python experiments/fp16_fp32_attention/build_report.py with NumPy/Matplotlib. The runner refuses to overwrite report.json. Source snapshots/hashes, raw measurements, loss scales, successful-step records, CUDA traces, compiler artifacts and checkpoint weight hashes are retained.'),
        ('Artifacts','FP16_FP32_ATTENTION_REPORT.pdf; RESULTS.md; standalone RESULTS.html; METRICS.csv; ALL_OUTPUTS.csv; raw report.json and per-case files. Read BACKENDS.md before comparing rows. Prior BF16 and forward-only FP32 artifacts remain in their original directories with original measured source snapshots.')])

    for filename,rows in [('METRICS.csv',metrics),('ALL_OUTPUTS.csv',outputs)]:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (OUT/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    (OUT/'RESULTS.md').write_text('\n'.join(md))
    (OUT/'PRESENTATION.md').write_text('# FP16/FP32: presentation notes\n\nStart with the backend glossary and fixed-master-weight policy, then show matched comparisons.\n\n'+highlights+'\n\nAll measured training updates were finite and successful. Backward support is validated, but performance and memory benefits must be read separately. This reset-step experiment does not measure convergence. Older BF16 measurements remain historical.\n')
    (OUT/'RESULTS.html').write_text('<!doctype html><html><meta charset="utf-8"><title>FP16/FP32 attention</title><style>body{font:16px/1.6 system-ui;max-width:1120px;margin:40px auto;padding:0 24px;color:#17252d}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px;border:1px solid #d5dfe6}th{background:#e6eff4}img{width:100%}h2{margin-top:42px}</style><body><h1>Main FP16/FP32 attention comparison</h1>'+''.join(parts)+'</body></html>')
    with PdfPages(OUT/'FP16_FP32_ATTENTION_REPORT.pdf') as pdf:
        for i,fig in enumerate(figures):
            fig.text(.98,.015,f'{i+1} / {len(figures)}',ha='right',size=8,color='#667780')
            pdf.savefig(fig);fig.savefig(OUT/f'page-{i+1}.png',dpi=110);plt.close(fig)
    print(highlights)
    print('Created',len(figures),'pages;',len(outputs),'task outputs;',successful,'finite successful measured steps')


if __name__=='__main__': main()
