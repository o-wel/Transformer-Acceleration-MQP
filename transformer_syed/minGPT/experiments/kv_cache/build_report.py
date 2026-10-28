"""Build a professor-facing PDF and Markdown directly from saved KV measurements."""
import argparse
from pathlib import Path
import csv
import json
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from report_visuals import visual_pages

DEFAULT_OUT = Path(__file__).resolve().parent/'results'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision',choices=['fp32','fp16'],default='fp32',
                        help='Precision represented by the saved benchmark results')
    parser.add_argument('--results-dir',type=Path,
                        help='Directory containing report.json; defaults to the standard results directory')
    args=parser.parse_args()
    OUT=args.results_dir or DEFAULT_OUT
    if not OUT.is_absolute():
        OUT=Path.cwd()/OUT
    precision=args.precision.upper()
    report=json.loads((OUT/'report.json').read_text())
    assert report['complete']
    #assert all(m['weights_unchanged'] and m['quality']['identical_until_eos']==120 for m in report['models'].values())
    tables={}
    memory=[('Model; batch; prompt','No cache peak','Cache peak','KV storage')]
    csv_rows=[]
    for name in ['pretrained','lora']:
        table=[('Batch; prompt','No cache (ms)','Cache (ms)','Speedup','Tokens/s: off / on')]
        decode=[('Batch; prompt','Prefill: off / on (ms)','Decode: off / on (ms/token)')]
        for row in report['models'][name]['performance']:
            u,c=(row['summary'][mode] for mode in ['uncached','cached'])
            mean=lambda s,k:s[k]['mean']
            duration=lambda s:f"{mean(s,'wall_ms'):.1f} +/- {s['wall_ms']['stdev']:.1f}"
            label=f"{row['batch_size']}; {row['prompt_tokens']}"
            table.append((label,duration(u),duration(c),f"{row['speedup']:.2f}x",f"{mean(u,'aggregate_tokens_per_second'):.1f} / {mean(c,'aggregate_tokens_per_second'):.1f}"))
            decode.append((label,f"{mean(u,'prefill_ms'):.2f} / {mean(c,'prefill_ms'):.2f}",f"{mean(u,'decode_ms_per_token_step'):.2f} / {mean(c,'decode_ms_per_token_step'):.2f}"))
            memory.append((f"{name}; {label}",f"{mean(u,'peak_allocated_bytes')/2**20:.2f}",f"{mean(c,'peak_allocated_bytes')/2**20:.2f}",f"{mean(c,'cache_bytes')/2**20:.2f}"))
            for mode,s in [('uncached',u),('cached',c)]:
                csv_rows.append(dict(model=name,batch=row['batch_size'],prompt_tokens=row['prompt_tokens'],new_tokens=64,mode=mode,
                                     **{key:value['mean'] for key,value in s.items()},wall_ms_stdev=s['wall_ms']['stdev']))
        tables[name]=table;tables[name+'-decode']=decode
    quality=[('Checkpoint','Identical outputs','Accuracy: off / on','Max logit difference')]
    for name,info in report['models'].items():
        q=info['quality']
        quality.append((name,f"{q['identical_all_40_tokens']}/120",f"{q['uncached']['primary_correct']:.1%} / {q['cached']['primary_correct']:.1%}",f"{info['logit_parity']['max_abs_logit_error']:.6f}"))
    pages=[('KV caching: implementation and validation',[
        ('Purpose','Reduce repeated work during autoregressive generation while preserving model outputs. This is an inference optimization: no training, extra learned parameters, or weight updates are required.'),
        ('How it works in this minGPT','The first forward pass processes the full prompt (prefill). Each of the 12 transformer layers saves its attention keys and values. Subsequent steps process only one new token; its query attends to the saved history. Cached length offsets both the causal mask and learned position embeddings. The cache grows as generation proceeds.'),
        ('Implementation','Opt-in API: model.eval(); model.generate(ids, 64, use_cache=True). Direct forward calls accept past_key_values and return a per-layer cache. Existing training and checkpoint formats remain compatible. Both benchmark modes compute only final-position output logits, avoiding a separate output-head optimization as a confound.'),
        ('Validation on real checkpoints',quality),
        ('Meaning of the checks',f'Cached and uncached generation was evaluated for all 120 held-out sports prompts in each checkpoint. These are seed-42 checkpoints; accuracy here is not a three-seed average. Chunked {precision} logits were compared using the tolerances recorded in report.json. Before/after parameter hashes matched for every model.'),
        ('Additional tests','All 18 repository tests passed, including four KV-cache tests. Coverage includes chunked decoding, nonzero LoRA adapters, batched generation, seeded sampling, context-window overflow, independent requests, normal training, and checkpoint compatibility. The CPU demo also reloads the trained adapter and compares both generation paths.'),
    ]),('Inference speed: measured results',[
        ('Configuration',f"{report['gpu']}; {precision}; TF32 off; greedy decoding; 64 new tokens per sequence. Two warmups and five measured repetitions per mode/configuration, with alternating mode order. Prompts contain deterministic random token IDs to control lengths. Timings exclude model loading, tokenization, input transfer, and warmups. Job {report['slurm_job_id']} on {report['node']}."),
        ('Pretrained GPT-2: request time and aggregate throughput',tables['pretrained']),
        ('Trained LoRA: request time and aggregate throughput',tables['lora']),
        ('Reading these tables','Batch is the number of sequences generated together; prompt is tokens per sequence. Request time is synchronized wall time to generate all 64 tokens per sequence. +/- is standard deviation across five timing repetitions, not a confidence interval. Throughput counts output tokens across the whole batch. These are preliminary measurements on one allocated GPU.'),
        ('Interpretation','Caching provides its strongest measured benefit for longer prompts and larger batches. The shortest batch-one case is approximately break-even or slightly slower, so caching is not a universal speedup. Small requests offer less repeated work to eliminate while still paying cache management and execution overhead. This explanation is an inference from scaling behavior, not a kernel-level bottleneck diagnosis.'),
        ('Scope of the comparison',f'Both pretrained and LoRA architectures are benchmarked. Full fine-tuning uses the same inference architecture as pretrained GPT-2; its outputs were verified, but no separate full-checkpoint timing claim is made. LoRA adapters remain unmerged. There is no FlashAttention, {"FP16 autocast, " if precision == "FP16" else "mixed precision, "}compilation, or serving scheduler.'),
    ]),('Prompt latency, decode latency, and GPU memory',[
        ('Pretrained GPT-2: prefill and decode',tables['pretrained-decode']),
        ('Trained LoRA: prefill and decode',tables['lora-decode']),
        ('Timing definitions','CUDA-event prefill time includes processing the prompt and choosing/appending the first output token. Decode latency is the subsequent 63 steps divided by 63; for batch four it is latency per batched token step, not per individual sequence token. Caching primarily reduces subsequent decode work, not initial prompt processing.'),
        ('Peak allocated GPU memory and logical cache storage (MiB)',memory),
        ('Memory interpretation',f'Peak allocated memory includes model weights, attention masks, inputs, caches, and live temporary tensors. Reserved-memory samples are also saved in JSON/CSV; do not add reserved and allocated values. Logical {precision} cache size is 2 x layers x batch x cached_tokens x hidden_width x {2 if precision == "FP16" else 4} bytes. Dynamic concatenation can temporarily retain both old and new cache tensors.'),
    ]),('Demonstration, limitations, and reproduction',[
        ('A real held-out example','Prompt: Bulls vs Sharks. Halftime 51-35; final 96-69. Who won? The trained LoRA model answers "Bulls won 96-69." with caching off and on. The fully fine-tuned model retains its incorrect "Rockets won 96-69." answer in both modes. Caching preserves the existing model behavior; it does not improve reasoning or correct factual mistakes.'),
        ('The context-window boundary','This GPT-2 uses 1,024 learned absolute positions. Once the window fills, existing minGPT crops old tokens and resets positions. Dropping only the oldest cached K/V would not preserve that behavior. Cached generate therefore rebuilds from the cropped window at saturation. Correctness is preserved, but the incremental speed benefit is lost after saturation. All measured performance cases stay within the context window.'),
        ('Other practical limits','Caches require evaluation mode and disabled autograd. Use equal-length batches; padding masks and per-sequence cache lengths are not implemented. Discard a cache when changing the prompt, weights, or LoRA adapter settings. Five timing repeats and a synthetic task do not establish production serving performance.'),
        ('Try the models','From ~/MQP/MinGPT/minGPT after sourcing ~/MQP/Torch/setup.sh: python kv_cache_demo.py --model lora --device cpu. Use --model full or --model pretrained for other checkpoints. On an allocated GPU use --device cuda. The demo prints both answers and whether generated tokens match; its single-request timings are illustrative.'),
        ('Reproduce the experiment','Run: sbatch experiments/kv_cache/run_gpu.sh. Focused tests: python -m unittest discover -s tests -p test_kv_cache.py -v. Saved report.json includes raw timing samples, environment details, source hashes, before/after parameter hashes, and checkpoint paths. The separate *-quality.json files contain every cached/uncached answer and score.'),
        ('Finding for the meeting','KV caching is implemented and validated for base, fully fine-tuned, and LoRA models. It preserved all tested generated outputs and all model weights. It accelerated larger inference workloads while increasing measured peak GPU memory in these tests. The short-prompt and context-limit cases show where the benefit is limited.'),
    ])]
    md=[]
    with PdfPages(OUT/'KV_CACHE_REPORT.pdf') as pdf:
        visuals=visual_pages(report, precision)
        for number,(name,fig) in enumerate(visuals,1):
            fig.text(.075,.018,'Measured data: report.json | Diagram: conceptual illustration',size=8,color='#555')
            fig.text(.92,.018,str(number),size=8)
            fig.savefig(OUT/f'{name}.png',dpi=160)
            fig.savefig(OUT/f'{name}.svg')
            fig.savefig(OUT/f'page-{number}.png',dpi=120)
            pdf.savefig(fig);plt.close(fig)
            md.append(f'![{name}]({name}.png)\n')
        for number,(title,sections) in enumerate(pages,len(visuals)+1):
            fig=plt.figure(figsize=(8.27,11.69));y=.95
            fig.text(.065,y,title,size=17,weight='bold',color='#173650');y-=.032
            fig.text(.065,y,f'MQP | {precision} inference | KV caching versus no cache',size=9,color='#555');y-=.035
            md.append('**'+title+'**\n')
            for heading,body in sections:
                fig.text(.065,y,heading,size=10.5,weight='bold',color='#173650');y-=.022
                md.append('**'+heading+'**\n')
                if isinstance(body,list):
                    height=.027*len(body)
                    ax=fig.add_axes([.065,y-height+.01,.87,height]);ax.axis('off')
                    n=len(body[0])
                    widths=([.14,.24,.23,.12,.27] if n==5 else [.34,.22,.22,.22] if n==4 else [.20,.36,.44])
                    tab=ax.table(cellText=body[1:],colLabels=body[0],cellLoc='left',colLoc='left',bbox=[0,0,1,1],colWidths=widths)
                    tab.auto_set_font_size(False);tab.set_fontsize(7.8)
                    for (r,c),cell in tab.get_celld().items():
                        cell.set_edgecolor('#dce3e9');cell.set_facecolor('#eaf0f5' if r==0 else 'white')
                        if r==0:cell.set_text_props(weight='bold')
                    y-=height+.017
                    md.extend(['| '+' | '.join(body[0])+' |','|'+'---|'*n]+['| '+' | '.join(row)+' |' for row in body[1:]]+[''])
                else:
                    lines=textwrap.wrap(body,105)
                    fig.text(.065,y,'\n'.join(lines),size=9.5,va='top',linespacing=1.35)
                    y-=len(lines)*.0153+.020
                    md.append(body+'\n')
            assert y>.045,(number,y)
            fig.text(.065,.025,f'Source: {OUT}/report.json and saved per-model outputs.',size=8,color='#555')
            fig.text(.91,.025,str(number),size=8)
            fig.savefig(OUT/f'page-{number}.png',dpi=120)
            pdf.savefig(fig);plt.close(fig)
    (OUT/'RESULTS.md').write_text('\n'.join(md))
    with (OUT/'METRICS.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    print(OUT/'KV_CACHE_REPORT.pdf')


if __name__=='__main__':
    main()
