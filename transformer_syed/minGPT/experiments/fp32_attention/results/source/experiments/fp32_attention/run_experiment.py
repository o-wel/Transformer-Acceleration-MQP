"""FP32 tiled-attention prototype, FastF32 efficient backend, and precision bridge."""
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel
import triton
from transformers import GPT2Tokenizer
from mingpt.tiled_attention import tiled_attention_fp32
from mingpt.measurement import parameter_counts
from experiments.flash_attention.run_experiment import load,digest
from experiments.kv_cache.benchmark import timed_generate
from experiments.flash_attention.benchmark import summary,measure
from experiments.sports.scoring import score,aggregate

OUT=ROOT/'experiments/fp32_attention/results'
VARIANTS={
    'fp32_eager':('eager',torch.float32),
    'fp32_tiled':('tiled_fp32',torch.float32),
    'fp32_efficient_fastf32':('efficient',torch.float32),
    'bf16_eager':('eager',torch.bfloat16),
    'bf16_flash':('flash',torch.bfloat16),
}
WARMUPS,REPEATS=3,7


def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2))


@torch.inference_mode()
def logits_check(model,x,reference):
    dense=model(x)[0].float()
    cache=None
    chunks=[]
    start=0
    for length in (11,1,7,18):
        logits,_,cache=model(x[:,start:start+length],past_key_values=cache,use_cache=True)
        chunks.append(logits.float())
        start+=length
    chunked=torch.cat(chunks,1)
    ref=reference.to(dense.device)
    metrics=dict(max_abs_error=float((dense-ref).abs().max()),mean_abs_error=float((dense-ref).abs().mean()),
                 cache_max_abs_error=float((chunked-dense).abs().max()),
                 argmax_agreement=float((dense.argmax(-1)==ref.argmax(-1)).float().mean()))
    if next(model.parameters()).dtype==torch.float32:
        torch.testing.assert_close(dense,ref,atol=3e-4,rtol=3e-4)
        torch.testing.assert_close(chunked,dense,atol=3e-4,rtol=3e-4)
    return metrics


@torch.inference_mode()
def evaluate(model,tokenizer,questions):
    buckets={}
    for i,row in enumerate(questions):
        ids=tokenizer.encode(f"User: {row['question']}\nAssistant:")
        buckets.setdefault(len(ids),[]).append((i,ids))
    outputs={'uncached':[],'cached':[]}
    for length,group in sorted(buckets.items()):
        for offset in range(0,len(group),8):
            batch=group[offset:offset+8]
            x=torch.tensor([ids for _,ids in batch],device='cuda')
            for cache in (False,True):
                mode='cached' if cache else 'uncached'
                tokens=model.generate(x,40,use_cache=cache)[:,length:].cpu().tolist()
                for (i,_),sequence in zip(batch,tokens):
                    until=sequence[:sequence.index(tokenizer.eos_token_id)] if tokenizer.eos_token_id in sequence else sequence
                    text=tokenizer.decode(until,skip_special_tokens=True).split('\nUser:')[0].strip()
                    outputs[mode].append(dict(index=i,tokens=sequence,tokens_until_eos=until,text=text,scores=score(questions[i],text)))
    for rows in outputs.values():
        rows.sort(key=lambda r:r['index'])
    return dict(outputs=outputs,metrics={m:aggregate(rows) for m,rows in outputs.items()})


@torch.inference_mode()
def profile(model,name,variant):
    x=torch.randint(50257,(1,512),device='cuda')
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        _,_,cache=model(x,use_cache=True,last_token_only=True)
        model(x[:,:1],past_key_values=cache,use_cache=True,last_token_only=True)
    filename=f'{name}-{variant}-trace.json'
    prof.export_chrome_trace(str(OUT/filename))
    return dict(trace=filename,operators=[dict(name=e.key,calls=e.count,device_time_us=e.device_time_total)
                for e in prof.key_averages() if any(w in e.key.lower() for w in ['attention','tiled','flash'])])


@torch.inference_mode()
def inference(model,name,variant):
    rows=[]
    for batch,length in [(1,128),(1,512),(1,896),(4,128)]:
        generator=torch.Generator().manual_seed(4200+batch+length)
        prompt=torch.randint(50257,(batch,length),generator=generator).cuda()
        samples={'uncached':[],'cached':[]}
        for repeat in range(WARMUPS+REPEATS):
            for cache in ((False,True) if repeat%2==0 else (True,False)):
                mode='cached' if cache else 'uncached'
                output,metrics=timed_generate(model,prompt,64,cache)
                metrics['incremental_peak_bytes']=metrics['peak_allocated_bytes']-metrics['baseline_allocated_bytes']
                if repeat>=WARMUPS:
                    samples[mode].append(metrics)
                del output
        row=dict(batch=batch,prompt_tokens=length,new_tokens=64,samples=samples,summary={m:summary(v) for m,v in samples.items()})
        rows.append(row)
        save(f'{name}-{variant}-performance.json',rows)
        print('INFERENCE',name,variant,batch,length,{m:round(v['wall_ms']['median'],2) for m,v in row['summary'].items()},flush=True)
    return rows


def training():
    result=[]
    for method in ['full','lora']:
        for length in [512,896]:
            for variant in ['fp32_eager','fp32_efficient_fastf32','bf16_eager','bf16_flash']:
                gc.collect();torch.cuda.empty_cache()
                torch.manual_seed(42)
                backend,dtype=VARIANTS[variant]
                model=load('pretrained' if method=='full' else 'lora',torch.float32).train().set_attention_backend(backend)
                for module in model.modules():
                    if isinstance(module,torch.nn.Dropout):
                        module.p=0.0
                optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5)
                generator=torch.Generator(device='cuda').manual_seed(4200+length)
                x=torch.randint(50257,(2,length),generator=generator,device='cuda')
                y=x.roll(-1,1)
                def step():
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dtype==torch.bfloat16):
                        _,loss=model(x,y)
                    loss.backward()
                    optimizer.step()
                    return loss.detach()
                samples=[]
                for repeat in range(WARMUPS+REPEATS):
                    optimizer.zero_grad(set_to_none=True)
                    loss,metrics=measure(step,torch.device('cuda'))
                    assert bool(torch.isfinite(loss))
                    metrics['tokens_per_second']=2*length*1000/metrics['wall_ms']
                    if repeat>=WARMUPS:
                        samples.append(metrics)
                row=dict(method=method,length=length,batch=2,variant=variant,parameters=parameter_counts(model),
                         samples=samples,summary=summary(samples),last_loss=float(loss))
                result.append(row)
                save('training.json',result)
                print('TRAIN',method,length,variant,round(row['summary']['wall_ms']['median'],2),flush=True)
                del loss,optimizer,model,x,y
    return result


@torch.inference_mode()
def kernel_bench():
    rows=[]
    for length in [128,512,896]:
        for phase in ['prefill','decode']:
            nq,nk=(length,length) if phase=='prefill' else (1,length+63)
            generator=torch.Generator(device='cuda').manual_seed(72+length)
            originals=[torch.randn(1,12,n,64,device='cuda',generator=generator) for n in [nq,nk,nk]]
            mask=torch.ones(nq,nk,device='cuda',dtype=torch.bool).tril() if phase=='prefill' else None
            samples={}
            for variant,(backend,dtype) in VARIANTS.items():
                q,k,v=[t.to(dtype) for t in originals]
                def attention():
                    if backend=='eager':
                        scores=(q@k.transpose(-1,-2))*(64**-.5)
                        if mask is not None:
                            scores=scores.masked_fill(~mask,float('-inf'))
                        return scores.softmax(-1)@v
                    if backend=='tiled_fp32':
                        return tiled_attention_fp32(q,k,v,nk-nq)
                    selected=SDPBackend.FLASH_ATTENTION if backend=='flash' else SDPBackend.EFFICIENT_ATTENTION
                    with sdpa_kernel(selected):
                        return F.scaled_dot_product_attention(q,k,v,is_causal=phase=='prefill')
                for _ in range(10):
                    out=attention()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(20):
                        out=attention()
                # Warm for at least 0.1s to reduce clock-ramp sensitivity.
                until=time.perf_counter()+.1
                while time.perf_counter()<until:
                    graph.replay()
                    torch.cuda.synchronize()
                timings=[]
                for _ in range(15):
                    start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    start.record();graph.replay();end.record();end.synchronize()
                    timings.append(start.elapsed_time(end)*1000/20)
                samples[variant]=dict(samples_us=timings,median_us=__import__('statistics').median(timings))
                if backend=='tiled_fp32' and length==512:
                    _,compiled=tiled_attention_fp32(q,k,v,nk-nq,return_kernel=True)
                    ptx=compiled.asm['ptx']
                    (OUT/f'tiled-{phase}.ptx').write_text(ptx)
                    (OUT/f'tiled-{phase}.cubin').write_bytes(compiled.asm['cubin'])
                    assert 'mma.sync' not in ptx.lower() and '.tf32' not in ptx.lower()
                    assert 'fma.rn.f32' in ptx.lower()
                del graph,out,q,k,v
                gc.collect();torch.cuda.empty_cache()
            rows.append(dict(length=length,phase=phase,query_tokens=nq,key_tokens=nk,backends=samples))
            save('kernels.json',rows)
            print('KERNEL',length,phase,{v:round(s['median_us'],2) for v,s in samples.items()},flush=True)
            del originals,mask
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'report.json').exists():
        raise RuntimeError('Refusing to overwrite previous experiment')
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    report=dict(complete=False,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
                triton=triton.__version__,node=platform.node(),job=os.environ.get('SLURM_JOB_ID'),
                started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),warmups=WARMUPS,repeats=REPEATS,
                variants={v:dict(backend=b,dtype=str(d)) for v,(b,d) in VARIANTS.items()},tf32_flags=False,
                nvidia_tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE'),
                efficient_arithmetic='FP32 tensors; CUTLASS OpMultiplyAddFastF32; not claimed IEEE FP32',
                tiled_arithmetic='IEEE FP32 dots, FP32 online-softmax state, FP32 output; no backward',
                models={},source_sha256={})
    for path in ['mingpt/model.py','mingpt/tiled_attention.py','experiments/fp32_attention/run_experiment.py',
                 'experiments/kv_cache/benchmark.py','experiments/flash_attention/run_experiment.py']:
        content=(ROOT/path).read_bytes()
        report['source_sha256'][path]=hashlib.sha256(content).hexdigest()
        target=OUT/'source'/path
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(content)
    save('report.json',report)
    suite=unittest.TestSuite([unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=p)
                             for p in ['test_flash_attention.py','test_tiled_attention.py']])
    tests=unittest.TextTestRunner(verbosity=2).run(suite)
    assert tests.wasSuccessful()
    report['tests']=dict(run=tests.testsRun,failures=len(tests.failures),errors=len(tests.errors),skipped=len(tests.skipped))
    report['kernels']=kernel_bench()
    tokenizer=GPT2Tokenizer.from_pretrained('gpt2',local_files_only=True)
    questions=[json.loads(s) for s in (ROOT/'experiments/sports/test.jsonl').read_text().splitlines()]
    for name in ['pretrained','lora']:
        info={}
        reference=None
        x=torch.randint(50257,(2,37),generator=torch.Generator().manual_seed(777)).cuda()
        # Fixed blocks allow one resident model, giving direct per-model memory peaks.
        # The block order is recorded; it is not a paired interleaved kernel trial.
        for variant,(backend,dtype) in VARIANTS.items():
            model=load(name,dtype).eval().set_attention_backend(backend)
            before=digest(model)
            if reference is None:
                with torch.inference_mode():
                    reference=model(x)[0].float().cpu()
            row=dict(parameters=parameter_counts(model),weights_before=before)
            row['logits']=logits_check(model,x,reference)
            print('LOGITS',name,variant,row['logits'],flush=True)
            row['performance']=inference(model,name,variant)
            row['profile']=profile(model,name,variant)
            row['quality']=evaluate(model,tokenizer,questions)
            row['weights_after']=digest(model)
            assert before==row['weights_after']
            info[variant]=row
            save(f'{name}-{variant}.json',row)
            report['models'][name]=info
            save('report.json',report)
            print('QUALITY',name,variant,row['quality']['metrics'],flush=True)
            del model
            gc.collect();torch.cuda.empty_cache()
        del x,reference
    report['training']=training()
    report['complete']=True
    save('report.json',report)
    print('COMPLETE',flush=True)


if __name__=='__main__':
    main()
