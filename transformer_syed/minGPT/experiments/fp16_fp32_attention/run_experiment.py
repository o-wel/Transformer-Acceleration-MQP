"""Paired same-model FP32/AMP-FP16 attention inference and reset-step training."""
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
import triton
from transformers import GPT2Tokenizer
from mingpt.measurement import parameter_counts
from mingpt.tiled_attention import _forward,_run_backward
from experiments.flash_attention.run_experiment import load,digest
from experiments.flash_attention.benchmark import summary,measure
from experiments.kv_cache.benchmark import timed_generate
from experiments.sports.scoring import score,aggregate

OUT=ROOT/'experiments/fp16_fp32_attention/results'
VARIANTS={'fp32_eager':('eager',False),'fp32_tiled':('tiled_fp32',False),
          'fp16_eager':('eager',True),'fp16_flash':('flash',True)}
WARMUPS,REPEATS=3,7


def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2))


def amp(variant):
    return torch.autocast('cuda',dtype=torch.float16,enabled=VARIANTS[variant][1])


def set_variant(model,variant):
    model.set_attention_backend(VARIANTS[variant][0])


def disable_dropout(model):
    for m in model.modules():
        if isinstance(m,torch.nn.Dropout):
            m.p=0.0


def gradient_check(model,name):
    """Same real weights/input: forward and unscaled gradients versus FP32 eager."""
    model.train();disable_dropout(model)
    x=torch.randint(50257,(2,128),generator=torch.Generator(device='cuda').manual_seed(711),device='cuda')
    y=x.roll(-1,1)
    ref_grads=None;ref_logits=None;ref_loss=None
    rows={}
    for variant in VARIANTS:
        set_variant(model,variant);model.zero_grad(set_to_none=True)
        with amp(variant):
            logits,loss=model(x,y)
        assert bool(torch.isfinite(logits).all()) and bool(torch.isfinite(loss)),f'{name}/{variant}: nonfinite forward'
        scale=128.0 if VARIANTS[variant][1] else 1.0
        (loss*scale).backward()
        grads={n:(p.grad.detach().float()/scale).cpu() for n,p in model.named_parameters() if p.requires_grad}
        assert all(bool(torch.isfinite(g).all()) for g in grads.values()),f'{name}/{variant}: nonfinite gradient'
        if ref_grads is None:
            ref_grads=grads
            ref_logits=logits.detach().float().cpu()
            ref_loss=float(loss)
        squared_error=sum(float((g-ref_grads[n]).double().square().sum()) for n,g in grads.items())
        squared_ref=sum(float(g.double().square().sum()) for g in ref_grads.values())
        row=dict(loss=float(loss),loss_abs_error=abs(float(loss)-ref_loss),
                 max_abs_logit_error=float((logits.detach().float().cpu()-ref_logits).abs().max()),
                 gradient_relative_l2=(squared_error/max(squared_ref,1e-30))**.5,
                 gradient_max_abs=max(float((g-ref_grads[n]).abs().max()) for n,g in grads.items()),
                 gradients_finite=True,loss_scale=scale)
        if variant=='fp32_tiled':
            torch.testing.assert_close(logits.detach().float().cpu(),ref_logits,atol=3e-4,rtol=3e-4)
            for n,g in grads.items():
                torch.testing.assert_close(g,ref_grads[n],atol=1e-4,rtol=2e-3,msg=n)
        rows[variant]=row
        del logits,loss,grads
    # Within FP16, compare against FP16 eager rather than only against FP32.
    rows['fp16_backend_gradient_l2_note']='Each row above references FP32 eager; direct FP16 kernel tests are included in the suite.'
    model.zero_grad(set_to_none=True);model.eval()
    return rows


@torch.inference_mode()
def inference(model,name):
    rows=[]
    for batch,length in [(1,128),(1,512),(1,896),(4,128)]:
        prompt=torch.randint(50257,(batch,length),generator=torch.Generator().manual_seed(4200+batch+length)).cuda()
        modes=[(v,c) for v in VARIANTS for c in [False,True]]
        samples={v+('-cached' if c else '-uncached'):[] for v,c in modes}
        for repeat in range(WARMUPS+REPEATS):
            order=modes.copy();random.Random(1000+repeat).shuffle(order)
            for variant,cached in order:
                set_variant(model,variant)
                # A full-request finite-logit check is restricted to the first
                # warmup and excluded from measured timing repetitions.
                finite=[]
                hook=model.lm_head.register_forward_hook(lambda m,i,o:finite.append(torch.isfinite(o).all())) if repeat==0 else None
                try:
                    with amp(variant):
                        output,metrics=timed_generate(model,prompt,64,cached)
                    if finite:
                        assert bool(torch.stack(finite).all()),f'{name}/{variant}: nonfinite generated logits'
                finally:
                    if hook is not None: hook.remove()
                metrics['incremental_peak_bytes']=metrics['peak_allocated_bytes']-metrics['baseline_allocated_bytes']
                if repeat>=WARMUPS:
                    samples[variant+('-cached' if cached else '-uncached')].append(metrics)
                del output
        row=dict(batch=batch,prompt_tokens=length,new_tokens=64,samples=samples,summary={m:summary(s) for m,s in samples.items()})
        rows.append(row);save(f'{name}-inference.json',rows)
        print('INFERENCE',name,batch,length,{m:round(s['wall_ms']['median'],2) for m,s in row['summary'].items()},flush=True)
    return rows


@torch.inference_mode()
def quality(model,tokenizer,name):
    questions=[json.loads(s) for s in (ROOT/'experiments/sports/test.jsonl').read_text().splitlines()]
    buckets={}
    for i,row in enumerate(questions):
        ids=tokenizer.encode(f"User: {row['question']}\nAssistant:")
        buckets.setdefault(len(ids),[]).append((i,ids))
    outputs={v+('-cached' if c else '-uncached'):[] for v in VARIANTS for c in [False,True]}
    for length,group in sorted(buckets.items()):
        for offset in range(0,len(group),8):
            batch=group[offset:offset+8]
            x=torch.tensor([ids for _,ids in batch],device='cuda')
            for variant in VARIANTS:
                set_variant(model,variant)
                for cached in [False,True]:
                    finite=[]
                    hook=model.lm_head.register_forward_hook(lambda m,i,o:finite.append(torch.isfinite(o).all()))
                    try:
                        with amp(variant):
                            tokens=model.generate(x,40,use_cache=cached)[:,length:].cpu().tolist()
                        assert bool(torch.stack(finite).all()),f'{name}/{variant}: nonfinite quality logits'
                    finally: hook.remove()
                    key=variant+('-cached' if cached else '-uncached')
                    for (i,_),seq in zip(batch,tokens):
                        until=seq[:seq.index(tokenizer.eos_token_id)] if tokenizer.eos_token_id in seq else seq
                        text=tokenizer.decode(until,skip_special_tokens=True).split('\nUser:')[0].strip()
                        outputs[key].append(dict(index=i,tokens=seq,tokens_until_eos=until,text=text,scores=score(questions[i],text)))
    for s in outputs.values(): s.sort(key=lambda r:r['index'])
    ref=outputs['fp32_eager-uncached']
    result=dict(questions=questions,outputs=outputs,summary={m:dict(metrics=aggregate(rows),
                identical_until_eos=sum(r['tokens_until_eos']==q['tokens_until_eos'] for r,q in zip(rows,ref)),
                identical_all_tokens=sum(r['tokens']==q['tokens'] for r,q in zip(rows,ref))) for m,rows in outputs.items()})
    save(f'{name}-quality.json',result)
    return result['summary']


def training(model,name):
    """Paired reset-step AdamW benchmark: identical weights, batch and zero moments."""
    model.train();disable_dropout(model)
    initial={n:t.detach().cpu().clone() for n,t in model.state_dict().items()}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5)
    rows=[]
    for length in [128,512,896]:
        x=torch.randint(50257,(2,length),device='cuda',generator=torch.Generator(device='cuda').manual_seed(4200+length))
        y=x.roll(-1,1)
        # Materialize all AdamW state tensors before any timing baseline.
        set_variant(model,'fp32_eager')
        optimizer.zero_grad(set_to_none=True)
        model(x,y)[1].backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
        samples={v:[] for v in VARIANTS};updates={v:[] for v in VARIANTS}
        loss_rows={v:[] for v in VARIANTS}
        for repeat in range(WARMUPS+REPEATS):
            order=list(VARIANTS);random.Random(500+repeat).shuffle(order)
            for variant in order:
                # All copies, optimizer-state zeroing and synchronization are
                # outside timing. Every trial starts from identical state.
                model.load_state_dict(initial)
                optimizer.zero_grad(set_to_none=True)
                for state in optimizer.state.values():
                    for value in state.values():
                        if isinstance(value,torch.Tensor): value.zero_()
                set_variant(model,variant)
                scaler=torch.amp.GradScaler('cuda',enabled=VARIANTS[variant][1],init_scale=128.0,growth_interval=1000000)
                # Initialize scaler's tiny persistent tensors outside timing.
                scaler.scale(torch.zeros((),device='cuda'))
                def step():
                    with amp(variant):
                        _,loss=model(x,y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    return loss.detach()
                loss,metrics=measure(step,torch.device('cuda'))
                successful=all(float(s['step'])==1.0 for s in optimizer.state.values())
                final_scale=float(scaler.get_scale())
                finite=bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.requires_grad)
                assert finite and successful and (not VARIANTS[variant][1] or final_scale==128.0),f'{name}/{variant}: invalid or skipped training update'
                metrics['tokens_per_second']=2*length*1000/metrics['wall_ms']
                if repeat>=WARMUPS:
                    samples[variant].append(metrics)
                    updates[variant].append(dict(successful=successful,loss_scale=final_scale,gradients_finite=finite))
                    loss_rows[variant].append(float(loss))
                del loss,scaler
        row=dict(method=name,batch=2,length=length,samples=samples,summary={v:summary(s) for v,s in samples.items()},
                 updates=updates,losses=loss_rows,parameters=parameter_counts(model))
        rows.append(row);save(f'{name}-training.json',rows)
        print('TRAINING',name,length,{v:round(s['wall_ms']['median'],2) for v,s in row['summary'].items()},flush=True)
    model.load_state_dict(initial);model.zero_grad(set_to_none=True);model.eval()
    del optimizer,initial
    return rows


def profile(model,name):
    result={}
    x=torch.randint(50257,(2,128),device='cuda')
    model.train();disable_dropout(model)
    for variant in VARIANTS:
        set_variant(model,variant);model.zero_grad(set_to_none=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            with amp(variant): _,loss=model(x,x.roll(-1,1))
            loss.backward()
        filename=f'{name}-{variant}-forward-backward-trace.json'
        prof.export_chrome_trace(str(OUT/filename))
        result[variant]=dict(trace=filename,operators=[dict(name=e.key,calls=e.count,device_time_us=e.device_time_total)
            for e in prof.key_averages() if any(s in e.key.lower() for s in ['attention','backward_q','backward_kv','row_delta','tiled'])])
        del loss
    model.zero_grad(set_to_none=True);model.eval()
    return result


def compiler_evidence():
    q=torch.randn(1,2,128,64,device='cuda')
    out,lse,fwd=_forward(q,q,q,0,True)
    _,backward=_run_backward(q,q,q,out,lse,torch.randn_like(out),0,True)
    for name,kernel in {'forward_training':fwd,**backward}.items():
        ptx=kernel.asm['ptx'].lower()
        assert 'mma.sync' not in ptx and '.tf32' not in ptx
        (OUT/f'{name}.ptx').write_text(kernel.asm['ptx'])
        (OUT/f'{name}.cubin').write_bytes(kernel.asm['cubin'])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'report.json').exists(): raise RuntimeError('Refusing to overwrite previous measurements')
    torch.manual_seed(42);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    report=dict(complete=False,gpu=torch.cuda.get_device_name(),node=platform.node(),job=os.environ.get('SLURM_JOB_ID'),
                torch=torch.__version__,cuda=torch.version.cuda,triton=triton.__version__,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                variants=VARIANTS,warmups=WARMUPS,repeats=REPEATS,tf32=False,
                precision_policy='FP32 master parameters for all variants; FP16 autocast only in fp16_*; training uses GradScaler(128) in both FP16 variants',
                inference_note='AMP weight-cast cache is part of FP16 request cost/memory; not native-FP16 weight storage',
                training_note='Paired reset-step: original weights, identical batch, AdamW zero moments and step=0 every trial; copies/reset excluded',
                models={},training={},source_sha256={})
    for name in ['mingpt/model.py','mingpt/tiled_attention.py','tests/test_tiled_backward.py','tests/test_tiled_attention.py',
                 'experiments/fp16_fp32_attention/run_experiment.py','experiments/kv_cache/benchmark.py']:
        content=(ROOT/name).read_bytes();report['source_sha256'][name]=hashlib.sha256(content).hexdigest()
        target=OUT/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    save('report.json',report)
    suite=unittest.TestSuite([unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=p)
                             for p in ['test_flash_attention.py','test_tiled_attention.py','test_tiled_backward.py']])
    tests=unittest.TextTestRunner(verbosity=2).run(suite)
    assert tests.wasSuccessful()
    report['tests']=dict(run=tests.testsRun,failures=len(tests.failures),errors=len(tests.errors),skipped=len(tests.skipped))
    compiler_evidence()
    tokenizer=GPT2Tokenizer.from_pretrained('gpt2',local_files_only=True)
    for name in ['pretrained','lora']:
        model=load(name,torch.float32).eval()
        before=digest(model)
        info=dict(parameters=parameter_counts(model),weight_hash_before=before)
        info['gradient_check']=gradient_check(model,name)
        print('GRADIENTS',name,info['gradient_check'],flush=True)
        info['inference']=inference(model,name)
        info['quality']=quality(model,tokenizer,name)
        info['weight_hash_after_inference']=digest(model)
        assert before==info['weight_hash_after_inference']
        report['models'][name]=info;save('report.json',report)
        report['training']['full' if name=='pretrained' else 'lora']=training(model,'full' if name=='pretrained' else 'lora')
        info['weight_hash_after_training_reset']=digest(model)
        assert before==info['weight_hash_after_training_reset']
        info['profiles']=profile(model,name)
        save('report.json',report)
        del model;gc.collect();torch.cuda.empty_cache()
    report['complete']=True;save('report.json',report)
    print('COMPLETE',flush=True)


if __name__=='__main__': main()
