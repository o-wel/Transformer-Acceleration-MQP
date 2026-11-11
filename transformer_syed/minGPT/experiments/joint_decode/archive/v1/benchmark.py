"""Run with python -m experiments.joint_decode.benchmark --phase discovery."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time
import torch
import torch.nn.functional as F
from mingpt.model import GPT
from experiments.joint_decode.kernel import decode
from experiments.joint_decode.runtime import mode, merge

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CHECKPOINT = ROOT / 'experiments/flash_attention/results/combined_optimization_review/lora-fp16-lr0.0001.pt'
MODES = [('dynamic_flash', False), ('static_flash', False), ('static_stream', False),
         ('fused_stream', False), ('static_flash', True), ('static_stream', True)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(model):
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        h.update(n.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def kernel_checks():
    rows = []
    for rank in (2, 4, 16, 32):
        for batch in (1, 4):
            for past in (0, 1, 31, 895, 1023):
                h, d, cap = 12, 64, 1024
                c = h * d
                qkv = torch.randn(batch, 1, 3*c, device='cuda', dtype=torch.float16) * .2
                z = torch.randn(batch, 1, rank, device='cuda', dtype=torch.float16) * .2
                up = torch.randn(3*c, rank, device='cuda', dtype=torch.float16) * .2
                cache = tuple(torch.randn(batch,h,cap,d,device='cuda',dtype=torch.float16)*.2 for _ in range(2))
                adapted = qkv + 2 * F.linear(z, up)
                q,k,v = [t.view(batch,1,h,d).transpose(1,2) for t in adapted.split(c,2)]
                refk = torch.cat((cache[0][:,:,:past], k), 2).float()
                refv = torch.cat((cache[1][:,:,:past], v), 2).float()
                ref = F.scaled_dot_product_attention(q.float(),refk,refv).transpose(1,2).reshape(batch,1,c)
                got = decode(qkv,cache,past,h,z=z,up=up,scale=2.)
                torch.testing.assert_close(got.float(),ref,atol=.012,rtol=.04)
                for actual, expected in zip(cache,(k,v)):
                    torch.testing.assert_close(actual[:,:,past:past+1], expected,atol=.003,rtol=.01)
                plain = decode(adapted,cache,past,h)
                torch.testing.assert_close(plain.float(),ref,atol=.012,rtol=.04)
                rows.append(dict(rank=rank,batch=batch,past=past,max_error=float((got.float()-ref).abs().max())))
    return rows


def model_checks(models):
    rows = []
    for prompt in (37, 896):
        ids = torch.randint(50257,(1,prompt+3),device='cuda')
        with mode(models[0], 'dynamic_flash'):
            refs=[]; cache=None
            for step in range(4):
                x = ids[:,:prompt] if step==0 else ids[:,prompt+step-1:prompt+step]
                logits,_,cache=models[0](x,past_key_values=cache,use_cache=True,last_token_only=True)
                refs.append(logits.float().clone())
        for name, merged in MODES:
            model=models[int(merged)]
            with mode(model,name):
                cache=None
                for step,ref in enumerate(refs):
                    x=ids[:,:prompt] if step==0 else ids[:,prompt+step-1:prompt+step]
                    logits,_,cache=model(x,past_key_values=cache,use_cache=True,last_token_only=True)
                    got=logits.float()
                    torch.testing.assert_close(got,ref,atol=.1,rtol=.02)
                    logp=ref.log_softmax(-1);logq=got.log_softmax(-1)
                    rows.append(dict(mode=name,merged=merged,prompt=prompt,step=step,
                        max_error=float((got-ref).abs().max()),
                        kl=float((logp.exp()*(logp-logq)).sum()),
                        argmax_agreement=bool(torch.equal(got.argmax(-1),ref.argmax(-1)))))
    return rows


def timed(model, ids, count):
    torch.cuda.synchronize()
    before=torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    wall=time.perf_counter();start.record()
    output=model.generate(ids,count,use_cache=True)
    end.record();end.synchronize()
    wall_ms=(time.perf_counter()-wall)*1000
    return dict(wall_ms=wall_ms,event_ms=start.elapsed_time(end),
                incremental_peak_bytes=torch.cuda.max_memory_allocated()-before,
                output_ids=output[:,ids.size(1):].tolist())


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['discovery','confirmation'],required=True)
    parser.add_argument('--replicate',type=int,default=0)
    args=parser.parse_args()
    job=os.environ.get('SLURM_JOB_ID','local')
    path=HERE/'results'/f'{args.phase}-{args.replicate}-{job}.json'
    path.parent.mkdir(exist_ok=True)
    if path.exists():raise RuntimeError('Refuse to overwrite evidence')
    torch.set_num_threads(1)
    torch.manual_seed(71000+args.replicate)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    sources={str(p.relative_to(ROOT)):sha(p) for p in [HERE/'kernel.py',HERE/'runtime.py',
             Path(__file__),HERE/'PROTOCOL.md',ROOT/'mingpt/model.py']}
    result=dict(complete=False,phase=args.phase,replicate=args.replicate,job=job,
        node=platform.node(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        gpu_telemetry=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version,clocks.sm,temperature.gpu',
                                             '--format=csv'],text=True),
        sources=sources,adapter_sha256=sha(CHECKPOINT),samples=[])
    save(path,result)
    try:
        result['kernel_checks']=kernel_checks()
        print('KERNEL GATES PASS',flush=True)
        model=GPT.from_pretrained('gpt2').enable_lora(4,8)
        missing,unexpected=model.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True),strict=False)
        assert not unexpected and all('.A.' not in n and '.B.' not in n for n in missing)
        for m in model.modules():
            if isinstance(m,torch.nn.Dropout):m.p=0.
        model=model.eval().half().cuda()
        merged=copy.deepcopy(model)
        merge(merged)
        models=[model,merged]
        result['weights_before']=[fingerprint(m) for m in models]
        result['model_checks']=model_checks(models)
        print('MODEL GATES PASS',flush=True)
        prompts=(32,896) if args.phase=='discovery' else (128,512)
        count=32 if args.phase=='discovery' else 64
        repeats=7 if args.phase=='discovery' else 10
        rng=random.Random(91234+args.replicate)
        for batch in (1,4):
            for prompt in prompts:
                for block in range(-3,repeats):
                    ids=torch.randint(50257,(batch,prompt),device='cuda')
                    order=list(MODES);rng.shuffle(order)
                    for name,is_merged in order:
                        with mode(models[int(is_merged)],name):
                            row=timed(models[int(is_merged)],ids,count)
                        if block>=0:
                            result['samples'].append(dict(batch=batch,prompt=prompt,new_tokens=count,block=block,
                                mode=name,merged=is_merged,prompt_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),**row))
                    if block>=0:save(path,result)
                print('WORKLOAD DONE',batch,prompt,count,flush=True)
        result['weights_after']=[fingerprint(m) for m in models]
        assert result['weights_before']==result['weights_after']
        result['complete']=True
    except Exception as e:
        result['failure']=repr(e)
        raise
    finally:
        save(path,result)
    print('COMPLETE',path,flush=True)


if __name__=='__main__':main()
