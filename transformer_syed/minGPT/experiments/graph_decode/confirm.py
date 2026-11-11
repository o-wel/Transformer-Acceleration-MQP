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
from mingpt.model import GPT
from experiments.joint_decode.benchmark import CHECKPOINT,fingerprint,sha,save
from experiments.joint_decode.runtime import mode,merge
from experiments.graph_decode.runtime import Runner
from experiments.graph_decode.pilot import timed,checks

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replicate',type=int,required=True)
    args=parser.parse_args();job=os.environ['SLURM_JOB_ID']
    out=HERE/'results'/f'confirmation-{args.replicate}-{job}.json'
    if out.exists():raise RuntimeError('Refuse overwrite')
    torch.set_num_threads(1);torch.manual_seed(82000+args.replicate)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    report=dict(complete=False,replicate=args.replicate,job=job,gpu=torch.cuda.get_device_name(),node=platform.node(),
        torch=torch.__version__,cuda=torch.version.cuda,
        gpu_telemetry=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version,clocks.sm,temperature.gpu','--format=csv'],text=True),
        sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),HERE/'runtime.py',HERE/'pilot.py',HERE/'CONFIRMATION_PROTOCOL.md',
            ROOT/'experiments/joint_decode/kernel.py',ROOT/'experiments/joint_decode/runtime.py',ROOT/'mingpt/model.py']},
        adapter_sha256=sha(CHECKPOINT),samples=[],setup={})
    save(out,report)
    try:
        model=GPT.from_pretrained('gpt2').enable_lora(4,8)
        missing,unexpected=model.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True),strict=False)
        assert not unexpected and all('.A.' not in x and '.B.' not in x for x in missing)
        for m in model.modules():
            if isinstance(m,torch.nn.Dropout):m.p=0.
        model=model.half().eval().cuda();model.set_attention_backend('flash')
        merged=copy.deepcopy(model);merge(merged)
        report['weights_before']=[fingerprint(m) for m in (model,merged)]
        runners={}
        for b in (1,4):
            runners[b]={}
            for name,fused,graph,which in [('eager_plain',False,False,model),('graph_plain',False,True,model),
                    ('eager_fused',True,False,model),('graph_fused',True,True,model),('merged_graph',False,True,merged)]:
                torch.cuda.synchronize();before=torch.cuda.memory_allocated();start=time.perf_counter()
                runner=Runner(which,b,fused,graph);torch.cuda.synchronize();runners[b][name]=runner
                report['setup'][f'{b}-{name}']=dict(total_ms=(time.perf_counter()-start)*1000,
                    warmup_capture_ms=runner.setup_ms,incremental_resident_mib=(torch.cuda.memory_allocated()-before)/2**20)
        report['checks']=checks(model,runners)
        # Avoid a thousand-token original rollout in the boundary smoke check:
        # exercise one graph step at the first and last positions separately.
        report['boundaries']={}
        for b in (1,4):
            runner=runners[b]['graph_fused'];values=[]
            for p in (1,1023):
                ids=torch.randint(50257,(b,p),device='cuda')
                runner.prefill(ids);runner.replay()
                x=torch.cat((ids,runner.output[:,:1]),dim=1)
                ref=model(x,last_token_only=True)[0]
                torch.testing.assert_close(runner.step_logits,ref,atol=.1,rtol=.02)
                values.append(dict(prompt=p,max_error=float((runner.step_logits-ref).abs().max())))
            ids=torch.randint(50257,(b,37),device='cuda');first=runner.generate(ids,4).clone()
            runner.generate(torch.randint(50257,(b,71),device='cuda'),5)
            assert torch.equal(first,runner.generate(ids,4))
            runner.position.zero_();runner.write_index.zero_();runner.token.fill_(42);runner.replay()
            ref=model(torch.full((b,1),42,device='cuda',dtype=torch.long))[0]
            torch.testing.assert_close(runner.step_logits,ref,atol=.1,rtol=.02)
            for bad in [(torch.zeros((b,1023),device='cuda',dtype=torch.long),2),
                        (torch.zeros((b+1,37),device='cuda',dtype=torch.long),2)]:
                try:runner.generate(*bad)
                except AssertionError:pass
                else:raise AssertionError('Unsupported shape/context accepted')
            report['boundaries'][str(b)]=values
        print('ALL NUMERICAL/BOUNDARY GATES PASS',flush=True);save(out,report)
        report['resident_allocated_mib']=torch.cuda.memory_allocated()/2**20
        rng=random.Random(82000+args.replicate)
        for b in (1,4):
            for p in (128,512):
                for count in (32,128):
                    for block in range(-3,10):
                        ids=torch.randint(50257,(b,p),device='cuda')
                        names=['original','previous_fusion','eager_fused','graph_plain','graph_fused','merged_graph'];rng.shuffle(names)
                        for name in names:
                            if name=='original':row=timed(lambda x,n:model.generate(x,n,use_cache=True),ids,count)
                            elif name=='previous_fusion':
                                with mode(model,'fused_stream'):row=timed(lambda x,n:model.generate(x,n,use_cache=True),ids,count)
                            else:row=timed(runners[b][name].generate,ids,count)
                            if block>=0:report['samples'].append(dict(batch=b,prompt=p,new_tokens=count,block=block,mode=name,
                                prompt_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),**row))
                        if block>=0:save(out,report)
                    print('WORKLOAD',b,p,count,'DONE',flush=True)
        if args.replicate==0:
            ids=torch.randint(50257,(1,128),device='cuda');report['profiles']={}
            for name in ('eager_fused','graph_fused'):
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    runners[1][name].generate(ids,4)
                trace=HERE/'results'/f'profile-{job}-{name}.json';prof.export_chrome_trace(str(trace))
                report['profiles'][name]=[dict(key=e.key,count=e.count) for e in prof.key_averages()
                    if 'graph' in e.key.lower() or '_decode' in e.key or 'mm' in e.key.lower()]
        report['weights_after']=[fingerprint(m) for m in (model,merged)]
        assert report['weights_after']==report['weights_before'];report['complete']=True
    except Exception as e:
        report['failure']=repr(e);raise
    finally:save(out,report)


if __name__=='__main__':main()
