import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time
import torch
from mingpt.model import GPT
from experiments.joint_decode.benchmark import CHECKPOINT,fingerprint,sha,save
from experiments.joint_decode.runtime import mode,merge
from experiments.graph_decode.runtime import Runner

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent


def timed(fn,ids,count):
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    before=torch.cuda.memory_allocated()
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start=time.perf_counter();a.record();out=fn(ids,count);b.record();b.synchronize()
    return dict(wall_ms=(time.perf_counter()-start)*1000,event_ms=a.elapsed_time(b),
        incremental_peak_mib=(torch.cuda.max_memory_allocated()-before)/2**20,
        output_ids=out[:,ids.size(1):].tolist())


def checks(model,runners):
    result=[]
    for b in (1,4):
        for p in (37,896):
            ids=torch.randint(50257,(b,p+3),device='cuda');refs=[];cache=None
            for step in range(4):
                x=ids[:,:p] if step==0 else ids[:,p+step-1:p+step]
                logits,_,cache=model(x,use_cache=True,past_key_values=cache,last_token_only=True)
                refs.append(logits.float().clone())
            by={}
            for name,runner in runners[b].items():
                runner.prefill(ids[:,:p]);vals=[runner.prefill_logits.float().clone()]
                for step in range(3):
                    runner.token.copy_(ids[:,p+step:p+step+1]);runner.replay()
                    vals.append(runner.step_logits.float().clone())
                for step,(got,ref) in enumerate(zip(vals,refs)):
                    torch.testing.assert_close(got,ref,atol=.1,rtol=.02)
                    result.append(dict(batch=b,prompt=p,mode=name,step=step,max_error=float((got-ref).abs().max()),
                        argmax_disagreements=int((got.argmax(-1)!=ref.argmax(-1)).sum())))
                by[name]=vals
            for suffix in ('plain','fused'):
                for a,c in zip(by['eager_'+suffix],by['graph_'+suffix]):
                    torch.testing.assert_close(a,c,atol=0.,rtol=0.)
    return result


@torch.inference_mode()
def main():
    torch.set_num_threads(1);torch.manual_seed(81007)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    job=os.environ['SLURM_JOB_ID'];out=HERE/'results'/f'pilot-{job}.json'
    out.parent.mkdir(exist_ok=True)
    if out.exists():raise RuntimeError('Refuse overwrite')
    report=dict(complete=False,job=job,gpu=torch.cuda.get_device_name(),node=platform.node(),
        sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),HERE/'runtime.py',HERE/'PILOT_PROTOCOL.md',
            ROOT/'experiments/joint_decode/kernel.py',ROOT/'mingpt/model.py']},adapter_sha256=sha(CHECKPOINT),samples=[])
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
        runners={};report['setup_ms']={}
        for b in (1,4):
            runners[b]={}
            for name,fused,graph,which in [('eager_plain',False,False,model),('graph_plain',False,True,model),
                    ('eager_fused',True,False,model),('graph_fused',True,True,model),('merged_graph',False,True,merged)]:
                runner=Runner(which,b,fused,graph);runners[b][name]=runner
                report['setup_ms'][f'{b}-{name}']=runner.setup_ms
                print('SETUP',b,name,runner.setup_ms,flush=True)
        report['checks']=checks(model,runners);save(out,report);print('NUMERICAL GATES PASS',flush=True)
        rng=random.Random(81007)
        for b in (1,4):
            for p in (32,896):
                for block in range(-3,7):
                    ids=torch.randint(50257,(b,p),device='cuda')
                    names=['original','previous_fusion']+list(runners[b]);rng.shuffle(names)
                    for name in names:
                        if name=='original':
                            row=timed(lambda x,n:model.generate(x,n,use_cache=True),ids,64)
                        elif name=='previous_fusion':
                            with mode(model,'fused_stream'):
                                row=timed(lambda x,n:model.generate(x,n,use_cache=True),ids,64)
                        else:row=timed(runners[b][name].generate,ids,64)
                        if block>=0:report['samples'].append(dict(batch=b,prompt=p,new_tokens=64,block=block,mode=name,
                            prompt_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),**row))
                    if block>=0:save(out,report)
                print('WORKLOAD DONE',b,p,flush=True)
        report['weights_after']=[fingerprint(m) for m in (model,merged)]
        assert report['weights_after']==report['weights_before'];report['complete']=True
    except Exception as e:
        report['failure']=repr(e);raise
    finally:save(out,report)


if __name__=='__main__':main()
