"""Validate all kernels and the full pipeline before releasing expensive jobs."""
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import torch
from experiments.paper_hyperparameters.common import save,read,configure
from experiments.paper_hyperparameters.protocol import DESIGN,TILES,BASE,BASE_TILE,WORKLOADS
from experiments.paper_hyperparameters.engine import (load_model,tokenizer,encoded,fingerprint,kernel_checks,cache_checks,
                                                       timed,setting,likelihood,generate,backend,amp,cleanup)


def run(study):
    env=configure(study);model=load_model(BASE,0);tok=tokenizer()
    result=dict(complete=False,environment=env,frozen_base_sha256=fingerprint(model),
                tokenizer_sha256=hashlib.sha256(json.dumps(tok.get_vocab(),sort_keys=True).encode()).hexdigest(),tiles={})
    for tile in TILES:
        name='fa-'+'-'.join(map(str,tile))
        try:result['tiles'][name]=dict(valid=True,tiles=tile,checks=kernel_checks(tile))
        except Exception as e:
            if isinstance(e,torch.cuda.OutOfMemoryError):raise
            result['tiles'][name]=dict(valid=False,tiles=tile,error=repr(e))
        print('PREFLIGHT',name,result['tiles'][name]['valid'],flush=True)
    assert result['tiles']['fa-'+'-'.join(map(str,BASE_TILE))]['valid']
    result['cache_checks']={}
    for chunk in DESIGN['cache_chunks']:
        try:result['cache_checks'][str(chunk)]=dict(valid=True,checks=cache_checks(model,chunk))
        except AssertionError as e:result['cache_checks'][str(chunk)]=dict(valid=False,error=str(e))
    assert result['cache_checks']['128']['valid']
    train=encoded('train',tok);val=encoded('validation',tok)
    result['initial_validation_nll']=likelihood(model,val)
    # Real forward/backward/update with the largest adapter rank; audit frozen tensors.
    del model;cleanup();wide=dict(rank=32,alpha=128,lr=.001);model=load_model(wide,0)
    before=fingerprint(model);model.train();params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=wide['lr'],weight_decay=0.);scaler=torch.amp.GradScaler('cuda',init_scale=128)
    x,y=(torch.stack([r[k] for r in train[:8]]).cuda() for k in (0,1))
    with amp():loss=model(x,y)[1]
    scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    scaler.step(opt);scaler.update()
    assert all(float(s['step'])==1 for s in opt.state.values());assert before==fingerprint(model)
    result['smoke_training_loss']=float(loss.detach());model.eval()
    for name,s in [('eager',setting()),('cache',setting(cached=True)),('prototype',setting(attention='prototype')),
                   ('combined',setting(cached=True,attention='prototype'))]:
        w=dict(batch=4,prompt=896,new_tokens=128)
        ids=torch.randint(50257,(w['batch'],w['prompt']),generator=torch.Generator().manual_seed(77)).cuda()
        result.setdefault('max_workload_smoke',{})[name]=timed(model,ids,w['new_tokens'],s)
    ids=torch.randint(50257,(1,128),device='cuda')
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        timed(model,ids,8,setting(cached=True,attention='prototype'))
    result['prototype_profiler_keys']=[e.key for e in prof.key_averages() if '_fwd' in e.key.lower() or 'triton' in e.key.lower()]
    from experiments.paper_hyperparameters import fa_kernel
    fa_kernel.TILES=tuple(BASE_TILE)
    q=torch.randn(1,1,37,64,device='cuda',dtype=torch.float16)
    with torch.no_grad():_,_,compiled=fa_kernel.forward(q,q,q,0)
    (study/'prototype-baseline.ptx').write_text(compiled.asm['ptx'])
    trace=study/'prototype.trace.json';prof.export_chrome_trace(str(trace))
    with trace.open('rb') as inp,gzip.open(str(trace)+'.gz','wb') as out:shutil.copyfileobj(inp,out)
    trace.unlink()
    (study/'software-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    (study/'torch-config.txt').write_text(torch.__config__.show())
    result['complete']=True;save(study/'preflight.json',result)
