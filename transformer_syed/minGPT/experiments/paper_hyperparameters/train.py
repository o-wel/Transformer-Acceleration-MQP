"""Equal-budget adapter training with exact resume and frozen-weight auditing."""
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import torch
from experiments.paper_hyperparameters.common import save,read,configure,sha
from experiments.paper_hyperparameters.protocol import DESIGN,GRID,BASE,SEARCH_SEEDS,CONFIRM_SEEDS,config_key
from experiments.paper_hyperparameters.engine import load_model,tokenizer,encoded,likelihood,amp,fingerprint,cleanup


def atomic_torch(path,obj):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)


def train_one(study,config,seed,phase):
    out=study/phase/config_key(config)/f'seed-{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'training.json').exists() and read(out/'training.json').get('complete'):return
    env=configure(study);tok=tokenizer(study);data=encoded('train',tok);val=encoded('validation',tok)
    model=load_model(config,seed,study=study);frozen=fingerprint(model)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=config['lr'],weight_decay=0.)
    scaler=torch.amp.GradScaler('cuda',init_scale=128,growth_interval=1000000)
    budget=DESIGN['train'];rng=random.Random(seed)
    schedule=[rng.choices(range(len(data)),k=budget['batch']) for _ in range(budget['steps'])]
    report=dict(complete=False,config=config,seed=seed,phase=phase,environment=env,validation=[],step_ms=[],warm_step_ms=[],
                schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),
                frozen_base_before=frozen,trainable_parameters=sum(p.numel() for p in params),
                dropout=[m.p for m in model.modules() if isinstance(m,torch.nn.Dropout)],peak_allocated_bytes=0)
    start_step=0;best=float('inf');torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    if (out/'resume.pt').exists():
        state=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
        assert state['config']==config and state['seed']==seed and state['schedule_sha256']==report['schedule_sha256']
        model.load_state_dict(state['adapter'],strict=False);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        report=state['report'];start_step=state['step'];best=report['validation_nll'];report.setdefault('resumed_environments',[]).append(env)
    elapsed_prior=report.get('training_elapsed_seconds',0.)
    begun=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for step in range(start_step+1,budget['steps']+1):
        x,y=(torch.stack([data[i][k] for i in schedule[step-1]]).cuda() for k in (0,1))
        model.train();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();begin=time.perf_counter()
        with amp():loss=model(x,y)[1]
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Non-finite training loss')
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params,budget['clip_norm'],error_if_nonfinite=True)
        old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
        if scaler.get_scale()<old_scale or not all(float(s['step'])==step for s in optimizer.state.values()):raise FloatingPointError('Skipped/non-finite update')
        report['step_ms'].append((time.perf_counter()-begin)*1000)
        if step>start_step+budget['warm_steps']:report['warm_step_ms'].append(report['step_ms'][-1])
        if step%budget['validate_every']==0:
            report['peak_allocated_bytes']=max(report['peak_allocated_bytes'],torch.cuda.max_memory_allocated())
            v=likelihood(model,val)
            if not float('-inf')<v<float('inf'):raise FloatingPointError('Non-finite validation loss')
            report['validation'].append(dict(step=step,nll=v))
            weights={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
            if v<best:
                best=v;report['selected_step']=step;atomic_torch(out/'adapter.pt',weights)
            report.update(validation_nll=best,successful_updates=step,warm_median_step_ms=statistics.median(report['warm_step_ms']),
                          measured_update_seconds=sum(report['step_ms'])/1000,training_elapsed_seconds=elapsed_prior+time.perf_counter()-begun)
            state=dict(config=config,seed=seed,step=step,schedule_sha256=report['schedule_sha256'],adapter=weights,optimizer=optimizer.state_dict(),
                       scaler=scaler.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),report=report)
            atomic_torch(out/'resume.pt',state);save(out/'training.json',report)
            print('TRAIN',phase,config_key(config),seed,step,v,flush=True)
            optimizer.zero_grad(set_to_none=True);cleanup();torch.cuda.reset_peak_memory_stats()
    report['frozen_base_after']=fingerprint(model);assert report['frozen_base_after']==frozen
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    report['adapter_sha256']=sha(out/'adapter.pt')
    report['complete']=True;save(out/'training.json',report)
    del model,optimizer,params;cleanup()


def run_search(study,index):
    c=GRID[index]
    for seed in SEARCH_SEEDS:train_one(study,c,seed,'search-training')


def run_confirmation(study,index):
    selected=read(study/'selection.json')['lora'];seed=CONFIRM_SEEDS[index]
    configs=[BASE,selected] if selected!=BASE else [BASE]
    random.Random(909+seed).shuffle(configs)
    for c in configs:train_one(study,c,seed,'confirmation-training')
