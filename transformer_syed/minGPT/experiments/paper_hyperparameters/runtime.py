"""Blocked runtime discovery and independently allocated factorial confirmation."""
import hashlib
import itertools
from pathlib import Path
import random
import statistics
import time
import torch
from experiments.paper_hyperparameters.common import save,read,configure,environment
from experiments.paper_hyperparameters.protocol import (DESIGN,BASE,BASE_CHUNK,BASE_TILE,CHUNKS,TILES,SEARCH_SEEDS,CONFIRM_SEEDS,SEARCH_WORKLOADS,WORKLOADS,config_key,workload_key)
from experiments.paper_hyperparameters.engine import (DATA,tokenizer,encoded,load_model,likelihood,kernel_checks,cache_checks,setting,timed,cleanup,backend,generate,amp,fingerprint)
from lora_experiment import load_rows,prompt as question_prompt
from experiments.sports.scoring import score,aggregate


def checkpoint(study,config,seed,phase):
    return study/phase/config_key(config)/f'seed-{seed}'/'adapter.pt'


def prompt_ids(w,seed):
    return torch.randint(50257,(w['batch'],w['prompt']),generator=torch.Generator().manual_seed(seed)).cuda()


def benchmark(study,path,models,modes,workloads,repeats,seed,env):
    if path.exists() and read(path).get('complete'):return read(path)
    weights_before={k:fingerprint(m,frozen=False) for k,m in models.items()}
    result=dict(complete=False,environment=env,modes=modes,workloads={},events=[],weights_before=weights_before)
    # Resume only completed workload blocks; incomplete blocks repeat in full.
    if path.exists():
        previous=read(path)
        if previous['modes']!=modes or previous['weights_before']!=weights_before:raise RuntimeError('Cannot resume altered timing modes/weights')
        if previous['environment']['job']!=env['job']:
            archived=path.with_name(path.stem+'-aborted-'+str(previous['environment']['job'])+path.suffix)
            path.replace(archived)
        else:
            result=previous
    for wi,w in enumerate(workloads):
        wk=workload_key(w)
        if wk in result['workloads']:continue
        samples={name:[] for name in modes}
        for rep in range(-DESIGN['timing']['warmups'],repeats):
            ids=prompt_ids(w,seed+wi*1000+rep+100)
            names=list(modes);random.Random(seed+wi*1000+rep+200).shuffle(names)
            for order,name in enumerate(names):
                m=modes[name]
                row=timed(models[m['model']],ids,w['new_tokens'],m['setting'])
                if rep>=0:samples[name].append(dict(repeat=rep,order=order,prompt_seed=seed+wi*1000+rep+100,**row))
            if rep>=0 and rep%5==0:print('TIMING',path.name,wk,rep,flush=True)
        result['workloads'][wk]=dict(workload=w,samples=samples)
        result['events'].append(dict(workload=wk,telemetry=environment()))
        save(path,result)
    result['weights_after']={k:fingerprint(m,frozen=False) for k,m in models.items()}
    assert result['weights_before']==result['weights_after']
    result['complete']=True;save(path,result);return result


def search_runtime(study,index):
    env=configure(study);seed=SEARCH_SEEDS[index]
    model=load_model(BASE,seed,checkpoint(study,BASE,seed,'search-training'),study)
    val=encoded('validation',tokenizer(study));reference=likelihood(model,val)
    valid_tiles={k for k,v in read(study/'preflight.json')['tiles'].items() if v['valid']}
    modes={};gates=dict(reference_nll=reference,cache={},fa={})
    for chunk in CHUNKS:
        name=f'cache-{chunk}'
        if not read(study/'preflight.json')['cache_checks'][str(chunk)]['valid']:
            gates['cache'][name]=dict(valid=False,error='Preflight disqualified');continue
        try:
            checks=cache_checks(model,chunk)
            nll=likelihood(model,val,setting(cached=True,chunk=chunk))
            if abs(nll-reference)>DESIGN['quality']['max_abs_nll_delta']:raise ValueError('Cache NLL quality gate failed')
            gates['cache'][name]=dict(valid=True,chunk=chunk,nll=nll,logit_checks=checks)
            modes[name]=dict(model='base',setting=setting(cached=True,chunk=chunk))
        except (AssertionError,ValueError) as e:gates['cache'][name]=dict(valid=False,error=str(e))
    for tile in TILES:
        name='fa-'+'-'.join(map(str,tile))
        if name not in valid_tiles:
            gates['fa'][name]=dict(valid=False,error='Preflight disqualified');continue
        nll=likelihood(model,val,setting(attention='prototype',tiles=tile))
        valid=abs(nll-reference)<=DESIGN['quality']['max_abs_nll_delta']
        gates['fa'][name]=dict(valid=valid,tiles=list(tile),nll=nll)
        if valid:modes[name]=dict(model='base',setting=setting(attention='prototype',tiles=tile))
    assert gates['cache'][f'cache-{BASE_CHUNK}']['valid']
    assert gates['fa']['fa-'+'-'.join(map(str,BASE_TILE))]['valid']
    save(study/'runtime-search'/f'gates-{index}.json',dict(environment=env,**gates))
    benchmark(study,study/'runtime-search'/f'session-{index}.json',{'base':model},modes,SEARCH_WORKLOADS,DESIGN['timing']['search_repeats'],410000+index*10000,env)


def confirmation_modes(selected):
    c=selected['chunk'];t=selected['tiles'];modes={}
    # 2^3 parameter-level factorial; all three methods stay enabled.
    for l,k,f in itertools.product([0,1],repeat=3):
        modes[f'factorial-{l}{k}{f}']=dict(model='tuned' if l else 'baseline',
                                            setting=setting(cached=True,chunk=c if k else BASE_CHUNK,attention='prototype',tiles=t if f else BASE_TILE))
    modes.update(lora_baseline=dict(model='baseline',setting=setting()),lora_tuned=dict(model='tuned',setting=setting()),
                 cache_baseline=dict(model='baseline',setting=setting(cached=True)),
                 cache_tuned=dict(model='baseline',setting=setting(cached=True,chunk=c)),
                 fa_baseline=dict(model='baseline',setting=setting(attention='prototype')),
                 fa_tuned=dict(model='baseline',setting=setting(attention='prototype',tiles=t)))
    return modes


def confirmation_runtime(study,index):
    env=configure(study);seed=CONFIRM_SEEDS[index//3];session=index%3
    selected=read(study/'selection.json')
    models={name:load_model(c,seed,checkpoint(study,c,seed,'confirmation-training'),study)
            for name,c in [('baseline',BASE),('tuned',selected['lora'])]}
    # Recheck selected kernels on each allocation; fail the confirmation if invalid.
    kernel_checks(selected['tiles'])
    benchmark(study,study/'confirmation-timing'/f'seed-{seed}-session-{session}.json',models,confirmation_modes(selected),WORKLOADS,
              DESIGN['timing']['confirmation_repeats'],800000+seed*1000+session*100000,env)


@torch.inference_mode()
def answers(model,tok,rows,config):
    result=[]
    with backend(model,config['backend'],config['tiles']),amp():
        for i,row in enumerate(rows):
            ids=tok.encode(question_prompt(row['question']),return_tensors='pt').cuda()
            output=generate(model,ids,DESIGN['quality']['decode_tokens'],config['cached'],config['chunk'])
            tokens=output[0,ids.size(1):].tolist()
            if tok.eos_token_id in tokens:tokens=tokens[:tokens.index(tok.eos_token_id)]
            text=tok.decode(tokens,skip_special_tokens=True).split('\nUser:')[0].strip()
            result.append(dict(match_id=row['match_id'],group_id=row['group_id'],kind=row['kind'],question=row['question'],
                               reference=row['answer'],text=text,token_ids=tokens,scores=score(row,text)))
            if i%100==0:print('QUALITY',config['backend'],i,len(rows),flush=True)
    return dict(scores=aggregate(result),answers=result)


def confirmation_quality(study,index):
    env=configure(study);seed=CONFIRM_SEEDS[index];selected=read(study/'selection.json')
    out=study/'confirmation-quality'/f'seed-{seed}.json'
    if out.exists() and read(out).get('complete'):return
    tok=tokenizer(study);val=encoded('validation',tok)
    rows=load_rows(DATA/'test.jsonl')+load_rows(DATA/'counterfactual.jsonl')
    result=dict(complete=False,environment=env,seed=seed,models={})
    if out.exists():result=read(out)
    for name,c in [('baseline',BASE),('tuned',selected['lora'])]:
        if name in result['models']:continue
        model=load_model(c,seed,checkpoint(study,c,seed,'confirmation-training'),study)
        configs=dict(eager=setting(),combined=setting(cached=True,chunk=selected['chunk'] if name=='tuned' else BASE_CHUNK,
                                                      attention='prototype',tiles=selected['tiles'] if name=='tuned' else BASE_TILE))
        record=dict(validation={},outputs={})
        for variant,s in configs.items():
            record['validation'][variant]=likelihood(model,val,s)
            record['outputs'][variant]=answers(model,tok,rows,s)
        # A gate failure is recorded, never used to choose another setting on test data.
        record['runtime_nll_gate_passed']=abs(record['validation']['eager']-record['validation']['combined'])<=DESIGN['quality']['max_abs_nll_delta']
        result['models'][name]=record;save(out,result);del model;cleanup()
    result['complete']=True;save(out,result)
