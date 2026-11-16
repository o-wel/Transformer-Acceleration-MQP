"""Frozen factorial extension: checkpoint quality x attention tuning x KV-cache tuning."""
import argparse
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENGINE = ROOT/'experiments/flash_attention/results/fp16_hyperparameter_search'
sys.path.insert(0,str(HERE))
from protocol import SEEDS, WORKLOADS, ATTENTION, CACHES, states, uid, selections, materialize
# Load the established engine under an explicit name to avoid this module's name.
import importlib.util
spec=importlib.util.spec_from_file_location('matrix_engine',ENGINE/'study.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
sys.path.insert(0,str(ENGINE))


def read(p): return json.loads(p.read_text())
def write(p,v): s.write(p,v)
def manifest_check(run):
    root=run.parents[0]
    manifest=read(root/'SNAPSHOT_SHA256.json')
    frozen_plan=root/'snapshot/repo/experiments/optimization_matrix/plans'/f'{run.name}.json'
    assert read(run/'plan.json')==read(frozen_plan), 'Run plan changed after freezing'
    for name,digest in manifest.items():
        f=root/'snapshot/repo'/name
        if hashlib.sha256(f.read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Frozen source changed: {name}')


def configure(model, config, workload):
    import cache
    # Always release previous static buffers, including when using the same policy.
    cache.enable(model,False)
    capacity=workload['prompt_tokens']+workload['new_tokens']-1 if config['cache_mode']=='static_request' else model.block_size
    if config['cache_mode'].startswith('static_'):
        for block in model.transformer.h:
            block.attn._matrix_capacity=capacity
        cache.enable(model,True)
    s.configure_attention(config['backend'],config['bm'],config['bn'],config['warps'])
    model.set_attention_backend('eager' if config['backend']=='eager' else 'tiled_fp32')


def install_capacity_cache():
    # Patch the frozen local cache module, not a previous study or its snapshot.
    import cache
    path=ENGINE/'cache.py'
    source=path.read_text()
    old='shape=(b,self.n_head,self.bias.size(-1),c//self.n_head)'
    new="shape=(b,self.n_head,getattr(self,'_matrix_capacity',self.bias.size(-1)),c//self.n_head)"
    assert source.count(old)==1
    namespace=dict(cache.__dict__)
    exec(compile(source.replace(old,new),str(path)+' [matrix capacity extension]','exec'),namespace)
    cache.forward=namespace['forward']


def load_model(out,label,seed):
    config=read(out/'plan.json')['configs'][label]
    r=read(out/'trials'/s.uid(dict(config=config,examples=9600,seed=seed))/'result.json')
    checkpoint=Path(r['checkpoint'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==r['checkpoint_sha256']
    return s.selected_model(out,config,seed)


def prepare(out):
    plan=read(out/'plan.json')
    results=[]
    for label,c in plan['configs'].items():
        for seed in SEEDS:
            r=s.trial(out,c,9600,seed)
            r['checkpoint_sha256']=hashlib.sha256(Path(r['checkpoint']).read_bytes()).hexdigest()
            write(out/'trials'/s.uid(dict(config=c,examples=9600,seed=seed))/'result.json',r)
            results.append(dict(label=label,**r))
    write(out/'training.json',dict(complete=True,results=results))


def reference(model,prompt,workload):
    from experiments.kv_cache.benchmark import timed_generate
    configure(model,dict(ATTENTION[0],cache_mode='off'),workload)
    with s.torch.inference_mode():
        logits=model(prompt,last_token_only=True)[0].detach().clone()
        output,_=timed_generate(model,prompt,workload['new_tokens'],False)
    return logits,output.detach().clone()


def measure(model,prompt,workload,config,ref,repeats):
    from experiments.kv_cache.benchmark import timed_generate
    import cache
    torch=s.torch
    cell=dict(config=config,status='pending',samples=[],wall_ms=None)
    try:
        configure(model,config,workload)
        with torch.inference_mode():
            got=model(prompt,last_token_only=True)[0]
            cell['logits_pass']=bool(torch.allclose(got,ref[0],atol=3e-4,rtol=3e-4))
            cell['max_abs_logit_error']=float((got-ref[0]).abs().max())
            del got
            cell['tokens_equal']=True
            for repeat in range(2+repeats):
                output,sample=timed_generate(model,prompt,workload['new_tokens'],config['cache_mode']!='off')
                cell['tokens_equal'] &= bool(torch.equal(output,ref[1]))
                del output
                if repeat>=2:
                    sample['static_capacity_bytes']=cache.allocated_bytes(model)
                    cell['samples'].append(sample)
        cell['wall_ms']=statistics.median(r['wall_ms'] for r in cell['samples'])
        cell['status']='ok' if cell['logits_pass'] and cell['tokens_equal'] else 'parity_failed'
        cell['dtype_audit']=model._dtype_audit.copy()
    except torch.cuda.OutOfMemoryError:
        cell.update(status='oom',samples=[])
    finally:
        cache.enable(model,False)
        gc.collect();torch.cuda.empty_cache()
    return cell


def calibrate(out,shard):
    torch=s.torch
    path=out/f'calibration-{shard}.json'
    existing=read(path) if path.exists() else []
    done={r['id'] for r in existing}
    for label in ('off','best','poor'):
        model=load_model(out,label,42).eval()
        for wi,workload in enumerate(WORKLOADS):
            if workload['batch']!=(1,4,8)[shard]:continue
            prompt=torch.randint(0,50257,(workload['batch'],workload['prompt_tokens']),generator=torch.Generator().manual_seed(123+wi)).cuda()
            ref=reference(model,prompt,workload)
            configs=[dict(a,cache_mode=c) for a,c in itertools.product(ATTENTION,CACHES)]
            random.Random(1000+wi).shuffle(configs)
            for config in configs:
                identity=uid(dict(lora=label,workload=workload,config=config))
                if identity in done:continue
                cell=measure(model,prompt,workload,config,ref,3)
                existing.append(dict(id=identity,lora=label,workload=workload,seed=42,**cell))
                write(path,existing)
            del prompt,ref
        del model;gc.collect();torch.cuda.empty_cache()


def select(out):
    rows=sum([read(out/f'calibration-{i}.json') for i in range(3)],[])
    expected=3*len(WORKLOADS)*len(ATTENTION)*len(CACHES)
    assert len(rows)==expected and len({r['id'] for r in rows})==expected
    picks=[dict(lora=lora,workload=w,**selections(rows,lora,w)) for lora,w in itertools.product(('off','best','poor'),WORKLOADS)]
    write(out/'serving_selection.json',dict(frozen_before_confirmation=True,calibration_prompt_seed=123,
        confirmation_prompt_seed=987,criterion='Passing calibration latency extrema; numerical failures diagnostic only',selections=picks))


def confirm(out,shard):
    torch=s.torch
    seed=SEEDS[shard//3];batch=(1,4,8)[shard%3]
    selection=read(out/'serving_selection.json');assert selection['frozen_before_confirmation']
    path=out/f'confirmation-{shard}.json'
    rows=read(path) if path.exists() else [];done={r['id'] for r in rows}
    for label in ('off','best','poor'):
        model=load_model(out,label,seed).eval()
        for wi,workload in enumerate(WORKLOADS):
            if workload['batch']!=batch:continue
            pick=next(p for p in selection['selections'] if p['lora']==label and p['workload']==workload)
            prompt=torch.randint(0,50257,(batch,workload['prompt_tokens']),generator=torch.Generator().manual_seed(987+wi)).cuda()
            ref=reference(model,prompt,workload)
            cases=[(state,materialize(state,pick)) for state in states() if state['lora']==label]
            # Also confirm joint calibration extrema, distinct from independently tuned factors.
            if label in ('best','poor'):
                cases += [(dict(lora=label,attention='joint_'+level,cache='joint_'+level),pick['joint'][level]) for level in ('best','poor')]
            random.Random(seed+wi).shuffle(cases)
            for state,config in cases:
                identity=uid(dict(state=state,workload=workload,seed=seed))
                if identity in done:continue
                if config is None:cell=dict(config=None,status='unavailable',samples=[],wall_ms=None)
                else:cell=measure(model,prompt,workload,config,ref,7)
                rows.append(dict(id=identity,state=state,lora=label,workload=workload,seed=seed,selection=pick,**cell))
                write(path,rows)
            del prompt,ref
        del model;gc.collect();torch.cuda.empty_cache()


def quality(out,shard):
    """All 27 states plus joint extrema, natural held-out prompts, canonical serving selection."""
    torch=s.torch;seed=SEEDS[shard]
    canonical=dict(batch=1,prompt_tokens=128,new_tokens=32)
    picks=read(out/'serving_selection.json')['selections']
    path=out/f'quality-{shard}.json'
    results=read(path) if path.exists() else [];done={uid(r['state']) for r in results}
    for label in ('off','best','poor'):
        model=load_model(out,label,seed).eval()
        pick=next(p for p in picks if p['lora']==label and p['workload']==canonical)
        cases=[(state,materialize(state,pick)) for state in states() if state['lora']==label]
        if label in ('best','poor'):
            cases += [(dict(lora=label,attention='joint_'+level,cache='joint_'+level),pick['joint'][level]) for level in ('best','poor')]
        for state,config in cases:
            if uid(state) in done:continue
            if config is None:
                results.append(dict(state=state,seed=seed,status='unavailable',metrics=None,outputs=[]));write(path,results);continue
            outputs=[]
            # Validate loss on the full original validation set for this backend.
            configure(model,config,canonical)
            validation_nll=s.nll(model,s.data(128,'validation'))
            if s.TASK=='spam':
                import spam_task
                # Single label prediction has no decode reuse; explicitly record this limitation.
                with torch.inference_mode():
                    metrics,outputs=spam_task.evaluate(model,s.load_rows(s.data_path('test')),s.tokenizer,torch)
                note='SMS is one next-token label decision; KV reuse is measured by the separate serving workloads.'
            else:
                from experiments.sports.scoring import score,aggregate
                for i,q in enumerate(s.load_rows(s.ROOT/'experiments/sports/test.jsonl')):
                    ids=s.tokenizer.encode(f"User: {q['question']}\nAssistant:")
                    x=torch.tensor([ids],device='cuda')
                    # Cache capacity follows actual natural prompt, not synthetic calibration length.
                    configure(model,config,dict(batch=1,prompt_tokens=len(ids),new_tokens=40))
                    with torch.inference_mode():
                        seq=model.generate(x,40,use_cache=config['cache_mode']!='off')[0,len(ids):].tolist()
                    if s.tokenizer.eos_token_id in seq:seq=seq[:seq.index(s.tokenizer.eos_token_id)]
                    text=s.tokenizer.decode(seq,skip_special_tokens=True).split('\nUser:')[0].strip()
                    outputs.append(dict(index=i,text=text,tokens=seq,scores=score(q,text)))
                metrics=aggregate(outputs);note='Natural task generation; serving settings frozen using canonical synthetic calibration workload.'
            results.append(dict(state=state,config=config,seed=seed,status='measured',validation_nll=validation_nll,
                metrics=metrics,outputs=outputs,note=note))
            write(path,results)
        del model;gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','prepare','calibrate','select','confirm','quality','report'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--shard',type=int,default=0)
    args=p.parse_args();out=args.out.resolve();manifest_check(out)
    plan=read(out/'plan.json');s.TASK=plan['task'];s.PRECISION=plan['precision']
    if args.stage=='select':select(out);return
    if args.stage=='report':
        from report import build
        build(out);return
    s.setup();install_capacity_cache()
    if args.stage=='smoke':
        import preflight
        preflight.run(s,out)
        # Explicitly exercise the newly added request-sized static cache.
        model=s.build(plan['configs']['best'],42).eval()
        w=dict(batch=1,prompt_tokens=39,new_tokens=8)
        x=s.torch.randint(0,50257,(1,39),device='cuda')
        ref=reference(model,x,w)
        for mode in ('static_request','static_context'):
            cell=measure(model,x,w,dict(ATTENTION[0],cache_mode=mode),ref,1)
            assert len(cell['samples'])==1 and cell['status'] in ('ok','parity_failed')
            assert cell['samples'][0]['static_capacity_bytes']>0
        write(out/'matrix_smoke.json',dict(complete=True,capacity_modes=['static_request','static_context']))
    elif args.stage=='prepare':prepare(out)
    elif args.stage=='calibrate':calibrate(out,args.shard)
    elif args.stage=='confirm':confirm(out,args.shard)
    elif args.stage=='quality':quality(out,args.shard)


if __name__=='__main__':main()
