import argparse
import copy
import hashlib
import json
import math
import random
import statistics
import time
import torch
from experiments.graph_decode_optimal.common import HERE,ROOT,GRID,environment,load,references,check_runner,fingerprint,sha,save
from experiments.graph_decode_optimal.runtime import Runner
from experiments.graph_decode.pilot import timed
from experiments.joint_decode.runtime import mode,merge
from experiments.graph_decode_optimal.quality import run as quality_run


@torch.inference_mode()
def search(seed):
    report=environment(seed,'search');path=HERE/'results'/f"search-{seed}-{report['job']}.json"
    if path.exists():raise RuntimeError('Refuse overwrite')
    save(path,report)
    try:
        model,report['checkpoint']=load(seed,'search');report['weights_before']=fingerprint(model)
        refs=references(model);runners={};report['configs']={}
        for bn,warps in GRID:
            key=f'{bn}-{warps}';record=dict(bn=bn,warps=warps,valid=False,checks=[],setup={})
            report['configs'][key]=record
            try:
                for b in (1,4):
                    torch.cuda.synchronize();start=time.perf_counter()
                    runner=Runner(model,b,True,True,bn,warps);torch.cuda.synchronize()
                    record['setup'][str(b)]=(time.perf_counter()-start)*1000
                    checks,_=check_runner(runner,refs);record['checks']+=checks;runners[(b,key)]=runner
                record['valid']=True
            except torch.cuda.OutOfMemoryError:raise
            except Exception as e:record['failure']=repr(e)
            print('CONFIG',key,record['valid'],flush=True);save(path,report)
        assert report['configs']['128-4']['valid'],'Original control must pass'
        eligible=[k for k,v in report['configs'].items() if v['valid']]
        report['samples']=[];rng=random.Random(20261110+seed)
        for b in (1,4):
            for p in (32,896):
                for block in range(-2,5):
                    ids=torch.randint(50257,(b,p),device='cuda');order=eligible[:];rng.shuffle(order)
                    for key in order:
                        row=timed(runners[(b,key)].generate,ids,32)
                        if block>=0:report['samples'].append(dict(batch=b,prompt=p,new_tokens=32,block=block,config=key,
                            prompt_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),**row))
                    if block>=0:save(path,report)
                print('SEARCH WORKLOAD',b,p,'DONE',flush=True)
        report['weights_after']=fingerprint(model);assert report['weights_before']==report['weights_after']
        report['complete']=True
    except Exception as e:report['failure']=repr(e);raise
    finally:save(path,report)


def select():
    files=sorted((HERE/'results').glob('search-*.json'));runs=[json.loads(p.read_text()) for p in files]
    assert len(runs)==3 and all(r['complete'] for r in runs)
    assert sorted(r['seed'] for r in runs)==[11,22,33]
    assert all(r['sources']==runs[0]['sources'] for r in runs)
    rows=[]
    for bn,w in GRID:
        key=f'{bn}-{w}';valid=all(r['configs'][key]['valid'] for r in runs);medians=[]
        if valid:
            for r in runs:
                for b in (1,4):
                    for p in (32,896):
                        vals=[s['wall_ms'] for s in r['samples'] if s['config']==key and s['batch']==b and s['prompt']==p]
                        assert len(vals)==5;medians.append(statistics.median(vals))
        rows.append(dict(config=key,bn=bn,warps=w,valid=valid,
            score_ms=math.exp(statistics.mean(map(math.log,medians))) if valid else None))
    winner=min((r for r in rows if r['valid']),key=lambda r:r['score_ms'])
    result=dict(selected=winner,candidates=rows,search_files={p.name:sha(p) for p in files},sources=runs[0]['sources'],
        protocol_sha256=sha(HERE/'PROTOCOL.md'),selection_uses_confirmation=False)
    path=HERE/'selection.json'
    if path.exists():raise RuntimeError('Refuse overwrite selection')
    save(path,result);(HERE/'selection.sha256').write_text(sha(path)+'\n');print(json.dumps(winner,indent=2))


@torch.inference_mode()
def confirm(seed):
    report=environment(seed,'confirmation');path=HERE/'results'/f"confirmation-{seed}-{report['job']}.json"
    if path.exists():raise RuntimeError('Refuse overwrite')
    selection=json.loads((HERE/'selection.json').read_text())
    assert sha(HERE/'selection.json')==(HERE/'selection.sha256').read_text().strip()
    assert report['sources']==selection['sources'],'Source drift after runtime discovery'
    selected=(selection['selected']['bn'],selection['selected']['warps'])
    report.update(selection_sha256=sha(HERE/'selection.json'),selected=selected,samples=[],setup={});save(path,report)
    try:
        model,report['checkpoint']=load(seed,'confirmation');merged=copy.deepcopy(model);merge(merged)
        report['weights_before']=[fingerprint(m) for m in (model,merged)];runners={}
        refs=references(model);report['checks']=[]
        configs=[('eager_fused',True,False,model,selected),('plain_graph',False,True,model,selected),
            ('default_graph',True,True,model,(128,4)),('selected_graph',True,True,model,selected),
            ('merged_graph',False,True,merged,selected)]
        for b in (1,4):
            runners[b]={};values={}
            for name,fused,graphed,which,tiles in configs:
                torch.cuda.synchronize();before=torch.cuda.memory_allocated();start=time.perf_counter()
                runner=Runner(which,b,fused,graphed,*tiles);torch.cuda.synchronize();runners[b][name]=runner
                report['setup'][f'{b}-{name}']=dict(total_ms=(time.perf_counter()-start)*1000,
                    warmup_capture_ms=runner.setup_ms,incremental_resident_mib=(torch.cuda.memory_allocated()-before)/2**20)
                checks,values[name]=check_runner(runner,refs)
                report['checks'] += [dict(mode=name,**c) for c in checks]
            for a,z in zip(values['eager_fused'],values['selected_graph']):
                for x,y in zip(a,z):torch.testing.assert_close(x,y,atol=0,rtol=0)
            runner=runners[b]['selected_graph']
            for p in (1,1023):
                ids=torch.randint(50257,(b,p),device='cuda');runner.prefill(ids);runner.replay()
                ref=model(torch.cat((ids,runner.output[:,:1]),dim=1),last_token_only=True)[0]
                torch.testing.assert_close(runner.step_logits,ref,atol=.1,rtol=.02)
            runner.position.zero_();runner.write_index.zero_();runner.token.fill_(42);runner.replay()
            ref=model(torch.full((b,1),42,device='cuda',dtype=torch.long))[0]
            torch.testing.assert_close(runner.step_logits,ref,atol=.1,rtol=.02)
            ids=torch.randint(50257,(b,37),device='cuda');first=runner.generate(ids,4).clone()
            runner.generate(torch.randint(50257,(b,71),device='cuda'),5)
            assert torch.equal(first,runner.generate(ids,4))
            for bad in [(torch.zeros((b,1023),device='cuda',dtype=torch.long),2),
                        (torch.zeros((b+1,37),device='cuda',dtype=torch.long),2)]:
                try:runner.generate(*bad)
                except AssertionError:pass
                else:raise AssertionError('Unsupported context/batch accepted')
        print('CONFIRMATION GATES PASS',flush=True);save(path,report)
        report['resident_allocated_mib']=torch.cuda.memory_allocated()/2**20
        rng=random.Random(20261110+seed)
        modes=['original','previous_fusion','eager_fused','plain_graph','default_graph','selected_graph','merged_graph']
        for b in (1,4):
            for p in (128,512):
                for count in (32,128):
                    for block in range(-3,10):
                        ids=torch.randint(50257,(b,p),device='cuda');order=modes[:];rng.shuffle(order)
                        for name in order:
                            if name=='original':row=timed(lambda x,n:model.generate(x,n,use_cache=True),ids,count)
                            elif name=='previous_fusion':
                                with mode(model,'fused_stream'):row=timed(lambda x,n:model.generate(x,n,use_cache=True),ids,count)
                            else:row=timed(runners[b][name].generate,ids,count)
                            if block>=0:report['samples'].append(dict(batch=b,prompt=p,new_tokens=count,block=block,mode=name,
                                prompt_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),**row))
                        if block>=0:save(path,report)
                    print('CONFIRM WORKLOAD',b,p,count,'DONE',flush=True)
        report['quality']=quality_run(model,merged,runners,selected);save(path,report)
        report['weights_after']=[fingerprint(m) for m in (model,merged)]
        assert report['weights_after']==report['weights_before']
        assert all(v['nll_gate_passed'] for v in report['quality']['summary']['comparisons'].values())
        report['complete']=True
    except Exception as e:report['failure']=repr(e);raise
    finally:save(path,report)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['search','select','confirm']);p.add_argument('--seed',type=int)
    args=p.parse_args()
    if args.phase=='search':search(args.seed)
    elif args.phase=='confirm':confirm(args.seed)
    else:select()


if __name__=='__main__':main()
