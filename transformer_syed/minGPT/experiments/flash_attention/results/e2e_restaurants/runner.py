"""GPU stages for the isolated E2E search and integration study."""
import argparse,hashlib,json,sys
from pathlib import Path
import matrix as m
s=m.s
HERE=Path(__file__).resolve().parent

def integrity(out):
    root=out.parent
    for name,digest in json.loads((root/'SNAPSHOT_SHA256.json').read_text()).items():
        assert hashlib.sha256((root/'snapshot/repo'/name).read_bytes()).hexdigest()==digest,name
    frozen=root/'snapshot/repo'/HERE.relative_to(s.ROOT)/'search_plans'/f'{out.name}.json'
    assert json.loads(frozen.read_text())==json.loads((out/'search_plan.json').read_text())

def freeze_matrix(out):
    chosen=s.stage_results(out,'selection');assert chosen['frozen_before_test']
    rows=[json.loads(f.read_text()) for f in (out/'trials').glob('*/result.json')]
    eligible=[r for r in rows if r.get('complete') and r['examples']==9600 and r['seed']==42 and r['config']['method']=='lora']
    poor=max(eligible,key=lambda r:(r['validation_nll'],s.uid(r['config'])))
    plan=dict(task='e2e',precision=s.PRECISION,seeds=[42,43,44],examples=9600,
        configs=dict(off=chosen['full_control'],best=chosen['winner'],poor=poor['config']),
        poor_selection='Worst observed full-budget seed-42 validation NLL; newly confirmed on seeds 43/44',
        selection='Best observed three-seed validation NLL, not a global optimum',workloads=m.WORKLOADS)
    s.write(out/'plan.json',plan)
    s.write(out/'MATRIX_PLAN_SHA256.json',dict(sha256=hashlib.sha256((out/'plan.json').read_bytes()).hexdigest()))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--out',type=Path,required=True);p.add_argument('--shard',type=int,default=0)
    a=p.parse_args();out=a.out.resolve();integrity(out)
    plan=json.loads((out/'search_plan.json').read_text());s.TASK='e2e';s.PRECISION=plan['precision']
    if a.stage=='report':
        from reporting import build
        build(out)
        assert json.loads((out/'REPORT_STATUS.json').read_text())['status']=='COMPLETE','Incomplete report cannot release next precision'
        return
    if a.stage=='freeze':freeze_matrix(out);return
    s.setup();m.install_capacity_cache()
    if a.stage=='smoke':
        import preflight,task
        preflight.run(s,out)
        # Natural task and supervised encoding checks before the expensive sweep.
        rows=s.load_rows(s.data_path('train'))[:4]
        encoded=task.encode(rows,s.tokenizer,256,s.torch)
        assert all(int((y!=-1).sum())>1 for x,y in encoded)
        model=s.build(dict(s.grid()[0],method='full'),42).eval()
        metrics,outputs=task.evaluate(model,s.load_rows(s.data_path('test')),s.tokenizer,s.torch,
            config=dict(backend='eager',bm=16,bn=32,warps=4,cache_mode='off'),configure=m.configure)
        s.write(out/'pretrained-test.json',dict(metrics=metrics,outputs=outputs))
        # New cache-capacity policy is verified before matrix use.
        w=dict(batch=1,prompt_tokens=39,new_tokens=8)
        x=s.torch.randint(0,50257,(1,39),device='cuda');ref=m.reference(model,x,w);checks=[]
        for mode in ('static_request','static_context'):
            c=m.measure(model,x,w,dict(m.ATTENTION[0],cache_mode=mode),ref,1)
            capacity=46 if mode=='static_request' else model.block_size
            expected=2*len(model.transformer.h)*capacity*768*(2 if s.PRECISION=='fp16' else 4)
            assert c['samples'][0]['static_capacity_bytes']==expected
            checks.append(dict(mode=mode,status=c['status'],expected_bytes=expected))
        s.write(out/'matrix_smoke.json',dict(complete=True,checks=checks))
    elif a.stage in ('screen','refine','regularize','batch','confirm_training'):
        s.train_stage(out,'confirm' if a.stage=='confirm_training' else a.stage,a.shard)
    else:
        assert hashlib.sha256((out/'plan.json').read_bytes()).hexdigest()==json.loads((out/'MATRIX_PLAN_SHA256.json').read_text())['sha256']
        {'prepare':m.prepare,'calibrate':lambda x:m.calibrate(x,a.shard),'select':m.select,
         'confirm':lambda x:m.confirm(x,a.shard),'quality':lambda x:m.quality(x,a.shard)}[a.stage](out)
if __name__=='__main__':main()
