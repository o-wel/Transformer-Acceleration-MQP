"""Short A100 follow-up: FP32 eager/tiled only; separate from full A30 experiment."""
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from experiments.flash_attention.run_experiment import load,digest
from experiments.kv_cache.benchmark import timed_generate
from experiments.flash_attention.benchmark import summary

OUT=ROOT/'experiments/fp32_attention/results/a100'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'report.json'
    if path.exists():
        raise RuntimeError('Refusing to overwrite A100 follow-up')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assert 'A100' in torch.cuda.get_device_name()
    report=dict(complete=False,gpu=torch.cuda.get_device_name(),node=platform.node(),job=os.environ.get('SLURM_JOB_ID'),
                torch=torch.__version__,cuda=torch.version.cuda,tf32=False,dtype='float32',warmups=3,repeats=7,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='FP32 eager vs custom tiled, B1, P512/896, 64 generated tokens; no BF16/training/quality reevaluation',models={})
    path.write_text(json.dumps(report,indent=2))
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_tiled_attention.py')
    tests=unittest.TextTestRunner(verbosity=2).run(suite)
    assert tests.wasSuccessful()
    report['tests']=dict(run=tests.testsRun,failures=len(tests.failures),errors=len(tests.errors),skipped=len(tests.skipped))
    for name in ['pretrained','lora']:
        model=load(name,torch.float32).eval()
        before=digest(model)
        rows=[]
        for length in [512,896]:
            generator=torch.Generator().manual_seed(4200+1+length)
            x=torch.randint(50257,(1,length),generator=generator).cuda()
            modes=[(backend,cache) for backend in ['eager','tiled_fp32'] for cache in [False,True]]
            samples={b+('-cached' if c else '-uncached'):[] for b,c in modes}
            for repeat in range(10):
                # Rotate all four modes: same model, precision and GPU, paired runs.
                order=modes[repeat%4:]+modes[:repeat%4]
                for backend,cache in order:
                    model.set_attention_backend(backend)
                    out,metrics=timed_generate(model,x,64,cache)
                    if repeat>=3:
                        samples[backend+('-cached' if cache else '-uncached')].append(metrics)
                    del out
            row=dict(length=length,batch=1,new_tokens=64,samples=samples,summary={m:summary(s) for m,s in samples.items()})
            rows.append(row)
            print('A100',name,length,{m:round(s['wall_ms']['median'],2) for m,s in row['summary'].items()},flush=True)
        after=digest(model)
        assert before==after
        report['models'][name]=dict(performance=rows,weights_before=before,weights_after=after)
        path.write_text(json.dumps(report,indent=2))
        del model,x
        gc.collect();torch.cuda.empty_cache()
    report['complete']=True
    path.write_text(json.dumps(report,indent=2))
    print('A100 COMPLETE',flush=True)


if __name__=='__main__':
    main()
