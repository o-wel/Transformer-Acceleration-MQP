"""Supplement: native-FP16 inference policy, paired eager/Flash, separate allocation."""
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from transformers import GPT2Tokenizer
import experiments.fp16_fp32_attention.run_experiment as primary
from experiments.flash_attention.run_experiment import load,digest
from experiments.fp32_attention.run_experiment import evaluate

OUT=ROOT/'experiments/fp16_fp32_attention/results/native_fp16'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'report.json'
    if path.exists(): raise RuntimeError('Refusing to overwrite native-FP16 supplement')
    torch.set_num_threads(1);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # Reuse the same timing/seeding/shuffling and finite-logit checks, with no
    # autocast: weights, embeddings, masks, Q/K/V and outputs are natively half.
    primary.OUT=OUT
    primary.VARIANTS={'fp16_native_eager':('eager',False),'fp16_native_flash':('flash',False)}
    report=dict(complete=False,gpu=torch.cuda.get_device_name(),node=platform.node(),job=os.environ.get('SLURM_JOB_ID'),
                torch=torch.__version__,cuda=torch.version.cuda,policy='native float16 weights/activations; no autocast; inference only',
                warmups=3,repeats=7,allocation_note='Separate allocation; do not pool samples with the primary AMP experiment',models={})
    report['source_sha256']={}
    for filename in ['experiments/fp16_fp32_attention/native_inference.py','experiments/fp16_fp32_attention/run_experiment.py','mingpt/model.py','mingpt/tiled_attention.py']:
        content=(ROOT/filename).read_bytes()
        report['source_sha256'][filename]=hashlib.sha256(content).hexdigest()
        target=OUT/'source'/filename;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    path.write_text(json.dumps(report,indent=2))
    tokenizer=GPT2Tokenizer.from_pretrained('gpt2',local_files_only=True)
    questions=[json.loads(s) for s in (ROOT/'experiments/sports/test.jsonl').read_text().splitlines()]
    for name in ['pretrained','lora']:
        model=load(name,torch.float16).eval();before=digest(model)
        info=dict(weight_hash_before=before,inference=primary.inference(model,name),quality={})
        for variant,(backend,_) in primary.VARIANTS.items():
            model.set_attention_backend(backend)
            finite=[]
            hook=model.lm_head.register_forward_hook(lambda m,i,o:finite.append(torch.isfinite(o).all()))
            try:
                result=evaluate(model,tokenizer,questions)
                assert bool(torch.stack(finite).all()),f'{name}/{variant}: nonfinite native-FP16 quality logits'
            finally: hook.remove()
            info['quality'][variant]=result
        info['weight_hash_after']=digest(model);assert before==info['weight_hash_after']
        report['models'][name]=info;path.write_text(json.dumps(report,indent=2))
        del model;gc.collect();torch.cuda.empty_cache()
    report['complete']=True;path.write_text(json.dumps(report,indent=2))
    print('NATIVE FP16 COMPLETE',flush=True)


if __name__=='__main__': main()
