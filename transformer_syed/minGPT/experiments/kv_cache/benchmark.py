"""FP32 cached/uncached inference: paired timing, CUDA memory, and output parity."""
import argparse
import gc
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from transformers import GPT2Tokenizer
from mingpt.model import GPT
from sports_experiment import load_parameters
from experiments.sports.scoring import score, aggregate

OUT = ROOT / 'experiments/kv_cache/results'


def precision_context(fp16):
    return torch.autocast(device_type='cuda', dtype=torch.float16, enabled=fp16) if fp16 else nullcontext()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2))


def digest(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def timed_generate(model, prompt, new_tokens, cached, fp16=False):
    """Dense equal-length batch; no EOS stopping; measure exactly N generated tokens."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    start, first, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    idx = prompt
    cache = None
    begin = time.perf_counter()
    start.record()
    for i in range(new_tokens):
        cond = idx[:, -model.block_size:]
        if cached:
            if cache is not None and cache[0][0].size(2) < model.block_size:
                with precision_context(fp16):
                    logits, _, cache = model(idx[:, -1:], past_key_values=cache,
                                              use_cache=True, last_token_only=True)
            else:
                cache = None
                with precision_context(fp16):
                    logits, _, cache = model(cond, use_cache=True, last_token_only=True)
        else:
            with precision_context(fp16):
                logits, _ = model(cond, last_token_only=True)
        idx = torch.cat((idx, logits[:, -1].argmax(-1, keepdim=True)), dim=1)
        if i == 0:
            first.record()
    end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - begin) * 1000
    result = dict(wall_ms=wall_ms, prefill_ms=start.elapsed_time(first),
                  decode_ms=first.elapsed_time(end),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  baseline_allocated_bytes=baseline,
                  end_allocated_bytes=torch.cuda.memory_allocated(),
                  cache_bytes=0 if cache is None else sum(t.numel()*t.element_size() for pair in cache for t in pair))
    result['decode_ms_per_token_step'] = result['decode_ms'] / (new_tokens - 1)
    result['aggregate_tokens_per_second'] = prompt.size(0) * new_tokens / (wall_ms / 1000)
    return idx, result


@torch.inference_mode()
def check_logits(model, tokenizer, fp16=False):
    ids = tokenizer.encode('Basketball match: Bulls vs Sharks. Halftime score: Bulls 51, Sharks 35. Final score: Bulls 96, Sharks 69. Who won the match, and what was the final score?')
    x = torch.tensor([ids, ids], device='cuda')
    with precision_context(fp16):
        ref = model(x)[0]
    cache = None
    pieces = []
    for start in range(0, x.size(1), 7):
        with precision_context(fp16):
            logits, _, cache = model(x[:, start:start+7], past_key_values=cache, use_cache=True)
        pieces.append(logits)
    got = torch.cat(pieces, dim=1)
    if fp16:
        rtol, atol = 0.05, 0.25
    else:
        rtol, atol = 2e-4, 2e-4
    torch.testing.assert_close(got, ref, rtol=rtol, atol=atol)
    return dict(max_abs_logit_error=(got-ref).abs().max().item(), rtol=rtol, atol=atol,
                all_argmax_equal=bool(torch.equal(got.argmax(-1), ref.argmax(-1))))


@torch.inference_mode()
def check_quality(model, tokenizer, name, fp16=False):
    rows = [json.loads(line) for line in (ROOT/'experiments/sports/test.jsonl').read_text().splitlines()]
    buckets = {}
    for i, row in enumerate(rows):
        ids = tokenizer.encode(f"User: {row['question']}\nAssistant:")
        buckets.setdefault(len(ids), []).append((i, ids))
    outputs = []
    for length, group in sorted(buckets.items()):
        for start in range(0, len(group), 8):
            batch = group[start:start+8]
            x = torch.tensor([ids for _, ids in batch], device='cuda')
            with precision_context(fp16):
                plain = model.generate(x, 40, use_cache=False)
                cached = model.generate(x, 40, use_cache=True)
            for j, (i, _) in enumerate(batch):
                texts=[]
                continuations=[]
                for sequence in (plain, cached):
                    tokens=sequence[j,length:].tolist()
                    if tokenizer.eos_token_id in tokens:
                        tokens=tokens[:tokens.index(tokenizer.eos_token_id)]
                    continuations.append(tokens)
                    texts.append(tokenizer.decode(tokens,skip_special_tokens=True).split('\nUser:')[0].strip())
                outputs.append(dict(index=i, question=rows[i]['question'], reference=rows[i]['answer'],
                                    uncached=texts[0], cached=texts[1],
                                    all_40_tokens_equal=bool(torch.equal(plain[j],cached[j])),
                                    tokens_until_eos_equal=continuations[0]==continuations[1],
                                    uncached_scores=score(rows[i], texts[0]), cached_scores=score(rows[i],texts[1])))
    outputs.sort(key=lambda r:r['index'])
    result=dict(questions=len(rows), identical_all_40_tokens=sum(o['all_40_tokens_equal'] for o in outputs),
                identical_until_eos=sum(o['tokens_until_eos_equal'] for o in outputs),
                identical_texts=sum(o['uncached']==o['cached'] for o in outputs),
                uncached=aggregate([dict(scores=o['uncached_scores']) for o in outputs]),
                cached=aggregate([dict(scores=o['cached_scores']) for o in outputs]), outputs=outputs)
    save(f'{name}-quality.json',result)
    if result['identical_until_eos'] != len(rows):
        mismatches = len(rows) - result['identical_until_eos']
        print(
            f'{name}: cached and uncached generation differed for '
            f'{mismatches}/{len(rows)} examples',
            flush=True,
        )
    return {k:v for k,v in result.items() if k!='outputs'}


def summarize(samples):
    return {key:dict(mean=statistics.mean(r[key] for r in samples),
                     stdev=statistics.stdev(r[key] for r in samples),
                     min=min(r[key] for r in samples),max=max(r[key] for r in samples))
            for key in samples[0]}


def display_path(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


@torch.inference_mode()
def benchmark_model(model, tokenizer, name, fp16=False):
    results=[]
    # All performance cases stay inside GPT-2's 1,024-token context window.
    for batch,length in [(1,32),(1,128),(1,512),(4,128)]:
        gen=torch.Generator().manual_seed(42+length)
        prompt=torch.randint(0,50257,(batch,length),generator=gen).cuda()
        samples={'uncached':[],'cached':[]}
        for warmup in range(2):
            for enabled in (False,True):
                output,_=timed_generate(model,prompt,64,enabled,fp16)
                del output
        for repeat in range(5):
            outputs={}
            for enabled in ((False,True) if repeat%2==0 else (True,False)):
                mode='cached' if enabled else 'uncached'
                output,measure=timed_generate(model,prompt,64,enabled,fp16)
                outputs[mode]=output.cpu()
                samples[mode].append(measure)
                del output
            token_agreement = torch.equal(
                outputs['uncached'],
                outputs['cached'],
            )

            if not token_agreement:
                print(
                    f'{name}: cached and uncached benchmark outputs differ '
                    f'under {"FP16 autocast" if fp16 else "FP32"}',
                    flush=True,
                )
        row=dict(model=name,batch_size=batch,prompt_tokens=length,new_tokens=64,repeats=5,
                 samples=samples,summary={mode:summarize(v) for mode,v in samples.items()})
        row['speedup']=row['summary']['uncached']['wall_ms']['mean']/row['summary']['cached']['wall_ms']['mean']
        results.append(row)
        save(f'{name}-performance.json',results)
        print(f'{name} B={batch} P={length}: {row["speedup"]:.2f}x speedup',flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 CUDA autocast while keeping model weights FP32')
    parser.add_argument('--full-model', type=Path,
                        help='Path to the full fine-tuned model.pt checkpoint')
    parser.add_argument('--lora-model', type=Path,
                        help='Path to the LoRA adapter.pt checkpoint')
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    selection=json.loads((ROOT/'experiments/sports/matched/selection.json').read_text())
    default_full = ROOT/selection['methods']['full']['runs']['42']/'model.pt'
    default_lora = ROOT/selection['methods']['lora']['runs']['42']/'adapter.pt'
    full_checkpoint = (args.full_model or default_full).expanduser()
    lora_checkpoint = (args.lora_model or default_lora).expanduser()
    if not full_checkpoint.is_absolute():
        full_checkpoint = ROOT/full_checkpoint
    if not lora_checkpoint.is_absolute():
        lora_checkpoint = ROOT/lora_checkpoint
    for name, path in [('full', full_checkpoint), ('lora', lora_checkpoint)]:
        if not path.is_file():
            parser.error(f'{name} checkpoint does not exist: {path}')
    tokenizer=GPT2Tokenizer.from_pretrained('gpt2',local_files_only=True)
    report=dict(complete=False,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                cuda=torch.version.cuda,python=platform.python_version(),node=platform.node(),
                slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                dtype='float16' if args.fp16 else 'float32', autocast=args.fp16, tf32=False,
                warmups=2,repeats=5,decode='greedy',timing='CUDA events and synchronized wall time; exclude tokenization and host/device input transfer',
                cache_implementation='dynamic per-layer K/V; rebuild cropped context after context limit',
                source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['mingpt/model.py','experiments/kv_cache/benchmark.py']},models={})
    for name in ['pretrained','full','lora']:
        model=GPT.from_pretrained('gpt2').float()
        checkpoint_path=None
        if name !='pretrained':
            checkpoint_path=lora_checkpoint if name=='lora' else full_checkpoint
            ckpt=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
            if name=='lora':
                model.enable_lora(ckpt['rank'],ckpt['alpha'])
            load_parameters(model,ckpt)
            del ckpt
        before=digest(model)
        model.cuda().eval()
        assert all(p.dtype==torch.float32 for p in model.parameters())
        info=dict(checkpoint=None if checkpoint_path is None else display_path(checkpoint_path),
                  total_parameters=sum(p.numel() for p in model.parameters()),weight_sha256_before=before)
        info['logit_parity']=check_logits(model,tokenizer,args.fp16)
        print(name,'logit parity',info['logit_parity'],flush=True)
        info['quality']=check_quality(model,tokenizer,name,args.fp16)
        print(name,'quality parity',info['quality'],flush=True)
        # Profile base and adapted architectures; full FT shares base inference architecture.
        if name!='full':
            info['performance']=benchmark_model(model,tokenizer,name,args.fp16)
        info['weight_sha256_after']=digest(model)
        assert info['weight_sha256_after']==before
        info['weights_unchanged']=True
        report['models'][name]=info
        save('report.json',report)
        del model
        gc.collect();torch.cuda.empty_cache()
    report['complete']=True
    save('report.json',report)
    print('COMPLETE: output parity, unchanged weights, and inference benchmarks saved.',flush=True)


if __name__=='__main__':
    main()
