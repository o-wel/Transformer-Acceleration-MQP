"""Validation-only staged FP32 search. Run stages in order; preserve old results."""
import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / 'combined_optimization_review'))


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def uid(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def grid():
    return [dict(method='lora', rank=r, scale=s, lr=lr, targets=t,
                 dropout=0., weight_decay=0., batch=8, microbatch=8, length=128)
            for r, s, lr, t in itertools.product(
                [2, 4, 8, 16, 32], [1., 2., 4.], [1e-4, 3e-4, 1e-3], ['attention', 'all'])]


def setup():
    global torch, GPT, LoRALinear, encode, load_rows, tokenizer
    import torch
    from mingpt.model import GPT, LoRALinear
    from lora_experiment import encode, load_rows
    from transformers import GPT2Tokenizer
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; CPU measurements cannot establish GPU winners')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)


def build(c, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    model = GPT.from_pretrained('gpt2').float()
    if c['method'] == 'lora':
        model.requires_grad_(False)
        for block in model.transformer.h:
            pairs = [(block.attn, 'c_attn'), (block.attn, 'c_proj')]
            if c['targets'] == 'all':
                pairs += [(block.mlp, 'c_fc'), (block.mlp, 'c_proj')]
            for parent, name in pairs:
                adapter = LoRALinear(getattr(parent, name), c['rank'], c['rank'] * c['scale'])
                # Adapter input dropout only; base projection sees the original x.
                adapter.A = torch.nn.Sequential(torch.nn.Dropout(c['dropout']), adapter.A)
                setattr(parent, name, adapter)
    return model.cuda()


def model_hash(model, frozen=False):
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        if not frozen or not p.requires_grad:
            h.update(n.encode())
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def data(length, split):
    return encode(load_rows(ROOT / f'experiments/sports/{split}.jsonl'), tokenizer, length)


@__import__('contextlib').contextmanager
def cleanup():
    try:
        yield
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def nll(model, rows):
    model.eval()
    total = count = 0
    with torch.inference_mode():
        for start in range(0, len(rows), 8):
            x, y = [torch.stack([r[k] for r in rows[start:start+8]]).cuda() for k in (0, 1)]
            loss = model(x, y)[1]
            n = int((y != -1).sum())
            total += float(loss) * n
            count += n
    value = total / count
    if not math.isfinite(value):
        raise ValueError('Nonfinite validation loss')
    return value


def trial(out, c, examples, seed=42):
    identity = dict(config=c, examples=examples, seed=seed)
    directory = out / 'trials' / uid(identity)
    result_path = directory / 'result.json'
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous.get('complete'):
            return previous
    directory.mkdir(parents=True, exist_ok=True)
    if examples % c['batch'] or c['batch'] % c['microbatch']:
        raise ValueError('Example budget and accumulation must divide exactly')
    train_rows, val_rows = data(c['length'], 'train'), data(128, 'validation')
    model = build(c, seed)
    frozen = model_hash(model, frozen=True)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=c['lr'], weight_decay=c['weight_decay'])
    steps = examples // c['batch']
    report = dict(**identity, steps=steps, complete=False, validation=[], step_ms=[],
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  job=os.environ.get('SLURM_JOB_ID'), trainable_parameters=sum(p.numel() for p in params),
                  total_parameters=sum(p.numel() for p in model.parameters()), checkpoint=str(directory / 'best.pt'))
    rng = random.Random(seed)
    torch.manual_seed(seed)
    sample_hash = hashlib.sha256()
    best = float('inf')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    peak = 0
    begun = time.perf_counter()
    for step in range(1, steps + 1):
        indices = rng.choices(range(len(train_rows)), k=c['batch'])
        for i in indices:
            sample_hash.update(i.to_bytes(8, 'little'))
        # Normalize each microbatch by supervised token count across the entire
        # effective batch, matching the unpartitioned cross-entropy objective.
        count = sum(int((train_rows[i][1] != -1).sum()) for i in indices)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for offset in range(0, len(indices), c['microbatch']):
            chunk = indices[offset:offset+c['microbatch']]
            x, y = [torch.stack([train_rows[i][k] for i in chunk]).cuda() for k in (0, 1)]
            loss = model(x, y)[1] * ((y != -1).sum() / count)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite training loss')
            loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        report['step_ms'].append(1000 * (time.perf_counter() - start))
        peak = max(peak, torch.cuda.max_memory_allocated())
        if step % max(1, steps // 4) == 0 or step == steps:
            value = nll(model, val_rows)
            report['validation'].append(dict(step=step, nll=value))
            if value < best:
                best = value
                report['selected_step'] = step
                torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                           directory / 'best.pt')
            report['validation_nll'] = best
            write(result_path, report)
            print(uid(identity), step, value, flush=True)
    report.update(complete=True, sample_sha256=sample_hash.hexdigest(), peak_training_bytes=peak,
                  elapsed_seconds=time.perf_counter()-begun,
                  median_step_ms=statistics.median(report['step_ms'][min(10, steps//2):]),
                  frozen_base_unchanged=model_hash(model, frozen=True) == frozen)
    if not report['frozen_base_unchanged']:
        raise AssertionError('Frozen parameters changed')
    write(result_path, report)
    del model, optimizer, params
    gc.collect()
    torch.cuda.empty_cache()
    return report


def stage_results(out, stage):
    path = out / f'{stage}.json'
    if not path.exists():
        raise RuntimeError(f'Missing prerequisite {path}')
    return json.loads(path.read_text())


def top(rows, n):
    return sorted(rows, key=lambda r: (r['validation_nll'], uid(r['config'])))[:n]


def train_stage(out, stage, shard):
    if stage == 'screen':
        configs, budget = grid()[shard::3], 1200
    elif stage == 'refine':
        rows = sum([stage_results(out, f'screen-{s}') for s in range(3)], [])
        configs, budget = [r['config'] for r in top(rows, 9)], 9600
    elif stage == 'regularize':
        base = top(stage_results(out, 'refine'), 1)[0]['config']
        configs = [dict(base, dropout=d, weight_decay=w) for d, w in itertools.product([0., .05, .1], [0., .01, .1])]
        budget = 9600
    elif stage == 'batch':
        base = top(stage_results(out, 'refine') + stage_results(out, 'regularize'), 1)[0]['config']
        # Equal 9,600 sampled examples; effective batch changes the update count.
        configs = [dict(base, batch=b, microbatch=min(b, 8)) for b in [4, 8, 16, 32]]
        configs += [dict(base, microbatch=m) for m in [1, 2, 4]]
        configs += [dict(base, length=l) for l in [256, 512]]
        budget = 9600
    elif stage == 'confirm':
        rows = stage_results(out, 'refine') + stage_results(out, 'regularize') + stage_results(out, 'batch')
        unique = {uid(r['config']): r for r in rows}
        configs = [r['config'] for r in top(list(unique.values()), 3)]
        results = [trial(out, c, 9600, seed) for c in configs for seed in [42, 43, 44]]
        # Untuned baseline and tuned full fine-tuning control have separate labels.
        controls = [dict(configs[0], method='full', lr=lr) for lr in [1e-4, 3e-4, 1e-3]]
        full = [trial(out, c, 9600) for c in controls]
        winner_full = top(full, 1)[0]['config']
        full += [trial(out, winner_full, 9600, seed) for seed in [43, 44]]
        groups = []
        for c in configs:
            rr = [r for r in results if r['config'] == c]
            groups.append(dict(config=c, mean_validation_nll=statistics.mean(r['validation_nll'] for r in rr),
                               stdev_validation_nll=statistics.stdev(r['validation_nll'] for r in rr)))
        groups.sort(key=lambda r: r['mean_validation_nll'])
        write(out/'selection.json', dict(frozen_before_test=True, criterion='mean validation NLL over seeds 42,43,44',
              ranked=groups, winner=groups[0]['config'], full_control=winner_full,
              note='Best observed in this staged search, not a global optimum'))
        write(out/'confirm.json', results + full)
        return
    else:
        raise ValueError(stage)
    results = [trial(out, c, budget) for c in configs]
    write(out / (f'screen-{shard}.json' if stage == 'screen' else f'{stage}.json'), results)


def selected_model(out, config, seed):
    r = json.loads((out/'trials'/uid(dict(config=config, examples=9600, seed=seed))/'result.json').read_text())
    model = build(config, seed).eval()
    incompatible = model.load_state_dict(torch.load(r['checkpoint'], map_location='cpu', weights_only=True), strict=False)
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    loaded = set(torch.load(r['checkpoint'], map_location='cpu', weights_only=True))
    assert loaded == expected and not incompatible.unexpected_keys
    return model


def configure_attention(mode, bm=16, bn=32, warps=4):
    # Route the existing model's tiled hook to our isolated configurable FA1-derived
    # implementation. No PyTorch SDPA/FlashAttention is used in this study.
    import mingpt.tiled_attention as original
    import attention
    attention.BM, attention.BN, attention.WARPS = bm, bn, warps
    if mode == 'hybrid':
        def kernel(q, k, v, past=0):
            if q.size(2) == 1:
                a = torch.softmax((q @ k.transpose(-2, -1)) * q.size(-1)**-.5, dim=-1)
                return a @ v
            return attention.attention(q, k, v, past)
    else:
        kernel = attention.attention
    original.tiled_attention_fp32 = kernel


def systems(out, shard):
    from experiments.kv_cache.benchmark import timed_generate
    from experiments.sports.scoring import score, aggregate
    import cache as static_cache
    selection = stage_results(out, 'selection')
    assert selection['frozen_before_test']
    rows = []
    # Per-cell raw measurements and correctness checks are retained, including OOMs.
    for method, c in [('full', selection['full_control']), ('lora', selection['winner'])]:
        model = selected_model(out, c, 42)
        before = model_hash(model)
        for batch, length, generated in itertools.product([[1,4,8][shard]], [32, 128, 512, 896], [32, 64]):
            static_cache.enable(model,False)
            prompt = torch.randint(0, 50257, (batch, length), generator=torch.Generator().manual_seed(123)).cuda()
            with torch.inference_mode():
                model.set_attention_backend('eager')
                reference_logits = model(prompt, last_token_only=True)[0]
                reference, _ = timed_generate(model, prompt, generated, False)
            cases = [('eager',16,32,4), ('hybrid',16,32,4)]
            cases += [('fa1',bm,bn,w) for bm,bn,w in itertools.product([16,32],[32,64],[4,8])]
            for mode, bm, bn, warps in cases:
                configure_attention(mode,bm,bn,warps)
                model.set_attention_backend('eager' if mode == 'eager' else 'tiled_fp32')
                for cache_mode in ['off', 'dynamic', 'static']:
                    cached=cache_mode!='off'
                    static_cache.enable(model,cache_mode=='static')
                    cell = dict(method=method,batch=batch,prompt_tokens=length,new_tokens=generated,
                                backend=mode,bm=bm,bn=bn,warps=warps,cached=cached,cache_mode=cache_mode,samples=[])
                    try:
                        with torch.inference_mode():
                            got = model(prompt,last_token_only=True)[0]
                            torch.testing.assert_close(got,reference_logits,rtol=3e-4,atol=3e-4)
                            cell['max_abs_logit_error'] = float((got-reference_logits).abs().max())
                            for repeat in range(9):
                                output, measure = timed_generate(model,prompt,generated,cached)
                                cell['tokens_equal'] = bool(torch.equal(output,reference))
                                if not cell['tokens_equal']:
                                    raise AssertionError('Greedy generation differs from eager uncached reference')
                                if repeat >= 2:
                                    measure['static_capacity_bytes']=static_cache.allocated_bytes(model)
                                    cell['samples'].append(measure)
                        cell['wall_ms'] = statistics.median(s['wall_ms'] for s in cell['samples'])
                        cell['status'] = 'ok'
                    except torch.cuda.OutOfMemoryError:
                        cell.update(status='oom',samples=[])
                        torch.cuda.empty_cache()
                    rows.append(cell)
                    write(out/f'systems-{shard}.json', rows)
        static_cache.enable(model,False)
        assert before == model_hash(model)
        # Held-out test is opened only after validation selection is persisted.
        model.set_attention_backend('eager')
        quality = []
        for seed in ([42,43,44] if shard==0 else []):
            if seed != 42:
                del model
                gc.collect(); torch.cuda.empty_cache()
                model = selected_model(out,c,seed)
            outputs = []
            with torch.inference_mode():
                for i, q in enumerate(load_rows(ROOT/'experiments/sports/test.jsonl')):
                    ids = tokenizer.encode(f"User: {q['question']}\nAssistant:")
                    x = torch.tensor([ids],device='cuda')
                    seq = model.generate(x,40,use_cache=True)[0,len(ids):].tolist()
                    if tokenizer.eos_token_id in seq:
                        seq = seq[:seq.index(tokenizer.eos_token_id)]
                    text = tokenizer.decode(seq,skip_special_tokens=True).split('\nUser:')[0].strip()
                    outputs.append(dict(index=i,text=text,tokens=seq,scores=score(q,text)))
            quality.append(dict(seed=seed,metrics=aggregate(outputs),outputs=outputs))
            write(out/f'{method}-test.json',quality)
        del model
        gc.collect(); torch.cuda.empty_cache()


def report(out):
    results = [json.loads(p.read_text()) for p in (out/'trials').glob('*/result.json')]
    completed = [r for r in results if r.get('complete')]
    fields = ['id','method','rank','scale','alpha','lr','targets','dropout','weight_decay','batch','microbatch','length',
              'examples','seed','validation_nll','selected_step','median_step_ms','peak_training_bytes','trainable_parameters','gpu']
    with (out/'TRIALS.csv').open('w') as f:
        w = csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in completed:
            row = {**r['config'],**{k:r[k] for k in fields if k in r}}
            row.update(id=uid(dict(config=r['config'],examples=r['examples'],seed=r['seed'])),
                       alpha=r['config']['rank']*r['config']['scale'])
            w.writerow({k:row.get(k) for k in fields})
    text = ['# Hyperparameter search results', '', f'Completed trials: {len(completed)}. Incomplete trials: {len(results)-len(completed)}.', '',
            'Rankings use validation NLL. Screening and full-budget trials are reported separately. Best observed is conditional on the staged search and this dataset.', '']
    eligible = [r for r in completed if r['examples']==9600 and r['seed']==42 and r['config']['method']=='lora']
    if eligible:
        text += ['| Rank | r | alpha | LR | Targets | Dropout | Decay | Batch | Length | Validation NLL | Step ms |',
                 '|---|---|---|---|---|---|---|---|---|---|---|']
        for i,r in enumerate(top(eligible,10),1):
            c=r['config']
            text.append(f"| {i} | {c['rank']} | {c['rank']*c['scale']:g} | {c['lr']:g} | {c['targets']} | {c['dropout']} | {c['weight_decay']} | {c['batch']} | {c['length']} | {r['validation_nll']:.6f} | {r['median_step_ms']:.2f} |")
    if (out/'selection.json').exists():
        text += ['', 'Frozen selection:', '```json',json.dumps(stage_results(out,'selection'),indent=2),'```']
    (out/'RESULTS.md').write_text('\n'.join(text)+'\n')
    if eligible:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8,5))
        for targets,marker in [('attention','o'),('all','s')]:
            rr=[r for r in eligible if r['config']['targets']==targets]
            ax.scatter([r['median_step_ms'] for r in rr],[r['validation_nll'] for r in rr],
                       s=[20+2*r['config']['rank'] for r in rr],marker=marker,label=targets,alpha=.7)
        ax.set(xlabel='Median training update (ms)',ylabel='Validation NLL (lower is better)',title='Quality versus update cost; marker size indicates LoRA rank')
        ax.legend();fig.tight_layout();fig.savefig(out/'quality_cost.png',dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['plan','screen','refine','regularize','batch','confirm','systems','report','smoke'])
    p.add_argument('--out',type=Path,default=HERE/'run_fp32')
    p.add_argument('--shard',type=int,choices=range(3),default=0)
    args=p.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.stage=='plan':
        write(out/'plan.json',dict(core_grid=grid(),screen_examples=1200,full_examples=9600,seeds=[42,43,44],
              stages=['screen x3 shards','refine top9','regularize 3x3','batch/accumulation/length','confirm top3 and full controls','systems'],
              dtype='FP32',tf32=False,test_selection=False,search_is_global_optimum=False))
        return
    if args.stage=='report':
        report(out);return
    setup()
    sources = [Path(__file__),HERE/'attention.py',HERE/'cache.py',ROOT/'mingpt/model.py',ROOT/'lora_experiment.py',ROOT/'experiments/kv_cache/benchmark.py']
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    manifest=out/'source_hashes.json'
    if manifest.exists() and json.loads(manifest.read_text()) != hashes:
        raise RuntimeError('Source changed during study; use a new output directory')
    if not manifest.exists():
        write(manifest,hashes)
        write(out/'data_hashes.json',{s:hashlib.sha256((ROOT/f'experiments/sports/{s}.jsonl').read_bytes()).hexdigest() for s in ['train','validation','test']})
    else:
        current={s:hashlib.sha256((ROOT/f'experiments/sports/{s}.jsonl').read_bytes()).hexdigest() for s in ['train','validation','test']}
        assert current==json.loads((out/'data_hashes.json').read_text()), 'Dataset changed during study'
    if args.stage=='smoke':
        trial(out,grid()[0],16)
        # Forward/backward, cached chunks, and tile geometry checked on GPU.
        import attention
        checks=[]
        for bm,bn,w in itertools.product([16,32],[32,64],[4,8]):
            configure_attention('fa1',bm,bn,w)
            for nq,past in [(35,0),(1,35),(7,35)]:
                q=torch.randn(1,2,nq,64,device='cuda',requires_grad=True)
                k=torch.randn(1,2,nq+past,64,device='cuda',requires_grad=True)
                v=torch.randn_like(k,requires_grad=True)
                mask=torch.arange(nq+past,device='cuda')[None,:] <= past+torch.arange(nq,device='cuda')[:,None]
                ref=torch.softmax((q@k.transpose(-2,-1))*.125+torch.where(mask,0.,-torch.inf),-1)@v
                got=attention.attention(q,k,v,past)
                torch.testing.assert_close(got,ref,rtol=3e-4,atol=3e-4)
                grad=torch.randn_like(got)
                a=torch.autograd.grad(got,(q,k,v),grad,retain_graph=True)
                b=torch.autograd.grad(ref,(q,k,v),grad)
                for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=1e-3,atol=1e-3)
                checks.append(dict(bm=bm,bn=bn,warps=w,nq=nq,past=past,max_error=float((got-ref).abs().max())))
        write(out/'smoke.json',dict(complete=True,checks=checks))
        import cache as static_cache
        from experiments.kv_cache.benchmark import timed_generate
        model=build(grid()[0],42).eval()
        cache_checks=[]
        for batch in [1,2,1]:
            x=torch.randint(0,50257,(batch,39),device='cuda')
            with torch.inference_mode():
                static_cache.enable(model,False)
                reference=model(x)[0]
                expected,_=timed_generate(model,x,8,False)
                for mode in ['eager','fa1','hybrid']:
                    configure_attention(mode)
                    model.set_attention_backend('eager' if mode=='eager' else 'tiled_fp32')
                    for static in [False,True]:
                        static_cache.enable(model,static)
                        past=None;pieces=[]
                        for start in range(0,39,7):
                            logits,_,past=model(x[:,start:start+7],past_key_values=past,use_cache=True)
                            pieces.append(logits)
                        got=torch.cat(pieces,dim=1)
                        torch.testing.assert_close(got,reference,rtol=3e-4,atol=3e-4)
                        actual,_=timed_generate(model,x,8,True)
                        assert torch.equal(actual,expected)
                        cache_checks.append(dict(batch=batch,backend=mode,static=static,max_error=float((got-reference).abs().max())))
                static_cache.enable(model,False)
                model.set_attention_backend('eager')
        write(out/'smoke.json',dict(complete=True,attention_checks=checks,cache_checks=cache_checks))
    elif args.stage=='systems':systems(out,args.shard)
    else:train_stage(out,args.stage,args.shard)
    if args.stage not in ['screen','systems']:report(out)


if __name__=='__main__':main()
