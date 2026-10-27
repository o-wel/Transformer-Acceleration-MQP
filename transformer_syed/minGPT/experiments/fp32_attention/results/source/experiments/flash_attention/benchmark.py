"""Paired attention benchmark; no downloads. See README.md for interpretation."""
import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from mingpt.model import GPT, LoRALinear
from mingpt.measurement import parameter_counts


def measure(fn, device):
    cuda = device.type == 'cuda'
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
    begin = time.perf_counter()
    result = fn()
    if cuda:
        end.record()
        torch.cuda.synchronize(device)
    row = {'wall_ms': (time.perf_counter() - begin) * 1000}
    if cuda:
        row.update(cuda_ms=start.elapsed_time(end), baseline_bytes=baseline,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                   incremental_peak_bytes=torch.cuda.max_memory_allocated(device) - baseline,
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return result, row


def summary(rows):
    return {k: {'median': statistics.median(r[k] for r in rows),
                'mean': statistics.mean(r[k] for r in rows),
                'stdev': statistics.stdev(r[k] for r in rows) if len(rows) > 1 else 0,
                'min': min(r[k] for r in rows), 'max': max(r[k] for r in rows)}
            for k in rows[0]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='bfloat16')
    p.add_argument('--backends', nargs='+', default=['eager', 'math', 'flash'],
                   choices=['eager', 'math', 'flash', 'sdpa'])
    p.add_argument('--lengths', nargs='+', type=int, default=[128, 512, 1024])
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--new-tokens', type=int, default=32)
    p.add_argument('--layers', type=int, default=6)
    p.add_argument('--heads', type=int, default=6)
    p.add_argument('--width', type=int, default=384)
    p.add_argument('--vocab-size', type=int, default=4096)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--output', type=Path, default=Path('experiments/flash_attention/results/report.json'))
    a = p.parse_args()
    if min(a.lengths + [a.batch_size, a.new_tokens, a.layers, a.heads, a.width,
                        a.vocab_size, a.rank, a.repeats]) < 1 or a.warmup < 0:
        p.error('sizes and repeats must be positive; warmup must be nonnegative')
    if 'eager' not in a.backends:
        p.error('include eager as the paired baseline')
    device = torch.device(a.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; use --device cpu --dtype float32 --backends eager math for a smoke test')
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    c = GPT.get_default_config()
    c.model_type = None
    c.n_layer, c.n_head, c.n_embd = a.layers, a.heads, a.width
    c.vocab_size, c.block_size = a.vocab_size, max(a.lengths) + a.new_tokens
    c.embd_pdrop = c.resid_pdrop = c.attn_pdrop = 0.0
    base = GPT(c).to(dtype=getattr(torch, a.dtype))
    report = dict(complete=False, args={**vars(a), 'output': str(a.output)},
                  torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version(),
                  gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                  weights='seeded random; measures execution, not task quality',
                  tf32=False, optimizer='SGD without momentum',
                  source_sha256={str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (Path(__file__), Path(__file__).resolve().parents[2] / 'mingpt/model.py')},
                  rows=[])
    a.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        a.output.write_text(json.dumps(report, indent=2))

    for lora in (False, True):
        model = copy.deepcopy(base).to(device)
        if lora:
            model.enable_lora(a.rank, a.rank)
            # Exercise active adapters, not just the zero-initialized update.
            for module in model.modules():
                if isinstance(module, LoRALinear):
                    torch.nn.init.normal_(module.B.weight, std=.01)
        counts = parameter_counts(model)
        for length in a.lengths:
            x = torch.randint(a.vocab_size, (a.batch_size, length), device=device)
            continuation = torch.randint(a.vocab_size, (a.batch_size, a.new_tokens), device=device)
            for task in ('train', 'prefill', 'decode_uncached', 'decode_cached'):
                model.train(task == 'train')
                samples = {b: [] for b in a.backends}
                failures, parity, profiles = {}, {}, {}
                model.set_attention_backend('eager')
                with torch.no_grad():
                    reference = model(x, last_token_only=True)[0]
                # Use stateless SGD to compare identical updates without duplicate optimizer buffers.
                initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()} if task == 'train' else None
                optimizers = {b: torch.optim.SGD([q for q in model.parameters() if q.requires_grad], lr=1e-4)
                              for b in a.backends} if task == 'train' else {}

                def run():
                    if task == 'train':
                        optimizers[backend].zero_grad(set_to_none=True)
                        _, loss = model(x, x)
                        loss.backward()
                        optimizers[backend].step()
                        return
                    if task == 'prefill':
                        return model(x, last_token_only=True)[0]
                    cache = None
                    if task == 'decode_cached':
                        # Prefill is excluded from decode timing, but cache storage is included.
                        _, _, cache = model(x, use_cache=True, last_token_only=True)
                    return cache

                for backend in a.backends:
                    model.set_attention_backend(backend)
                    try:
                        model.eval()
                        with torch.no_grad():
                            got = model(x, last_token_only=True)[0]
                        tol = 2e-2 if a.dtype != 'float32' else 2e-4
                        torch.testing.assert_close(got, reference, atol=tol, rtol=tol)
                        with torch.no_grad():
                            _, _, prefix = model(x, use_cache=True, last_token_only=True)
                            cached = model(continuation[:, :1], past_key_values=prefix, use_cache=True)[0]
                            full = model(torch.cat((x, continuation[:, :1]), 1))[0][:, -1:]
                            torch.testing.assert_close(cached, full, atol=tol, rtol=tol)
                        parity[backend] = {'max_abs_logit_error': float((got-reference).abs().max()),
                                           'cache_max_abs_error': float((cached-full).abs().max()), 'tolerance': tol}
                        del got, prefix, cached, full
                    except RuntimeError as e:
                        failures[backend] = str(e)
                    except AssertionError:
                        save()
                        raise
                model.train(task == 'train')
                # Rotate execution order to reduce temperature/order bias.
                for repeat in range(a.warmup + a.repeats):
                    order = a.backends[repeat % len(a.backends):] + a.backends[:repeat % len(a.backends)]
                    for backend in order:
                        if backend in failures:
                            continue
                        model.set_attention_backend(backend)
                        if initial is not None:
                            model.load_state_dict(initial)
                        model.zero_grad(set_to_none=True)
                        with nullcontext() if task == 'train' else torch.inference_mode():
                            cache = run() if task.startswith('decode') else None
                            def operation():
                                nonlocal cache
                                if not task.startswith('decode'):
                                    return run()
                                result = None
                                for i in range(a.new_tokens):
                                    if task == 'decode_cached':
                                        result, _, cache = model(continuation[:, i:i+1], past_key_values=cache,
                                                                 use_cache=True, last_token_only=True)
                                    else:
                                        context = torch.cat((x, continuation[:, :i+1]), 1)
                                        result = model(context, last_token_only=True)[0]
                                return result
                            result, metrics = measure(operation, device)
                            if task == 'decode_cached':
                                metrics['cache_bytes'] = sum(t.numel()*t.element_size() for pair in cache for t in pair)
                            tokens = a.batch_size * (a.new_tokens if task.startswith('decode') else length)
                            metrics['tokens_per_second'] = tokens * 1000 / metrics['wall_ms']
                            if task.startswith('decode'):
                                metrics['ms_per_decode_step'] = metrics['wall_ms'] / a.new_tokens
                            if repeat >= a.warmup:
                                samples[backend].append(metrics)
                            if a.profile and repeat == a.warmup:
                                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU] +
                                        ([torch.profiler.ProfilerActivity.CUDA] if device.type == 'cuda' else [])) as prof:
                                    # Fresh cache required after the measured decode loop.
                                    cache = run() if task.startswith('decode') else None
                                    operation()
                                trace = a.output.parent / f'{"lora" if lora else "full"}-{length}-{task}-{backend}.json'
                                prof.export_chrome_trace(str(trace))
                                profiles[backend] = str(trace)
                            del result, cache
                if initial is not None:
                    model.load_state_dict(initial)
                model.zero_grad(set_to_none=True)
                summaries = {b: summary(rows) for b, rows in samples.items() if rows}
                baseline = summaries['eager']['wall_ms']['median']
                row = dict(lora=lora, length=length, task=task, parameters=counts, samples=samples,
                           summary=summaries, unavailable=failures, parity=parity, traces=profiles,
                           speedup_vs_eager={b: baseline/s['wall_ms']['median'] for b, s in summaries.items()})
                report['rows'].append(row)
                save()
                print(f'LoRA={lora} length={length} {task}: {row["speedup_vs_eager"]}; unavailable={list(failures)}', flush=True)
        del model
    report['interactions'] = []
    for row in report['rows']:
        for backend, stats in row['summary'].items():
            comparison = dict(lora=row['lora'], length=row['length'], task=row['task'], backend=backend)
            if row['task'] == 'decode_cached':
                uncached = next(r for r in report['rows'] if r['lora'] == row['lora'] and
                                r['length'] == row['length'] and r['task'] == 'decode_uncached')
                if backend in uncached['summary']:
                    comparison['cache_speedup_same_backend'] = uncached['summary'][backend]['wall_ms']['median'] / stats['wall_ms']['median']
                    comparison['combined_speedup_vs_eager_uncached'] = uncached['summary']['eager']['wall_ms']['median'] / stats['wall_ms']['median']
            if row['lora']:
                full = next(r for r in report['rows'] if not r['lora'] and r['length'] == row['length'] and r['task'] == row['task'])
                if backend in full['summary']:
                    comparison['lora_speedup_same_backend'] = full['summary'][backend]['wall_ms']['median'] / stats['wall_ms']['median']
            report['interactions'].append(comparison)
    report['complete'] = True
    save()


if __name__ == '__main__':
    main()
