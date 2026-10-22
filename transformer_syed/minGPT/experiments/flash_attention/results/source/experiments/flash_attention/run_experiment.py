"""Real GPT-2 checkpoints: paired GPU timing, quality, training, and kernel evidence."""
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from transformers import GPT2Tokenizer
from mingpt.model import GPT
from mingpt.measurement import parameter_counts
from sports_experiment import load_parameters
from experiments.sports.scoring import score, aggregate
from experiments.kv_cache.benchmark import timed_generate
from experiments.flash_attention.benchmark import measure, summary

BACKENDS = ['eager', 'math', 'flash']
OUT = ROOT / 'experiments/flash_attention/results'
WARMUPS, REPEATS = 3, 7


def save(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2))


def digest(model):
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        h.update(n.encode())
        h.update(p.detach().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load(name, dtype):
    model = GPT.from_pretrained('gpt2')
    if name != 'pretrained':
        selection = json.loads((ROOT/'experiments/sports/matched/selection.json').read_text())
        directory = ROOT/selection['methods'][name]['runs']['42']
        ckpt = torch.load(directory/('adapter.pt' if name == 'lora' else 'model.pt'), map_location='cpu', weights_only=True)
        if name == 'lora':
            model.enable_lora(ckpt['rank'], ckpt['alpha'])
        load_parameters(model, ckpt)
        del ckpt
    return model.to(device='cuda', dtype=dtype)


@torch.inference_mode()
def parity(model):
    x = torch.randint(50257, (2, 37), device='cuda')
    model.set_attention_backend('eager')
    reference = model(x)[0].float()
    rows = {}
    for backend in BACKENDS:
        model.set_attention_backend(backend)
        logits = model(x)[0].float()
        cache, chunks, start = None, [], 0
        for length in (11, 1, 7, 18):
            got, _, cache = model(x[:, start:start+length], past_key_values=cache, use_cache=True)
            chunks.append(got.float())
            start += length
        cached = torch.cat(chunks, 1)
        error = (logits-reference).abs()
        cache_error = (cached-logits).abs()
        rows[backend] = dict(max_abs_error=error.max().item(), mean_abs_error=error.mean().item(),
                             cached_max_abs_error=cache_error.max().item(),
                             argmax_agreement=float((logits.argmax(-1) == reference.argmax(-1)).float().mean()),
                             cached_argmax_agreement=float((cached.argmax(-1) == logits.argmax(-1)).float().mean()))
        # A safety check for mask mistakes; precise BF16 errors remain in the report.
        torch.testing.assert_close(logits, reference, rtol=.05, atol=.25)
        torch.testing.assert_close(cached, logits, rtol=.05, atol=.25)
    return rows


@torch.inference_mode()
def quality(model, tokenizer, name):
    questions = [json.loads(s) for s in (ROOT/'experiments/sports/test.jsonl').read_text().splitlines()]
    buckets = {}
    for i, row in enumerate(questions):
        ids = tokenizer.encode(f"User: {row['question']}\nAssistant:")
        buckets.setdefault(len(ids), []).append((i, ids))
    outputs = {b+('-cached' if cache else '-uncached'): [] for b in BACKENDS for cache in (False, True)}
    for length, group in sorted(buckets.items()):
        for offset in range(0, len(group), 8):
            batch = group[offset:offset+8]
            x = torch.tensor([ids for _, ids in batch], device='cuda')
            for backend in BACKENDS:
                model.set_attention_backend(backend)
                for cache in (False, True):
                    generated = model.generate(x, 40, use_cache=cache)[:, length:].cpu().tolist()
                    key = backend+('-cached' if cache else '-uncached')
                    for (i, _), tokens in zip(batch, generated):
                        before_eos = tokens[:tokens.index(tokenizer.eos_token_id)] if tokenizer.eos_token_id in tokens else tokens
                        text = tokenizer.decode(before_eos, skip_special_tokens=True).split('\nUser:')[0].strip()
                        outputs[key].append(dict(index=i, tokens=tokens, tokens_until_eos=before_eos,
                                                 text=text, scores=score(questions[i], text)))
    for rows in outputs.values():
        rows.sort(key=lambda r:r['index'])
    ref = outputs['eager-uncached']
    summaries = {}
    for key, rows in outputs.items():
        summaries[key] = dict(metrics=aggregate(rows),
                              identical_all_40_tokens=sum(r['tokens'] == q['tokens'] for r, q in zip(rows, ref)),
                              identical_until_eos=sum(r['tokens_until_eos'] == q['tokens_until_eos'] for r, q in zip(rows, ref)),
                              identical_texts=sum(r['text'] == q['text'] for r, q in zip(rows, ref)))
    save(f'{name}-quality.json', dict(questions=questions, summary=summaries, outputs=outputs))
    return summaries


@torch.inference_mode()
def performance(model, name):
    cases = [(1, 32), (1, 128), (1, 512), (4, 128), (1, 896)]
    rows = []
    for batch, length in cases:
        generator = torch.Generator().manual_seed(42+length+batch)
        prompt = torch.randint(50257, (batch, length), generator=generator).cuda()
        modes = [(backend, cache) for backend in BACKENDS for cache in (False, True)]
        samples = {b+('-cached' if c else '-uncached'): [] for b, c in modes}
        for repeat in range(WARMUPS+REPEATS):
            order = modes.copy()
            random.Random(1000+repeat).shuffle(order)
            for backend, cache in order:
                model.set_attention_backend(backend)
                output, result = timed_generate(model, prompt, 64, cache)
                result['incremental_peak_bytes'] = result['peak_allocated_bytes'] - result['baseline_allocated_bytes']
                if repeat >= WARMUPS:
                    samples[backend+('-cached' if cache else '-uncached')].append(result)
                del output
        row = dict(model=name, batch=batch, prompt_tokens=length, new_tokens=64,
                   samples=samples, summary={k: summary(v) for k, v in samples.items()})
        rows.append(row)
        save(f'{name}-performance.json', rows)
        print('PERFORMANCE', name, batch, length, {k: round(v['wall_ms']['median'], 2) for k, v in row['summary'].items()}, flush=True)
    return rows


@torch.inference_mode()
def profile_inference(model, name):
    model.set_attention_backend('eager')
    x = torch.randint(50257, (1, 512), device='cuda')
    result = {}
    for backend in BACKENDS:
        model.set_attention_backend(backend)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            _, _, cache = model(x, use_cache=True, last_token_only=True)
            model(x[:, :1], past_key_values=cache, use_cache=True, last_token_only=True)
        torch.cuda.synchronize()
        trace = f'{name}-{backend}-prefill-decode-trace.json'
        prof.export_chrome_trace(str(OUT/trace))
        ops = [dict(name=e.key, calls=e.count, device_time_total_us=e.device_time_total)
               for e in prof.key_averages() if any(word in e.key.lower() for word in ('attention', 'flash', 'bmm', 'softmax'))]
        result[backend] = dict(trace=trace, attention_operators=ops)
        del cache
    return result


def training(name):
    """FP32 master weights + BF16 autocast, AdamW; fresh model per kernel/shape."""
    rows = []
    for batch, length in [(2, 128), (2, 512), (2, 896)]:
        sample_by_backend = {}
        losses = {}
        profiles = {}
        params = None
        for backend in BACKENDS:
            gc.collect()
            torch.cuda.empty_cache()
            torch.manual_seed(42)
            model = load('pretrained' if name == 'full' else 'lora', torch.float32).train()
            model.set_attention_backend(backend)
            # Disable dropout so gradient checks and timing are reproducible.
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.0
            params = parameter_counts(model)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
            x = torch.randint(50257, (batch, length), device='cuda')
            y = torch.roll(x, shifts=-1, dims=1)
            def step():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    _, loss = model(x, y)
                loss.backward()
                optimizer.step()
                return loss.detach()
            samples = []
            for repeat in range(WARMUPS+REPEATS):
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = measure(step, torch.device('cuda'))
                metrics['tokens_per_second'] = batch*length*1000/metrics['wall_ms']
                if repeat >= WARMUPS:
                    samples.append(metrics)
                del loss
            sample_by_backend[backend] = samples
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                losses[backend] = float(model(x, y)[1])
            if length == 512:
                optimizer.zero_grad(set_to_none=True)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as prof:
                    step()
                trace = f'{name}-{backend}-training-trace.json'
                prof.export_chrome_trace(str(OUT/trace))
                profiles[backend] = dict(trace=trace, attention_operators=[
                    dict(name=e.key, calls=e.count, device_time_total_us=e.device_time_total)
                    for e in prof.key_averages() if 'attention' in e.key.lower() or 'flash' in e.key.lower()])
            del optimizer, model, x, y
        row = dict(method=name, batch=batch, length=length, parameters=params, samples=sample_by_backend,
                   summary={b:summary(v) for b,v in sample_by_backend.items()}, final_loss=losses, profiles=profiles)
        rows.append(row)
        save(f'{name}-training.json', rows)
        print('TRAINING', name, batch, length, {k:round(v['wall_ms']['median'],2) for k,v in row['summary'].items()}, flush=True)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'report.json').exists():
        raise RuntimeError('Existing experiment results; choose a new output directory instead of overwriting')
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = dict(complete=False, gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  python=platform.python_version(), node=platform.node(), job=os.environ.get('SLURM_JOB_ID'),
                  date_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  inference_dtype='bfloat16 weights and activations', training_dtype='float32 master weights, bfloat16 autocast, float32 AdamW states',
                  warmups=WARMUPS, repeats=REPEATS, tf32=False, backends=BACKENDS,
                  source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                                 ['mingpt/model.py','experiments/flash_attention/run_experiment.py','experiments/kv_cache/benchmark.py']}, models={})
    save('report.json', report)
    # GPU correctness suite includes forced flash forward/backward and chunked-cache tests.
    import unittest
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_flash_attention.py')
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    report['gpu_tests'] = dict(run=outcome.testsRun, failures=len(outcome.failures), errors=len(outcome.errors), skipped=len(outcome.skipped))
    assert outcome.wasSuccessful()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    for name in ['pretrained', 'lora', 'full']:
        model = load(name, torch.bfloat16).eval()
        before = digest(model)
        info = dict(parameters=parameter_counts(model), weights_sha256_before=before)
        info['parity'] = parity(model)
        print('PARITY', name, info['parity'], flush=True)
        if name != 'full':
            info['performance'] = performance(model, name)
            info['profiles'] = profile_inference(model, name)
        info['quality'] = quality(model, tokenizer, name)
        print('QUALITY', name, {k:v['metrics']['primary_correct'] for k,v in info['quality'].items()}, flush=True)
        info['weights_sha256_after'] = digest(model)
        info['weights_unchanged'] = before == info['weights_sha256_after']
        assert info['weights_unchanged']
        report['models'][name] = info
        save('report.json', report)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    report['training'] = {}
    for name in ['full', 'lora']:
        report['training'][name] = training(name)
        save('report.json', report)
    report['complete'] = True
    save('report.json', report)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
