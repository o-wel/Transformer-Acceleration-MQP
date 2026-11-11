"""Quality-constrained individual tuning followed by a frozen combined comparison."""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from transformers import GPT2Tokenizer
from mingpt.model import GPT
from lora_experiment import encode, load_rows, prompt as question_prompt
from experiments.sports.scoring import score, aggregate
import mingpt.tiled_attention as tiled
from experiments.hyperparameter_tuning import fa1_tunable as fa

BASE_LORA = dict(rank=4, alpha=8, lr=0.0001)
BASE_TILE = (16, 32, 4)
BASE_CHUNK = 128
LORA_GRID = [dict(rank=r, alpha=r*s, lr=lr)
             for r, s, lr in itertools.product((2, 4), (1, 2), (0.0001, 0.001))]
TILE_GRID = list(itertools.product((16, 32), (32, 64), (4, 8)))
CHUNK_GRID = (64, 128, 256, 896)


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def key(c):
    return f"r{c['rank']}-a{c['alpha']}-lr{c['lr']:g}"


def amp():
    return torch.autocast('cuda', dtype=torch.float16)


@contextmanager
def backend(model, attention, tiles=BASE_TILE):
    old = tiled.tiled_attention_fp32
    fa.TILES = tuple(tiles)
    tiled.tiled_attention_fp32 = fa.attention
    model.set_attention_backend('tiled_fp32' if attention else 'eager')
    try:
        yield
    finally:
        tiled.tiled_attention_fp32 = old
        model.set_attention_backend('eager')


@torch.inference_mode()
def nll(model, data):
    model.eval()
    total = count = 0
    for start in range(0, len(data), 8):
        x, y = (torch.stack([r[k] for r in data[start:start+8]]).cuda() for k in (0, 1))
        with amp():
            loss = model(x, y)[1]
        n = int((y != -1).sum())
        total += float(loss) * n
        count += n
    return total/count


def load_model(config, checkpoint=None):
    torch.manual_seed(42)
    model = GPT.from_pretrained('gpt2').float()
    model.enable_lora(config['rank'], config['alpha'])
    if checkpoint:
        missing, unexpected = model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=False)
        assert not unexpected and all(not n.endswith(('A.weight', 'B.weight')) for n in missing)
    return model.cuda().eval()


def train(config, train_data, validation, out, steps):
    model = load_model(config)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config['lr'], weight_decay=0)
    scaler = torch.amp.GradScaler('cuda', init_scale=128, growth_interval=1000000)
    rng = random.Random(42)
    torch.manual_seed(42)
    times = []
    best = float('inf')
    result = dict(config=config, validation=[], steps=steps, trainable_parameters=sum(p.numel() for p in params))
    checkpoint = out/(key(config)+'.pt')
    sample_hash = hashlib.sha256()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, steps+1):
        indices = rng.choices(range(len(train_data)), k=8)
        sample_hash.update(json.dumps(indices).encode())
        x, y = (torch.stack([train_data[i][k] for i in indices]).cuda() for k in (0, 1))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with amp():
            loss = model(x, y)[1]
        assert bool(torch.isfinite(loss))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        times.append((time.perf_counter()-start)*1000)
        assert all(float(s['step']) == step for s in optimizer.state.values())
        if step % 300 == 0 or step == steps:
            v = nll(model, validation)
            result['validation'].append(dict(step=step, nll=v))
            if v < best:
                best = v
                result['selected_step'] = step
                torch.save({n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}, checkpoint)
            result.update(validation_nll=best, step_ms=times, warm_median_step_ms=statistics.median(times[10:]),
                          sample_sha256=sample_hash.hexdigest(), peak_bytes=torch.cuda.max_memory_allocated())
            save(out/(key(config)+'.json'), result)
            print('TRAIN', key(config), step, v, flush=True)
    del model, optimizer, params
    gc.collect()
    torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def generate(model, ids, tokens, cached, chunk):
    if ids.size(1)+tokens > model.block_size:
        raise ValueError('This fixed-workload experiment must fit inside the context window')
    cache = None
    if cached:
        for start in range(0, ids.size(1), chunk):
            logits, _, cache = model(ids[:, start:start+chunk], past_key_values=cache, use_cache=True, last_token_only=True)
    for i in range(tokens):
        if not cached:
            logits, _ = model(ids, last_token_only=True)
        elif i:
            logits, _, cache = model(ids[:, -1:], past_key_values=cache, use_cache=True, last_token_only=True)
        ids = torch.cat((ids, logits[:, -1].argmax(-1, keepdim=True)), dim=1)
    return ids


@torch.inference_mode()
def latency(model, ids, cached, chunk, attention, tiles):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with backend(model, attention, tiles), amp():
        output = generate(model, ids, 64, cached, chunk)
    torch.cuda.synchronize()
    return dict(wall_ms=(time.perf_counter()-start)*1000,
                peak_bytes=torch.cuda.max_memory_allocated()), output


def measure_modes(model, ids, modes, repeats, seed):
    """Interleave modes; JIT and warmup excluded; retain every repeat."""
    samples = {name:[] for name in modes}
    outputs = {}
    for rep in range(repeats+3):
        order = list(modes)
        random.Random(seed+rep).shuffle(order)
        for name in order:
            measurement, output = latency(model, ids, **modes[name])
            if rep >= 3:
                samples[name].append(measurement)
                outputs[name] = output.cpu().tolist()
    return {name:dict(samples=rows, median_ms=statistics.median(r['wall_ms'] for r in rows),
                      output_tokens=outputs[name]) for name,rows in samples.items()}


def mode(cached=False, chunk=BASE_CHUNK, attention=False, tiles=BASE_TILE):
    return dict(cached=cached, chunk=chunk, attention=attention, tiles=tiles)


@torch.inference_mode()
def validate_tiles():
    rows = []
    for tiles in TILE_GRID:
        fa.TILES = tiles
        for nq, past in ((37, 0), (17, 21), (1, 36)):
            torch.manual_seed(42)
            q = torch.randn(1, 2, nq, 64, device='cuda', dtype=torch.float16)
            k = torch.randn(1, 2, nq+past, 64, device='cuda', dtype=torch.float16)
            v = torch.randn_like(k)
            got = fa.attention(q, k, v, past)
            mask = torch.arange(nq+past, device='cuda')[None, :] <= past+torch.arange(nq, device='cuda')[:, None]
            ref = torch.softmax((q.double()@k.double().transpose(-2,-1)/8).masked_fill(~mask, -float('inf')), -1)@v.double()
            torch.testing.assert_close(got.double(), ref, atol=.012, rtol=.04)
            rows.append(dict(tiles=tiles, queries=nq, past=past, max_error=float((got.double()-ref).abs().max())))
    return rows


@torch.inference_mode()
def quality(model, tokenizer, rows, setting):
    answers = []
    with backend(model, setting['attention'], setting['tiles']), amp():
        for row in rows:
            ids = tokenizer.encode(question_prompt(row['question']), return_tensors='pt').cuda()
            generated = generate(model, ids, 64, setting['cached'], setting['chunk'])
            answer_ids = generated[0, ids.size(1):].tolist()
            if tokenizer.eos_token_id in answer_ids:
                answer_ids = answer_ids[:answer_ids.index(tokenizer.eos_token_id)]
            text = tokenizer.decode(answer_ids, skip_special_tokens=True).split('\nUser:')[0].strip()
            answers.append(dict(question=row['question'], reference=row['answer'], answer=text,
                                token_ids=answer_ids, scores=score(row, text)))
    return dict(scores=aggregate(answers), answers=answers)


def build_report(out, report):
    selected = report['selected']
    lines = ['# Hyperparameter tuning results', '',
             'Best tested settings, selected on validation data and tuning timings before confirmation/test evaluation. Single training seed; speed and quality claims are limited to this grid, workload, GPU, and FA1-derived prototype.', '',
             'All timings use FP32 stored weights with FP16 autocast, batch 1, prompt 896, 64 generated tokens. LoRA is unmerged. Cache tuning changes prefill chunk size without evicting tokens. FA tuning changes query/key tile sizes and warp count; every FA configuration uses the FA1-derived algorithm, not the original optimized FA1 CUDA library.', '',
             f"Selected LoRA: `{selected['lora']}`. Selected cache prefill chunk: `{selected['chunk']}`. Selected FA tiles/warps: `{selected['tiles']}`.", '',
             '| Comparison | Untuned ms | Tuned ms | Speedup |', '|---|---:|---:|---:|']
    for name, row in report['comparisons'].items():
        lines.append(f"| {name} | {row['baseline_ms']:.3f} | {row['tuned_ms']:.3f} | {row['speedup']:.3f}× |")
    lines += ['', 'Ratios below 1 are slowdowns. The predeclared untuned controls may themselves win; we do not deliberately choose the slowest setting. LoRA inference and training speedups are reported separately. KV cache has no role in the training step.', '',
              'The combined row holds all three methods enabled and compares the individual winners against rank 4 / alpha 8 / LR 0.0001, cache prefill chunk 128, and FA tiles 16×32 / 4 warps. Individual winners are not jointly reoptimized. Confirmation uses fresh prompts, seven shuffled paired repeats, and bootstrap intervals. Prompt length, batch size, output length, precision and cache retention are held fixed.', '',
              'LoRA eligibility requires validation NLL within 0.02 of the best grid candidate; eligible candidates are ranked by tuning inference latency. The untuned LoRA control is reported even if it fails that quality constraint. Runtime settings must pass tensor correctness and full validation NLL checks. Test data does not select any settings.', '',
              '| Model | Validation NLL | Test fields accuracy |', '|---|---:|---:|']
    for label in ('baseline','tuned'):
        lines.append(f"| {label} | {report['combined_validation'][label]:.5f} | {report['quality'][label]['scores']['fields_correct']:.3f} |")
    lines += ['', 'Raw timing samples, token outputs, source/data hashes, frozen selections, quality scores and checkpoints are preserved beside this report. Training/search overhead is not included in inference latency. There is no claim that tuning amortizes its search cost.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def comparison(base, tuned, seed=42):
    b = [r['wall_ms'] for r in base['samples']]
    t = [r['wall_ms'] for r in tuned['samples']]
    rng = random.Random(seed)
    boot = sorted(statistics.median(b[i] for i in ix)/statistics.median(t[i] for i in ix)
                  for ix in ([rng.randrange(len(b)) for _ in b] for _ in range(2000)))
    return dict(baseline_ms=statistics.median(b), tuned_ms=statistics.median(t),
                speedup=statistics.median(b)/statistics.median(t), bootstrap_95=[boot[50],boot[1949]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path(__file__).parent/'results')
    parser.add_argument('--steps', type=int, default=1200)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    train_data = encode(load_rows(ROOT/'experiments/sports/train.jsonl'), tokenizer, 128)
    val_data = encode(load_rows(ROOT/'experiments/sports/validation.jsonl'), tokenizer, 128)
    report = dict(complete=False, gpu=torch.cuda.get_device_name(), node=platform.node(),
                  job=os.environ.get('SLURM_JOB_ID'), torch=torch.__version__, cuda=torch.version.cuda,
                  protocol=json.loads((Path(__file__).parent/'protocol.json').read_text()),
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__), Path(fa.__file__), ROOT/'mingpt/model.py',ROOT/'lora_experiment.py',ROOT/'experiments/sports/scoring.py']},
                  data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'experiments/sports').glob('*.jsonl')})
    save(out/'report.json', report)
    report['tile_correctness'] = validate_tiles()
    report['training'] = [train(c, train_data, val_data, out, args.steps) for c in LORA_GRID]
    assert len({r['sample_sha256'] for r in report['training']}) == 1
    save(out/'report.json', report)
    tuning_ids = torch.randint(50257, (1,896), generator=torch.Generator().manual_seed(7001)).cuda()
    cutoff = min(r['validation_nll'] for r in report['training'])+.02
    candidates = []
    for r in report['training']:
        c = r['config']
        if r['validation_nll'] <= cutoff or c == BASE_LORA:
            model = load_model(c, out/(key(c)+'.pt'))
            timing = measure_modes(model, tuning_ids, {'lora':mode()}, 5, 100)['lora']
            candidates.append(dict(config=c, eligible=r['validation_nll']<=cutoff, **timing))
            del model
            gc.collect(); torch.cuda.empty_cache()
    selected_lora = min((r for r in candidates if r['eligible']), key=lambda r:r['median_ms'])['config']
    # Cache and FA are tuned independently on the same fixed untuned-LoRA weights.
    model = load_model(BASE_LORA, out/(key(BASE_LORA)+'.pt'))
    cache_modes = {str(c):mode(cached=True, chunk=c) for c in CHUNK_GRID}
    cache_timings = measure_modes(model, tuning_ids, cache_modes, 5, 200)
    fa_modes = {str(t):mode(attention=True, tiles=t) for t in TILE_GRID}
    fa_timings = measure_modes(model, tuning_ids, fa_modes, 5, 300)
    # Check every runtime candidate on teacher-forced validation likelihood, not just random tensors.
    reference_nll = nll(model, val_data)
    for t in TILE_GRID:
        with backend(model, True, t):
            v = nll(model, val_data)
        fa_timings[str(t)]['validation_nll'] = v
        assert abs(v-reference_nll) <= .02
    # Same retained context: chunked prefill must preserve greedy benchmark output.
    reference = cache_timings[str(896)]['output_tokens']
    assert all(r['output_tokens'] == reference for r in cache_timings.values())
    selected_chunk = min(CHUNK_GRID, key=lambda c:cache_timings[str(c)]['median_ms'])
    selected_tiles = min(TILE_GRID, key=lambda t:fa_timings[str(t)]['median_ms'])
    selected = dict(lora=selected_lora, chunk=selected_chunk, tiles=selected_tiles)
    report.update(tuning=dict(lora=candidates, cache=cache_timings, fa=fa_timings), selected=selected,
                  validation_cutoff=cutoff, choices_frozen_before_confirmation_and_test=True)
    save(out/'selection.json', selected)
    save(out/'report.json', report)
    del model
    gc.collect(); torch.cuda.empty_cache()
    # Fresh prompts + fresh paired repetitions. Separate models interleaved for LoRA/combined.
    models = {label:load_model(c, out/(key(c)+'.pt')) for label,c in [('baseline',BASE_LORA),('tuned',selected_lora)]}
    settings = dict(lora_baseline=('baseline',mode()), lora_tuned=('tuned',mode()),
                    cache_baseline=('baseline',mode(cached=True)), cache_tuned=('baseline',mode(cached=True,chunk=selected_chunk)),
                    fa_baseline=('baseline',mode(attention=True)), fa_tuned=('baseline',mode(attention=True,tiles=selected_tiles)),
                    combined_baseline=('baseline',mode(cached=True,attention=True)),
                    combined_tuned=('tuned',mode(cached=True,chunk=selected_chunk,attention=True,tiles=selected_tiles)))
    samples = {n:dict(samples=[], outputs=[]) for n in settings}
    for rep in range(10):
        ids = torch.randint(50257,(1,896),generator=torch.Generator().manual_seed(9000+rep)).cuda()
        order = list(settings)
        random.Random(500+rep).shuffle(order)
        for name in order:
            label, setting = settings[name]
            measurement, output = latency(models[label], ids, **setting)
            if rep >= 3:
                samples[name]['samples'].append(measurement)
                samples[name]['outputs'].append(output.cpu().tolist())
        save(out/'confirmation-timings.json', samples)
        print('CONFIRM',rep,flush=True)
    report['confirmation'] = samples
    report['comparisons'] = {name:comparison(samples[name+'_baseline'],samples[name+'_tuned']) for name in ('lora','cache','fa','combined')}
    training = {key(r['config']):r for r in report['training']}
    b,t = (training[key(c)] for c in (BASE_LORA,selected_lora))
    report['comparisons']['lora_training_step'] = dict(baseline_ms=b['warm_median_step_ms'],tuned_ms=t['warm_median_step_ms'],speedup=b['warm_median_step_ms']/t['warm_median_step_ms'])
    report['combined_validation'] = {}
    report['quality'] = {}
    test_rows = load_rows(ROOT/'experiments/sports/test.jsonl')
    for label in ('baseline','tuned'):
        setting = settings['combined_'+label][1]
        with backend(models[label], setting['attention'], setting['tiles']):
            report['combined_validation'][label] = nll(models[label], val_data)
        report['quality'][label] = quality(models[label],tokenizer,test_rows,setting)
    report['complete'] = True
    save(out/'report.json', report)
    build_report(out,report)
    print('COMPLETE',json.dumps(report['comparisons']),flush=True)


if __name__ == '__main__':
    main()
