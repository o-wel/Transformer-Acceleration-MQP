"""FP32 minGPT LoRA training, accounting, held-out evaluation, and interactive comparison."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from transformers import GPT2Tokenizer
from mingpt.model import GPT, LoRALinear
from mingpt.measurement import ChangeAudit, parameter_counts, cuda_memory


def prompt(question):
    return f'User: {question}\nAssistant:'


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def encode(rows, tokenizer, length):
    result = []
    for row in rows:
        prefix = tokenizer.encode(prompt(row['question']))
        ids = prefix + tokenizer.encode(' ' + row['answer']) + [tokenizer.eos_token_id]
        if len(ids) > length + 1:
            raise ValueError('Example exceeds sequence length; increase --length')
        x = ids[:-1] + [tokenizer.eos_token_id]*(length-len(ids)+1)
        y = [-1]*(len(prefix)-1) + ids[len(prefix):] + [-1]*(length-len(ids)+1)
        result.append((torch.tensor(x), torch.tensor(y)))
    return result


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    total, count = 0., 0
    for x, y in data:
        y = y.unsqueeze(0).to(device)
        loss = model(x.unsqueeze(0).to(device), y)[1]
        n = int((y != -1).sum())
        total += loss.item()*n
        count += n
    return total/count


@torch.no_grad()
def answer(model, tokenizer, question, device, tokens):
    model.eval()
    ids = tokenizer.encode(prompt(question), return_tensors='pt').to(device)
    if ids.shape[1] + tokens > model.block_size:
        raise ValueError('Question is too long for this model context')
    start = ids.shape[1]
    for _ in range(tokens):
        logits, _ = model(ids)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat((ids, nxt), dim=1)
        if nxt.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(ids[0, start:].tolist(), skip_special_tokens=True).split('\nUser:')[0].strip()


def scales(model, enabled):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.scale = module.alpha/module.rank if enabled else 0.


def comparisons(model, tokenizer, rows, device, tokens):
    outputs = []
    for row in rows:
        pair = dict(question=row['question'], reference=row['answer'])
        for enabled, name in [(False, 'base'), (True, 'lora')]:
            scales(model, enabled)
            pair[name] = answer(model, tokenizer, row['question'], device, tokens)
            pair[name+'_contains_expected'] = row['expected'].lower() in pair[name].lower()
        outputs.append(pair)
    scales(model, True)
    return outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['train', 'compare'])
    p.add_argument('--out', default='out/lora-sports')
    p.add_argument('--train', default='experiments/lora/train.jsonl')
    p.add_argument('--test', default='experiments/lora/test.jsonl')
    p.add_argument('--device', default='cuda')
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--length', type=int, default=64)
    p.add_argument('--rank', type=int, default=4)
    p.add_argument('--alpha', type=float, default=8)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--audit-every', type=int, default=50)
    p.add_argument('--new-tokens', type=int, default=40)
    p.add_argument('--full', action='store_true', help='Full fine-tuning memory baseline, separate output directory')
    p.add_argument('--prompt')
    a = p.parse_args()
    if min(a.steps, a.batch_size, a.length, a.audit_every, a.new_tokens) < 1 or a.length > 1024:
        p.error('Positive sizes required and length must be <= 1024')
    torch.set_num_threads(a.threads)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(a.device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT.from_pretrained('gpt2').float()
    out = Path(a.out)
    if a.mode == 'compare':
        ckpt = torch.load(out/'adapter.pt', map_location='cpu', weights_only=True)
        model.enable_lora(ckpt['rank'], ckpt['alpha'])
        missing, unexpected = model.load_state_dict(ckpt['adapter'], strict=False)
        assert not unexpected and all(not n.endswith(('A.weight', 'B.weight')) for n in missing)
        model.to(device).eval()
        while True:
            try:
                question = a.prompt or input('You (blank to quit): ').strip()
            except EOFError:
                break
            if not question:
                break
            for enabled, name in [(False, 'BASE'), (True, 'LORA')]:
                scales(model, enabled)
                print(f'{name}: {answer(model, tokenizer, question, device, a.new_tokens)}')
            if a.prompt:
                break
        return
    out.mkdir(parents=True, exist_ok=True)
    if (out/'report.json').exists() or (out/'adapter.pt').exists():
        raise FileExistsError('Use a new --out directory to preserve prior results')
    train_rows, test_rows = load_rows(a.train), load_rows(a.test)
    assert not ({r['question'] for r in train_rows} & {r['question'] for r in test_rows})
    train, test = encode(train_rows, tokenizer, a.length), encode(test_rows, tokenizer, a.length)
    model.eval()
    probe = train[0][0][:16].unsqueeze(0)
    with torch.no_grad():
        base_logits = model(probe)[0]
    base_counts = parameter_counts(model)
    if not a.full:
        model.enable_lora(a.rank, a.alpha)
        with torch.no_grad():
            torch.testing.assert_close(model(probe)[0], base_logits, rtol=0, atol=0)
    del base_logits, probe
    audit = ChangeAudit(model)
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.)
    report = dict(config=vars(a), torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                  base_parameters=base_counts, parameters=parameter_counts(model),
                  data_sha256={str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in [a.train, a.test]},
                  initial_model_memory=cuda_memory(device), dtype='float32', tf32=False,
                  base_test_answer_nll=evaluate(model, test, device), history=[])
    if not a.full:
        report['base_responses'] = [answer(model, tokenizer, r['question'], device, a.new_tokens) for r in test_rows]
    optimizer.zero_grad(set_to_none=True)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    total_seconds = 0.
    for step in range(1, a.steps+1):
        batch = random.choices(train, k=a.batch_size)
        x, y = (torch.stack([row[i] for row in batch]).to(device) for i in range(2))
        model.train()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y)[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter()-start
        total_seconds += elapsed
        if step == 1:
            report['first_training_step_memory'] = cuda_memory(device)
        if step == 1 or step % a.audit_every == 0 or step == a.steps:
            changes = audit.measure(model)
            (out/f'changes-{step:04d}.json').write_text(json.dumps(changes, indent=2))
            row = dict(step=step, loss=loss.item(), changed_elements=changes['changed_elements'],
                       frozen_changed_elements=changes['frozen_changed_elements'])
            report['history'].append(row)
            print(json.dumps(row), flush=True)
    report['training_memory'] = cuda_memory(device)
    report['training_step_seconds'] = total_seconds
    report['processed_tokens_per_second'] = a.steps*a.batch_size*a.length/total_seconds
    report['timing_scope'] = 'forward/backward/clip/optimizer with synchronization; excludes data transfer, audits, evaluation; includes cold first step and padding tokens'
    report['gradient_bytes'] = sum(p.grad.numel()*p.grad.element_size() for p in model.parameters() if p.grad is not None)
    report['optimizer_tensor_bytes'] = sum(v.numel()*v.element_size() for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v))
    report['final_changes'] = {k:v for k,v in changes.items() if k != 'per_tensor'}
    report['final_test_answer_nll'] = evaluate(model, test, device)
    if not a.full:
        torch.save(dict(rank=a.rank, alpha=a.alpha, base='gpt2',
                        adapter={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}), out/'adapter.pt')
        report['comparisons'] = comparisons(model, tokenizer, test_rows, device, a.new_tokens)
        report['base_responses_match_initial'] = report['base_responses'] == [r['base'] for r in report['comparisons']]
        if not report['base_responses_match_initial']:
            raise RuntimeError('Disabling LoRA did not reproduce the original base responses')
        for name in ['base', 'lora']:
            report[name+'_expected_phrase_accuracy'] = sum(r[name+'_contains_expected'] for r in report['comparisons'])/len(test_rows)
    (out/'report.json').write_text(json.dumps(report, indent=2))
    print(f'Saved {out}/report.json', flush=True)


if __name__ == '__main__':
    main()
