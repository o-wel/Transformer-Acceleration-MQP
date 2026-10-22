"""Render measured JSON results into a meeting handout; no inferred GPU numbers."""
import argparse
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--lora', default='out/lora-sports-run1')
parser.add_argument('--full', default='out/full-sports-run1')
parser.add_argument('--output', default='experiments/lora/RESULTS.md')
args = parser.parse_args()
lora_path = root/args.lora/'report.json'
full_path = root/args.full/'report.json'
if not lora_path.exists():
    raise SystemExit('Training report is not available yet.')
r = json.loads(lora_path.read_text())
c = r['parameters']
lines = ['# Preliminary FP32 LoRA results', '',
         f"Device: {r['gpu'] or r['config']['device']}; PyTorch: {r['torch_version']}; CUDA: {r['cuda_version']}.",
         f"Seed {r['config']['seed']}, rank {r['config']['rank']}, alpha {r['config']['alpha']}, batch {r['config']['batch_size']}, sequence length {r['config']['length']}, {r['config']['steps']} steps.", '',
         '| Measurement | LoRA |', '|---|---:|',
         f"| Total parameters | {c['total']:,} |",
         f"| Trainable parameters | {c['trainable']:,} ({c['trainable_percent']:.4f}%) |",
         f"| Frozen parameters | {c['frozen']:,} |",
         f"| Actually changed scalars | {r['final_changes']['changed_elements']:,} |",
         f"| Frozen scalars changed | {r['final_changes']['frozen_changed_elements']} |",
         f"| All parameter storage (MiB) | {c['parameter_bytes']/2**20:.2f} |",
         f"| Adapter parameter storage (MiB) | {c['trainable_parameter_bytes']/2**20:.2f} |"]
for title, key in [('Model placement','initial_model_memory'),('First training step','first_training_step_memory'),('Training window','training_memory')]:
    m = r[key]
    if m:
        for label, metric in [('allocated','peak_allocated_bytes'), ('reserved','peak_reserved_bytes')]:
            lines.append(f"| {title} peak {label} (MiB) | {m[metric]/2**20:.2f} |")
lines += ['', f"Held-out answer NLL: {r['base_test_answer_nll']:.4f} → {r['final_test_answer_nll']:.4f} nats/token.",
          f"Expected-phrase score: {r['base_expected_phrase_accuracy']:.1%} → {r['lora_expected_phrase_accuracy']:.1%} (12 fixed questions).",
          '', 'This is a fictional FAQ memorization/paraphrase demonstration, not broad sports-chat quality. No test-based checkpoint selection. See README for measurement scope.', '']
if full_path.exists():
    full = json.loads(full_path.read_text())
    if full['training_memory']:
        lines += [f"Full fine-tuning reference ({full['config']['steps']} steps): peak allocated {full['training_memory']['peak_allocated_bytes']/2**20:.2f} MiB; peak reserved {full['training_memory']['peak_reserved_bytes']/2**20:.2f} MiB. Different training duration: compare memory only, not quality or speed.", '']
lines += ['## All held-out outputs', '']
for row in r['comparisons']:
    lines += [f"### {row['question']}", '', f"Reference: {row['reference']}", '', 'Base:', '```text',row['base'],'```','LoRA:','```text',row['lora'],'```','']
(root/args.output).write_text('\n'.join(lines))
print(root/args.output)
