# Reproduce the minGPT LoRA demo

This guide runs the small GPT-2 LoRA FAQ experiment in `lora_experiment.py`.
It trains on 48 examples about fictional sports clubs and evaluates 12 held-out
paraphrases. It demonstrates adaptation to known facts, not general sports knowledge.
For the larger full-fine-tuning-versus-LoRA experiment, see
[the matched experiment guide](experiments/sports/matched/README.md).

## 1. Get the code and dependencies

```bash
git clone https://github.com/o-wel/Transformer-Acceleration-MQP.git
cd Transformer-Acceleration-MQP/transformer_syed/minGPT
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch 'transformers<5' numpy
python -m pip install -e .
python -c "import torch, transformers; print('torch:', torch.__version__); print('transformers:', transformers.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Use Python 3.12 for consistency with the original environment. The GPU commands
require a CUDA-enabled PyTorch installation compatible with your GPU and driver.
If the check prints `False`, use the CPU commands below or configure your CUDA
PyTorch environment first. Initial training and comparison need internet access
to download GPT-2 weights and its tokenizer from Hugging Face, unless cached.
`setup.py` only declares PyTorch; the extra dependencies above are needed by the demo.
The installation commands are a portable setup, not a locked original environment.

Run all remaining commands from this `minGPT` directory. The dataset files
`experiments/lora/train.jsonl` and `test.jsonl` are already included. Their original
generator is `python experiments/lora/make_data.py`; regeneration is optional.

## 2. Check the implementation without downloading GPT-2

```bash
python -m unittest discover -s tests -p test_lora.py -v
python -m unittest discover -s tests -p test_measurement.py -v
```

These use tiny local models to check LoRA math, zero-change initialization,
frozen-weight isolation, checkpoint restoration, and parameter accounting.

## 3. Train an adapter

On a CUDA GPU:

```bash
python lora_experiment.py train \
  --device cuda --out out/lora-repro-seed42 \
  --steps 300 --batch-size 4 --length 64 \
  --rank 4 --alpha 8 --lr 0.001 --seed 42 \
  --threads 4 --audit-every 50 --new-tokens 40
```

For CPU, run the same command with `--device cpu`. CPU training is slower and
CUDA memory metrics will be null. A short execution check is:

```bash
python lora_experiment.py train --device cpu \
  --out out/lora-repro-smoke --steps 2 --audit-every 1 --new-tokens 8
```

The smoke run only checks execution; it does not reproduce the 300-step quality
results. Always use a fresh `--out` path: training refuses to overwrite an existing
`report.json` or `adapter.pt`. The published repository excludes `.pt` checkpoints,
so train first even if older reports already exist.

Training uses FP32 with TF32 disabled, AdamW with zero weight decay, gradient
clipping at 1.0, and answer-only loss (prompts/padding are ignored). Rank-4 LoRA
adapts fused QKV, attention output, and both MLP projections in every GPT-2 layer.
Only 589,824 adapter scalars train; base weights remain frozen. This script uses
eager attention, without FlashAttention or a KV cache.

## 4. Compare base and adapted responses

Use the exact output directory from training:

```bash
python lora_experiment.py compare --device cpu \
  --out out/lora-repro-seed42 --new-tokens 40 \
  --prompt 'Who is the coach of the Harbor Comets?'
```

For interactive questions, omit `--prompt`:

```bash
python lora_experiment.py compare --device cpu --out out/lora-repro-seed42
```

Each question prints `BASE` and `LORA`; enter a blank line to exit. Replace
`cpu` with `cuda` for GPU inference. Questions are independent, with no chat
history. Comparison reloads GPT-2 and the saved `adapter.pt`, then toggles the
adapter off/on to compare the same base weights.

## 5. Inspect the evidence

The training output directory contains:

- `adapter.pt`: only adapter weights plus rank, alpha, and base-model metadata.
- `report.json`: configuration, library/GPU information, dataset hashes, parameter
  counts, training time, memory metrics, held-out answer NLL, and base/LoRA outputs.
- `changes-NNNN.json`: exact changed-scalar counts and per-tensor audits relative
  to initialization, at step 1, each audit interval, and the last step.

Inspect the main checks:

```bash
python - <<'PY'
import json
from pathlib import Path
r = json.loads(Path('out/lora-repro-seed42/report.json').read_text())
for key in ('parameters', 'final_changes', 'base_responses_match_initial',
            'base_test_answer_nll', 'final_test_answer_nll',
            'base_expected_phrase_accuracy', 'lora_expected_phrase_accuracy'):
    print(key, r[key])
PY
```

Expect zero frozen changed scalars and `base_responses_match_initial: True`.
Inspect every answer as well as aggregate scores: expected-phrase matching can
miss contradictions. [The original results](experiments/lora/RESULTS.md) recorded
NLL 2.9957 → 0.0213 and expected-phrase accuracy 0% → 100% on these 12 questions,
using an NVIDIA L40S, PyTorch 2.14.0+cu130, seed 42, and the settings above.
These are reference observations; hardware/library differences can affect results
and timings. Preserve your report and record `python -m pip freeze` with each run.

## 6. Optional full fine-tuning memory reference

Run in a separate process and output directory:

```bash
python lora_experiment.py train --device cuda --full \
  --steps 10 --batch-size 4 --length 64 --seed 42 \
  --out out/full-repro-memory
```

This 10-step run is a memory reference for the 300-step LoRA run, not a matched
quality or speed comparison. It writes audits and a report, but no adapter for
`compare`. For a quality comparison use the matched experiment linked above.

The existing [detailed LoRA README](experiments/lora/README.md) defines measurement
scopes. Its Slurm wrapper contains author-specific environment paths, partition
names, and fixed output paths; adapt those before submitting on another cluster.
