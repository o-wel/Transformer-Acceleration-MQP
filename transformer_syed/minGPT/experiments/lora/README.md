# FP32 LoRA demonstration

Run from `~/MQP/MinGPT/minGPT`, after `source ~/MQP/Torch/setup.sh`.

```bash
srun -p short --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=00:20:00 \
  python lora_experiment.py train --out out/lora-sports-run1

# Interactive: each question prints BASE and LORA responses.
srun --pty -p short --gres=gpu:1 --cpus-per-task=4 --mem=12G --time=00:20:00 \
  python lora_experiment.py compare --out out/lora-sports-run1

# A single question (also works with --device cpu without srun).
python lora_experiment.py compare --device cpu --out out/lora-sports-run1 \
  --prompt 'Who is the coach of the Harbor Comets?'

# Separate process, same batch/sequence size, short full-training memory baseline.
srun -p short --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=00:10:00 \
  python lora_experiment.py train --full --steps 10 --out out/full-sports-run1
```

For queued cluster execution, `sbatch experiments/lora/run_gpu.sh` runs both GPU experiments and generates `experiments/lora/RESULTS.md`. Submit it from the minGPT root.

Use a new output directory for each run. `report.json` records settings, data hashes,
parameter counts, memory, quality, timing, and every held-out response.
`changes-NNNN.json` records per-tensor exact changed-element counts and delta norms
relative to initialization, at step 1, every 50 steps, and the last step.
`adapter.pt` contains only adapter weights and rank/alpha metadata. Reloading downloads
or uses cached GPT-2 base weights. Existing `GPT.from_pretrained('gpt2')` now returns
an unadapted base model; opt in with `.enable_lora(...)` or `lora_rank=4`.

## What this demonstrates

The included 48 training examples and 12 held-out questions describe three **fictional**
sports clubs. `make_data.py` is the original data source. Held-out questions ask about
training facts using different wording: this measures memorization and paraphrase
transfer, not unseen-fact generalization or broad sports knowledge. The 12 prompts
are fixed before training. No checkpoint selection based on test results is performed.
Report every response, including failures. Exact expected-phrase presence is only a
rough factual score; also inspect answers for contradictions. Answer-only cross entropy
(NLL, nats/token) excludes user prompts and padding. Generation is greedy for both
models and uses the same formatting and token limit. The interactive mode answers
independent questions; it does not retain multi-turn history.

For a broader experiment, supply independently curated `--train` and `--test` JSONL
files with `question`, `answer`, and `expected` fields. Split by conversation or source
before creating paraphrases, reserve validation data for tuning, and keep a final
untouched test set. A tiny fictional FAQ is a useful first demonstration, not evidence
that a small GPT-2 is a general sports assistant.

## Accounting and measurement definitions

- Total/trainable/frozen counts include all stored parameter scalars, including minGPT's
  separate input embedding and output head. This implementation does not tie their storage,
  so its physical parameter count exceeds the commonly quoted tied GPT-2 count.
  
- LoRA adapts fused QKV, attention output, and both MLP projections, in all 12 layers.
  Rank 4 adds `12 * 16 * 768 * 4 = 589824` scalars. All other parameters freeze.

- Changed parameters means individual scalar values unequal to their initial values;
  changed tensor counts alone are not sufficient. Audits compare exact FP32 values on CPU,
  verify no frozen gradients, and raise on frozen-weight changes. Intermediate snapshots
  are differences from initialization, not a cumulative count of every update event.

- FP32 storage is **4 bytes** per scalar (32 bits). Buffers are reported separately.
  Gradient and optimizer tensor storage is measured after training, not inferred from
  the trainable count. These components do not sum to runtime peak usage.

- Initial model memory is captured after model placement and optimizer creation, before
  evaluation or training; Adam states are allocated lazily on the first step.

- Training peak counters reset after base evaluation/generation and cache clearing.
  They include forward, backward, clipping, optimizer steps, activations and workspaces
  managed by PyTorch, including the first optimizer-state allocation. CPU audit snapshots
  do not contribute to GPU storage. Final evaluation/generation is outside this window.

- Peak allocated memory measures live PyTorch tensor allocations; peak reserved memory
  measures its caching allocator. Neither is the whole-device/driver footprint, and the
  two must not be added. See [PyTorch peak memory documentation](https://docs.pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html).

- Runtime sums synchronized training steps, excluding data transfer, audit overhead,
  evaluation and generation. It includes the cold first step, so this is preliminary
  throughput, not a warmed repeated benchmark. Processed tokens include padding.

- Both experiments use FP32, with TF32 disabled, no AMP, no FlashAttention, and no KV cache.
  A 10-step full-training run is a memory reference only, not a matched quality comparison
  to 300 steps of LoRA. Use equal steps and repeated seeds for quality/runtime comparisons.

CPU validation:

```bash
python -m unittest discover -s tests -p test_lora.py
python -m unittest discover -s tests -p test_measurement.py
```

A separately labeled CPU demonstration is available at `out/lora-sports-cpu` after its run completes (200 steps, 24-token generation limit). Use that path with `compare --device cpu` while GPU measurements are pending. CPU runs report null CUDA metrics.
