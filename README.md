# Transformer Acceleration MQP

This project implements and evaluates decoder-only transformers, **LoRA**,
**KV caching**, and **FlashAttention-style attention**, with controlled studies
of their interactions, precision, model quality, memory, and inference latency.

The repository contains Owen's from-scratch transformer variants and Syed's
minGPT-based research. Results below belong to the specified implementation and
experiment; a minGPT speedup is not a measured speedup of
[`transformer_owen/decoder_transformer.py`](transformer_owen/decoder_transformer.py).

## Latest report: joint decoding with selected hyperparameters

Read the **[nine-page PDF report](transformer_syed/minGPT/experiments/graph_decode_optimal/MINGPT_OPTIMIZATION_REPORT.pdf)**,
[standalone HTML](transformer_syed/minGPT/experiments/graph_decode_optimal/REPORT.html),
or [Markdown](transformer_syed/minGPT/experiments/graph_decode_optimal/REPORT.md).
The [reproducibility package](transformer_syed/minGPT/experiments/MINGPT_OPTIMIZATION_REPORT_PACKAGE.zip)
includes protocols, source snapshots, figures, raw measurements and outputs,
selection records, and content hashes. Pretrained and adapter checkpoint weights
are not included in this update.

The latest experiment holds the trained model fixed while comparing seven
inference implementations. Five independently trained GPT-2/minGPT checkpoints
were measured in five fresh NVIDIA A30 allocations, using eight held-out workload
shapes and ten randomized paired timing blocks per shape.

| Matched comparison | Measured speedup | Paired 95% interval |
|---|---:|---:|
| Original dynamic-cache FlashAttention path / selected fused graph | **4.695×** | **4.667–4.720×** |
| Previous Python-driven joint fusion / selected fused graph | 4.182× | 4.159–4.206× |
| Same fused runtime without / with graph replay | 3.754× | 3.731–3.774× |
| Ordinary LoRA graph / fused LoRA graph | 1.0355× | 1.0305–1.0406× |
| Default graph launch settings / selected settings | 1.0064× | 1.0020–1.0106× |
| Original path / merged-LoRA graph control | 7.415× | 7.363–7.466× |

Ratios are comparison latency divided by candidate latency: values above one
mean faster execution. These are measured request-level comparisons, not products
of isolated speedups. The merged control is faster and can be preferable when
retaining an unmerged adapter is unnecessary.

**Quality:** original, selected-graph, and merged-graph inference each achieved
100% normalized exact-reference accuracy on 360 fresh synthetic questions per
checkpoint. All 1,800 model/question cases had identical trimmed answers across
the three modes. There are **120 unique match groups**, not 1,800 independent
questions. Raw 64-token continuations were not always identical. This is a
limited synthetic task, not evidence of universal language-model accuracy.
The stronger accuracy relative to earlier exploratory checkpoints comes from
training/checkpoint selection; graph replay did not improve model accuracy.

### What the optimization actually does

1. **Full prefill:** forced PyTorch FlashAttention processes the prompt and builds
   adapter-specific K/V. No previous prompt's cache is reused.
2. **Static KV buffers:** subsequent tokens write into preallocated cache slots
   instead of concatenating and copying the growing history.
3. **GPU-resident state:** the next token, learned absolute position, output buffer
   and write index keep stable addresses while their values change on the GPU.
4. **Fused Triton attention:** each head expands the current token's LoRA Q/K/V
   corrections, appends adapted K/V, and streams through valid historical K/V
   with online softmax. The base projection and LoRA down-projection remain
   separate operations.
5. **CUDA graph replay:** Python captures one complete decoding step and replays
   its dependent GPU operations with one host graph launch per generated token.

**This is not a whole-model CUDA megakernel.** The attention fusion is a Triton
GPU kernel; embeddings, other projections, MLPs and sampling still execute as
multiple GPU kernels. Python constructs and replays a CUDA graph containing
those kernels. The large gain comes mainly from consolidating CPU launch work,
with smaller additional gains from fusion and launch-parameter tuning.

For a 128-token answer, prefill predicts the first answer token, followed by 127
one-token graph replays. The graph records computations, not a fixed answer.
The growing valid cache length is read from a device tensor; unused reserved
positions are not treated as context. All valid history is retained.

### Novelty and relationship to prior work

The engineering contribution investigated here is the **specific integration of
current-token LoRA Q/K/V expansion, full-cache append and streaming attention in
a graph-compatible minGPT decoding step**, together with controlled ablations
and reproducible evaluation.

Publication-level novelty is **not established**. KV caching, LoRA, online-softmax
attention and CUDA graphs are established techniques. Simply enabling all three
optimizations is not new. [LRAgent](https://arxiv.org/abs/2602.01053) already
combines low-rank cache corrections with Flash-LoRA-Attention, and
[EpiLoRA](https://arxiv.org/abs/2611.03899) addresses LoRA expansion fusion.
[CUDA graph replay](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
and graph-compatible adapter serving also have production precedents. A large
improvement over this repository's Python loop does not establish a new graph
algorithm or state-of-the-art serving performance.

The decoder uses a **custom FlashAttention-style online-softmax kernel**, not an
official FA1/FA2 decode release. Prefill uses forced PyTorch FlashAttention.
Earlier FA1-derived prototypes, FP32 kernels and production backend comparisons
are labeled separately in their reports.

### Best-tested hyperparameters

| Component | Selected setting | Selection basis |
|---|---|---|
| LoRA rank / alpha | **2 / 8**, scaling 4 | Inherited validation-based training study |
| Training learning rate | **0.001**, constant | Checkpoint training configuration |
| LoRA targets | QKV, attention output, both MLP projections in every block | 294,912 trainable adapter parameters |
| Context / KV capacity | **1,024** tokens, full retained context | GPT-2 learned-position limit |
| Prefill | One full prompt pass; chunk setting 1,024 | Inherited cache selection |
| Decoder history tile / warps | **256 / 8** | New 16-configuration runtime search |
| Serving policy | Native FP16; TF32 off; greedy softmax/top-k | Identical across matched runtime modes |

The inherited study searched 45 rank/scaling/LR settings with three discovery
seeds and selected the fastest training setting within 0.01 validation NLL of
the quality-best setting. Its operational choice is distinct from its
quality-best rank-32/alpha-64/LR-0.0003 setting. Learning rate affects the trained
checkpoint, not inference-time kernel launch behavior.

The graph-specific search tested history tiles `{32,64,128,256}` crossed with
warps `{1,2,4,8}` on three discovery allocations. Selection was frozen before
five-seed confirmation and fresh task evaluation. The selected launch setting
was slightly slower than 128/4 on one of eight held-out shapes, despite its
positive aggregate result. **Best tested under a declared objective is not a
global optimum or a per-workload optimum.**

See the [frozen protocol](transformer_syed/minGPT/experiments/graph_decode_optimal/PROTOCOL.md),
[selection record](transformer_syed/minGPT/experiments/graph_decode_optimal/selection.json),
and [machine-readable summary](transformer_syed/minGPT/experiments/graph_decode_optimal/SUMMARY.json).

## Implementations and completed work

| Area | Work completed | Source / evidence |
|---|---|---|
| From-scratch transformer | Decoder-only baseline, rotary embeddings and absolute-position variant; training, tokenization and generation | [Owen baseline](transformer_owen/decoder_transformer.py), [absolute positions](transformer_owen/decoder_transformer_absolute.py), [Syed transformer](transformer_syed/transformer.py) |
| Separate optimization implementations | LoRA, KV-cache and attention variants alongside the baseline | [LoRA](LoRA_transformer_owen/), [KV cache](KV_transformer_owen/), [attention](flash_attention_owen/) |
| minGPT LoRA and measurement | Frozen-base adapters, exact weight-change audits, trainable/storage accounting, optimizer integration and tests | [minGPT guide](transformer_syed/minGPT/README.md), [model](transformer_syed/minGPT/mingpt/model.py), [measurement](transformer_syed/minGPT/mingpt/measurement.py) |
| Character-level training data | Literature and everyday-dialogue preparation with source/hash/vocabulary manifests; separate character and GPT-2 tokenization workflows | [charGPT guide](transformer_syed/minGPT/projects/chargpt/readme.md) |
| Initial LoRA demonstration | Fictional FAQ adaptation and base/adapter comparisons, with explicit memorization/generalization limits | [FAQ protocol](transformer_syed/minGPT/experiments/lora/README.md) |
| Sports reasoning and matched controls | Unseen names/scores, counterfactual score swaps, three-seed evaluation, equal-budget full fine-tuning versus LoRA, and stopping-rule studies | [sports](transformer_syed/minGPT/experiments/sports/README.md), [matched results](transformer_syed/minGPT/experiments/sports/matched/results/RESULTS.md), [stopping](transformer_syed/minGPT/experiments/sports/stopping/RESULTS.md) |
| Real-data classification | Deduplicated/template-grouped UCI SMS Spam study; equal LR searches, three seeds, accuracy/F1/memory/time comparisons | [SMS results](transformer_syed/minGPT/experiments/sms_spam/RESULTS.md) |
| KV-cache evaluation | Cached versus uncached generation, prefill/decode separation, full-context retention and learned-position rollover correctness | [cache benchmark](transformer_syed/minGPT/experiments/kv_cache/README.md) |
| Attention and precision attribution | Eager/SDPA/forced FlashAttention, FP32 tiled inference, first-order tiled backward, matched FP16/FP32 studies and kernel/profiler evidence | [attention](transformer_syed/minGPT/experiments/flash_attention/README.md), [FP32](transformer_syed/minGPT/experiments/fp32_attention/README.md), [precision glossary](transformer_syed/minGPT/experiments/fp16_fp32_attention/BACKENDS.md) |
| Combined-method review | Matched FA1-derived comparison, deployment-backend comparison, quality checks and merged-LoRA ablation | [combined review](transformer_syed/minGPT/experiments/flash_attention/results/combined_optimization_review/OPTIMIZATION_COMPARISON.md) |
| Exploratory tuning | Initial separate LoRA/cache/attention selection, confirmation and recorded protocol amendments | [pilot tuning](transformer_syed/minGPT/experiments/hyperparameter_tuning/results/RESULTS.md) |
| Independent hyperparameter confirmation | 45 training settings × three discovery seeds; validation-only selection; five confirmation seeds and three runtime allocations per seed | [methods](transformer_syed/minGPT/experiments/paper_hyperparameters/METHODS.md), [results](transformer_syed/minGPT/experiments/paper_hyperparameters/runs/study-20261105-v2/RESULTS.md) |
| Broader FP32 staged search | 127 unique training trials and 1,440 serving cells; rank/scaling/LR/targets, regularization, batching, accumulation, padding and backend/cache ablations | [FP32 study](transformer_syed/minGPT/experiments/flash_attention/results/hyperparameter_search/README.md), [final report](transformer_syed/minGPT/experiments/flash_attention/results/hyperparameter_search/run_fp32/FINAL_REPORT.pdf) |
| Joint decode fusion | Current-token LoRA expansion + KV append + streaming attention, with static-cache and standalone-expansion controls | [fusion research](transformer_syed/minGPT/experiments/joint_decode/RESEARCH.md), [results](transformer_syed/minGPT/experiments/joint_decode/RESULTS.md) |
| Graph-compatible decoding | Device-resident decode state, reusable full-step graphs, eight held-out workloads and fresh task checks | [graph research](transformer_syed/minGPT/experiments/graph_decode/RESEARCH.md), [results](transformer_syed/minGPT/experiments/graph_decode/RESULTS.md) |
| Selected-model graph confirmation | New decoder launch search, five selected training checkpoints, fresh matched timing and quality, setup/memory accounting | [latest report](transformer_syed/minGPT/experiments/graph_decode_optimal/REPORT.md), [implementation](transformer_syed/minGPT/experiments/graph_decode_optimal/runtime.py) |

The broader FP32 staged search selected rank 32 / alpha 64 / LR 0.0003, all
projections, adapter dropout 0.05, weight decay 0.01, effective/microbatch 4 and
padded length 128 under its validation-quality objective. It is a **different
study** from the native-FP16 graph operating point above. Its 93.06% mean
held-out field accuracy and the graph study's 100% fresh exact accuracy use
different evaluation sets and must not be treated as a matched accuracy gain.

## Methodology and interpretation

- **Separate discovery from confirmation.** Validation selects hyperparameters
  and checkpoints; fresh confirmation timings and test answers do not retune them.
- **Use matched model weights.** Kernel/cache/graph contrasts hold the checkpoint
  fixed. Training-method comparisons disclose their training and tuning budgets.
- **Measure the complete request.** Prefill, cache initialization/copy, decoding
  and output assembly are included. Loading, compilation and graph setup are
  accounted separately; reported warm speedups are not clean-machine cold starts.
- **Preserve failures and numerical differences.** Teacher-forced logits,
  eager/graph parity, cache boundaries, stale-cache isolation, raw outputs and
  exact parameter hashes supplement task scoring. FP16 paths need not be
  bitwise identical to a different attention implementation.
- **Respect statistical units.** The latest timing bootstrap resamples five
  composite training-seed/allocation units and paired blocks. Quality resamples
  seeds and match groups. Repeated questions, timings and GPUs are not inflated
  into independent samples. One allocation per seed cannot separate those two
  variance components.
- **Report limitations and regressions.** The graph results concern this
  GPT-2/minGPT implementation, A30 GPUs, native FP16 and the declared workload
  family. No fixed GPU clocks or comprehensive co-tenant isolation were imposed.
  A perfect small synthetic test is not proof of population-level equality.

The graph runner supports one dense fixed-batch request stream, one fixed
model/adapter, greedy decoding and a 1,024-position context. A new prompt resets
the cache. Adapter changes require a new runner; context rollover, continuous
batching, concurrent requests and arbitrary sampling are not implemented.

## Setup and reproduction

Install minGPT from this repository's copy in an appropriate PyTorch environment:

```bash
cd transformer_syed/minGPT
python -m pip install -e .
python -m unittest discover -s tests -p 'test_lora.py' -v
python -m unittest discover -s tests -p 'test_kv_cache.py' -v
```

CUDA attention experiments need a compatible CUDA GPU, PyTorch backend and,
for custom kernels, Triton. Exact experiment versions are recorded in the
reports and `software-freeze.txt` files. The A30 study's Slurm scripts retain
cluster-specific paths/account settings and must be adapted on another machine.
Pretrained GPT-2 assets and the recorded trained adapters must be available
before reproducing checkpoint-based comparisons; reports and hashes do not
replace the weights.

After loading a selected rank-2/alpha-8 checkpoint into an eval-mode native-FP16
CUDA model with the `flash` backend:

```python
import torch
from experiments.graph_decode_optimal.runtime import Runner

with torch.inference_mode():
    runner = Runner(model, batch=1, fused=True, graph=True, bn=256, warps=8)
    output_ids = runner.generate(input_ids, count=64)
```

`input_ids` must be a dense CUDA `int64` batch of the declared size, and prompt
plus output must fit the context. Construct the runner once and reuse it across
prompts. Start with the [latest experiment README](transformer_syed/minGPT/experiments/graph_decode_optimal/README.md)
and [frozen protocol](transformer_syed/minGPT/experiments/graph_decode_optimal/PROTOCOL.md)
for the full commands, evidence layout, checkpoint provenance and limitations.

## Combined optimization correctness test

Run LoRA, forced CUDA FlashAttention, and KV caching together:

```bash
cd transformer_syed/minGPT
python -m unittest discover -s tests -p 'test_flash_attention.py' -k combined -v
```

This offline test uses a tiny model with nonzero, unmerged LoRA adapters. It
compares adapter gradients and cached prefill/decode logits against eager
attention, then checks cached versus uncached generation across context-window
overflow. It requires a CUDA GPU and PyTorch with FlashAttention support; without
CUDA it reports a skip. This is a correctness test, not a performance benchmark.
