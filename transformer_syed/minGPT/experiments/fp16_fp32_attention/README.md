# Main FP16/FP32 attention comparison

Read [BACKENDS.md](BACKENDS.md) first: it defines eager, tiled, Flash, math and
efficient, and explains which comparisons isolate precision or implementation.
This experiment is the primary FP16 comparison. Prior BF16/forward-only reports
remain historical evidence and are not renamed or overwritten.

## Four main variants

| Variant | Master parameters | Execution policy | Attention |
|---|---|---|---|
| fp32_eager | FP32 | No autocast | Original separate PyTorch operations |
| fp32_tiled | FP32 | No autocast | Custom Triton forward and backward, IEEE FP32 dot products |
| fp16_eager | FP32 | FP16 autocast | Original separate PyTorch operations |
| fp16_flash | FP32 | FP16 autocast | PyTorch SDPA forced FlashAttention |

The same model object and checkpoint are used for all four main rows. Both FP16 rows
have the same autocast settings and, for training, GradScaler initialized to 128
with a large growth interval. FP32 rows do not require loss scaling. The FP32-to-
FP16 ratio therefore measures the precision **policy**, including its required
casting/scaling costs, rather than just a theoretical change in matrix arithmetic.
Internal accumulation and rounding can still differ between attention backends.

Inference is AMP FP16 with FP32 master weights, **not native FP16 weight storage**.
Autocast's weight-cast cache cost/memory is included. We keep weights identical
instead of changing model storage precision between rows. Do not use these AMP
memory results to claim halved model storage or compare them directly with the
older native-BF16 inference-memory results. Frozen LoRA and trainable full weights
can also have different autocast-cache behavior.

## FP32 backward implementation

`mingpt/tiled_attention.py` now supports first-order autograd. Forward saves Q/K/V,
the output and one log-normalizer per query. Backward computes the row reduction
`delta = sum(output * doutput)` and recomputes probabilities in tiles to form dQ,
dK and dV. It does not save or globally materialize an attention-score matrix, and
its key/value tiles own their gradients without atomic accumulation.

Q/K/V and upstream-gradient copies for noncontiguous layouts are linear-size and
included in training time/memory. All dot products request `input_precision="ieee"`.
FP32 GPU exp/log approximations and reduction order still differ from eager.
Attention dropout and higher-order derivatives are unsupported; benchmark dropout
is zero for every variant. Other PyTorch model operations retain normal autograd.

```python
model = model.float().cuda().set_attention_backend('tiled_fp32')
# Construct the model with config.attn_pdrop = 0 for training.
logits, loss = model(tokens, targets)
loss.backward()
```

Validation compares forward and dQ/dK/dV against float64, tests noncontiguous
layouts, odd and cached lengths, real full/LoRA model gradients and four optimizer
updates, verifies linear saved state, and checks compiled forward/backward PTX.
Real-checkpoint gradient errors and finite FP16 gradients are recorded separately.

## Measurement protocol

Run `sbatch experiments/fp16_fp32_attention/run_gpu.sh`. It refuses to overwrite
an existing `results/report.json`. After it finishes, run
`/home/sashazli/torch_env/bin/python experiments/fp16_fp32_attention/build_report.py`.
The report environment needs NumPy and Matplotlib; the GPU environment needs
PyTorch, Transformers and Triton. GPT-2 weights/tokenizer and trained LoRA adapter
must already be present at the paths used by the existing experiments.

Inference: pretrained and trained unmerged LoRA; B1/P128, B1/P512, B1/P896 and
B4/P128; 64 generated tokens with no early EOS stopping; cache off/on. Three
warmups, seven measured repetitions, all eight variant/cache modes shuffled per
repetition. Same deterministic prompts and output budget. A complete finite-logit
check runs in the first warmup and is excluded from measured samples.

Training: batch 2, lengths 128/512/896, full and LoRA. Every individual timed trial
starts from the original checkpoint, identical synthetic batch, zero AdamW moments
and step=0. Model copies, optimizer-state reset and gradient clearing are outside
timing. This paired reset-step protocol isolates update cost; it is not a training
trajectory or convergence benchmark. Both FP16 variants use the same loss scaling;
all gradients must be finite and every optimizer state must advance exactly one
step. Skipped/overflow updates invalidate the comparison rather than counting as
fast steps. AdamW state is materialized before timing.

One resident model provides direct PyTorch allocation peaks. Inference weights are
audited before/after; diagnostic training restores them afterward. Profiling and
compilation are outside measured samples. CUDA events and synchronized wall time,
throughput, peak/baseline/incremental allocation, cache bytes, raw samples, loss
scales and successful updates are retained. TF32 is disabled.

Quality: all 120 held-out sports questions on each checkpoint, each variant with
cache off/on; record every token sequence, primary/field score and agreement
through EOS. Compare both within precision and against FP32 eager. Do not infer
long-run training quality from gradient checks or one-step timings.

## Artifacts

- [PDF report](results/FP16_FP32_ATTENTION_REPORT.pdf)
- [Results and charts](results/RESULTS.md), standalone [HTML](results/RESULTS.html)
- [Metrics CSV](results/METRICS.csv), [all answers](results/ALL_OUTPUTS.csv)
- Raw report, source hashes/snapshots, CUDA traces, forward/backward PTX/cubins,
  disassembly, static resource usage and instruction audit under `results/`.

## Native FP16 inference supplement

A separate native-FP16 inference run pairs eager and Flash with half-precision
weights and no autocast. Queue it after the primary job:

```bash
sbatch --dependency=afterok:PRIMARY_JOB_ID experiments/fp16_fp32_attention/run_native_gpu.sh
```

This runs on a separate allocation and writes `results/native_fp16/`. All four
shapes, both checkpoints, raw samples and 120-question quality checks are retained.
The report includes it as a separate deployment-policy supplement. Do not pool
its samples with the primary AMP experiment or attribute cross-policy differences
solely to attention kernels or weight casting. No native-FP16 training is claimed.

The old [BF16 experiment](../flash_attention/README.md) and
[forward-only FP32 experiment](../fp32_attention/README.md) are preserved with
their original source snapshots. Their timing protocols differ from this paired
AMP/reset-step comparison and must not be pooled into its statistics.
