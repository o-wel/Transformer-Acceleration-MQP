# FP32 tiled attention and precision controls

This experiment adds two opt-in minGPT backends:

- `tiled_fp32`: our inference-only Triton implementation of tiled online-softmax
  attention. FP32 Q/K/V, FP32 softmax state/output, both dot products explicitly
  use `input_precision="ieee"`. It supports causal prefill, one-token KV decode,
  and offset multi-token cache chunks, with dense equal-length batches.
- `efficient`: forces PyTorch `SDPBackend.EFFICIENT_ATTENTION`, without fallback.
  It accepts FP32 tensors but the installed SM80+ source selects CUTLASS
  `OpMultiplyAddFastF32` Tensor Core arithmetic. Do not call this an IEEE-FP32
  implementation just because its inputs/outputs are FP32.

```python
model = model.float().cuda().eval().set_attention_backend('tiled_fp32')
with torch.inference_mode():
    output = model.generate(ids.cuda(), 64, use_cache=True)
```

The custom kernel is a small FlashAttention-algorithm prototype, **not** a port of
PyTorch's optimized FlashAttention CUDA backend. Triton is required only when this
backend is selected. Supported head widths are 1..128; the measured GPT-2 head
width is 64. It has no backward or dropout implementation and raises on autograd
inputs; it never silently substitutes an eager backward pass. CPU execution and
non-FP32 inputs are rejected. GPU FP32 exponential approximations and reduction
order mean FP32 outputs need not be bitwise identical to eager.

## Measured experiment

A100 was requested using the authorized `yli15` account. The first request
initially queued, then briefly received an allocation during cancellation without
reaching the experiment. A short follow-up remained pending with an estimated
wait of roughly two hours and was canceled while pending. The actual experiment
ran on A30 via account `isp` / partition `academic`. No A100 results are claimed.
The optional `run_a100.sh` follow-up remains available for a later allocation.

```bash
sbatch experiments/fp32_attention/run_gpu.sh
/home/sashazli/torch_env/bin/python experiments/fp32_attention/build_report.py
```

The runner refuses to overwrite an existing `results/report.json`. Use a new output
location for a new run. Existing BF16 FlashAttention and earlier FP32 KV-cache
artifacts remain in their original directories.

Five variants run on the same GPU: FP32 eager, FP32 custom tiled, FP32 efficient
(FastF32 caveat), BF16 eager, BF16 FlashAttention. Same real pretrained/seed-42
LoRA checkpoints, prompts, shapes, cache semantics and 64-token generation budget.
Use these pairs:

| Comparison | Interpretation |
|---|---|
| FP32 eager / FP32 tiled | Custom algorithm/implementation effect with IEEE FP32 dots |
| FP32 eager / BF16 eager | Precision change in the original implementation |
| BF16 eager / BF16 flash | Optimized FA backend at fixed BF16 input precision |
| FP32 eager / BF16 flash | Combined precision + implementation change |

All variants are measured with cache off/on. Warmups: 3; measured repetitions: 7.
Cache order alternates. Variants run in recorded sequential blocks to allow one
resident model and direct memory measurement; clock/load drift between blocks is
a limitation. Timings include prompt prefill, selection/append and all generation
steps; exclude transfers, model loading, compilation, cleanup and warmups. CUDA
and synchronized wall measurements, allocation peaks, logical cache size, every
sample and short-term variability are saved.

Attention-only diagnostics use 20 operations captured in a CUDA graph, >=0.1s
warmup and 15 measured replays. This reduces host dispatch but does not enable
CUDA graphs in the real model. FP32 tiled is an untuned 16-query x 32-key kernel;
one-token decode still processes the 16-row query tile.

Training controls cover full/LoRA at batch 2, lengths 512/896, with FP32
parameters/AdamW states. FP32 eager/efficient receive FP32 activations; BF16
variants use BF16 autocast. Identical seeded batches across methods/backends,
dropout off, lr 1e-5, 3 warmups + 7 measured steps. There is no custom-tiled
training row because its backward is not implemented. These diagnostic updates
do not replace trained checkpoints or measure convergence.

Task quality checks every backend/cache combination on the same 120 held-out
sports questions for pretrained and LoRA. Logits compare to FP32 eager, and tokens
compare to FP32 eager uncached through EOS. No full-checkpoint quality conclusion
is inferred from these two models. Source snapshots, weight hashes and all raw
responses are retained.

## Evidence and artifacts

- [PDF report](results/FP32_ATTENTION_REPORT.pdf)
- [Markdown results](results/RESULTS.md), standalone [HTML](results/RESULTS.html)
- [Metrics CSV](results/METRICS.csv), [all task outputs](results/ALL_OUTPUTS.csv)
- [Presentation notes](results/PRESENTATION.md)
- `results/report.json`: raw data and environment/provenance
- `results/tiled-*.ptx`, cubins and SASS: compiled custom-kernel evidence
- `results/*trace.json`: actual CUDA kernel traces

GPU tests compare against float64 attention and check real fused-QKV strides,
odd lengths, cached chunks, active LoRA updates, context overflow, and errors.
Compiled PTX must contain FP32 FMA and must not contain TF32 or Tensor Core MMA.
A separate SASS audit checks actual GPU opcodes. Zero-input HFMA2.MMA instructions
used to materialize constants are documented separately from Tensor Core HMMA
instructions and FP32 Q/K/V arithmetic.

```bash
python -m unittest discover -s tests -p 'test_tiled_attention.py' -v
python -m unittest discover -s tests -p 'test_flash_attention.py' -v
```

References: [FlashAttention algorithm](https://arxiv.org/abs/2205.14135),
[Triton dot precision](https://triton-lang.org/main/python-api/generated/triton.language.dot.html),
[PyTorch installed-version GEMM selection](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm_kernel_utils.h).
