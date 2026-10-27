# Reading the attention comparisons

All rows use the same minGPT PyTorch model. Only the attention implementation and
explicitly labeled precision change; none is a different language-model architecture.

| Name | Implementation | Precision and role |
|---|---|---|
| Eager | Original QK matmul, causal mask, softmax, and probability-times-V operations | Main no-FA baseline, tested independently at FP32 and FP16 |
| Tiled FP32 | Our Triton implementation of tiled attention with online softmax | FP32 inputs/state/output and IEEE FP32 dot products; custom implementation, not the official FA CUDA backend |
| Flash | PyTorch SDPA with FLASH_ATTENTION forced, no fallback | FP16 attention in this experiment; optimized fused CUDA implementation |
| Math | PyTorch SDPA's general math backend | Numerical reference; FP16 inputs normally use FP32 intermediates, so not interchangeable with FP16 eager |
| Efficient | PyTorch SDPA's EFFICIENT_ATTENTION backend | Separate backend; prior FP32 tests selected FastF32 Tensor Core arithmetic, not an IEEE-dot baseline |

Eager does not mean CPU: all measured variants execute on a CUDA GPU. Tiled and
Flash compute the same mathematical attention using tiled execution; they differ
in implementation, hardware instructions, optimization maturity, and supported
precisions. Floating-point operation order can change results.

## Comparisons to report

1. FP32 eager versus FP32 tiled: custom implementation at fixed precision.
2. FP16 eager versus FP16 FlashAttention: optimized FlashAttention at fixed input
   precision and the same full-model precision policy.
3. FP32 eager versus FP16 eager: precision-policy effect with the baseline backend
   held fixed.
4. FP32 eager versus FP16 FlashAttention: combined practical precision/backend
   change; do not label the whole ratio a FlashAttention-only speedup.
5. FP32 tiled versus FP16 FlashAttention: practical precision/implementation tradeoff,
   not an isolated test of either algorithm or arithmetic precision.

For inference, distinguish FP32 weights/activations from native FP16 weights/
activations. For training, keep master parameters and AdamW states FP32 for ALL
rows; FP16 rows use identical autocast and GradScaler policies. Compare losses,
finite gradients, successful optimizer steps, memory, and time separately.
Compilation, warmup and profiling must be excluded from timings. Cache on/off
comparisons hold precision/backend fixed. Quality checks use identical held-out
questions and preserve raw answers. Prior BF16 reports remain historical BF16
measurements; they must not be renamed FP16.
