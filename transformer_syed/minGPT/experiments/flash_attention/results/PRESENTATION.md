# FlashAttention: presentation notes

At batch 1 / prompt 512 / 64 generated tokens, forced FlashAttention changes pretrained uncached generation from 403.9 to 289.5 ms (1.40x). Flash plus cache takes 326.4 ms (1.24x versus eager without cache).
For the trained unmerged LoRA model, the corresponding flash-only and combined speedups are 1.19x and 1.18x.
At training batch 2 / length 512, FlashAttention speeds up full updates by 1.16x and LoRA updates by 1.16x. LoRA + flash versus full eager changes step latency by 1.28x and lowers peak allocation by 42.5%. These are execution measurements, not time-to-quality results.

After KV caching is already enabled, flash's measured request speedup ranges from 1.14x to 1.33x over the ten model/shape cases. Do not multiply standalone LoRA, flash and cache speedups. Read the combined measured configurations. Small batch-one decode can be dominated by projection, Python dispatch, sampling or cache-management costs; this is an inference from workload scaling, not an occupancy/bandwidth diagnosis.

Show generation-speed.png first, then attention-kernels.png to distinguish kernel and full-model gains. Use training-speed.png and gpu-memory.png for adaptation costs. The task-quality table discloses all BF16 output changes. Every comparison here uses the same precision within its phase; do not compare the older FP32 timing report directly. Flash was forced and verified in real CUDA traces. These are one-GPU measurements, not convergence or production-serving claims.
