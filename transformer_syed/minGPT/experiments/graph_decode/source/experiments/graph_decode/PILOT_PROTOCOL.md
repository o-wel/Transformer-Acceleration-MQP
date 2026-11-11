# Larger-speedup discovery pilot: CUDA graphs and joint decode fusion

Fixed before first measurement, 2026-11-07. Objective: determine whether
whole-step replay removes the dominant Python/dispatch cost and whether LoRA
QKV/attention fusion still adds value after replay. This is exploratory, not
a new-algorithm claim. Earlier experiments and their artifacts stay intact.

Reuse the same rank-4/alpha-8 trained adapter and native-FP16 GPT-2 weights as
joint_decode. Fixed full-context KV cache, learned positions, greedy softmax /
topk selection and forced PyTorch FlashAttention prefill. Decode reuses the
same tiled online-softmax attention kernel. Position, input token, write index
and KV buffers reside at stable device addresses; a graph updates positions
and token state without reading them on the CPU. No static all-context mask,
truncation, precision change or adapter change. Capacity 1024; no rollover.

Controls: original dynamic-Flash path; earlier Python fusion; a new fixed
buffer/device-position runtime with ordinary LoRA expansion, with and without
graph replay; the same with fused expansion, with and without replay; and
merged-adapter graph replay. All methods share the same resident model set.
Use batches {1,4}, prompts {32,896}, output 64, three excluded warmups and
seven paired blocks with randomized method order and fresh identical prompts.
Request timing includes prefill, cache reset/copy, generation and final token
accumulation; setup/capture/model load/host IO are excluded and reported
separately. Record synchronized wall and CUDA-event latency, peak allocated
memory, raw output IDs, setup costs, versions, weights/source hashes and Slurm
metadata. Report warm speedups and graph setup amortization separately.

Numerical gates: compare each runtime's prefill and three teacher-forced decode
logits with the original dynamic-Flash path at batches {1,4}, prompts {37,896};
atol=.1, rtol=.02, matching the previous study. Exact equality is required
between eager and graph variants using the same fused or ordinary kernel.
Greedy token divergence from the original path is recorded; downstream fresh
task quality must be checked before a quality-preserving claim. Failures are
preserved and never resolved by loosening a threshold after inspecting it.

No result from this pilot is confirmation evidence. A successful pilot must
be followed by a frozen confirmation protocol, fresh workload shapes, new
requests and multiple fresh allocations, plus capture-amortization and
task-quality checks. Treat generic graph gains separately from joint-fusion
gains, and benchmark merged graph serving as the strong practical control.
