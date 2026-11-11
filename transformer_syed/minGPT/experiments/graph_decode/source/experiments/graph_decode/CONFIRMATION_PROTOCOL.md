# Frozen confirmation: larger gains from graph-compatible joint decoding

Date: 2026-11-07. The discovery pilot suggested approximately fourfold warm
request speedups on the first two batch-1 shapes. Those observations select
the existing graph_fused candidate, not a new kernel or a timing threshold.
No pilot sample enters confirmation. runtime.py and its imported joint_decode
kernel are unchanged from the successful pilot; all source hashes are frozen.

## Estimands and controls

Primary: end-to-end wall-latency ratio `previous_fusion / graph_fused`, equally
weighting eight new workload shapes and three fresh A30 allocations. A larger
improvement is operationally defined as a primary 95% interval wholly above
2x. Report the original dynamic-Flash / graph_fused ratio separately.

All modes retain the same rank-4/alpha-8 trained adapter and native-FP16 model
weights. Modes: original dynamic cache + forced FlashAttention; previous
Python joint fusion; new eager fixed-buffer fused runtime; graph replay with
ordinary LoRA expansion; graph replay with joint LoRA expansion; merged graph
runtime. Both model variants and all runtime buffers remain resident throughout
each comparison. Merged graph is the strong serving control, not omitted from
the claim because it may be faster. Adapter changes are unsupported within a
runner; any adapter change requires cache invalidation and runner recreation.

Important contrasts: eager_fused / graph_fused isolates graph replay using
identical runtime mathematics; graph_plain / graph_fused isolates LoRA QKV
fusion after graphing; original / previous_fusion checks the earlier result
in the current allocations. No products of isolated speedups are reported.
Other contrasts and workload intervals are secondary/descriptive.

## Workloads, pairing and cost

Three separately allocated A30 jobs; repeat physical devices are allowed and
identified by UUID. Batch {1,4}, prompt {128,512}, output {32,128}: eight shapes
absent from this discovery pilot. Three warmup blocks excluded; ten measured
blocks per shape. Methods are shuffled within each block, with the same fresh
synthetic prompt in each mode. Fixed output counts and greedy softmax/topk;
no EOS timing shortcut. Prompt plus output fits the 1024 learned positions.

Record synchronized host wall time and CUDA-event time, raw output IDs,
incremental peak allocated memory, total resident memory, model/source hashes,
GPU UUID/node/versions and failures. Request timing includes prefill, initial
K/V buffer copy/reset, all decode replays, and final output concatenation.
No persistent prompt-cache reuse across requests. Model load, graph/runtime
construction, compilation, host IO, hashing, mode installation and profiling
are outside request timing and accounted separately.

Time each runtime's complete construction, including buffer allocation,
warmup and graph capture, with synchronization. Compilation may hit the on-disk
Triton cache populated in discovery; call these observed setup costs, not
clean-machine cold-start measurements. Report `(setup_ms + N*request_ms)/N`
for N={1,10,100}, and break-even requests against previous_fusion and
eager_fused when possible. No assumption of infinite setup amortization.
Resident graph/cache memory is reported separately from incremental request
peak; zero incremental memory does not imply zero runtime footprint.

## Gates and analysis

Run the pilot's teacher-forced model gates at batches {1,4}, prompts {37,896},
prefill plus three subsequent tokens: atol=.1, rtol=.02 versus dynamic Flash.
Require bitwise equality of eager and graph logits for identical fused/plain
kernels. Check before/after weights exactly. Record original-path argmax
differences and generated raw sequence differences. These numerical gates do
not themselves establish answer-level equivalence.

Analyze paired log ratios. Bootstrap three allocations, then paired blocks
within sampled allocation/workload, for 5000 draws with fixed analysis seed
20261108. Workload shapes retain equal fixed weight. With only three
allocations and one GPU class, uncertainty does not imply broad hardware
generalization. Report all workloads and regressions. Never retune on
confirmation outcomes or loosen numerical gates in response to them.

## Fresh task and termination checks

Generate 120 new synthetic matches with seed 20261108, original training team
names and the existing question templates: 360 questions. Exclude quadruples
from original sports splits, the completed hyperparameter study, and the
previous joint_decode fresh set. Save and hash inputs before outputs.
Compare original, previous_fusion and graph_fused using the same native-FP16
adapter; 64-token greedy answers, trim EOS and next User turn. Retain raw IDs,
decoded answers, exact reference and field scores. Compute reference-answer
NLL including EOS with cached teacher forcing. The NLL equivalence gate is
absolute difference <=.01 nats/token relative to original. Accuracy changes
and match-cluster bootstrap intervals are descriptive; no post-hoc margin.
If all decoded answers agree, report the one-sided 95% match-discordance upper
bound rather than interpreting a zero-width bootstrap as population equality.

Exercise graph_fused at past lengths 0 and 1023, and requests ending at the
context boundary. Verify stale cache from a previous prompt cannot affect a
new prompt. Requests requiring rollover are rejected explicitly. A new
batch size requires its own runner/graph; no unsupported batch is padded
silently. These checks happen before confirmation timing.

## Claim boundaries

This combines established graph replay with the earlier experimental
LoRA-QKV/cache/online-attention fusion. Graph-compatible stable-address
caches and LoRA graphs are production precedents. A large speedup over this
repository's Python loop is not evidence of a new CUDA graph algorithm or
state-of-the-art serving performance. Forced PyTorch FlashAttention handles
prefill; decode is a custom FlashAttention-style kernel, not official FA1/FA2.
Official fused-KV attention, additional ranks/models/GPUs and multi-adapter
serving remain external-validity work. No accuracy improvement is predicted
from a mathematically equivalent inference implementation.
