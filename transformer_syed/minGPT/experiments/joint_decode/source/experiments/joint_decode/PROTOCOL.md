# Joint decode fusion: exploratory protocol v2, 2026-11-07

Amendment before confirmation: the first discovery allocation used v1 (six
modes); its sources/protocol are preserved in archive/v1. After inspecting the
two batch-1 pilot workloads, add a seventh, stronger control: standalone fused
LoRA expansion/scale/add followed by the same streaming cache/attention kernel
(`expand_stream`). No kernel tile, workload, checkpoint, primary contrast or
threshold changes. Version-1 samples are exploratory only and never pooled
with v2 confirmation. Record v2 source hashes identically across confirmation
allocations. Profile four modes outside timed intervals on confirmation 0.
This amendment adds a potentially disconfirming comparison rather than
selecting a faster candidate. The expand_stream/fused_stream contrast is a
required secondary check for any claim about fusion across the attention
boundary, beyond ordinary fused LoRA arithmetic.

Question: can fusing the current token's unmerged LoRA QKV expansion, in-place
KV append and streaming exact attention reduce GPT-2 request latency beyond
ordinary static caching and streaming attention alone?

Status: an exploratory systems candidate, not an established novelty claim.
No training, rank search, cache eviction, approximate sharing or precision
change between methods. No previous study's timing samples are pooled here.
The prototype modifies no production model file and is inference-only.

## Design fixed before the first GPU measurement

Use the existing trained rank-4/alpha-8 FP16-autocast sports adapter, with the
locally cached GPT-2 base. Load both into FP32 and then cast the same model to
FP16 for all modes. This is native FP16 serving, distinct from earlier studies'
FP32 weights plus autocast. Form the merged control in FP32 from those same
FP16-serving weights, then round once to FP16. No retraining.

Modes: unmerged dynamic cache + forced PyTorch FlashAttention; unmerged static
cache + forced FlashAttention; unmerged static cache + prototype streaming
decode; unmerged static cache + fused LoRA/streaming decode; merged static cache
+ forced FlashAttention; merged static cache + prototype streaming decode.
Seventh mode: unmerged standalone fused LoRA expansion + static streaming
attention. The streaming prototype uses a FlashAttention-style tiled online softmax but
is neither official FA1 nor FA2. All modes use forced PyTorch FlashAttention
for prefill. The installed production flash-attn package is absent; its fused
KV append API is an important missing control, not assumed inferior.

Discovery workloads: batch {1,4}, prompt {32,896}, output 32. Seven measured
 paired blocks after three excluded warmups; shuffle all modes per block.
Confirmation workloads: batch {1,4}, prompt {128,512}, output 64; ten measured
paired blocks after three warmups on each of three fresh GPU allocations.
Independent allocations need not imply distinct physical GPUs; retain node and
GPU UUID. No kernel/dispatch selection uses confirmation results. No rollout
crosses the learned 1024-position limit. Greedy decoding generates a fixed
number of tokens; different outputs cannot shorten a request.

Same resident model set throughout comparisons. Timed requests include cache
allocation, prefill, decode, sampling and output accumulation; exclude model
loading, compilation, host transfer, mode setup, hashing and telemetry.
Record synchronized host wall time, CUDA-event time, incremental peak allocated
memory and raw output IDs. Generate identical synthetic prompts per paired
block. CUDA event duration includes host dispatch gaps; it is not summed
kernel execution time. Compilation is warmed before timing. Thread count 1;
TF32 off; eval and inference_mode; zero dropout. No CUDA graphs in this pilot.

Numerical gates, before timing: compare the fused and unfused kernels against
FP32 PyTorch attention using FP16 inputs, ranks {2,4,16,32}, batches {1,4},
past lengths {0,1,31,895,1023}, head width 64. Tolerance atol=0.012, rtol=0.04.
For actual GPT-2, compare teacher-forced logits at prompt lengths {37,896}
and three cached steps, batch 1; atol=0.1, rtol=0.02. Report argmax disagreement,
maximum error and KL divergence rather than claiming bitwise equivalence.
All modes must pass; a gate failure blocks any quality-preserving claim.
No teacher-forced test can establish general downstream task noninferiority.

Store sources, protocol, dependency/model hashes, GPU/Slurm identity, failures
and raw paired samples. Source hashes are recorded before each run and must
match between discovery and confirmation. Fix bugs only in a documented new
version and rerun every affected gate and measurement. Preserve failed logs.

Primary confirmation contrast: static_stream latency / fused_stream latency,
equal weighting across the four held-out shapes and three allocations, using
the geometric mean of paired ratios. This isolates the incremental LoRA
fusion rather than attributing ordinary cache/attention improvements to it.
Secondary: dynamic_flash / fused_stream and static_flash / fused_stream.
Also report both merged controls; losing to merged serving limits the use case
to scenarios where adapters need to remain separate.

Uncertainty: resample allocations, then paired blocks within allocation and
shape, 5000 bootstrap draws; retain workload shapes as a fixed design. Three
allocations give limited uncertainty estimation and establish no hardware
population claim. A positive pilot result needs the primary interval above 1
and numerical gates passing; downstream quality remains unconfirmed. Report
every workload, including regressions. Never multiply isolated speedups.

## Novelty boundaries and disconfirmation

S-LoRA/Punica already optimize unmerged adapter serving; official
FlashAttention already fuses cache updates; LRAgent already combines low-rank
KV corrections with Flash-LoRA-Attention; EpiLoRA already fuses LoRA expansion
into GEMM epilogues. The candidate is the specific *current-token* QKV
expansion/cache-append/attention fusion with fully adapted historical K/V.
That implementation boundary is a hypothesis to investigate, not proof of a
new algorithm. Search terms and primary references are retained in RESEARCH.md.

Reject a strong contribution claim if ordinary static caching explains the
gain; the unfused streaming control is as fast; native fused-KV FlashAttention
or CUDA graphs erase it; numerical gates fail; or closer prior art implements
the same boundary. One model, one rank, short contexts and a custom kernel
cannot establish broadly superior serving or publication novelty.
