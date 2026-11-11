# Research continuation: removing the larger decode bottleneck

Research date: November 7, 2026. This work follows the measured current-token
LoRA QKV/cache/attention fusion in `experiments/joint_decode`. The goal is a
substantially larger speedup while preserving the same trained model.

## Diagnosis and choice of next experiment

The first experiment improved the full request by 1.1245x and isolated LoRA
fusion by 1.0578x. Its stored four-token profiler traces show 1,816 kernels
for the original runtime versus 1,660 for joint fusion. Summing the union of
GPU kernel intervals gives 8.88 ms over a 58.71 ms original trace span and
11.91 ms over a 53.01 ms fused span. These are *instrumented trace* statistics,
not unprofiled utilization estimates: profiling itself changes dispatch cost.
They nevertheless show that accelerating only kernel arithmetic leaves large
gaps between many small operations.

This motivated testing whole-step replay before building a trained drafter.
CUDA graph replay can remove per-operation host dispatch while retaining the
same target model. A learned speculative drafter introduces extra training,
acceptance-rate and cache-consistency questions. It remains a later option,
not a speedup assumed in this experiment.

## Primary-source investigation and novelty boundary

Search terms included `speculative decoding LoRA draft adapter shared KV
cache`, `prompt lookup decoding speculative n gram exact verification`,
`LoRA speculative decoding FlashAttention self speculative adapter`, `LoRA
CUDA graph decode`, and `CUDA graphs inference launch overhead static cache`.
These are targeted searches and primary-source follow-up, not an exhaustive
systematic review. No speedups from different papers or models are pooled.

* [NVIDIA's CUDA graph introduction](https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/introduction.html)
  describes recording dependent operations and replaying the workflow to
  reduce CPU launch overhead. That mechanism is established prior art.
* [NVIDIA's PyTorch graph best practices](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/best-practices.html)
  recommend profiling small kernels/launch gaps and maintaining static graph
  inputs. This informs the experiment's diagnosis and stable-buffer design.
* [TensorRT-LLM's LoRA documentation](https://nvidia.github.io/TensorRT-LLM/1.4.0rc3/features/lora.html)
  describes graph-compatible LoRA execution metadata. LoRA graph serving is
  therefore also prior art; this is not a first-LoRA-graph claim.
* [PyGraph](https://arxiv.org/abs/2503.19779) studies compiler support for
  graph use in PyTorch. Capturing an inference loop is not by itself a new
  systems algorithm.
* [Leviathan et al., speculative decoding](https://arxiv.org/abs/2211.17192)
  establish draft-and-verify acceleration. The [prompt lookup author
  implementation](https://github.com/apoorvumang/prompt-lookup-decoding)
  establishes string matching as a proposal mechanism. Neither is claimed
  as a discovery here; neither is used in the measured graph candidate.
* The earlier [LRAgent](https://arxiv.org/abs/2602.01053) and
  [EpiLoRA](https://arxiv.org/abs/2611.03899) overlap findings still apply:
  neither a broad LoRA/KV/attention combination nor generic adapter fusion
  is new solely because this repository implements it.

The useful result here can be a large, reproducible minGPT implementation
improvement even if no publication-level novelty is established. A claim of
a new graph algorithm or state-of-the-art inference would not be supported.

## How the three methods interact in the implementation

A runner is tied to one model/adapter, batch, rank, scale, precision and
device. Its captured graph does not fix the current context length.
Each request owns preallocated K/V, next-token, output-index and learned-
position tensors. Prefill runs forced PyTorch FlashAttention and copies the
resulting adapted K/V into the reusable cache. It does not reuse previous
prompts' content or hidden states.

Each graph replay reads the device's current position, embeds its current
token at the original learned absolute position, and runs all GPT-2 layers.
In every attention layer, the fused kernel expands the current token's LoRA
Q/K/V corrections, appends adapted K/V and streams the valid historical
prefix using online softmax. A device-loaded loop bound means only the valid
prefix is processed, rather than attending to the full reserved capacity.
The softmax/topk sampler matches the existing generation policy. The graph
writes the next token, updates its input and increments the position/index.

KV caching supplies persistent history and reduces each new query to one
token. LoRA fusion avoids a separate QKV expansion intermediate. Streaming
attention consumes current corrected K/V directly. Stable memory and
device-resident positions allow that combination to replay as one host graph
launch per token, independent of growing sequence length. There is no CPU
read of the current decode position and no attention-matrix materialization.

The transformer/output/MLP mathematics is otherwise unchanged. In eval mode
all dropout calls are identities. Numerical comparison is still required
because this attention prototype has different floating-point reductions
from PyTorch FlashAttention. Exact equality of eager and graphed versions of
the *same* kernel is a separate, stronger implementation check.

This is a custom FlashAttention-style decode implementation and official
PyTorch FlashAttention prefill, not an official FA1/FA2 decode release.
Full context is retained; no eviction, quantization, approximate cross-model
cache reuse, skipped layers or speculative acceptance assumption is used.

## Why the comparisons matter

`original / graph_fused` measures the full repository improvement.
`previous_fusion / graph_fused` measures progress beyond the first candidate.
`eager_fused / graph_fused` isolates replay on the identical fixed-buffer
runtime. `graph_plain / graph_fused` tests whether LoRA fusion still matters
once host overhead is reduced. `merged_graph / graph_fused` reveals whether
keeping adapters separate is required for the candidate to be practical.
All contrasts use actual paired request measurements, not multiplied ratios.

A large replay gain with a smaller remaining fusion gain means the dominant
effect is established graph engineering. That distinction should remain
visible even if the combined headline improvement is attractive.

## Methodology and practical limits

The discovery pilot's two inspected batch-1 results motivated freezing the
candidate for confirmation. Pilot samples do not enter its estimates.
Confirmation uses three new allocations, eight new workload shapes, paired
fresh prompts, randomized mode order and a hierarchical paired bootstrap.
Only three allocations on one GPU class limit uncertainty inference. Fixed
clocks and complete co-tenant isolation are not imposed.

The protocol includes graph construction time, observed cache-backed JIT
setup, amortization at 1/10/100 requests, cache/graph resident memory, boundary
positions, stale-cache checks and rejected unsupported shapes. Fresh task
data are generated and hashed before outputs, with match-level clustering.
The one fixed adapter is held constant; this is an inference implementation
study, not a new accuracy/training result.

Graph construction is not a clean-machine cold-start measurement. Model
loading is a common excluded cost. Fixed-batch runners do not establish
continuous batching or concurrent serving. Adapter replacement requires a
new runner and fresh cache in this implementation. Context rollover is
rejected; full prompt-plus-output fits 1024 positions. Sampling other than
the original greedy policy is not implemented. Native-FP16 serving here must
not be conflated with the earlier paper study's FP32 weights plus autocast.

Before claiming a broadly new contribution, compare production fused-KV
FlashAttention, production LoRA graph engines and compiler-generated graphs;
test multiple ranks, models, GPU families and concurrent adapter workloads.
Those missing controls limit novelty and external validity even if this
controlled minGPT speedup is large.
