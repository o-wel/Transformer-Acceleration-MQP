# Research audit: a candidate optimization spanning KV cache, LoRA and attention

Research date: November 7, 2026. This is a targeted prior-art investigation and
an exploratory implementation study, not an exhaustive systematic review or
proof of first publication. See RESULTS.md for measured outcomes and
PROTOCOL.md for the design fixed before confirmation.

## What the repository establishes

`mingpt/model.py` adapts fused QKV, attention output, and both MLP projections
in every block. The adapter is `linear(x) + scale * B(A(x))`. Cached attention
concatenates historical and new K/V with two `torch.cat` calls per layer per
decode step. Backend selection already handles the important right-aligned
causal-mask distinction between prefill and cached decoding. Generation uses
the last-position LM head, retains the full context, and rebuilds the cache
when learned absolute positions roll over.

The completed `paper_hyperparameters/runs/study-20261105-v2/RESULTS.md` reports
1.0288x combined held-out speedup from parameter tuning. Its scope explicitly
excludes a production FlashAttention comparison. This investigation changes
the implementation boundary; its native-FP16 serving measurements must not be
pooled with that earlier study's autocast measurements. Existing data,
checkpoints, running configuration and production model sources are preserved.

## Literature search and inclusion rules

Searches used the web search service, followed by primary papers, author
repositories and official implementation documentation. Terms included:

* `LoRA KV cache FlashAttention optimization low rank attention inference`
* `LoRA shared prefix KV cache reuse FlashAttention activated LoRA`
* `LoRA attention epilogue fusion inference`
* `LoRA KV fused append`
* `FlashAttention LoRA rank kernel fusion`
* `LoRA decode CUDA graphs fused cache`
* `LoRA expansion attention fuse decode kernel`
* `LoRA QKV attention single kernel`
* `LoRA cache append`
* `LoRA current-token attention fusion`
* `LoRA up-projection attention kernel`
* `LoRA cache prologue attention`

Citation follow-up included the [LRAgent author repository](https://github.com/hjeon2k/LRAgent).

Include work available by the research date that intersects two or more of
adapter execution, KV storage/reuse, and attention/kernel fusion; include the
original component papers and selection-bias methodology. Read closest
matches directly. Search snippets identify leads; secondary summaries do not
establish priority or support performance claims. Repository/documentation
pages are mutable and their content date may be uncertain. Search indexing,
terminology and inaccessible source code limit recall; failure to find the
exact implementation is not evidence that it is new. No cross-paper speedup
numbers are used to estimate the candidate's speedup.

## Prior-art matrix

| Primary source | Relevant established idea | Implication for this project |
|---|---|---|
| [Hu et al., LoRA](https://arxiv.org/abs/2106.09685), 2021 | Frozen base plus low-rank adaptation; weights can be merged for inference | Merged serving is a necessary strong control; adding an adapter is not inherently an inference optimization |
| [Dao et al., FlashAttention](https://arxiv.org/abs/2205.14135), 2022; [FlashAttention-2](https://arxiv.org/abs/2307.08691), 2023 | IO-aware exact attention; improved work partitioning | Streaming softmax and avoiding the full score matrix are prior art |
| [Punica](https://arxiv.org/abs/2310.18547), 2023 | Efficient batched serving of distinct LoRA adapters | Efficient unmerged adapter execution is prior art |
| [S-LoRA](https://arxiv.org/abs/2311.03285), 2023/2024 | Unified paging of adapters and KV caches plus specialized serving kernels | Joint adapter/cache memory management is prior art |
| [Activated LoRA](https://arxiv.org/abs/2504.12397), 2025; [multi-adapter aLoRA serving](https://arxiv.org/abs/2512.17910), 2025 | Activation boundaries permit reuse of preceding base-model cache | Adapter-aware prefix reuse is prior art and has semantic/training requirements |
| [ELoRA](https://arxiv.org/abs/2505.03756), 2025 | Coordinated LoRA and KV-cache management | A shared memory pool or cache-aware scheduler alone is not a new contribution |
| [LRAgent v2](https://arxiv.org/html/2602.01053v2), May 31, 2026 | Base/low-rank cache sharing and Flash-LoRA-Attention avoid expanding a historical low-rank value cache | The broad three-way idea is already published. Its cross-agent sharing assumptions differ from this candidate's fully adapter-specific history |
| [EpiLoRA](https://arxiv.org/abs/2611.03899), November 2, 2026 preprint | Schedule LoRA shrink before a low-precision GEMM and expand in its epilogue | Generic LoRA expansion fusion is already proposed; distinguish the attention boundary and avoid a broad priority claim |
| [Official FlashAttention implementation](https://github.com/Dao-AILab/flash-attention#how-to-use-flashattention) | `flash_attn_with_kvcache` updates K/V in place and performs attention | Static cache and fused append/attention are production features, not discoveries here |
| [Modular MAX kernel API](https://max.modular.com/stable/api/python/nn.kernels/) | Official documentation's search-indexed description says QKV LoRA expansion writes K/V directly into the paged cache and returns Q | A close implementation precedent for expansion/cache fusion. Direct page retrieval failed during the audit; integration with attention itself remains unresolved |
| [AWS Neuron MLA attention block](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/library/api/mla-attention-block-gather-decode.html) | Combines MLA QKV preparation and decode attention in one kernel | Projection/attention fusion has broader architectural precedent; MLA architectural low rank is distinct from this GPT-2 adapter |

The LRAgent overlap is decisive: **do not call combining LoRA, KV cache and
FlashAttention itself novel.** The candidate's narrower distinction is
current-token QKV adapter expansion fused with ordinary adapter-specific KV
append and streaming attention. Establishing a publishable distinction still
requires deeper code comparison, related-work citation tracing and comparison
with production serving kernels.

## Candidate: expand the current token inside the decode attention kernel

For one token and one layer, let `u = x W0^T + b`, `z = x A^T`, and
`delta = s z B^T`, where the rows of B are grouped into Q, K and V slices.
Ordinary execution materializes the full expanded update, scales/adds it to
u, splits Q/K/V, appends K/V, and launches attention. The prototype leaves
the base projection and shrink projection outside the kernel, then performs:

1. Load this token's base Q/K/V slices, z and the corresponding B rows.
2. Expand the three low-rank corrections and add them in registers.
3. Store adapted current K/V into the next positions of preallocated buffers.
4. Initialize the softmax accumulator from the current token in registers.
5. Stream historical fully adapted K/V in tiles, updating online softmax.
6. Write the attention output, then run the ordinary output projection.

For a score tile with maximum `mt`, sum `lt` and weighted value sum `at`,
merge it with prior state `(m,l,a)` using
`mnew=max(m,mt)`, `lnew=exp(m-mnew)l+exp(mt-mnew)lt`, and
`anew=exp(m-mnew)a+exp(mt-mnew)at`. The final output is `a/l`.
This is the standard online-softmax identity. No attention probability matrix
is written to global memory. Each batch/head program owns its cache row;
the current token is consumed from registers, avoiding a cross-program
read-after-write dependency.

Why the interaction matters: the KV cache reduces each decode query to one
token, making small adapter operations and launch overhead visible. The
adapter's low-rank structure lets the attention program expand only its own
head slices. Streaming attention provides a place to consume those corrected
values immediately. A prefill matrix has different parallelism and reuse;
the prototype deliberately uses production PyTorch FlashAttention for that
phase. Historical adapter corrections are never recomputed.

This transformation is algebraically exact in real arithmetic. Floating-point
reductions differ; the kernel preserves separate FP16 expansion, scale and
add rounding boundaries and disables contraction. Actual numerical errors
are measured. It does not infer that two adapters share hidden states merely
because they share base weights. With LoRA throughout the network, previous
layers alter hidden states; `x_adapter W0` generally differs from
`x_base W0`. Reusing base-model history in ordinary LoRA would therefore need
additional assumptions, approximation, or an activation-aware training design.

## Falsifiable mechanism and expected limits

Relative to naive dynamic caching, static buffers eliminate copying the
historical cache on every token. That is ordinary cache engineering and must
not be attributed to LoRA fusion. Relative to static streaming attention,
the candidate removes the QKV up-projection launch and intermediate
scale/add operations. Relative to a standalone fused expansion kernel, it
removes one more launch and the materialized adapted-QKV boundary. These are
different contrasts, all retained.

Per-token attention still reads O(Td) historical elements per head; this does
not change attention's asymptotic cost or compress the cache. Static buffers
reserve the whole context and can cost more memory on short requests.
The prototype uses one program per batch/head and no split-K reduction, so
larger contexts or different hardware may favor production FlashAttention.
Higher ranks increase register pressure and repeated per-head reads. The
three other LoRA projections per block remain unfused and may dominate.
CUDA graphs can reduce launch overhead and could shrink the observed gain.

If merging is allowed, the merged controls may be faster: they remove all
adapter branches. If the candidate only wins with unmerged adapters, the
potential application is serving that retains separate adapters. This pilot
does not implement multi-tenant batching, adapter switching, heterogeneous
ranks, graph capture, concurrent requests, chunked cached prefill, training
gradients or rollover. Its patch is an isolated research runtime, not a
production-ready replacement for `GPT.generate`.

## Methodology and what would justify stronger claims

The protocol declares workloads, tolerances, controls, ordering, estimands
and uncertainty before confirmation. The initial pilot is exploratory;
its partial results motivated an additional disconfirming control, explicitly
documented as protocol v2. None of its timing samples enter confirmation.
This separation addresses the model-selection problem discussed by
[Cawley and Talbot](https://jmlr.org/papers/v11/cawley10a.html).

Fresh confirmation allocations use workload shapes not timed in discovery,
paired prompts and randomized method order. Source/checkpoint hashes,
raw timings, token sequences, numerical failures and model invariance checks
are preserved. Bootstrap allocation/block clusters rather than treating
every repeated timer as an independent GPU experiment. End-to-end ratios are
measured directly. Report regressions and merged-model comparisons.

The limited model-logit checks are implementation evidence, not a claim of
unchanged task quality. Before a paper claims quality preservation, evaluate
fresh task instances with a predefined noninferiority margin, token NLL and
answer-level scores clustered by match, keeping the same trained weights.
Before a paper claims systems novelty or superior serving, compare against
official fused-KV FlashAttention, an ordinary fused LoRA serving implementation
and CUDA graphs; test additional ranks, contexts, batches, GPUs and model
families. Include prefill/decode breakdowns, throughput under concurrent
adapters, memory and setup/amortization cost. A winning limited pilot is a
reason to continue that work, not a substitute for it.
