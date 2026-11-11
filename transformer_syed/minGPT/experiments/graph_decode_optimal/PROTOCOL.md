# Best-tested hyperparameter report: frozen protocol, November 10, 2026

Purpose: produce a report using the previously selected training settings and
a new finite runtime search appropriate to the graph-compatible decoder.
The earlier rank-4/alpha-8 graph experiment remains historical evidence and
is not pooled with this study. No global-optimality claim is permitted.

## Inherited training/cache choice

Use study-20261105-v2's frozen selection: rank 2, alpha 8, learning rate .001,
full prefill (chunk 1024), and a full 1024-token context/cache. That study
searched 45 LoRA settings with training seeds 11/22/33; selected the fastest
training setting within .01 validation NLL of the quality-best setting; and
independently retrained seeds 101/202/303/404/505. Its quality-best setting
(rank32/alpha64/LR.0003) is distinct from its operational cost/quality choice.
Report this distinction. Load existing completed checkpoint files only after
verifying config, training metadata and SHA256. No retraining in this study.

Its FA1-derived prototype launch choice (query tile64/key tile128/warps4)
cannot be called optimal for this different decode kernel. Query length is
one here. Official PyTorch FlashAttention performs prefill; custom tiled
online-softmax attention performs decode. All serving uses native FP16;
the old study used FP32 weights plus autocast. Report that policy difference.

## New runtime search

Search key/history tile BN in {32,64,128,256} crossed with num_warps in
{1,2,4,8}: 16 configurations. Head width64, capacity1024, scalealpha/rank4,
precision, batch, prompt and output lengths are not optimized. No eviction,
quantization, context reduction, skipped layers or approximate cache sharing.

Three fresh A30 search allocations use the selected setting's existing
training-search checkpoints for seeds 11/22/33. Workloads: batch{1,4},
prompt{32,896}, output32. Per workload/config: two excluded warmups and five
paired measured blocks with identical prompts and randomized config order.
Warm every configuration before timing. Construction, graph capture,
compilation and IO excluded from warm latency and recorded separately.
All 16 configs/runners remain resident for equal comparison state.

Numerical gate for every config: prefill and three teacher-forced logits
versus the same model under dynamic-cache forced FlashAttention, batches1/4
and prompts37/896, atol=.1 and rtol=.02. Disqualify compile/resource or
numerical failures; do not loosen gates. Record all errors/argmax differences.
Selection minimizes the geometric mean of per-allocation/workload median
wall latency across all three search allocations. Tie follows grid order.
The original BN128/warps4 is retained as a control regardless of selection.
Write/hash selection.json before confirmation runs. No test-driven tuning.

## Independent confirmation

Five fresh A30 allocations use the five independent confirmation-training
seeds101/202/303/404/505, one allocation per seed. These composite seed/
allocation units do not identify model-seed and hardware-allocation variance
separately. Report the limitation. Repeat devices are allowed and retained
by UUID. All model variants/runners remain resident throughout comparisons.

Eight held-out shapes: batch{1,4}, prompt{128,512}, output{32,128}. Each uses
three excluded warmup blocks and ten measured paired blocks with randomized
mode order, identical fresh synthetic prompts, and fixed output counts.
Modes: original dynamic cache + forced Flash; earlier Python joint fusion;
eager fixed-buffer joint decoder at selected tiles; ordinary-LoRA graph
decoder at selected tiles; fused graph decoder at original128/4; fused graph
decoder at selected tiles; merged-LoRA graph decoder at selected tiles.
Identical weights within all runtime contrasts; merged controls formed from
the same FP16-serving weights via FP32 merge then rounded once to FP16.

Primary endpoint: original/selected fused-graph wall-latency ratio on held-out
shapes. Important secondary contrasts: previous fusion/selected graph;
eager/selected graph (replay); ordinary graph/fused graph (fusion); original
tile graph/selected tile graph (tuning); merged graph/fused graph. Ratios are
measured directly, not products. Report every workload and any regressions.

Record synchronized wall and CUDA-event time, output IDs, incremental request
peak memory, graph construction and resident memory, versions/telemetry,
source/selection/checkpoint hashes, and exact before/after parameter hashes.
Request time includes prefill/cache copy, decode and final output assembly;
excludes model load, setup, host IO and mode installation. Model loading is
common excluded deployment cost. Observed setup may hit compiled-kernel disk
cache and must not be labeled a clean-machine cold start. Report amortization
for1/10/100 requests and capture break-even. No prompt cache reused across
requests. Fixed GPU clocks and comprehensive co-tenant isolation not imposed.

Before timing, recheck selected/plain/merged runtime logits under the same
tolerances, eager/graph equality for the same fused kernel, first/last cache
positions, stale-cache isolation and rejected overflow/batch mismatch.

Analyze paired log latency ratios, equal workload weight. 5000 bootstrap draws
resample the five composite seed/allocation units and then paired blocks
within each workload. Seed20261110. Shapes fixed, not resampled. Confidence
intervals are conditional on this model/task/GPU/workload family; secondary
intervals descriptive, without multiple-testing adjustment.

## Fresh task quality

Before search, freeze120 new matches (360 questions), seed20261110, familiar
training team names/templates; exclude score quadruples from original sports,
the completed tuning study, and both prior joint/graph fresh sets. Same fresh
set across all five model seeds. No fresh output may select any setting.

Evaluate original, selected fused graph and merged graph using64-token greedy
generation, trim first EOS and next User turn; retain raw IDs/text/reference
and field scores. For efficient evaluation, batch up to4 prompts of exactly
the same token length, with identical grouping across modes; no padding mask
or truncated context. Separate request timing uses the declared shapes only.
Compute answer-token NLL including EOS with cached teacher forcing. Require
absolute NLL difference<=.01 from original for each seed; retain answer and
raw-sequence agreement separately. Exact/field accuracy differences and
crossed bootstrap over five seeds and120 match groups are descriptive.
Zero observed differences do not establish population equality. Do not
attribute accuracy gains from inherited checkpoint selection to graph replay.

## Report and reproducibility

Produce PDF, standalone HTML, Markdown, CSV/JSON summaries, figures, protocols,
source/data/hash manifest and raw evidence. Separate the earlier rank-4 graph
result, inherited hyperparameter study and this new matched confirmation.
Include limitations, production-control gaps and prior art. CUDA graphs,
LoRA serving graphs and streaming attention are established; neither the
large repository speedup nor finite-grid tuning establishes research novelty.
Preserve failed runs and all compute accounting. Report best-tested settings
under declared objectives, not universal/global optimality.
