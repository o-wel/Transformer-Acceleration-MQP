# Transformer hyperparameter study

**Complete:** final deliverables are [PDF](run_fp32/FINAL_REPORT.pdf), [standalone HTML](run_fp32/FINAL_REPORT.html), and [editable Markdown](run_fp32/FINAL_REPORT.md). The study completed 127 training trials and 1,440 serving cells. See STATUS.md for verified completion.

Primary goal: identify the best observed LoRA configurations and the quality/cost tradeoff, then explain how LoRA, exact tiled attention, and KV caching interact. Existing experiments remain intact. All new artifacts live here.

Run `bash submit.sh` to launch the dependency-controlled GPU pipeline. It runs at most two jobs concurrently. A failed prerequisite prevents downstream jobs from executing. `run_fp32/jobs.tsv` records submission IDs; `run_fp32/logs/` holds progress and failures. A second submission requires a new output directory. Partial trials restart from scratch; completed trials are reused only with matching source and dataset hashes.

## Search and selection

| Stage | Dimensions | Budget and selection |
|---|---|---|
| Core grid | rank 2/4/8/16/32; alpha/r 1/2/4; LR 1e-4/3e-4/1e-3; attention-only/all transformer projections | 90 configurations; 1,200 examples each; promote best 9 |
| Refinement | Best 9 core configurations | 9,600 examples each, from original pretrained weights |
| Regularization | LoRA input dropout 0/.05/.1 × AdamW decay 0/.01/.1 | 9 configurations around best refined candidate; 9,600 examples each |
| Batch and accumulation | Effective batch 4/8/16/32; microbatch 1/2/4/8 at fixed effective batch 8 | Equal 9,600 examples; report update count, time, memory and validation loss |
| Sequence length | 128/256/512 | Equal 9,600 examples; fixed validation encoding length 128 |
| Confirmation | Best 3 unique full-budget configurations × seeds 42/43/44 | Winner by mean validation NLL; report seed variability |
| Full fine-tuning control | LR 1e-4/3e-4/1e-3; selected LR confirmed on seeds 43/44 | Same example budget; baseline gets less tuning compute, disclosed explicitly |
| Combined systems | full/LoRA × eager/FA1-derived × cache off/on | Additional hybrid backend, dynamic/static cache, 8 tile/warp configurations, batch 1/4/8, prompt 32/128/512/896, output 32/64 |

Alpha is `rank * scale`. Targets include the fused Q/K/V and attention output projections; `all` also includes both MLP projections. Attention-only is **not** Q/V-only. Adapter dropout applies only before A, with the base projection unchanged. All base parameters are frozen for LoRA; only adapter A/B weights train.

The core grid is exhaustive over its four dimensions. The whole study uses staged selection, not the Cartesian product of every dimension. Short-budget screening can discard late-improving candidates. Results support “best observed within this search,” not a universal optimum.

## Fairness and measurement

All primary trials use FP32, eager attention for training, disabled TF32, gradient clipping 1, fixed LR, zero pretrained dropout, and the repository's train/validation/test split. Identical seeds produce an identical flattened sample stream, including across batch sizes. Microbatch losses are weighted by the number of supervised tokens so accumulation matches the effective-batch loss. Equal examples do not mean equal optimizer updates; both are recorded. Learning-rate schedules, rsLoRA, per-layer rank allocation, quantization, and lossy cache compression remain follow-up dimensions rather than silently mixed interventions.

The existing examples are shorter than 128 tokens. Raising padded training length measures extra computation and padding cost, **not** learning from longer real contexts. Synthetic inference prompts provide a separate workload-length experiment. GPT-2's 1,024-token position limit is respected. Generation always emits exactly the configured number of tokens in timing tests; quality generation decodes up to EOS.

Validation loss selects each checkpoint and candidate. No test scores are used for promotion. The final configuration is persisted before test evaluation. Test quality is measured on all three trained seeds using the reference eager/cached policy; each systems cell separately checks logit tolerances and complete greedy-token parity. Training data, sources, sampled indices, selected checkpoint, GPU, software, timing samples, and memory are recorded. Hashes detect changes but do not replace a reproducible environment lock. Only the frozen winner and selected full control receive test evaluation.

The custom kernel is a configurable copy of the existing FA1-derived Algorithm 1/4 prototype: K/V tiles outer, queries inner, identical FP32 arithmetic, no SDPA. It is **not** the original optimized FA1 CUDA implementation, and results must not be generalized to FA2/3/4. The hybrid policy uses tiled prefill and eager single-token attention. Static caching allocates full context capacity, appends in place, and supports one serial request stream per model. Its retained allocation is reported separately from logical used-cache bytes; dynamic caches use concatenation. Contiguous copies inside the prototype still count in end-to-end timing.

Each system case gets two warmups and seven measured repetitions, with prefill, decode, wall time, allocated/reserved memory, logical KV bytes, static capacity, and token throughput. Kernel JIT is excluded through warmup. Results are grouped by GPU; cross-GPU speed rankings are not valid. Cache mode and tile settings are deterministic experimental factors; this initial runner uses a fixed case order, so thermal/order effects remain a limitation. Cache quantization, eviction/window length, and paging would change precision or implementation and are not claimed as implemented.

Reports: `run_fp32/TRIALS.csv`, `run_fp32/RESULTS.md`, `run_fp32/quality_cost.png`, `run_fp32/selection.json`, `run_fp32/systems-{0,1,2}.json`, and per-method test outputs. Regenerate available summaries with `python study.py report`. Until stages complete, the report is explicitly partial. Use `METHODS.md` for the framework diagram and `LITERATURE.md` for related work and interpretation.

`analyze.py` generates `ANALYSIS.md`, `core_grid_heatmaps.png`, `PARETO.csv`, `SYSTEMS.csv`, `INTERACTIONS.csv`, and `combined_latency.png`; the submission pipeline runs it after the study finishes. It can also be run on partial results with `python analyze.py --out run_fp32`. Plot dependencies are installed locally under `.plot_deps` and exposed through ignored symlinks; they do not alter the shared Python environment or its NumPy. On a fresh checkout, activate the GPU Python environment and run `bash setup_plots.sh` before submitting. Versions are pinned in `plot-requirements.txt`.

FP16 is optional and deliberately excluded from the initial search. A later matched study should use the same custom algorithm in both precisions or the original `flash-attn` package, never the PyTorch FlashAttention backend, with separate labels and equal tuning budgets.

Rebuild the final report with `python build_final_report.py` after installing `report-requirements.txt` under `.plot_deps` (alongside the plotting dependencies). The builder verifies complete trial counts, frozen base weights, serving parity, repetition counts and three-seed test outputs before exporting.
