# Matched FP16 hyperparameter performance review

This study repeats the completed `../hyperparameter_search/` FP32 methodology on basketball and the existing real SMS spam task. A new matched FP32 spam sweep provides the second task's precision control. The old experiments are preserved.

The research message is integration of three established paradigms—LoRA, an FA1-derived attention prototype, and KV caching—and the importance of careful joint configuration. Best/worst differences are measured, not guaranteed. This is a finite staged search, not an exhaustive Cartesian search of all training and serving hyperparameters together.

**Precision:** FP16 CUDA autocast for all Linear projections, attention operands and outputs, LM-head logits, and dynamic/static KV storage. Master parameters, gradients, AdamW state, residual/normalization and stable reductions retain FP32 where required. It is mixed precision, not pure FP16 storage/arithmetic. TF32 is disabled. GradScaler starts at 128; skipped/nonfinite updates fail a trial. The legacy model API name `tiled_fp32` routes to the same locally copied configurable algorithm with asserted FP16 operands; it does not imply FP32 kernel inputs. No SDPA/FA2 substitution is used.

GPU preflight checks 32 custom forward/backward cases against FP64; real full/LoRA updates; every first-forward Linear output; and eager/custom/hybrid dynamic/static cache dtypes. Its numerical diagnostics use atol=.012/rtol=.04 (kernel) and atol=.1/rtol=.02 (cache logits). These are distinct from serving eligibility: the original FP32 atol=rtol=3e-4 logit gate and exact complete greedy-token parity remain unchanged. Failing cells retain all diagnostic timing samples as `parity_failed` and cannot support quality-preserving acceleration claims. Raw `status` fields distinguish numerical failures from OOM.

Search budgets, grids, sample stream, accumulation weighting, validation checkpoint opportunities, three confirmation seeds, full controls, systems workloads and 2 warmups/7 measurements match FP32. SMS retains its original split, 96-token content cap and ham/spam label-logit classifier; it receives the same staged 90-core-config search. Basketball/spam NLL are different objectives and must not be pooled. Existing held-out splits are reused as requested and are not fresh confirmation datasets.

Launch from the configured Torch environment:

```bash
python submit.py --out runs/NEW_ID --report-deps /path/to/compatible/report/libraries
python status.py runs/NEW_ID
```

Submission snapshots/hashes source and datasets. GPU jobs verify the entire snapshot before execution. Never edit a submitted snapshot; use a new study for amendments. `jobs.json` records each submission immediately. Do not blindly resubmit a partial submission. Each study is dependency gated; studies run serially and screen/serving arrays allow two concurrent GPU jobs. Each task repeats preflight before its expensive stages. All jobs use the existing `isp` account and academic A30 GPU partition. GPU-local temp and Triton cache paths use the study's writable directory.

Reports build automatically after successful jobs. Per task: `PERFORMANCE_REVIEW.md`, `.html`, `.pdf`, `TRIALS.csv`, `SYSTEMS.csv`, sensitivity figures, extreme configurations, frozen validation selections, three-seed held-out outputs and raw dtype audits/timings. The top-level `PERFORMANCE_REVIEW.md` and `PRECISION_PAIRS.csv/json` compare exact matched configurations and distinguish independently retuned winners. Reports say PARTIAL until required artifacts exist. A failed dependency prevents later jobs; inspect logs and use a new snapshot or explicit carefully scoped retry rather than treating a queued job as completed.

The initial independent GPU validation is retained under `validation/attempt01/`: job 2509470, NVIDIA A30, PyTorch 2.14.0+cu130, successful exit. All 32 kernel cases passed, both optimizers completed updates, and six backend/cache combinations had FP16 K/V. Diagnostic cache maximum absolute logit differences reached .15625, within the predeclared combined absolute/relative diagnostic tolerance; this does not establish the stricter serving parity gate.

Owen's source graphs are absent from this checkout. This report produces analogous sensitivity graphs from its own measurements and does not claim to have reviewed missing external figures.

A frozen report-only amendment in the active run’s `report_source/` corrects the inherited analysis failure labels, limits diagnostic PDF tables to representative cells, and adds audit/repetition checks before labeling a report complete. It changes no measurements, hyperparameters, numerical gates or selection rules. Supplementary report jobs regenerate the final deliverables after each original report job.
