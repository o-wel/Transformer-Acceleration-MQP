# Precision results: paper readiness

The existing measurements support inclusion now, with bounded claims. No new GPU run is required to report training quality, FP32 SMS classification, measured precision sensitivity, completed ablations, or the FP16 numerical limitations.

## Gaps filled in this update

- `PRECISION_RESULTS.tex` supplies a paper section with explicit mixed-precision policy, three-seed LoRA quality and numerical acceptance tables. The section is embedded in `MQP_REPORT_UPDATED.tex`; upload that file as a self-contained source, or paste the smaller section into the current Overleaf project.
- Both methodology sources distinguish the primary FP32 study from the FP16 extension. If using `METHODOLOGY_SECTIONS.tex` separately, include `PRECISION_RESULTS.tex` too, so its cross-reference resolves.
- `build_precision_evidence.py` regenerates the section and CSV/JSON tables from raw artifacts using Python's standard library. It verifies record counts, three quality seeds, unique ablation state/seed pairs, seven timing repetitions, and acceptance status coverage, and records input SHA256 hashes. These checks are not an independent rerun of GPU correctness.
- The newer optimization matrix is complete in all four task/precision studies: 2,592 calibration cells, 2,232 confirmation cells and 93 natural-task quality records each. This closes the earlier missing best/poor seed replications and factorial coverage. The matrix is a separate experiment; its samples must not be pooled into the earlier search's timing estimates.
- FP32 SMS results exist both in the earlier two-rate baseline and the newer matched staged sweep. Use the matched sweep for FP16 comparisons. The actual test set is 771 messages, including 93 spam.

## Claims still requiring new evidence

1. **Output-preserving FP16 custom-attention acceleration.** All custom and hybrid cells fail the original matched study's strict serving gate. Diagnose discrepancies on fixed checkpoints and identical token inputs, recording attention outputs, layer outputs, logits, top-two margins, first greedy divergence and cache position handling. Compare eager and tiled softmax/scaling/reduction behavior, including probability downcasts. These are investigation targets, not a diagnosed defect. Preserve the existing snapshots; any implementation change requires a new named run, preflight, and rerun of affected serving and task-quality cells. Do not widen the historical gate retrospectively. A prospective approximate-quality claim needs a separately justified criterion and fresh confirmation.
2. **FP16/FP32 equivalence or noninferiority.** The reported winners have different learned weights and configurations. Three-seed descriptive summaries do not establish equivalence. Predeclare a task-specific acceptable quality difference, freeze selections, and evaluate paired predictions with uncertainty appropriate to messages or basketball matches. Reused test splits are not fresh confirmation. Fixed-configuration training pairs and fixed-weight casting experiments answer different questions.
3. **Broadly repeatable timing improvements.** Existing repetitions measure short-term variation. Repeat a small, predeclared representative workload subset in independent A30 allocations with randomized execution order; report between-allocation variation. A full hyperparameter sweep need not be repeated just to check representative speed claims.
4. **Generalization beyond the current tasks.** A new frozen group-disjoint SMS set and new basketball matches can strengthen this claim. Keep the existing results and explicitly label a new evaluation. This is optional for a paper limited to the observed tasks/splits.

## Submission preparation

- Choose representative valid workload rows/figures from the completed matrix reports to substantiate any integration speed claim. Give adaptation checkpoint, task, batch, prompt/output length, backend, cache policy, memory scope and acceptance status. Never multiply isolated speedups or use failed cells as preserving output.
- Training search uses eager attention: do not imply attention-kernel hyperparameters were jointly tuned during adaptation. SMS's one-token classifier does not demonstrate KV reuse speedup.
- Resolve the existing Ax prose: `EDITOR_NOTES.md` says no completed Ax run or matching runner was identified. Present the completed work as the staged search unless its author supplies the separate Ax evidence.
- Preserve the existing report assets in Overleaf. `LoRA.png`, `KVCache.png`, and `FlashAttention.png` were already missing locally. A LaTeX compiler is unavailable in this environment; compile and visually check the resulting paper in Overleaf.
- Review bibliography and export the final PDF. This update audits local measurements; it does not verify all external background references.

## Regeneration

Run `python3 transformer_syed/reports/mqp_draft/build_precision_evidence.py` from the repository root. It refreshes the standalone results section and evidence tables. The embedded section in `MQP_REPORT_UPDATED.tex` is the version generated during this update; if raw inputs change, replace its generated section with the regenerated standalone section.

## Current execution limit

This environment has no `sbatch`/`sacct` commands on PATH and no configured GPU execution was established. No new cluster run has been submitted by this update. The existing completed data were sufficient for the included findings; the stronger claims above remain open.
