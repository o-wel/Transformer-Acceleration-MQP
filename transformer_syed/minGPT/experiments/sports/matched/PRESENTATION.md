# Revised professor presentation

Use this matched comparison instead of the previous LoRA-versus-untouched-model quality report.

1. **Explain the three models.** All start from the same pretrained GPT-2. The reference receives no new training. Full fine-tuning and LoRA both receive the same new data and the same 1,200-step budget. Full fine-tuning updates all base weights; LoRA updates only A/B adapters.
2. **Show the fairness checks.** Each method gets two learning-rate candidates on seed 42. Validation loss selects the rate and checkpoint; test outputs do not select them. Final results use seeds 42, 43 and 44. Initial-weight and batch-order hashes match between paired runs.
3. **Show held-out quality first.** Open `results/PROFESSOR_REPORT.pdf`. Compare full fine-tuning directly with LoRA. Use the pretrained model to show each method's improvement. Show per-task scores and score-swapped pairs; do not hide weak comeback performance or invalid answers.
4. **Show efficiency at that measured quality.** Compare trainable parameter counts, actually changed counts, peak allocated/reserved training memory, and training time. Distinguish storage (4 bytes per FP32 scalar) from runtime peak memory. Distinguish total training-step time from wall time with audits and checkpoint writes.
5. **Run the three-way demo.** Start `python sports_demo.py --matched --device cpu --interactive` from the minGPT directory after sourcing `~/MQP/Torch/setup.sh`. Supply names and scores; ask winner, halftime, or comeback. Change a score and inspect all three outputs. Seed 42 was chosen as the demo seed before test evaluation.
6. **Discuss the actual trade-off.** If LoRA matches or exceeds full fine-tuning here, say “on this controlled task and budget,” not “LoRA is always as good.” If full fine-tuning has better quality, show how much quality is traded for efficiency. Three seeds and a small synthetic benchmark do not establish statistical equivalence or broad chat capability.

For raw evidence, open the standalone `results/RESULTS.html` or `ALL_OUTPUTS.csv`. Both preserve every original and score-swapped output for every seed and model. `selection.json` records all candidate validation scores and selected run paths. Full-model weights are saved on the cluster as `model.pt`; LoRA weights as `adapter.pt`.

Old artifacts are preserved for provenance. Their 40-step full-training reference was a memory measurement only; do not cite it as the corrected quality comparison.
