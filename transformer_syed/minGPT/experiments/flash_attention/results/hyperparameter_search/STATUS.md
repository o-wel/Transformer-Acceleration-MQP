# Study complete — November 3, 2026

The FP32 study and final report are complete. All submitted jobs exited successfully; no study jobs remain queued or running.

- 127 unique training trials completed, including 90 core-grid screening configurations and three-seed finalist confirmation.
- 1,440 combined serving cells completed with seven measured repetitions each: 10,080 measured repetitions, plus warmups.
- Every serving cell passed its logit tolerance and greedy-token parity checks. No OOM cells occurred.
- The selected LoRA configuration is rank 32, alpha 64, LR 0.0003, all attention/MLP projections, adapter dropout 0.05, weight decay 0.01, batch/microbatch 4, length 128.
- Mean validation NLL: 0.016746; seed SD: 0.009248. Mean held-out answer-field accuracy: 93.06%; full-fine-tuning control: 24.17%. Unequal tuning budgets and task-specific scope are discussed in the report.

Final deliverables are run_fp32/FINAL_REPORT.pdf (8 pages), run_fp32/FINAL_REPORT.html (standalone figures), and run_fp32/FINAL_REPORT.md (editable). Data tables, raw outputs, source snapshots and job logs remain in run_fp32. FINAL_REPORT_HASHES.json records report integrity and completed counts. The PDF was rendered and inspected, and its extracted text checked for principal findings and the final section.

Optional FP16 experiments were not run. The report distinguishes the custom FA1-derived prototype from production FlashAttention and discusses all search, timing and evaluation limitations.
