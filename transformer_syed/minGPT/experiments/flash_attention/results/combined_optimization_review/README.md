# Combined optimization review

The report separates a matched FA1-derived prototype comparison from the available deployment backends (custom tiled FP32 versus PyTorch FA2). Original reports and source code are preserved.

- `OPTIMIZATION_COMPARISON.pdf`: presentation-ready report.
- `OPTIMIZATION_COMPARISON.html`: standalone report with embedded figures.
- `OPTIMIZATION_COMPARISON.md`: editable report.
- `FA1_MATCHED_METRICS.csv`: all 16 matched FA1-derived inference cells.
- `METRICS.csv`: all 96 deployment-backend inference configurations.
- `ALL_OUTPUTS.csv`: main held-out answers and token IDs.
- `confirmation.json`: main experiment and selected hyperparameters.
- `candidate-quality.json`: both learning-rate candidates, tested only after validation choices froze.
- `merged-confirmation.json`: merged-LoRA serving ablation.
- `validation.json`: independent GPU correctness-test results.
- `owen_evidence/`: reviewed PDFs, supporting cache JSON, repository commit and hashes.
- `source/`: dependency snapshots and hashes.

Fresh experiment protocol: one seed, 1,200 successful updates per candidate, batch 8, length 128, identical data order, two learning rates per method and precision. FP16 uses autocast and FP32 master weights. Forced FlashAttention is the FP16 attention variant; custom IEEE-FP32 tiled attention is the FP32 variant. Training and inference have separate cost protocols.

GPU launch: `sbatch run_gpu.sh`. After completion, `sbatch run_merge_gpu.sh` and `sbatch run_candidates_gpu.sh`. Run `sbatch validate_gpu.sh` independently. Use a new output directory for a new measurement, since the main runner refuses to overwrite existing results.

Report dependencies: NumPy, Matplotlib, Markdown, ReportLab. Run `python build_report.py` after the experiments finish. The GPU environment uses the existing `~/MQP/Torch/setup.sh` configuration and locally cached GPT-2 weights/tokenizer.

Attention generation: the installed PyTorch registry identifies its default FlashAttention implementation as FA2. `ATTENTION_VERSION_AUDIT.json` records the local version and registry evidence. `fa1_reference.py` is a separate FA1-derived Algorithm 1/4 prototype, with the same key-outer/query-inner code in FP32 and FP16. It is not the original optimized FA1 CUDA implementation. Run `sbatch run_fa1_gpu.sh` after the tuning choices have frozen; its results stay in `fa1_matched/`.
