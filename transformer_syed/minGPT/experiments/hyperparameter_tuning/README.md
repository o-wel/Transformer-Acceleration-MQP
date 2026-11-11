# Individual and combined hyperparameter tuning

Run from the minGPT root:

```bash
sbatch experiments/hyperparameter_tuning/run_gpu.sh
```

The complete preregistered grid, fixed controls, quality gates, and selection criteria are in `protocol.json`. Results go into a new `results/` directory; the runner refuses to overwrite it. Use `--out experiments/hyperparameter_tuning/results-new` for another run.

1. Train all eight LoRA rank/alpha/learning-rate candidates with the same 1,200 updates, examples and checkpoint opportunities. Select checkpoints using validation NLL only. Among candidates within 0.02 NLL of the best candidate, select the fastest eager uncached inference configuration. Also retain training-step latency.
2. On the fixed baseline LoRA checkpoint, independently tune full-context cache prefill chunks and FA1-derived tile sizes/warp counts. Cache remains enabled throughout its sweep; the attention algorithm remains enabled throughout its sweep. Neither batch size nor prompt length is optimized.
3. Freeze those individual choices. Measure tuned versus untuned LoRA, cache and FA individually, then tuned versus untuned configurations with all three enabled. Confirmation interleaves seven fresh paired repeats, with fresh prompts, after three warmups. Report speedup as untuned latency / tuned latency, including slowdowns and bootstrap intervals.
4. Evaluate combined configurations on held-out sports questions after selection. Preserve raw outputs and scoring. Training cost, inference cost and search cost have distinct meanings; a fast combined inference configuration does not establish a training speedup or recover tuning cost automatically.

The untuned configuration is predeclared: rank 4, alpha 8, LR 0.0001; cache prefill chunk 128; FA query/key tiles 16/32 and four warps. It is not selected after seeing timings. If it wins a sweep, report that honestly instead of forcing a non-optimal control. The LoRA baseline may fail the quality gate; its validation and test quality are still reported.

FlashAttention tuning uses a locally copied, configurable version of the existing FA1-derived prototype (`fa1_tunable.py`). This is not the original optimized FA1 CUDA library, and does not measure PyTorch FA2. Original experiments and their artifacts are preserved.

Artifacts include `selection.json`, `confirmation-timings.json`, `report.json`, `RESULTS.md`, eight adapter checkpoints and per-candidate training logs. A finite grid, one training seed and one A30 support best-tested findings, not global optimality.

To export PDF, standalone HTML, CSV and plots after the GPU run:

```bash
python -m pip install matplotlib markdown reportlab
python experiments/hyperparameter_tuning/build_report.py experiments/hyperparameter_tuning/results
```

`combined-configs.json` records both complete combinations, including adapter checkpoint, LoRA parameters, cache chunk size, and FA launch configuration.

Protocol amendment: the first runtime sweep stopped because FP16 chunk sizes changed greedy token sequences. The resumed sweep reuses all eight completed training runs, checks cache validation NLL against the same eager model (absolute tolerance 0.02) and same-input final-prefill logits (atol 0.1, rtol 0.02), and retains token-divergence flags. This change is recorded in the protocol before confirmation and test evaluation. CPU cache tests still require exact greedy parity.
