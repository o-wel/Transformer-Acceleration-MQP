# Paper hyperparameter study

This is a fresh prospective study. The earlier pilot is preserved in `experiments/hyperparameter_tuning/` and is not pooled with these results. Read [METHODS.md](METHODS.md) for the paper-ready design and [protocol.py](protocol.py) for executable constants.

The design uses 45 LoRA candidates × 3 discovery seeds, 6 cache settings, 24 prototype attention settings, 5 new confirmation seeds, 3 independently allocated timing jobs per confirmation seed, 14 workloads, 20 paired measured repeats, and fresh match-level held-out data. Selection is quality constrained and frozen before confirmation. LoRA is selected for training cost; cache/attention for inference cost; the combined primary endpoint is inference on eight unseen workload shapes.

Create and submit a new frozen study from the minGPT root:

```bash
python experiments/paper_hyperparameters/prepare.py experiments/paper_hyperparameters/runs/STUDY_ID
python experiments/paper_hyperparameters/submit.py experiments/paper_hyperparameters/runs/STUDY_ID
```

Submission records every job ID immediately in `jobs.json`; never blindly rerun submission after a partial failure. The preflight GPU job gates the full dependency DAG. Sources and data live under `snapshot/repo/`; jobs execute those copies. Do not edit a submitted snapshot or its protocol. Use a new study for an amendment.

Check the latest submission:

```bash
python experiments/paper_hyperparameters/status.py
```

Or pass a study directory explicitly. Completion produces `results.json`, `RESULTS.md`, standalone `RESULTS.html`, `TIMINGS.csv`, `FACTORIAL.csv`, `FACTORIAL_CONTRASTS.json`, `combined-configs.json`, `ARTIFACT_SHA256.json`, and raw training/runtime/quality JSON. Task JSON includes exact outputs, per-repeat timings, GPU/software metadata, selection provenance and numerical gates. `selection.json` does not exist until all discovery stages succeed. The final report job depends on all confirmation jobs.

The paper must call the tunable implementation an **FA1-derived Triton prototype**. Per the requested scope, there is no production-backend comparison. No results are promised in advance. A narrow grid, one seed, old test reuse, or repeated timings from a single allocation are insufficient for the claims this study is designed to support.
