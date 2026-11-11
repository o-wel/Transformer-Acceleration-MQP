# LoRA on a different task: real SMS spam classification

This experiment classifies messages as spam or legitimate (ham). It uses no sports examples, score reasoning, or free-text answer grading. All results below are measured on the same held-out messages.

## Data and protocol

[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), Almeida and Hidalgo (2011). Downloaded 5574 messages; 5159 remain after case/whitespace deduplication. Lowercased templates with URL/digit normalization are assigned to just one split before subsampling. This reduces common-template leakage; it does not prove all near duplicates are removed.

- Training: 1200 messages (151 spam).
- Validation: 400 messages (48 spam).
- Test: 771 messages (93 spam).
- Split seed: 20261026. Training seeds: 42, 43, 44. One fixed split across seeds.
- Same pretrained GPT-2, batch 8, five shuffled epochs, FP32/TF32 off, AdamW, no weight decay, gradient clipping 1.0. LoRA rank 4, alpha 8 on QKV, attention output, and MLP projections.
- Equal LR search for each method: 0.001 and 0.0001, seed 42, five epochs each. Select by final validation label NLL, before test evaluation. Selected seed-42 candidates are reused.
- Message content is truncated to the first 96 GPT-2 tokens, with the same instruction and label suffix for all methods. Right padding follows real tokens. Train loss is full-vocabulary next-token cross entropy for one correct label token.
- Prediction chooses the higher logit of the two single-token labels ` ham` and ` spam`. This constrained classification score is used for both pretrained and trained models; it is not free-form generation.
- The selected LR is shared by both stopping rules; the early-stopping rule does not receive a separate LR search.
- Fixed budget: checkpoint after five epochs. Loss stopping: first epoch with full-training-set label NLL ≤ 0.05, or five-epoch cap. Both checkpoints come from the same trajectory, with no test-based selection.

## Held-out classification quality

Percentages are mean ± sample SD across three training seeds; SD is not a confidence interval. The pretrained and majority baselines are single deterministic evaluations.

| Model | Rule | Accuracy (%) | Balanced accuracy (%) | Spam F1 (%) | Spam precision (%) | Spam recall (%) |
|---|---|---:|---:|---:|---:|---:|
| Always ham | No training | 87.94 | 50.00 | 0.00 | 0.00 | 0.00 |
| Pretrained GPT-2 | Two-label scoring | 21.01 | 36.07 | 14.59 | 8.39 | 55.91 |
| lora | epochs | 98.88 ± 0.15 | 96.89 ± 0.99 | 95.29 ± 0.63 | 96.40 ± 2.17 | 94.27 ± 2.24 |
| lora | loss_stop | 98.79 ± 0.30 | 97.15 ± 0.93 | 94.99 ± 1.14 | 95.11 ± 3.56 | 94.98 ± 2.24 |
| full | epochs | 98.31 ± 0.57 | 93.78 ± 2.25 | 92.59 ± 2.66 | 97.99 ± 0.72 | 87.81 ± 4.48 |
| full | loss_stop | 98.62 ± 0.15 | 95.97 ± 0.47 | 94.16 ± 0.60 | 95.94 ± 1.62 | 92.47 ± 1.08 |

## Training budget, time, and memory

| Method | Rule | Epochs | Training compute (s) | Trajectory wall time (s) | Peak training allocated (MiB) |
|---|---|---:|---:|---:|---:|
| lora | epochs | 5.00 ± 0.00 | 42.19 ± 0.89 | 67.10 ± 0.94 | 1915.44 ± 1.05 |
| lora | loss_stop | 4.67 ± 0.58 | 39.38 ± 4.43 | 62.63 ± 7.32 | 1915.40 ± 1.08 |
| full | epochs | 5.00 ± 0.00 | 58.39 ± 0.27 | 81.77 ± 0.32 | 3309.37 ± 1.28 |
| full | loss_stop | 2.00 ± 0.00 | 23.61 ± 0.15 | 33.11 ± 0.18 | 3309.02 ± 1.77 |

Compute time sums synchronized forward/backward/clipping/optimizer steps; it excludes transfers, audits, evaluation and checkpoint I/O. Wall time is elapsed within the training trajectory, including validation, full-training loss checks and checkpoint writes, but excludes model setup and final test/audit. Both rules share the same monitored trajectory; these are not independent end-to-end stopping-rule timing trials. Final wall time includes an earlier stopping-checkpoint write when applicable. Peak allocated GPU memory measures PyTorch allocations, not total device memory.

Total across all 8 training runs, including the learning-rate search: 403.4 seconds of training compute and 596.6 seconds summed training-loop wall time. These totals exclude model setup, final audits, and held-out testing. The comparison table reports per-run means for the selected rates.

## Learning-rate selection

| Method | Candidate LR | Final validation label NLL | Selected |
|---|---:|---:|---|
| lora | 0.001 | 0.270363 | no |
| lora | 0.0001 | 0.081884 | yes |
| full | 0.001 | 0.274684 | yes |
| full | 0.0001 | 0.365629 | no |

## What changed relative to the sports experiment?

- lora: loss stopping changed accuracy from 98.88% to 98.79% (-0.09 percentage points), with 6.7% less training compute. Target reached in 3/3 seeds.
- full: loss stopping changed accuracy from 98.31% to 98.62% (+0.30 percentage points), with 59.6% less training compute. Target reached in 3/3 seeds.
- Fixed epochs: LoRA/full accuracy ratio is 1.006×; difference +0.56 percentage points. A sports-task advantage should not be assumed to transfer to this task.
- lora: 589,824 trainable scalars out of 163,627,008; final audit found 0 frozen scalars changed.
- full: 163,037,184 trainable scalars out of 163,037,184; final audit found 0 frozen scalars changed.

A training-loss threshold controls fit to training labels, not guaranteed held-out accuracy. The numerical threshold 0.05 is reused to probe transfer of the stopping rule; one label token per example is a different objective from average loss across multi-token sports answers. Do not interpret equal numerical thresholds as equal task difficulty.

Classification asks for one of two labels, whereas the sports task required generating correct teams, scores and explanations. This experiment can show whether the observed LoRA advantage persists; it does not isolate a causal reason for any gap. Learning rates are selected separately and results depend on this dataset, split, prompt, training subset and budget.

Accuracy is inflated by class imbalance: always predicting ham already achieves the majority baseline above. Spam F1, recall and false negatives matter. The corpus is public and old; unknown GPT-2 pretraining overlap cannot be excluded. This is not evidence of performance on current SMS attacks or other classification datasets. Three seeds share a single test set, so small differences do not establish equivalence or a universal winner.

## Reproduction and artifacts

From the minGPT root, use the existing PyTorch/Transformers environment (see `../../LORA_REPRO.md` for setup). Then:
```bash
python experiments/sms_spam/test_protocol.py -v
sbatch experiments/sms_spam/run_gpu.sh
# After all runs finish (matplotlib required):
python experiments/sms_spam/build_report.py
```

The Slurm wrapper uses the author’s environment paths and cluster account; adapt them on another system. Runs refuse to overwrite the existing `runs` directory; preserve it and copy code files to a fresh sibling experiment directory for a rerun, updating the wrapper path. Dataset preparation downloads the UCI archive; GPT-2 loading uses cached weights/tokenizer. See README.md for the cache-preparation command.

- `protocol.json`: settings declared before training; `selection.json`: validation-only LR choices.
- `data/manifest.json`: source checksum, split counts and SHA256 checksums.
- `METRICS.csv`: each seed/rule result, confusion counts, loss, time and memory.
- `runs/*/training.json`, `changes.json`, `test.json`: histories, source/initial-weight hashes, audits, and every test prediction keyed by message hash.
- `verification.json`: acceptance checks; `summary.json`: aggregate results.
- `REPORT.pdf`: standalone report; plots saved as PNG/SVG.
- `runs/*/*.pt`: local model/adapter checkpoints; do not commit large full checkpoints.

Hardware: NVIDIA A30; PyTorch 2.14.0+cu130; CUDA 13.0.
All acceptance checks passed. The three offline tests cover scorer confusion counts, template/split separation, causal padding positions, and adapter-only training.
