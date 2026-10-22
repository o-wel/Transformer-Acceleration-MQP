# Matched full fine-tuning versus LoRA


Correct comparison: pretrained GPT-2 (reference), full fine-tuning on the new data, and LoRA fine-tuning on the same data. Full fine-tuning is now trained and evaluated for the same budget as LoRA; the previous 40-step full-training run was a memory-only baseline.


Controls: same GPT-2 starting weights, training examples and sample order, FP32 with TF32 off, batch 8, padded length 128, 1,200 steps, seeds 42/43/44, and NVIDIA A30. Starting-weight and batch-order SHA256 fingerprints match between methods for every seed. Each run processes 1,228,800 padded token positions; supervised answer-token budgets also match by seed.


Equal tuning budget: both methods tried learning rates 0.001 and 0.0001 for 1,200 steps on seed 42. Validation NLL at steps 300/600/900/1200 selected the rate and checkpoint; both rate choices were saved before new test evaluation. Selected rates: full 0.0001, LoRA 0.001. Seeds 43/44 then used the chosen rate. Four training runs per method including the unselected candidate; six selected runs in the final comparison.


Data: 480 train matches, 40 validation matches, 40 test matches, three questions per match, plus 120 score-swapped test questions. Test team names and score combinations are absent from training. This is the existing disclosed synthetic benchmark, not a newly blinded test set; no new test-based tuning was used.


Primary-answer score: pretrained 25.8%; full 31.9 ± 14.4%; LoRA 81.7 ± 0.8%. Answer + score fields: pretrained 0.0%; full 31.9 ± 14.4%; LoRA 80.8 ± 1.4%. ± is sample SD across three training seeds, not a confidence interval.


Trainable scalar parameters: full 163,037,184 (100%); LoRA 589,824 (0.3605% of the LoRA model). LoRA total 163,627,008; full total 163,037,184. LoRA updates 276.4 times fewer trainable parameters. All LoRA base-weight audits passed.


Mean full-budget peak allocated GPU memory: full 3747.0 MiB; LoRA 2481.6 MiB (33.8% lower). Mean synchronized training-step time: full 149.8 s; LoRA 105.2 s. Timing excludes transfer, validation, audits, checkpoint I/O and test generation. End-to-end training wall times are reported separately in JSON.


Limits: metrics are transparent but format-sensitive field checks, not broad conversational quality. Three questions share each match. Generated templates can be reused, so score-swap consistency and failures matter. Both methods are trained for the full budget but evaluated at their validation-selected checkpoint. This does not establish quality equivalence or time-to-quality without further experiments.

| Metric | Pretrained reference | Full fine-tuning | LoRA fine-tuning |
|---|---:|---:|---:|
| Primary answer (%) | 25.8 | 31.9 ± 14.4 | 81.7 ± 0.8 |
| Answer + scores (%) | 0.0 | 31.9 ± 14.4 | 80.8 ± 1.4 |
| Exact reference (%) | 0.0 | 26.4 ± 15.9 | 80.8 ± 1.4 |
| winner primary (%) | 57.5 | 18.3 ± 25.5 | 100.0 ± 0.0 |
| halftime primary (%) | 20.0 | 25.8 ± 17.7 | 96.7 ± 1.4 |
| comeback primary (%) | 0.0 | 51.7 ± 5.2 | 48.3 ± 1.4 |
| Both score-swap answers correct (%) | 0.0 | 6.4 ± 7.7 | 65.8 ± 0.8 |
| Total stored parameters | 163,037,184 | 163,037,184 | 163,627,008 |
| Trainable parameters during adaptation | 0 | 163,037,184 | 589,824 |
| Actually changed scalars (selected seed 42) | 0 | 123,851,224 | 589,824 |
| Parameter storage (MiB) | 621.94 | 621.94 | 624.19 |
| Mean peak allocated training memory (MiB) | N/A | 3747.0 | 2481.6 |
| Mean peak reserved training memory (MiB) | N/A | 4060.0 | 2758.0 |
| Mean training-step time (s) | N/A | 149.8 | 105.2 |

## Validation selection
| Method | Seed | Learning rate | Selected step | Validation NLL | Test NLL | Changed scalars |
|---|---:|---:|---:|---:|---:|---:|
| full | 42 | 0.0001 | 1200 | 0.04948 | 0.31978 | 123,851,224 |
| full | 43 | 0.0001 | 1200 | 0.05356 | 0.22493 | 123,826,232 |
| full | 44 | 0.0001 | 1200 | 0.04830 | 0.15950 | 123,851,454 |
| lora | 42 | 0.001 | 1200 | 0.02853 | 0.04067 | 589,824 |
| lora | 43 | 0.001 | 1200 | 0.02598 | 0.02646 | 589,824 |
| lora | 44 | 0.001 | 1200 | 0.02720 | 0.02770 | 589,824 |

## All learning-rate candidates (validation only)
| Method | Rate | Best validation NLL | Step |
|---|---:|---:|---:|
| lora | 0.001 | 0.02853 | 1200 |
| lora | 0.0001 | 0.07750 | 900 |
| full | 0.001 | 0.15015 | 900 |
| full | 0.0001 | 0.04948 | 1200 |

## Exploratory familiar-name control
Exploratory familiar-name control: added after observing the seed-42 unseen-name failures. Only team names were replaced; scores, question types and checkpoints were unchanged. No retraining or tuning. This is a diagnostic, not a replacement for the original test.
Reference answers overlapping training: 9 / 120.
primary_correct: pretrained 30.8%; full 82.5 ± 2.5%; LoRA 82.5 ± 0.0%.
fields_correct: pretrained 5.0%; full 82.5 ± 2.5%; LoRA 81.7 ± 0.8%.
reference_exact: pretrained 0.0%; full 82.5 ± 2.5%; LoRA 81.7 ± 0.8%.