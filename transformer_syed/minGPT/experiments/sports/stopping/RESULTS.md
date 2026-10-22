# Epoch versus training-loss stopping

Loss target: 0.05; epoch limit: 7. Final checkpoints; same held-out sports scoring.

| Method | Stop rule | Mean steps | Primary accuracy | Answer + scores |
|---|---|---:|---:|---:|
| lora | epochs | 1260 | 85.00% ± 5.83 | 85.00% ± 5.83 |
| lora | train_loss | 960 | 72.50% ± 6.61 | 72.50% ± 6.61 |
| full | epochs | 1260 | 27.50% ± 8.04 | 27.50% ± 8.04 |
| full | train_loss | 840 | 21.67% ± 2.50 | 21.11% ± 2.93 |

± denotes sample SD across training seeds. A run hitting the epoch cap did not reach the loss target. This is the existing synthetic benchmark.
