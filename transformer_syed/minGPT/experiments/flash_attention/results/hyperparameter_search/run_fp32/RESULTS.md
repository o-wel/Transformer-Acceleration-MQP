# Hyperparameter search results

Completed trials: 127. Incomplete trials: 0.

Rankings use validation NLL. Screening and full-budget trials are reported separately. Best observed is conditional on the staged search and this dataset.

| Rank | r | alpha | LR | Targets | Dropout | Decay | Batch | Length | Validation NLL | Step ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 32 | 64 | 0.0003 | all | 0.05 | 0.01 | 4 | 128 | 0.008009 | 62.01 |
| 2 | 32 | 64 | 0.0003 | all | 0.05 | 0.01 | 8 | 128 | 0.018789 | 120.18 |
| 3 | 32 | 64 | 0.0003 | all | 0.05 | 0.01 | 8 | 128 | 0.020831 | 237.96 |
| 4 | 32 | 64 | 0.0003 | all | 0.05 | 0.01 | 8 | 128 | 0.022593 | 90.11 |
| 5 | 32 | 64 | 0.0003 | all | 0.0 | 0.0 | 8 | 128 | 0.023162 | 87.77 |
| 6 | 32 | 32 | 0.001 | attention | 0.0 | 0.0 | 8 | 128 | 0.023164 | 82.80 |
| 7 | 32 | 64 | 0.0003 | all | 0.05 | 0.01 | 8 | 256 | 0.023247 | 187.88 |
| 8 | 4 | 16 | 0.001 | all | 0.0 | 0.0 | 8 | 128 | 0.023254 | 87.96 |
| 9 | 32 | 128 | 0.0003 | attention | 0.0 | 0.0 | 8 | 128 | 0.023444 | 82.24 |
| 10 | 32 | 64 | 0.0003 | all | 0.05 | 0.01 | 8 | 128 | 0.024037 | 471.19 |

Frozen selection:
```json
{
  "frozen_before_test": true,
  "criterion": "mean validation NLL over seeds 42,43,44",
  "ranked": [
    {
      "config": {
        "method": "lora",
        "rank": 32,
        "scale": 2.0,
        "lr": 0.0003,
        "targets": "all",
        "dropout": 0.05,
        "weight_decay": 0.01,
        "batch": 4,
        "microbatch": 4,
        "length": 128
      },
      "mean_validation_nll": 0.016746423939030006,
      "stdev_validation_nll": 0.009248221563686128
    },
    {
      "config": {
        "method": "lora",
        "rank": 32,
        "scale": 2.0,
        "lr": 0.0003,
        "targets": "all",
        "dropout": 0.05,
        "weight_decay": 0.01,
        "batch": 8,
        "microbatch": 4,
        "length": 128
      },
      "mean_validation_nll": 0.021946173039139278,
      "stdev_validation_nll": 0.0027511931248480515
    },
    {
      "config": {
        "method": "lora",
        "rank": 32,
        "scale": 2.0,
        "lr": 0.0003,
        "targets": "all",
        "dropout": 0.05,
        "weight_decay": 0.01,
        "batch": 8,
        "microbatch": 2,
        "length": 128
      },
      "mean_validation_nll": 0.030421684171141702,
      "stdev_validation_nll": 0.011979631948876131
    }
  ],
  "winner": {
    "method": "lora",
    "rank": 32,
    "scale": 2.0,
    "lr": 0.0003,
    "targets": "all",
    "dropout": 0.05,
    "weight_decay": 0.01,
    "batch": 4,
    "microbatch": 4,
    "length": 128
  },
  "full_control": {
    "method": "full",
    "rank": 32,
    "scale": 2.0,
    "lr": 0.0001,
    "targets": "all",
    "dropout": 0.05,
    "weight_decay": 0.01,
    "batch": 4,
    "microbatch": 4,
    "length": 128
  },
  "note": "Best observed in this staged search, not a global optimum"
}
```
