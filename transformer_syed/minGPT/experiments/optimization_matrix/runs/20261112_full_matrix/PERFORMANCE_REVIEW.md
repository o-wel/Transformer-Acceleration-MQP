# LoRA, attention and KV-cache optimization matrix

Overall status: COMPLETE
Each task/precision contains 27 off/best/poor combinations plus four joint serving selections.
Both independently tuned factors and jointly tuned attention/cache settings are confirmed.
Off LoRA is full fine-tuning; attention and cache are inference factors in this study.
Best/poor training choices are frozen from prior validation. Numerical failures stay diagnostic.
New measurements are not available until jobs complete. Submission is not completion.

## basketball_fp16: COMPLETE
{"calibration": 2592, "confirmation": 2232, "quality": 93, "training": 9}
Report: basketball_fp16/PERFORMANCE_REVIEW.pdf
Raw comparison table: basketball_fp16/COMPARISONS.csv

## basketball_fp32: COMPLETE
{"calibration": 2592, "confirmation": 2232, "quality": 93, "training": 9}
Report: basketball_fp32/PERFORMANCE_REVIEW.pdf
Raw comparison table: basketball_fp32/COMPARISONS.csv

## spam_fp16: COMPLETE
{"calibration": 2592, "confirmation": 2232, "quality": 93, "training": 9}
Report: spam_fp16/PERFORMANCE_REVIEW.pdf
Raw comparison table: spam_fp16/COMPARISONS.csv

## spam_fp32: COMPLETE
{"calibration": 2592, "confirmation": 2232, "quality": 93, "training": 9}
Report: spam_fp32/PERFORMANCE_REVIEW.pdf
Raw comparison table: spam_fp32/COMPARISONS.csv

The four studies retune separately. Precision contrasts between selected winners combine precision
and configuration changes; do not interpret them as isolated FP16 effects.
Input provenance and frozen source/data hashes: SNAPSHOT_SHA256.json. Scheduler IDs: jobs.json.
