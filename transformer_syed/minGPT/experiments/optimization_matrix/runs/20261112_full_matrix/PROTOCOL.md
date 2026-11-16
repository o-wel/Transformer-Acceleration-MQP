# Complete optimization ablation and tuning matrix

This extension measures LoRA, the local FA1-derived attention kernel, and KV caching individually, in every pair, and together. Each method has **off / best observed / poor observed** states: 27 combinations, including the eight ordinary on/off ablations. Four extra states cross best/poor LoRA with jointly best/poor attention/cache settings. Both basketball and SMS spam run in FP16 mixed precision and FP32 with TF32 disabled.

## Frozen training choices

For each task/precision, prior validation fixes three training configurations before new held-out evaluation:

- LoRA off: the existing selected full-fine-tuning control.
- LoRA best: the existing winner selected by three-seed mean validation NLL.
- LoRA poor: the existing worst seed-42 LoRA validation NLL at the full 9,600-example budget.

All three configurations are freshly trained for 9,600 examples with seeds 42/43/44 under this frozen implementation. This avoids mixing old checkpoint implementations and provides the previously missing poor-configuration seed replications. Poor selection has weaker prior seed evidence than the selected winner; it is not claimed to be the population worst. Full tuning inherited fewer search candidates than LoRA. Off is full adaptation, not an unadapted pretrained model. Batch changes optimizer update count at a fixed example budget. Attention and caching are serving factors; this does not sweep attention kernels during adaptation training.

## Serving hyperparameters and calibration

The original 24 workloads are retained: batch 1/4/8, prompt 32/128/512/896, output 32/64. Attention candidates are eager off and the eight custom tile/warp combinations BM 16/32, BN 32/64, warps 4/8. KV-cache candidates are off, dynamic, static request-sized capacity, and static full-context capacity. Request capacity is prompt + output - 1, matching actual stored positions. The isolated cache extension affects only this new frozen runner; old studies remain untouched.

Per training configuration and workload, calibrate every attention/cache setting on seed 42, with two warmups and three measured repetitions. Independently best/poor attention comes from custom tiles with cache off; independently best/poor cache comes from policies with eager attention. Joint extrema select the paired custom-attention/cache combination. Selection is latency-based among passing cells, with memory retained as a separate objective. There is no universal scalar optimum across quality, latency and memory.

Freeze selections before confirmation. The 27 independently selected states and four joint states run on all three model seeds using a separate prompt stream, two warmups and seven measured repetitions. Case order is randomized reproducibly. This provides 2,592 calibration and 2,232 confirmation cells per task/precision. Labels may coincide where there is only one eligible candidate; this is explicitly recorded and is not an invented tuning gain.

## Quality and numerical validity

Each serving cell compares strict logits (atol=rtol=3e-4) and complete greedy tokens with its own checkpoint on eager/no-cache. Training configurations intentionally have different learned functions and are compared by task metrics, not by requiring cross-checkpoint token equality. If no serving candidate passes, diagnostic extrema remain visible and explicitly ineligible. Existing FP16 custom attention is known to fail the strict gate; the gate is unchanged.

Natural held-out task evaluation covers all 31 states and all three seeds (93 quality records). Serving settings use the frozen canonical workload (batch 1, prompt 128, output 32), with static capacity adjusted to each actual natural prompt. All original validation examples are evaluated for the selected backend. SMS is a single label prediction: no repeated cache reuse exists in its task classifier; cache acceleration is assessed separately on the synthetic serving grid. Original splits are reused. Seed variation is not a dataset confidence interval, and repetitions within an allocation are not independent GPU replications.

## Execution and artifacts

```bash
python -m unittest discover -s experiments/optimization_matrix -p 'test_*.py' -v
python experiments/optimization_matrix/submit.py --out experiments/optimization_matrix/runs/NEW_ID
python experiments/optimization_matrix/status.py experiments/optimization_matrix/runs/NEW_ID
```

Submission snapshots and hashes Python sources, datasets, and frozen plans. Every job checks them. Studies are serial, with at most two concurrent GPU array tasks, using account `isp` and the academic A30 partition. Source plans are immutable; amendments require a new run. `jobs.json` is persisted after each submission, and partial submissions cannot be blindly repeated. Invalid dependency jobs are cancelled; the final report runs after quality termination and remains PARTIAL if measurements are missing.

Per-task CSVs: TRAINING, CALIBRATION, CONFIRMATION, QUALITY, and COMPARISONS. Markdown/HTML/PDF reports include all 27 states and joint choices by workload, task quality, passing coverage, peak memory, and speed ratios versus all-off only when all three paired seeds pass. Raw JSON retains repetitions, audits, selection eligibility, failures, checkpoint hashes, and natural-task outputs. A report marked COMPLETE indicates complete coverage, not correctness of every cell. No measurements are inferred from isolated gains or multiplied together.
