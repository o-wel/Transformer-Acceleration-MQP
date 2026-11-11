# Joint KV-cache / LoRA / attention research

This directory contains a targeted prior-art audit, an isolated Triton decode
prototype, a frozen comparison protocol and raw GPU experiments. The candidate
fuses **current-token LoRA QKV expansion + static KV append + streaming
attention**. It retains fully adapted historical K/V and uses forced PyTorch
FlashAttention for prefill. The decode kernel is an experimental implementation
of tiled online-softmax attention, not an official FlashAttention release.

Read [RESULTS.md](RESULTS.md) for measured conclusions and
[RESEARCH.md](RESEARCH.md) for the novelty assessment, mathematics, prior-art
matrix, limitations and next research steps. A speedup in this implementation
does not by itself establish publication novelty.

Files:

* `PROTOCOL.md`: timing/correctness design and documented v2 control amendment.
* `QUALITY_PROTOCOL.md`: supplementary fresh-task check fixed before outputs.
* `kernel.py`, `runtime.py`: isolated experimental implementation and controls.
* `benchmark.py`, `analyze.py`: paired GPU measurement and clustered analysis.
* `prepare_quality.py`, `quality.py`: fresh task inputs, paired NLL and answers.
* `results/`: per-allocation JSON, raw outputs, logs and profiler traces.
* `archive/v1/`: original discovery implementation and protocol, not pooled.
* `SUMMARY.json`, `TIMINGS.csv`: machine-readable confirmation summaries.

Run from the minGPT repository root with the existing Torch environment:

```bash
source /home/sashazli/MQP/Torch/setup.sh
sbatch experiments/joint_decode/run_gpu.sh --phase discovery
sbatch experiments/joint_decode/run_gpu.sh --phase confirmation --replicate 0
sbatch experiments/joint_decode/run_gpu.sh --phase confirmation --replicate 1
sbatch experiments/joint_decode/run_gpu.sh --phase confirmation --replicate 2
/home/sashazli/torch_env/bin/python -m experiments.joint_decode.analyze
```

The existing analysis expects exactly three confirmation files; archive old
results in a separate directory before a new independent run. Existing files
are not overwritten. Quality inputs are already frozen; rerun only the quality
GPU job to evaluate them again. Full reproduction requires the cached GPT-2
weights/tokenizer, existing trained adapter checkpoint, PyTorch/CUDA/Triton and
an A30 allocation. Source/data/checkpoint hashes are retained in the artifacts.
No new Python dependency or production-model change is required.
Analysis/plotting uses the existing `torch_env` environment because the GPU
environment does not contain Matplotlib; its versions are in plotting-environment.json.

The prototype supports one dense equal-length request batch at a time with
native-FP16 serving. It is not a concurrent serving engine and does not support
training, mixed adapters, cached multi-token chunks or context rollover.
