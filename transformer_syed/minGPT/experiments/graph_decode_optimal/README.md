# Report using best-tested hyperparameters

Read `MINGPT_OPTIMIZATION_REPORT.pdf`, `REPORT.html` or `REPORT.md` after the
complete study finishes. The HTML embeds all figures and works offline.

This experiment inherits rank2/alpha8/LR.001 and full prefill from the completed
45-setting validation-based study, then searches 16 launch configurations for
the actual graph decoder. It freezes runtime selection before measuring the
five independent confirmation-training checkpoints on held-out workloads and
360 fresh synthetic questions. “Best tested under the declared objective” is
the supported wording; no global-optimality claim is made.

Earlier rank-4 graph results and production sources are preserved. New code
and evidence live only in this directory. The independent training settings,
checkpoint provenance and serving precision are described in PROTOCOL.md.

From the minGPT repository root:

```bash
source /home/sashazli/MQP/Torch/setup.sh
# Inputs and selection already exist; do not overwrite them.
sbatch experiments/graph_decode_optimal/run_gpu.sh search --seed 11
sbatch experiments/graph_decode_optimal/run_gpu.sh search --seed 22
sbatch experiments/graph_decode_optimal/run_gpu.sh search --seed 33
python -m experiments.graph_decode_optimal.study select
sbatch experiments/graph_decode_optimal/run_gpu.sh confirm --seed 101
sbatch experiments/graph_decode_optimal/run_gpu.sh confirm --seed 202
sbatch experiments/graph_decode_optimal/run_gpu.sh confirm --seed 303
sbatch experiments/graph_decode_optimal/run_gpu.sh confirm --seed 404
sbatch experiments/graph_decode_optimal/run_gpu.sh confirm --seed 505
/home/sashazli/torch_env/bin/python -m experiments.graph_decode_optimal.analyze
PYTHONPATH=/tmp/mqp-paper-report-deps python -m experiments.graph_decode_optimal.build_report
```

The analysis expects three complete search files and five complete confirmation
files. For a separate reproduction, use a fresh directory/archive prior outputs
rather than mixing independent runs. Never rerun `prepare` over frozen inputs.
Executables refuse to overwrite evidence. Report building uses existing local
ReportLab/Matplotlib/Pillow dependencies; its versions are archived separately
from the GPU environment. Cached GPT-2 weights/tokenizer and the original
checkpoint files are required and referenced by SHA256, not duplicated here.

The runner remains inference-only, native FP16, fixed model/adapter and fixed
dense batch, with full context up to1024 positions. Context rollover, arbitrary
sampling, concurrent serving, continuous batching and adapter replacement in a
live runner are unsupported. Official PyTorch FlashAttention performs prefill;
decode uses an experimental tiled online-softmax kernel. CUDA graph replay and
LoRA graph serving are established methods; novelty is not established by
this repository speedup.

To use the selected setting after loading one of the recorded rank-2/alpha-8
checkpoints into an eval-mode native-FP16 CUDA model with the `flash` backend:

```python
from experiments.graph_decode_optimal.runtime import Runner
with torch.inference_mode():
    runner = Runner(model, batch=1, fused=True, graph=True, bn=256, warps=8)
    output = runner.generate(ids, count=64)
```

Construct once and reuse across prompts. `ids` must be a dense CUDA int64 batch
of the declared size; prompt plus output must fit the 1024-position context.
