# Larger-speedup continuation

This experiment combines reusable CUDA graph execution with the preceding
current-token LoRA-QKV / KV-append / streaming-attention fusion. It preserves
the same trained adapter, full context, precision and greedy selection policy.
The graph's position, next-token and output-index tensors change on the device
without changing tensor addresses or allocating a growing cache each step.

Read [RESULTS.md](RESULTS.md) for measured confirmation results and setup costs,
[RESEARCH.md](RESEARCH.md) for prior art and interpretation, and
[CONFIRMATION_PROTOCOL.md](CONFIRMATION_PROTOCOL.md) for the frozen design.
No production model source or prior experiment artifact is modified.

CUDA graph replay, LoRA graph serving and streaming attention are established
methods. The result is a larger minGPT implementation improvement; it is not
presented as a newly discovered graph algorithm or production speed record.

## Reproduce

From the minGPT repository root, using the cached base/tokenizer and existing
rank-4/alpha-8 trained checkpoint:

```bash
source /home/sashazli/MQP/Torch/setup.sh
sbatch experiments/graph_decode/run_pilot.sh
sbatch experiments/graph_decode/run_confirmation.sh --replicate 0
sbatch experiments/graph_decode/run_confirmation.sh --replicate 1
sbatch experiments/graph_decode/run_confirmation.sh --replicate 2
sbatch experiments/graph_decode/run_quality.sh
/home/sashazli/torch_env/bin/python -m experiments.graph_decode.analyze
```

The analysis expects exactly three completed confirmation files and one fresh
quality file. Archive existing outputs before starting a separate reproduction;
programs refuse to overwrite evidence. Fresh quality inputs are already frozen
and should not be regenerated. `prepare_quality.py` documents their creation.
The GPU environment lacks Matplotlib; analysis uses the existing `torch_env`
environment. Dependency/build versions and source hashes are archived.

## Use the experimental runner

After loading the adapter into an eval-mode native-FP16 CUDA GPT instance and
selecting the `flash` attention backend:

```python
from experiments.graph_decode.runtime import Runner

with torch.inference_mode():
    runner = Runner(model, batch=1, fused=True, graph=True)
    # ids: CUDA int64 tensor, shape (1, prompt_length)
    output = runner.generate(ids, count=64)
```

Construct once and reuse across prompts of different lengths. Every request
resets its cache; none of the measured benefit relies on repeated prompt
content. A different batch requires another runner. This prototype supports
one equal-length dense batch at a time, fixed adapter weights and native FP16.
It rejects rollover rather than silently changing learned-position semantics.
Changing adapters requires a new runner and invalidated cache. It has no
training, continuous batching, concurrent requests or sampling API.

The decode attention is an experimental tiled online-softmax kernel, not an
official FlashAttention release. Prefill uses forced PyTorch FlashAttention.
Fully adapted historical K/V are retained; no cross-adapter cache approximation
or quantization is used. Merged-adapter graph serving is included as the strong
performance control and may be preferable for a fixed adapter.

Artifacts include raw per-request outputs/timings, graph/eager numerical gates,
context-boundary and stale-cache checks, graph setup/memory costs, source/data
hashes, paired task scores and standalone SVG/PNG speedup figures.
