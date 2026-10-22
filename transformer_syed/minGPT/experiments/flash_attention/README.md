# FlashAttention measurements

The real-checkpoint A30 experiment is run with `sbatch experiments/flash_attention/run_gpu.sh`.
It measures pretrained GPT-2 and the trained seed-42 LoRA adapter, checks task quality
on pretrained/full/LoRA checkpoints, and measures full/LoRA AdamW training costs.
This is separate from the configurable synthetic smoke/architecture benchmark below.
The runner refuses to overwrite existing `results/report.json` evidence.

A second job isolates attention-only CUDA execution:
`sbatch --dependency=afterok:JOB_ID experiments/flash_attention/run_kernels.sh`.
After both finish, run `python experiments/flash_attention/build_report.py` in an
environment with Matplotlib and NumPy. On this account, the report environment is
`/home/sashazli/torch_env/bin/python`.

The saved report is [results/FLASH_ATTENTION_REPORT.pdf](results/FLASH_ATTENTION_REPORT.pdf),
with [Markdown](results/RESULTS.md), a standalone [HTML report](results/RESULTS.html),
[metrics CSV](results/METRICS.csv), and [presentation notes](results/PRESENTATION.md).
Raw samples, quality outputs, checkpoint fingerprints, source snapshots, and CUDA
traces are retained under `results/`.

The original explicit attention remains the default. Select a backend at construction
with `config.attention_backend`, or on a loaded model (including pretrained/LoRA models):

```python
model.set_attention_backend('flash')
```

`eager` is the original matmul/mask/softmax implementation. `math` forces PyTorch
SDPA's math backend; `flash` forces PyTorch's built-in CUDA FlashAttention backend.
`sdpa` allows automatic selection and must not be presented as proof of FlashAttention.
No separate `flash-attn` package is needed. This measures PyTorch's FlashAttention
implementation, not a pinned Dao-AILab FlashAttention release. Requires modern
PyTorch with `torch.nn.attention.sdpa_kernel` and `causal_lower_right` support.
Backend choice is not stored in the state dict; select it again after loading.

```bash
python experiments/flash_attention/benchmark.py --profile
```

Defaults compare eager/math/flash at prompt lengths 128, 512, 1024, batch 2,
bfloat16, 6 layers, 384 embedding width, 6 heads, vocabulary 4096. For a GPT-2-sized
architecture use `--layers 12 --heads 12 --width 768 --vocab-size 50257` (requires
more memory). Use an identical dtype for all paired backends. The existing KV-cache
experiment uses FP32; comparing those results directly with BF16 flash results
would conflate precision and kernel changes. Unsupported flash configurations are
reported under `unavailable`, never silently rerun using a different backend.
CUDA availability is checked before the benchmark starts.

The matrix covers base and active, unmerged LoRA (rank 8 by default), each with:

- Training: forward, loss, backward, and a stateless SGD update. Base trains all
  parameters; LoRA trains adapters. Dropout is disabled. This isolates attention
  and gradient costs; it does not estimate AdamW optimizer-state memory.
- Prefill: prompt forward with only the final-token vocabulary projection.
  This is model prefill latency, not user-facing time to first token.
- Uncached decode: repeatedly forward the entire growing context.
- Cached decode: prefill outside the timed region, then one-token forwards with
  growing K/V. Both decode paths consume the same predetermined continuation,
  keeping input shapes and tokens identical across kernels.

Decode processes `--new-tokens` additional input tokens and excludes token sampling,
text tokenization, transfers, and prefill. It is a teacher-forced decode benchmark,
not end-to-end `generate()` latency. All cases fit inside the context window, so no
cache rebuilds occur. Cache concatenation overhead is included in cached decode.

JSON reports contain raw repetitions, median/mean/stdev/range, synchronized wall
latency, CUDA-event elapsed latency, tokens/second, milliseconds/decode step,
peak allocated/reserved GPU bytes, baseline allocation, additional peak allocation,
cache bytes, and total/trainable parameter counts. CUDA event intervals include
stream gaps from host dispatch; they are not sums of kernel runtimes. Memory is
PyTorch allocator memory, not total device memory. Reserved memory reflects
allocator history; use allocated peaks for backend comparisons. Training baseline
excludes gradients; optimizer buffers are absent with stateless SGD.

Backend order rotates each repetition, after per-backend warmups. All training
steps restart from identical weights. Speedups divide median eager wall time by
median candidate wall time; `interactions` additionally reports cache speedup at a
fixed backend, combined backend+cache speedup versus eager uncached, and LoRA
speedup at a fixed backend. LoRA's main benefit is training parameter/gradient
reduction; unmerged adapters can increase inference latency. Do not multiply
independent speedups: use the measured combinations. Flash can help long prefill
more than one-token decode; KV caching already removes most repeated computation.

`--profile` writes a Chrome trace per configuration. Open in Perfetto or the Chrome
trace viewer to inspect CUDA kernels and SDPA forward/backward dispatch. Profiling
runs are excluded from timing samples. Forced flash selection prevents fallback;
traces provide evidence of which kernels executed. For kernel occupancy, bandwidth,
or hardware counters, use Nsight Compute separately because instrumentation changes
timing. No FLOP/s or bandwidth values are inferred from wall time.

Inputs and weights are seeded synthetic data; this benchmark does not measure task
quality or claim trained-adapter quality. Logit parity and cached/uncached logit
parity are checked at each configuration, with relaxed BF16/FP16 tolerances. Small
rounding differences can still change greedy text. Run the sports quality evaluation
with the selected backend on actual checkpoints before making quality claims.

CPU smoke test (no FlashAttention performance conclusions):

```bash
python experiments/flash_attention/benchmark.py --device cpu --dtype float32 \
  --backends eager math --lengths 8 --batch-size 1 --new-tokens 3 \
  --layers 1 --heads 2 --width 16 --vocab-size 31 --warmup 1 --repeats 2 \
  --output /tmp/mingpt-fa-smoke.json
python -m unittest discover -s tests -p 'test_flash_attention.py' -v
```

Tests compare gradients and cached chunk masks with nonzero LoRA updates, check
context-window rebuilds, and run forced CUDA flash parity when CUDA is available.
The existing MLP lambda was replaced with a method so deep-copied benchmark models
use their own MLP parameters and gradients.

PyTorch references: [SDPA semantics and dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html),
[backend selection](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html),
[non-square causal alignment](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html).
