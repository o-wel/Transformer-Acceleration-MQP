![mechanism](mechanism.png)

![generation-speed](generation-speed.png)

![gpu-memory](gpu-memory.png)

**KV caching: implementation and validation**

**Purpose**

Reduce repeated work during autoregressive generation while preserving model outputs. This is an inference optimization: no training, extra learned parameters, or weight updates are required.

**How it works in this minGPT**

The first forward pass processes the full prompt (prefill). Each of the 12 transformer layers saves its attention keys and values. Subsequent steps process only one new token; its query attends to the saved history. Cached length offsets both the causal mask and learned position embeddings. The cache grows as generation proceeds.

**Implementation**

Opt-in API: model.eval(); model.generate(ids, 64, use_cache=True). Direct forward calls accept past_key_values and return a per-layer cache. Existing training and checkpoint formats remain compatible. Both benchmark modes compute only final-position output logits, avoiding a separate output-head optimization as a confound.

**Validation on real checkpoints**

| Checkpoint | Identical outputs | Accuracy: off / on | Max logit difference |
|---|---|---|---|
| pretrained | 100/120 | 23.3% / 21.7% | 0.250000 |
| full | 119/120 | 17.5% / 17.5% | 0.031250 |
| lora | 96/120 | 23.3% / 23.3% | 0.250000 |

**Meaning of the checks**

Cached and uncached generation was evaluated for all 120 held-out sports prompts in each checkpoint. These are seed-42 checkpoints; accuracy here is not a three-seed average. Chunked FP16 logits were compared using the tolerances recorded in report.json. Before/after parameter hashes matched for every model.

**Additional tests**

All 18 repository tests passed, including four KV-cache tests. Coverage includes chunked decoding, nonzero LoRA adapters, batched generation, seeded sampling, context-window overflow, independent requests, normal training, and checkpoint compatibility. The CPU demo also reloads the trained adapter and compares both generation paths.

**Inference speed: measured results**

**Configuration**

NVIDIA GeForce RTX 4070 Laptop GPU; FP16; TF32 off; greedy decoding; 64 new tokens per sequence. Two warmups and five measured repetitions per mode/configuration, with alternating mode order. Prompts contain deterministic random token IDs to control lengths. Timings exclude model loading, tokenization, input transfer, and warmups. Job None on OH-MSI.

**Pretrained GPT-2: request time and aggregate throughput**

| Batch; prompt | No cache (ms) | Cache (ms) | Speedup | Tokens/s: off / on |
|---|---|---|---|---|
| 1; 32 | 812.4 +/- 69.5 | 803.7 +/- 31.1 | 1.01x | 79.2 / 79.7 |
| 1; 128 | 781.4 +/- 55.6 | 791.5 +/- 25.3 | 0.99x | 82.2 / 80.9 |
| 1; 512 | 760.0 +/- 20.9 | 801.4 +/- 25.3 | 0.95x | 84.3 / 79.9 |
| 4; 128 | 802.6 +/- 7.7 | 850.1 +/- 67.4 | 0.94x | 319.0 / 302.6 |

**Trained LoRA: request time and aggregate throughput**

| Batch; prompt | No cache (ms) | Cache (ms) | Speedup | Tokens/s: off / on |
|---|---|---|---|---|
| 1; 32 | 1200.9 +/- 7.3 | 1196.8 +/- 24.8 | 1.00x | 53.3 / 53.5 |
| 1; 128 | 1437.2 +/- 117.3 | 1476.0 +/- 117.2 | 0.97x | 44.8 / 43.6 |
| 1; 512 | 1259.9 +/- 209.5 | 1227.7 +/- 28.5 | 1.03x | 51.7 / 52.2 |
| 4; 128 | 1217.9 +/- 31.7 | 1309.1 +/- 60.1 | 0.93x | 210.3 / 195.9 |

**Reading these tables**

Batch is the number of sequences generated together; prompt is tokens per sequence. Request time is synchronized wall time to generate all 64 tokens per sequence. +/- is standard deviation across five timing repetitions, not a confidence interval. Throughput counts output tokens across the whole batch. These are preliminary measurements on one allocated GPU.

**Interpretation**

Caching provides its strongest measured benefit for longer prompts and larger batches. The shortest batch-one case is approximately break-even or slightly slower, so caching is not a universal speedup. Small requests offer less repeated work to eliminate while still paying cache management and execution overhead. This explanation is an inference from scaling behavior, not a kernel-level bottleneck diagnosis.

**Scope of the comparison**

Both pretrained and LoRA architectures are benchmarked. Full fine-tuning uses the same inference architecture as pretrained GPT-2; its outputs were verified, but no separate full-checkpoint timing claim is made. LoRA adapters remain unmerged. There is no FlashAttention, FP16 autocast, compilation, or serving scheduler.

**Prompt latency, decode latency, and GPU memory**

**Pretrained GPT-2: prefill and decode**

| Batch; prompt | Prefill: off / on (ms) | Decode: off / on (ms/token) |
|---|---|---|
| 1; 32 | 29.74 / 30.29 | 12.42 / 12.27 |
| 1; 128 | 26.39 / 29.77 | 11.98 / 12.09 |
| 1; 512 | 21.94 / 22.39 | 11.71 / 12.36 |
| 4; 128 | 24.87 / 30.51 | 12.34 / 13.01 |

**Trained LoRA: prefill and decode**

| Batch; prompt | Prefill: off / on (ms) | Decode: off / on (ms/token) |
|---|---|---|
| 1; 32 | 31.75 / 32.90 | 18.55 / 18.47 |
| 1; 128 | 35.68 / 38.49 | 22.24 / 22.81 |
| 1; 512 | 37.97 / 37.08 | 19.39 / 18.90 |
| 4; 128 | 38.21 / 38.09 | 18.72 / 20.17 |

**Timing definitions**

CUDA-event prefill time includes processing the prompt and choosing/appending the first output token. Decode latency is the subsequent 63 steps divided by 63; for batch four it is latency per batched token step, not per individual sequence token. Caching primarily reduces subsequent decode work, not initial prompt processing.

**Peak allocated GPU memory and logical cache storage (MiB)**

| Model; batch; prompt | No cache peak | Cache peak | KV storage |
|---|---|---|---|
| pretrained; 1; 32 | 757.87 | 763.69 | 3.34 |
| pretrained; 1; 128 | 758.71 | 770.44 | 6.71 |
| pretrained; 1; 512 | 762.79 | 802.83 | 20.21 |
| pretrained; 4; 128 | 762.66 | 823.77 | 26.86 |
| lora; 1; 32 | 760.99 | 766.81 | 3.34 |
| lora; 1; 128 | 761.84 | 773.56 | 6.71 |
| lora; 1; 512 | 765.92 | 805.96 | 20.21 |
| lora; 4; 128 | 765.78 | 828.92 | 26.86 |

**Memory interpretation**

Peak allocated memory includes model weights, attention masks, inputs, caches, and live temporary tensors. Reserved-memory samples are also saved in JSON/CSV; do not add reserved and allocated values. Logical FP16 cache size is 2 x layers x batch x cached_tokens x hidden_width x 2 bytes. Dynamic concatenation can temporarily retain both old and new cache tensors.

**Demonstration, limitations, and reproduction**

**A real held-out example**

Prompt: Bulls vs Sharks. Halftime 51-35; final 96-69. Who won? The trained LoRA model answers "Bulls won 96-69." with caching off and on. The fully fine-tuned model retains its incorrect "Rockets won 96-69." answer in both modes. Caching preserves the existing model behavior; it does not improve reasoning or correct factual mistakes.

**The context-window boundary**

This GPT-2 uses 1,024 learned absolute positions. Once the window fills, existing minGPT crops old tokens and resets positions. Dropping only the oldest cached K/V would not preserve that behavior. Cached generate therefore rebuilds from the cropped window at saturation. Correctness is preserved, but the incremental speed benefit is lost after saturation. All measured performance cases stay within the context window.

**Other practical limits**

Caches require evaluation mode and disabled autograd. Use equal-length batches; padding masks and per-sequence cache lengths are not implemented. Discard a cache when changing the prompt, weights, or LoRA adapter settings. Five timing repeats and a synthetic task do not establish production serving performance.

**Try the models**

From ~/MQP/MinGPT/minGPT after sourcing ~/MQP/Torch/setup.sh: python kv_cache_demo.py --model lora --device cpu. Use --model full or --model pretrained for other checkpoints. On an allocated GPU use --device cuda. The demo prints both answers and whether generated tokens match; its single-request timings are illustrative.

**Reproduce the experiment**

Run: sbatch experiments/kv_cache/run_gpu.sh. Focused tests: python -m unittest discover -s tests -p test_kv_cache.py -v. Saved report.json includes raw timing samples, environment details, source hashes, before/after parameter hashes, and checkpoint paths. The separate *-quality.json files contain every cached/uncached answer and score.

**Finding for the meeting**

KV caching is implemented and validated for base, fully fine-tuned, and LoRA models. It preserved all tested generated outputs and all model weights. It accelerated larger inference workloads while increasing measured peak GPU memory in these tests. The short-prompt and context-limit cases show where the benefit is limited.
