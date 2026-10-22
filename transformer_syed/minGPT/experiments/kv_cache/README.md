# KV caching in minGPT

KV caching accelerates autoregressive inference without training or modifying weights. The original model recomputes every prompt token at each generation step. The cached model processes the prompt once (**prefill**), stores each layer's attention keys and values, then processes just the new token (**decode**). That token's query attends to the stored history and its own key/value.

Each layer stores K and V with shape `(batch, heads, cached_tokens, head_width)`. Attention masks are offset by the cached length, as are GPT-2's learned position embeddings. A multi-token continuation uses a causal mask within the new chunk, so it cannot see future tokens. This is ordinary attention, not FlashAttention.

## Usage

The existing training API and uncached generation remain available. Caching is opt-in:

```python
model.eval()
output = model.generate(prompt_ids, max_new_tokens=64, use_cache=True)
```

For explicit prefill and decode:

```python
with torch.inference_mode():
    logits, _, cache = model(prompt_ids, use_cache=True, last_token_only=True)
    next_token = logits[:, -1].argmax(-1, keepdim=True)
    logits, _, cache = model(next_token, past_key_values=cache,
                            use_cache=True, last_token_only=True)
```

Normal `model(input_ids, targets)` still returns `(logits, loss)`. Cache-enabled calls return `(logits, None, cache)`. Caching requires `eval()` and disabled autograd; caches belong to the caller and are not model parameters or checkpoint entries. Discard caches when changing prompts, model weights, or LoRA adapter settings. Batches must have equal real sequence lengths; padding masks and per-example cache lengths are not implemented.

**Context limit:** GPT-2 has 1,024 learned positions. Existing minGPT generation crops a long sequence and renumbers positions from zero. Reusing old cached keys after cropping would not preserve that behavior. `generate()` therefore rebuilds the entire cropped window whenever the cache is full. It stays correct, but loses the incremental speed benefit after saturation. Direct `forward()` rejects an overflowing cache.

## Verification

```bash
source ~/MQP/Torch/setup.sh
cd ~/MQP/MinGPT/minGPT
python -m unittest discover -s tests -p 'test_kv_cache.py' -v
python kv_cache_demo.py --model lora --device cpu
```

Unit tests cover chunked and single-token logit parity, batch size two, nonzero LoRA adapters, greedy and seeded sampled generation, prompts beyond the context limit, repeated independent requests, unchanged training/checkpoint behavior, and invalid cache usage. The complete 18-test suite also passed, including the Hugging Face import comparison.

The GPU experiment compares cached and uncached outputs on all 120 held-out sports questions for each of the pretrained, fully fine-tuned, and LoRA seed-42 models. It saves raw outputs, primary-answer and score-field metrics, logit differences, and before/after parameter hashes. FP32 computations of different matrix shapes can differ slightly from floating-point rounding; numerical logit parity uses absolute/relative tolerances of 0.0002, while generated token parity is checked separately.

## GPU benchmark protocol

```bash
sbatch experiments/kv_cache/run_gpu.sh
```

- One allocated NVIDIA GPU; FP32; TF32 disabled; no mixed precision, FlashAttention, compilation, or adapter merging.
- Base GPT-2 and the trained LoRA model; full fine-tuning has the same inference architecture as base GPT-2, and is separately checked for output correctness.
- Batch 1 with 32, 128, and 512 prompt tokens; batch 4 with 128 prompt tokens; 64 generated tokens per request.
- Deterministic random token prompts isolate length and batch effects. Sports questions provide the separate meaningful-output check.
- Two warmup requests per mode/configuration; five measured requests per mode, alternating cached/uncached order. These repeats measure runtime variation, not independent training runs.
- Both paths compute LM-head logits only for the final position. The measured loop uses greedy argmax and does not stop at EOS, so both paths perform exactly the same output-token budget.
- CUDA-event prompt processing time includes choosing/appending the first generated token. Decode time covers the remaining 63 token steps; decode latency is divided by 63, not 64.
- Synchronized request wall time includes Python generation overhead; aggregate throughput is `batch * 64 / seconds`. Tokenization, model loading, prompt transfer, warmups, and cache cleanup are outside timing.
- Peak allocated and reserved CUDA memory are reset per request after clearing unused allocator blocks. Allocated includes model, masks, inputs, cache, and live temporaries. Reserved is not added to allocated; neither is a whole-driver/device measurement.

The logical FP32 KV cache size is `2 * layers * batch * cached_tokens * hidden_width * 4` bytes. For this model it is **72 KiB per cached token per sequence**. The final prediction of 64 generated tokens has cached `prompt_length + 63` tokens: the final output token need not be fed back into the model. Dynamic concatenation can transiently hold old and new caches; logical cache bytes are not the same as peak runtime memory.

Saved results and a generated meeting report go in `results/`. Inference speedup should preserve output quality, not improve it. Effects of cache storage and reduced intermediate computation must be measured rather than assuming total GPU memory increases or decreases.
