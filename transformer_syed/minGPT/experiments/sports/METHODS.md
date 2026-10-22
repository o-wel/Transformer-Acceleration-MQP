# Explaining the implementation to the professor

## Why only LoRA trains

Each adapted projection computes

`y = W_base x + (alpha / rank) B(A(x))`.

In `mingpt/model.py`, `enable_lora()` calls `self.requires_grad_(False)` on the existing GPT, then creates new A/B layers. The new layers are trainable; the original layers, embeddings, LayerNorms and output head remain frozen. `LoRALinear` also freezes the wrapped original linear layer explicitly.

The optimizer is constructed only from parameters whose `requires_grad` flag is true:

```python
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.001, weight_decay=0.0,
)
```

This is the mechanism, not the proof: proof comes from the exact audits below. `model.eval()` changes dropout behavior and does not freeze weights.

Source: [PyTorch autograd/freeze documentation](https://docs.pytorch.org/docs/stable/notes/autograd).

## FP32 baseline

The runner loads `GPT.from_pretrained('gpt2').float()`. The adapters use `.to(linear.weight)`, inheriting the base layer's dtype/device. The runner asserts every parameter is `torch.float32`, disables TF32 for matmul/cuDNN, and uses no autocast, AMP, half conversion, or gradient scaler.

```python
assert all(p.dtype == torch.float32 for p in model.parameters())
```

FP32 stores each scalar in 32 bits = 4 bytes. Parameter storage is measured using `sum(p.numel() * p.element_size() for p in model.parameters())`, so it reflects the actual tensors.

Source: [PyTorch Module.float](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.float).

## Total, trainable and changed

`mingpt/measurement.py` counts:

```python
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
percent = 100 * trainable / total
```

`numel()` counts scalar entries, not parameter tensors. The existing minGPT startup print excludes its separate output head; use the report's counts instead. This minGPT implementation stores the input embeddings and output head as separate parameters, even though imported GPT-2 initializes them with matching values. That is why the physical base count is 163,037,184 rather than the familiar approximately 124 million tied-weight count. We preserve that implementation for this experiment and disclose it for external comparisons.

Before training:

```python
initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
```

At each audit:

```python
current = param.detach().cpu()
changed_scalars = torch.count_nonzero(current != initial[name]).item()
delta = current - initial[name]
```

`detach()` avoids graph tracking, `cpu()` keeps snapshots out of GPU memory, and `clone()` makes independent storage. We record scalar counts, maximum absolute change, and L2 change for every parameter tensor. For frozen parameters, the audit raises if any scalar changed, any gradient exists, or `requires_grad` became true. A test deliberately mutates a frozen weight and confirms detection.

Exact comparison is intentional: even a tiny represented FP32 value change counts. It is not a thresholded definition of “significant change.” A parameter that changed and later returned exactly to initialization would not count at that snapshot; this is not a cumulative update-event counter.

Source: [PyTorch tensor operations](https://docs.pytorch.org/docs/stable/tensors.html).

## Why fewer trainable scalars change on step 1

B starts at zero, while A starts random. Since the update is B times A, the initial gradient with respect to A is zero when B is zero. B can update first. On later steps, nonzero B lets gradients reach A.

For these rank-4 adapters, B contains 331,776 scalars and A contains 258,048, totaling 589,824. The observed first update changes B only. This is why “trainable” and “actually changed” must be reported separately.

## GPU memory is a different measurement

After model placement, the runner records initial memory. During training it synchronizes CUDA and queries:

```python
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

The same first 40 steps form the main LoRA/full-training comparison. The window includes lazy Adam state allocation, forward activations, backward gradients and optimizer updates. Validation runs after the memory window is captured; counters reset afterward. The whole-run LoRA peak is also saved separately.

Allocated memory tracks live PyTorch tensor allocations. Reserved memory tracks the caching allocator. They are not additive and do not include all CUDA driver/context allocations. Report them with their labels, not as total physical device use. Model parameter storage is only one component and cannot predict the runtime peak alone.

Source: [PyTorch peak allocated memory](https://docs.pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html).
