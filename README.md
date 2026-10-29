# MQP Project intro: LLMs Acceleration

This repo will contain our work and demonstrations for this project.

This README will be updated as we go.

## Combined optimization correctness test

Run LoRA, forced CUDA FlashAttention, and KV caching together:

```bash
cd transformer_syed/minGPT
python -m unittest discover -s tests -p 'test_flash_attention.py' -k combined -v
```

This offline test uses a tiny model with nonzero, unmerged LoRA adapters. It
compares adapter gradients and cached prefill/decode logits against eager
attention, then checks cached versus uncached generation across context-window
overflow. It requires a CUDA GPU and PyTorch with FlashAttention support; without
CUDA it reports a skip. This is a correctness test, not a performance benchmark.
