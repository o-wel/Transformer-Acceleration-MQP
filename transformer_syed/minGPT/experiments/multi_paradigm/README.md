# Testing all three paradigms

This will let us test LoRA, KV cache and FlashAttention all together or on/off. It will also making hyperparameter testing easier.

#### To Run:
```
python benchmark.py --lora --kvcache --flash
```