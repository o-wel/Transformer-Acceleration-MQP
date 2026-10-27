# FP32 attention: measured findings

At batch 1 / prompt 896 / 64 generated tokens, pretrained FP32 eager takes 2329.8 ms; custom FP32 tiled attention takes 2134.0 ms (1.09x). With caching already enabled, that ratio is 1.01x.
The same-GPU precision bridge at this shape gives 3.60x for FP32 eager to BF16 eager, 1.75x for BF16 eager to BF16 flash, and 6.31x for the combined change.
The custom FP32 model's max logit difference from FP32 eager is 0.000534; its implementation avoids the full score matrix but is an untuned inference prototype. It matched FP32 eager tokens through EOS on 480/480 comparisons across both checkpoints and cache settings.

Keep the custom FP32 prototype, PyTorch FastF32 efficient attention, and BF16 FlashAttention distinctly labeled. The custom kernel has no backward. See attention-kernels.png, generation-speed.png and gpu-memory.png, then the per-checkpoint answer tables.
