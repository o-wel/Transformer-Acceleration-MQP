# FP16/FP32: presentation notes

Start with the backend glossary and fixed-master-weight policy, then show matched comparisons.

At batch 1 / prompt 896 / 64 generated tokens, pretrained FP32 eager takes 2306.5 ms; FP32 tiled takes 2124.8 ms (1.09x). FP16 AMP eager takes 943.9 ms and FP16 Flash takes 521.0 ms (1.81x at fixed FP16 policy).
At training batch 2 / length 896, full FP32 eager takes 242.5 ms and tiled FP32 262.8 ms (0.92x); peaks are 5997 and 5168 MiB. A ratio below one means slower, even if memory improves.

All measured training updates were finite and successful. Backward support is validated, but performance and memory benefits must be read separately. This reset-step experiment does not measure convergence. Older BF16 measurements remain historical.
