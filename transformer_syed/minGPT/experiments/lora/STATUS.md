# Run status (2026-09-29)

Completed: `out/lora-sports-cpu`, 200 FP32 training steps on compute-4-02.
See [CPU results and all responses](CPU-RESULTS.md). The saved adapter is ready for interactive comparison.
All 589,824 adapter scalars changed; zero of 163,037,184 frozen base scalars changed.
Disabling adapters reproduced every original base response exactly (checked against the saved report).
The offline LoRA tests (3) and measurement tests (2) passed; a separate two-step CPU smoke test also completed.

GPU batch job: **2333468** completed successfully on gpu-5-28 (exit 0; 46 seconds allocation elapsed).
It completed 300 LoRA steps and a separate 10-step full-training memory baseline; see [GPU results](RESULTS.md).
Logs: `experiments/lora/gpu-2333468.log`. Check `squeue -j 2333468` or, after completion, `sacct -j 2333468`.
Measured GPU results are now available in the two GPU report.json files and RESULTS.md; CPU and GPU results are labeled separately.
The GPU run used 300 steps and 40 generated tokens, whereas the completed CPU demonstration uses 200 and 24.

Try the completed demo from the minGPT directory:

```bash
source ~/MQP/Torch/setup.sh
python lora_experiment.py compare --device cpu --out out/lora-sports-cpu --new-tokens 24
```

Example: `Who is the coach of the Harbor Comets?`

This is an explicitly fictional club FAQ demonstration, not broad sports conversation evaluation.
