# Completed experiment status

Slurm job **2333691** completed successfully (exit 0) on **gpu-4-22**, NVIDIA A30, in **8 minutes 42 seconds**.
Account `isp`, partition `academic`, job QoS `short`.

All three 1,200-step runs (seeds 42/43/44) and the separate 40-step full-training memory reference completed.
All three checkpoints were selected at step 1,200 by minimum validation NLL.
Every frozen-base audit passed. Data/code hashes agree across all runs.
All held-out complete reference answers are absent from training; raw generated repeats are reported separately.
Base outputs agree across seeds. Batched-vs-single decoding checks passed. The CSV contains all 720 seed/question/pair rows (both model outputs per row).

The live seed-42 adapter was reloaded on CPU. Both the successful winner-score swap and unsuccessful comeback-score swap are preserved in `live-winner-demo.txt` and `live-demo.txt`.

Four sports tests passed: split isolation, scoring/counterfactual logic, answer-only target masking, and batched generation consistency. The two shared parameter-audit tests also passed.

Start with `results/PROFESSOR_REPORT.pdf`, then `PRESENTATION.md`. `results/RESULTS.html` contains every output and can be opened locally without a server.
