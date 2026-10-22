# Corrected experiment: full fine-tuning versus LoRA

The earlier comparison showed LoRA against an untouched pretrained model, with only a 40-step full-training memory reference. It did not compare the quality of two trained methods. This experiment supersedes that comparison for full-versus-LoRA claims.

Both methods start from identical pretrained GPT-2 weights and train on the same existing sports data. Each candidate runs 1,200 steps, batch 8, padded length 128, FP32 (TF32 off), AdamW, zero weight decay, gradient clipping 1.0. Initial-weight and sampled-batch SHA256 fingerprints verify matching weights and exposure for paired seeds. Both methods run sequentially on the same allocated A30 GPU.

## Equal learning-rate search

Before seeing new test outputs, use these two rates for **each** method: 0.001 and 0.0001. Train each for 1,200 steps on seed 42, recording validation answer NLL at 300/600/900/1200. Select each method's best rate/checkpoint using only validation NLL. Save both choices to `selection.json` before evaluating test outputs. Evaluate the selected seed-42 checkpoints and train seeds 43/44 at their method's selected rate, again selecting their checkpoint by validation loss.

That is four training runs per method: two rate candidates plus seeds 43 and 44. The chosen seed-42 candidate is reused, not retrained. Report the three selected runs per method and show all four tuning candidates' validation results. This is a small, equal-effort search, not exhaustive optimization. The test set was already disclosed in the previous experiment; it is not a newly blinded benchmark. No new test-based tuning is used.

Pretrained GPT-2 remains an untrained quality reference. Full fine-tuning updates every parameter; LoRA updates only its low-rank matrices. Both receive the same new training data. Seed 42 is the live-demo model for each method; never pick a seed for its test score.

## Run

```bash
source ~/MQP/Torch/setup.sh
cd ~/MQP/MinGPT/minGPT
sbatch experiments/sports/matched/run_gpu.sh
```

Run directories start with `out/matched-`. Existing directories and previous results are preserved. Reruns require new output paths.

After completion, generate the report:

```bash
python -m pip install --target /tmp/mqp-report-deps matplotlib
PYTHONPATH=/tmp/mqp-report-deps python experiments/sports/matched/build_report.py
```

Run `python experiments/sports/matched/verify.py` for the independent acceptance checks. The report checks identical data/code, initial weights, sample order, budgets and GPU type between matched runs. It reports quality, validation selections, total/trainable/changed scalar counts, parameter storage, allocated/reserved GPU peaks, and synchronized training-step time. Timing excludes data transfers, audits, validation, checkpoint serialization and test generation; measured training wall time is retained separately. Full-model checkpoint I/O is much larger than adapter checkpoint I/O, so do not confuse these timing scopes. Warm steps/second is also recorded in JSON.

Peak training memory is the maximum across training windows over all 1,200 steps, measured before validation and reset afterward. It is the absolute PyTorch allocator peak, not an increment above model storage and not whole-device memory. The same first-40-step measurement remains in the raw JSON for reference. All runs process 1,228,800 padded token positions; actual supervised token counts match by seed and are recorded separately.

Changed counts compare individual FP32 values at the selected checkpoint with initialization. Full fine-tuning can leave some trainable values unchanged (for example unused input-embedding rows). LoRA's frozen base must remain exactly unchanged. The physical model count includes minGPT's separate output head, unlike tied-weight GPT-2.

## Show the professor

Open `results/PROFESSOR_REPORT.pdf`, then `results/RESULTS.html` to inspect all three outputs on any test question. `ALL_OUTPUTS.csv` has 720 rows: 3 seeds × (120 test + 120 paired questions), each with all three model outputs.

```bash
python sports_demo.py --matched --device cpu --interactive
```

For faster interactive use:

```bash
srun --pty -A isp -p academic --qos=short --gres=gpu:1 --cpus-per-task=4 --mem=12G --time=00:15:00 \
  python sports_demo.py --matched --device cuda --interactive
```

Use the report's measured examples, then change scores live. Compare full fine-tuning with LoRA directly. Show a disagreement or failure as well as successes. Do not claim LoRA matches full quality merely because their scores look close; three seeds on a small synthetic benchmark do not establish equivalence. If one method wins on quality, present the actual quality/resource trade-off.

The existing dataset and scoring limitations still apply: [dataset protocol](../README.md), [implementation methods](../METHODS.md). Field checks are format-sensitive; exact reference matches can reject valid paraphrases. Questions share matches. This is controlled task adaptation, not unrestricted sports-chat evaluation.

## Epoch versus training-loss stopping

`sbatch experiments/sports/matched/run_stopping.sh` runs both methods and seeds
42/43/44 under two rules: seven complete shuffled epochs, or full-training-set
answer NLL ≤ 0.05 checked at each epoch boundary, with a seven-epoch cap.
The target is an exploratory fixed choice, not selected from test accuracy.
Batch size is eight, so 1,440 examples give 180 steps per epoch and a
1,260-step cap. Each pair uses the same initialization and shuffled example
order. Learning rates come from the existing validation-only search.
Both rules evaluate the final checkpoint, so validation selection cannot hide
the effect of stopping. The original 1,200-step results are preserved.
Raw reports, stopping reasons, losses, timings and held-out outputs are saved
under `experiments/sports/stopping/`; `RESULTS.md` compares accuracy across seeds.
Training loss measures fit to the training answers; accuracy is measured on
held-out generated answers. KV caching affects inference and does not define
the training stopping rule.

After all runs finish, save the accuracy comparison, training budget and loss
curves with `python experiments/sports/matched/plot_stopping.py` (requires
matplotlib). Graphs are kept in `experiments/sports/stopping/graphs/` as PNG
and SVG, with all figures in `stopping-comparison.pdf`. Dots show individual
seeds; error bars are sample standard deviations. Existing graphs from other
experiments are preserved.

## Added diagnostic (exploratory)

After inspecting the selected seed-42 full model's unseen-team failures, we added a fixed name-substitution control. The same 120 held-out questions and scores are evaluated with Bulls→Bears, Eagles→Comets, Owls→Falcons, Sharks→Hawks. These replacement names occur in training. Neither model is retrained or retuned. Reference-answer overlap with training is explicitly counted. This test is post-hoc and does not replace the original unseen-team benchmark. Run `sbatch experiments/sports/matched/run_control.sh` after the main experiment completes. Results and all raw outputs are in `familiar-control.json`; the report includes a separate diagnostic section.
