> **Updated comparison:** use [the matched full-fine-tuning versus LoRA experiment](matched/README.md) and [its presentation guide](matched/PRESENTATION.md). This earlier experiment used full fine-tuning only as a 40-step memory reference.

# Unseen-match LoRA experiment

This extends the frozen-weight/FAQ validation with conditional answers about new fictional basketball matches. It does not claim broad sports-chat competence.

## Protocol fixed before test evaluation

- Generate data with `python experiments/sports/data.py` (seed 20260929).
- Split matches first: 480 train, 40 validation, 40 test; three question types each. Never split questions from one match across splits.
- Training/validation use eight team names; testing uses four different names. Numeric score quadruples do not overlap across splits. No test reference answer occurs verbatim in the training references.
- Test questions use prompt forms also seen in training. This measures new-match/name generalization within a controlled task, **not** unseen instructions or arbitrary dialogue.
- Counterfactual pairs reverse final scores for winner questions, or halftime scores for halftime/comeback questions. The final result is unchanged for comeback pairs. The correct primary answer flips in every pair. Counterfactuals are deliberately related to test examples, not independent samples.
- FP32, TF32 disabled, no AMP/FlashAttention/KV cache. GPT-2 imported into minGPT with untied stored embeddings/head; all 163,037,184 base scalars freeze.
- Rank 4, alpha 8, all four projection groups per block, 589,824 adapter scalars; AdamW lr .001, no weight decay, clipping 1.0, batch 8, padded length 128, 1,200 steps.
- Three predeclared seeds: 42, 43, 44. Choose each checkpoint by validation answer NLL at steps 300/600/900/1200. Do not select using test outputs. Seed 42 is the demonstration model, regardless of relative test scores.
- Greedy decoding, maximum 40 new tokens, same prompt for base and LoRA. Batched decoding uses right-padding strictly after each actual prefix, hidden by causal attention. The first three LoRA outputs are checked against independent single-example decoding.

## Run

```bash
source ~/MQP/Torch/setup.sh
cd ~/MQP/MinGPT/minGPT
sbatch experiments/sports/run_gpu.sh
```

The batch script uses your authorized `isp` account on the academic partition (job QoS `short`), which has A30 GPUs. The batch job runs the three seeds, then a separate 40-step full-training memory reference with identical batch, length, data and seed. Existing run directories are never overwritten. To repeat, choose new `--out` paths in the runner or a copied batch script.

Results live in `out/sports-seed{42,43,44}/report.json`, `adapter.pt`, and exact audit snapshots `changes-*.json`. `selected-changes.json` audits the checkpoint actually evaluated. `out/sports-full-memory/report.json` contains the full-training memory reference.

## Scoring and limitations

- Primary answer: for winner/halftime, first sentence names the correct team and not the other team, with a simple negative-word guard; for comeback, leading Yes/No is correct. This is format-sensitive and not an exhaustive semantic grader.
- Answer + scores: primary correct and the exact multiset of expected score numbers, with no extra numbers. This allows punctuation/order variation but does not prove all score-to-team relationships are correct.
- Exact reference match: identical words/numbers after case, punctuation and whitespace normalization. Strong for this controlled target format, but can reject valid paraphrases.
- Paired accuracy: both original and score-swapped primary answers are correct; also record a stricter paired answer+scores metric. Merely changing the answer does not earn credit.
- Keep raw outputs. Inspect successes and failures rather than presenting any single automated score as broad quality or reasoning ability.
- Outputs may reuse sentence templates. The evidence against whole-answer memorization is unseen names/matches, absent complete training answers, and correct paired score edits. This is compositional task transfer within the generator's distribution, not proof of general reasoning.
- Report mean and sample SD across three training seeds. The 120 questions correspond to 40 independent generated matches; do not treat the 120 paired counterfactuals as additional independent matches.

## Memory, parameters and runtime

Parameter storage counts actual scalar storage including the separate minGPT output head. FP32 means 4 bytes per scalar. Buffers are separate. Audits compare exact values with independent CPU snapshots and require every frozen tensor to have `requires_grad=False`, no gradient, and zero changes. Changed counts are differences from initialization at each audit, not a count of all intermediate update events.

Peak allocated/reserved CUDA memory is captured during training windows, before validation, with peaks reset after validation and cache clearing. All peak values are absolute allocated/reserved bytes, not deltas. Main comparison uses the **same first 40 steps** in separate processes on the same GPU. This includes optimizer initialization, gradients, activations and PyTorch workspaces. These are PyTorch allocator measurements, not whole-device/driver measurements. Do not add allocated and reserved values. CPU snapshots do not consume GPU memory.

The full baseline is for memory only, not a matched quality comparison. JSON includes per-step synchronized timing; warm throughput excludes the first 10 steps and excludes data transfer, audits, validation, and decoding. No inference latency or throughput claim is made from toggling adapters in one resident model.

## Live demo

```bash
python sports_demo.py --device cpu --interactive
# Or one reproducible input:
python sports_demo.py --teams Eagles Owls --half 30 40 --final 85 80 --question comeback
# Change only halftime scores: the Yes/No answer should flip.
python sports_demo.py --teams Eagles Owls --half 40 30 --final 85 80 --question comeback
```

Use a GPU allocation and `--device cuda` for faster live responses. The supplied facts are included in both model prompts; reference answers are computed separately and are never included in their prompts.

## Presentation artifacts

After all runs finish, generate PDF, PNG plots, a Markdown summary, all-output CSV, and a self-contained searchable HTML viewer:

```bash
python -m pip install --target /tmp/mqp-report-deps matplotlib
PYTHONPATH=/tmp/mqp-report-deps python experiments/sports/build_report.py
```

The plotting dependency is separate from the PyTorch environment. Outputs are in `experiments/sports/results/`. See `PRESENTATION.md` for the meeting sequence.

Sources: [PyTorch freezing](https://docs.pytorch.org/docs/stable/notes/autograd), [parameter dtype conversion](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.float), [peak memory](https://docs.pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html), [original LoRA implementation and generation evaluations](https://github.com/microsoft/LoRA).
