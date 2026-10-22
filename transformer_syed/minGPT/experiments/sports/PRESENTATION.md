> **Updated comparison:** use [the matched full-fine-tuning versus LoRA experiment](matched/README.md) and [its presentation guide](matched/PRESENTATION.md). This earlier experiment used full fine-tuning only as a 40-step memory reference.

# Five-minute professor walkthrough

Open `results/PROFESSOR_REPORT.pdf` for the meeting and keep `results/RESULTS.html` open to inspect all raw outputs. The HTML file is self-contained and can be copied to another computer. The report is generated from actual completed runs; do not replace missing measurements with estimates.

## 1. State the research question (30 seconds)

“Can a small LoRA adapter teach pretrained GPT-2 to answer questions about unseen basketball matches while the pretrained weights stay frozen?”

Explain that the old fictional club FAQ validated memorization and parameter isolation. This new test supplies unseen match facts in the prompt and asks the model to interpret them. It is still a controlled synthetic task, not unrestricted sports conversation.

## 2. Show the split and quality results (60 seconds)

- 480 training matches, 40 validation matches, 40 held-out test matches.
- Each match has winner, halftime-leader, and comeback questions.
- Held-out team names and complete score combinations are absent from training.
- Three fixed training seeds; checkpoint selection uses validation loss only.
- Show primary-answer, answer-plus-score, and exact-reference metrics together. Say that the first two are automated field checks, not a human semantic-quality rating.
- Explain that the error bars are standard deviation across seeds, not a confidence interval.

Say only what the measured report supports. A small improvement or poor comeback accuracy is a useful limitation, not something to hide.

## 3. Demonstrate changed input → changed answer (90 seconds)

From the minGPT directory:

```bash
source ~/MQP/Torch/setup.sh
python sports_demo.py --device cpu --interactive
```

Enter two team names, halftime scores, final scores, and a question type. Both models receive exactly the same facts. The reference is printed after the model outputs.

For a repeatable example:

```bash
python sports_demo.py --teams Bulls Sharks --half 51 35 --final 96 69 --question winner
python sports_demo.py --teams Bulls Sharks --half 51 35 --final 69 96 --question winner
```

The correct winner switches from Bulls to Sharks when the final scores swap. This is the first held-out match, and seed 42 answered both versions correctly on GPU. It is a measured demonstration, not an estimate of all-task performance. The PDF states exactly how its examples were selected. Also show the comeback failure in `live-demo.txt`: seed 42 gave the same No answer after halftime scores swapped, exposing a real limitation.

Invite the professor to change a score or choose new names. If the model fails, compare the supplied facts with the reference and discuss the limit. CPU loading/decoding takes a moment; start the interactive process before the meeting. For a GPU demo:

```bash
srun --pty -A isp -p academic --qos=short --gres=gpu:1 --cpus-per-task=4 --mem=12G --time=00:15:00 \
  python sports_demo.py --device cuda --interactive
```

## 4. Explain parameter and memory evidence (60 seconds)

“We distinguish total stored parameters, trainable parameters, and scalar values actually different from initialization. The base is frozen using `requires_grad_(False)`, the optimizer contains only adapter weights, and exact snapshots verify zero base changes.”

Show the changed-scalar chart: A starts random and B starts zero, so the first update changes B but not A. Subsequent updates can change both. Show the actual snapshot counts rather than assuming all trainable scalars changed.

“FP32 parameter storage is four bytes per scalar. It is not training memory. We also measured peak PyTorch GPU allocated and reserved memory during matched first-40-step windows, including gradients, optimizer states, activations and workspaces.”

Do not add reserved and allocated memory, and do not call either whole-device memory. The 40-step full-training run is a memory reference; its quality is not compared with 1,200 LoRA steps.

## 5. Show a failure and propose the next experiment (40 seconds)

Use the report's first-failure example and the searchable raw outputs. State that sentence templates can be learned even when full answers are new. Correct counterfactual pairs strengthen the evidence that outputs depend on input facts, but do not prove general reasoning.

Next research steps: held-out question wording, new score ranges, real match summaries, a blinded human factuality rubric, then sequence-length sweeps. Keep FP32 fixed for A-term. Add FlashAttention and KV caching as separately controlled factors later; do not attribute their benefits to LoRA.

## Files to bring

- `results/PROFESSOR_REPORT.pdf`: concise report, figures and measured examples.
- `results/RESULTS.html`: searchable, standalone viewer containing every base/LoRA output for all seeds and paired tests.
- `results/ALL_OUTPUTS.csv`: raw responses and automated scoring fields.
- `results/RESULTS.md`: numerical summary including checkpoint steps and losses.
- `README.md`: exact experimental and measurement definitions.
- `out/sports-seed42/adapter.pt`: adapter used by the live demo; cached pretrained GPT-2 weights/tokenizer are also required.
