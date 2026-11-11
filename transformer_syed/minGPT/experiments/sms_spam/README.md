# LoRA on SMS spam classification

A separate real-data task: classify SMS messages as spam or legitimate (ham).
No sports data or sports answer grader is used.

The experiment compares pretrained GPT-2, full fine-tuning, and rank-4 LoRA.
Both training methods get the same validation-only learning-rate search and
three training seeds. It also compares final five-epoch checkpoints with the
first checkpoint reaching training label NLL ≤ 0.05.

From the minGPT root, in the existing PyTorch/Transformers environment:

```bash
# Populate the GPT-2 cache once if it is not already present:
python -c "from transformers import GPT2Tokenizer; from mingpt.model import GPT; GPT2Tokenizer.from_pretrained('gpt2'); GPT.from_pretrained('gpt2')"
python experiments/sms_spam/experiment.py prepare
python experiments/sms_spam/test_protocol.py -v
sbatch experiments/sms_spam/run_gpu.sh
# After completion, with matplotlib installed:
python experiments/sms_spam/build_report.py
```

The batch wrapper assumes the author's cluster paths/account; adapt on another
system. GPU training uses CUDA and cached GPT-2 weights/tokenizer. See
[LoRA setup](../../LORA_REPRO.md) for environment installation. The UCI archive is
downloaded during preparation. Existing runs are preserved; a repeat needs a
fresh sibling experiment directory containing copies of the code files, with no `runs/` directory. Update the wrapper to point to that directory.

[RESULTS.md](RESULTS.md) and [REPORT.pdf](REPORT.pdf) are generated after completion.
All settings, data checksums, validation selection, per-seed metrics, predictions,
and change audits are retained alongside the report. Large `.pt` checkpoints
remain local and are ignored by Git.

Source: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
