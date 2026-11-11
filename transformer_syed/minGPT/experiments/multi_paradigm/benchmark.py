"""Fine-tuning and inference benchmark for the sports question-answer task.

The benchmark runs one training job followed by held-out generation.  LoRA,
KV-cache generation, and forced FlashAttention are independent command-line
options so runs can be compared from the resulting report.json files.
"""
import argparse
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers import GPT2Tokenizer

from mingpt.measurement import parameter_counts
from mingpt.model import GPT
from experiments.sports.scoring import aggregate, score


def prompt(question):
    return f"User: {question}\nAssistant:"


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def encode(rows, tokenizer, length):
    encoded = []
    for row in rows:
        prefix = tokenizer.encode(prompt(row["question"]))
        ids = prefix + tokenizer.encode(" " + row["answer"]) + [tokenizer.eos_token_id]
        if len(ids) > length + 1:
            raise ValueError(f"Example {row['match_id']} exceeds --length={length}")
        padding = length - len(ids) + 1
        x = ids[:-1] + [tokenizer.eos_token_id] * padding
        y = [-1] * (len(prefix) - 1) + ids[len(prefix):] + [-1] * padding
        encoded.append((torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)))
    return encoded


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


DEFAULT_HYPERPARAMETERS = {
    "device": "cuda",
    "dtype": "float32",
    "steps": 300,
    "batch_size": 4,
    "length": 128,
    "new_tokens": 40,
    "lr": 0.001,
    "clip_norm": 1.0,
    "rank": 4,
    "alpha": 8.0,
    "inference_repeats": 1,
    "seed": 42,
    "threads": 4,
    "lora": False,
    "kvcache": False,
    "flash": False,
}


def report_path_with_hyperparameters(path, args):
    changed = []
    for name, default in DEFAULT_HYPERPARAMETERS.items():
        value = getattr(args, name)
        if value != default:
            label = name.replace("_", "-")
            changed.append(label if isinstance(value, bool) else f"{label}-{value}")
    if not changed:
        return path
    suffix = "_" + "_".join(changed)
    return path.with_name(path.stem + suffix + path.suffix)


def compact_report(report):
    """Remove per-step and per-example records while retaining aggregate metrics."""
    excluded = {"history", "memory_steps", "outputs"}
    if isinstance(report, dict):
        return {key: compact_report(value) for key, value in report.items()
                if key not in excluded}
    if isinstance(report, list):
        return [compact_report(value) for value in report]
    return report


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_snapshot(device):
    if device.type != "cuda":
        return {"allocated_bytes": None, "peak_allocated_bytes": None}
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


@torch.no_grad()
def evaluate_loss(model, data, device):
    model.eval()
    total, tokens = 0.0, 0
    for x, y in data:
        y = y.unsqueeze(0).to(device)
        loss = model(x.unsqueeze(0).to(device), y)[1]
        count = int((y != -1).sum())
        total += float(loss) * count
        tokens += count
    return total / tokens


@torch.no_grad()
def generate_answer(model, tokenizer, question, device, new_tokens, use_cache):
    ids = tokenizer.encode(prompt(question), return_tensors="pt").to(device)
    if ids.size(1) + new_tokens > model.block_size:
        raise ValueError("Question plus --new-tokens exceeds the model context window")
    generated = model.generate(ids, new_tokens, use_cache=use_cache)
    answer_ids = generated[0, ids.size(1):].tolist()
    if tokenizer.eos_token_id in answer_ids:
        answer_ids = answer_ids[:answer_ids.index(tokenizer.eos_token_id)]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).split("\nUser:")[0].strip()


@torch.no_grad()
def inference(model, tokenizer, rows, device, new_tokens, use_cache, repeats):
    model.eval()
    # Warm up both the model and, on CUDA, the selected attention kernel.
    for row in rows[: min(2, len(rows))]:
        generate_answer(model, tokenizer, row["question"], device, new_tokens, use_cache)
    samples = []
    outputs = []
    memory_steps = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        current = []
        for index, row in enumerate(rows):
            reset_peak_memory(device)
            synchronize(device)
            step_start = time.perf_counter()
            text = generate_answer(model, tokenizer, row["question"], device, new_tokens, use_cache)
            synchronize(device)
            current.append(text)
            memory_steps.append({
                "repeat": len(samples) + 1,
                "question_index": index,
                "wall_ms": (time.perf_counter() - step_start) * 1000,
                "memory": memory_snapshot(device),
            })
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)
        if not outputs:
            outputs = current
    scored = [dict(scores=score(row, text)) for row, text in zip(rows, outputs)]
    return {
        "use_kv_cache": use_cache,
        "questions": len(rows),
        "repeats": repeats,
        "wall_ms": {
            "mean": statistics.mean(samples),
            "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            "min": min(samples),
            "max": max(samples),
        },
        "tokens_per_second": len(rows) * new_tokens / (statistics.mean(samples) / 1000),
        "scores": aggregate(scored),
        "memory_steps": memory_steps,
        "peak_memory_allocated_bytes": max(
            (step["memory"]["peak_allocated_bytes"] for step in memory_steps
             if step["memory"]["peak_allocated_bytes"] is not None),
            default=None,
        ),
        "outputs": [
            {"question": row["question"], "reference": row["answer"], "generated": text,
             "scores": score(row, text)}
            for row, text in zip(rows, outputs)
        ],
    }


def train(model, data, validation_data, device, steps, batch_size, learning_rate, seed, clip_norm):
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=0.0,
    )
    history = []
    generator = random.Random(seed)
    total_seconds = 0.0
    progress_interval = max(1, steps // 10)
    print(f"Training started: {steps} steps on {device}", flush=True)
    for step in range(1, steps + 1):
        batch = generator.choices(data, k=batch_size)
        x = torch.stack([item[0] for item in batch]).to(device)
        y = torch.stack([item[1] for item in batch]).to(device)
        model.train()
        reset_peak_memory(device)
        synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y)[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            clip_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        synchronize(device)
        elapsed = time.perf_counter() - start
        total_seconds += elapsed
        history.append({
            "step": step,
            "loss": float(loss),
            "wall_ms": elapsed * 1000,
            "memory": memory_snapshot(device),
        })
        if step == 1 or step % progress_interval == 0 or step == steps:
            validation_loss = evaluate_loss(model, validation_data, device)
            history[-1]["validation_loss"] = validation_loss
            print(f"Training step {step}/{steps}: loss={float(loss):.4f}, "
                  f"validation_loss={validation_loss:.4f}, step_time={elapsed:.2f}s", flush=True)
    return {
        "steps": steps,
        "batch_size": batch_size,
        "seconds": total_seconds,
        "tokens_per_second": steps * batch_size * data[0][0].numel() / total_seconds,
        "history": history,
        "peak_memory_allocated_bytes": max(
            (row["memory"]["peak_allocated_bytes"] for row in history
             if row["memory"]["peak_allocated_bytes"] is not None),
            default=None,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="experiments/sports/train.jsonl")
    parser.add_argument("--validation", default="experiments/sports/validation.jsonl")
    parser.add_argument("--test", default="experiments/sports/test.jsonl")
    parser.add_argument("--output", type=Path, default=Path("experiments/multi_paradigm/results/report.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--inference-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lora", action="store_true", help="Train only LoRA adapter weights")
    parser.add_argument("--kvcache", "--kv-cache", dest="kvcache", action="store_true",
                        help="Use the KV cache during inference")
    parser.add_argument("--flash", "--flash-attention", dest="flash", action="store_true",
                        help="Force the CUDA FlashAttention backend")
    parser.add_argument("--no-report", action="store_true",
                        help="Do not write a report file; print a compact result for a search runner")
    args = parser.parse_args()

    if min(args.steps, args.batch_size, args.length, args.new_tokens,
           args.inference_repeats, args.rank) < 1 or args.lr <= 0 or args.clip_norm <= 0:
        parser.error("steps, batch size, lengths, rank, repeats, lr, and clip norm must be positive")
    if args.length > 1024:
        parser.error("--length must be <= 1024")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    if device.type == "cpu" and args.dtype != "float32":
        parser.error("CPU runs require --dtype float32")
    if args.flash and args.dtype == "float32" and device.type != "cuda":
        parser.error("FP32 custom flash attention requires CUDA")
    if args.flash and device.type != "cuda":
        parser.error("--flash requires a CUDA device")

    train_path = resolve_path(args.train)
    validation_path = resolve_path(args.validation)
    test_path = resolve_path(args.test)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = report_path_with_hyperparameters(output, args)
    train_rows = load_rows(train_path)
    validation_rows = load_rows(validation_path)
    test_rows = load_rows(test_path)
    question_sets = {
        "train": {row["question"] for row in train_rows},
        "validation": {row["question"] for row in validation_rows},
        "test": {row["question"] for row in test_rows},
    }
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if question_sets[first] & question_sets[second]:
            parser.error(f"{first} and {second} questions must not overlap")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    train_data = encode(train_rows, tokenizer, args.length)
    validation_data = encode(validation_rows, tokenizer, args.length)
    test_data = encode(test_rows, tokenizer, args.length)
    model = GPT.from_pretrained("gpt2").to(dtype=getattr(torch, args.dtype))
    if args.lora:
        model.enable_lora(args.rank, args.alpha)
    attention_backend = (
        "tiled_fp32" if args.flash and args.dtype == "float32"
        else "flash" if args.flash
        else "eager"
    )
    model.set_attention_backend(attention_backend)
    if attention_backend == "tiled_fp32":
        for block in model.transformer.h:
            block.attn.attn_dropout.p = 0.0
    model.to(device)

    report = {
        "complete": False,
        "config": {**vars(args), "output": str(output), "train": str(train_path),
                   "validation": str(validation_path), "test": str(test_path)},
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "enhancements": {"lora": args.lora, "kvcache": args.kvcache, "flash_attention": args.flash},
        "attention_backend": attention_backend,
        "data": {
            "train_examples": len(train_rows),
            "validation_examples": len(validation_rows),
            "test_examples": len(test_rows),
            "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
            "validation_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
            "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        },
        "parameters": parameter_counts(model),
        "training": {},
        "inference": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        if not args.no_report:
            output.write_text(json.dumps(report, indent=2))

    report["training"]["initial_validation_loss"] = evaluate_loss(model, validation_data, device)
    report["training"]["initial_test_loss"] = evaluate_loss(model, test_data, device)
    print("Training is running...", flush=True)
    report["training"].update(train(
        model, train_data, validation_data, device, args.steps, args.batch_size,
        args.lr, args.seed, args.clip_norm
    ))
    report["training"]["final_validation_loss"] = evaluate_loss(model, validation_data, device)
    report["training"]["final_test_loss"] = evaluate_loss(model, test_data, device)
    save()
    print("Inference is running...", flush=True)
    report["validation_inference"] = inference(
        model, tokenizer, validation_rows, device, args.new_tokens,
        args.kvcache, args.inference_repeats
    )
    report["inference"] = inference(
        model, tokenizer, test_rows, device, args.new_tokens, args.kvcache, args.inference_repeats
    )
    report["peak_memory"] = {
        "training_allocated_bytes": report["training"]["peak_memory_allocated_bytes"],
        "inference_allocated_bytes": report["inference"]["peak_memory_allocated_bytes"],
    }
    report["complete"] = True
    save()
    if args.no_report:
        print("PARAMETER_SEARCH_RESULT=" + json.dumps(compact_report(report)), flush=True)
    else:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()