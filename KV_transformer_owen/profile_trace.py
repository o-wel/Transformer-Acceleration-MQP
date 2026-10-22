"""Create a PyTorch profiler trace for transformer inference.

The prompt is processed once, then the profiler records one-token decode
steps. KV caching is enabled by default and can be disabled with
``--no-use-kv``.

Example:
  python profile_trace.py --input training_data/input.txt --steps 5

Open the resulting trace with ``tensorboard --logdir traces`` or Perfetto.
"""

import argparse
import os
from pathlib import Path

import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)

from KV_transformer import build_transformer
from generate import create_causal_mask
from tokenizer import encode, get_vocab_info


def profiler_for(args, device):
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    trace_dir = Path(args.trace_dir) / (
        f"decode_seq{args.seq_length}_H{args.num_heads}_N{args.num_dblocks}"
    )
    trace_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing traces to {trace_dir}")

    return profile(
        activities=activities,
        schedule=schedule(
            wait=1,
            warmup=args.warmup,
            active=args.steps,
            repeat=1,
        ),
        on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_prompt(args, device):
    text = Path(args.input).read_text(encoding="utf-8")
    vocab_size, chars = get_vocab_info(text)
    encoded = encode(chars)(text)

    if len(encoded) < args.seq_length:
        raise ValueError("the input must contain at least --seq-length tokens")

    prompt = torch.tensor(
        encoded[:args.seq_length],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    return vocab_size, prompt


def profile_inference(args, device):
    vocab_size, prompt = load_prompt(args, device)
    model = build_transformer(
        tgt_vocab_size=vocab_size,
        tgt_seq=args.seq_length,
        d_model=args.d_model,
        N=args.num_dblocks,
        h=args.num_heads,
        dropout=args.dropout,
        d_ff=args.d_ff,
        use_kv=args.use_kv,
    ).to(device)
    model.eval()
    model.reset_cache()

    prompt_mask = create_causal_mask(prompt.shape[1], device).unsqueeze(0)
    position = prompt.shape[1]
    context = prompt

    with torch.inference_mode():
        # Prefill the cache before recording one-token decode steps.
        output = model(prompt, prompt_mask, position=0)

        with profiler_for(args, device) as prof:
            for _ in range(1 + args.warmup + args.steps):
                next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                if args.use_kv:
                    output = model(next_token, mask=None, position=position)
                    position += 1
                else:
                    context = torch.cat((context, next_token), dim=1)
                    model_context = context[:, -args.seq_length:]
                    mask = create_causal_mask(model_context.shape[1], device)
                    output = model(model_context, mask.unsqueeze(0), position=0)
                prof.step()

    print(f"Last generated token: {output[:, -1, :].argmax(dim=-1).item()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", default="traces")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="number of recorded one-token decode steps",
    )
    parser.add_argument(
        "--input",
        default=os.path.join(
            os.path.dirname(os.path.relpath(__file__)),
            "training_data/input.txt",
        ),
    )
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-dblocks", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument(
        "--no-use-kv",
        dest="use_kv",
        action="store_false",
        help="disable KV caching for comparison",
    )
    parser.set_defaults(use_kv=False)
    args = parser.parse_args()

    if args.warmup < 0 or args.steps < 1:
        parser.error("--warmup must be nonnegative and --steps must be positive")
    if args.d_model % args.num_heads:
        parser.error("--d-model must be divisible by --num-heads")

    device = resolve_device(args.device)
    print(f"Profiling inference on {device} (use_kv={args.use_kv})")
    profile_inference(args, device)


if __name__ == "__main__":
    main()
