"""Create a PyTorch profiler trace for the transformer_owen training step.

Example:
  python profile_trace.py --input training_data/input.txt --steps 5

Open the resulting trace with ``tensorboard --logdir traces`` and select the
PyTorch Profiler tab. JSON trace files can also be opened with Perfetto.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)
from torch.utils.data import DataLoader

from decoder_transformer import build_transformer
from tokenizer import encode, get_vocab_info
from train import create_causal_mask, textDataset


def profiler_for(args, device):
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    trace_dir = Path(args.trace_dir) / (
        f"seq{args.seq_length}_H{args.num_heads}_N{args.num_dblocks}"
    )
    trace_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing traces to {trace_dir}")

    return profile(
        activities=activities,
        schedule=schedule(wait=1, warmup=args.warmup, active=args.steps, repeat=1),
        on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def profile_training(args, device):
    text = Path(args.input).read_text(encoding="utf-8")
    vocab_size, chars = get_vocab_info(text)
    dataset = textDataset(text, encode(chars), seq_len=args.seq_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    if not len(dataloader):
        raise ValueError("the input must contain at least one full training batch")

    model = build_transformer(
        tgt_vocab_size=vocab_size,
        tgt_seq=args.seq_length,
        d_model=args.d_model,
        N=args.num_dblocks,
        h=args.num_heads,
        dropout=args.dropout,
        d_ff=args.d_ff,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    batches = iter(dataloader)
    model.train()

    with profiler_for(args, device) as prof:
        for _ in range(1 + args.warmup + args.steps):
            try:
                x, y = next(batches)
            except StopIteration:
                batches = iter(dataloader)
                x, y = next(batches)

            x = x.to(device)
            y = y.to(device)
            mask = create_causal_mask(args.seq_length, device)

            optimizer.zero_grad()
            outputs = model(x, mask.unsqueeze(0))
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), y.reshape(-1))
            loss.backward()
            optimizer.step()
            prof.step()

    print(f"Last training loss: {loss.item():.4f}")

# set up path for logging and saving results
relative_path = os.path.dirname(os.path.relpath(__file__))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", default="traces")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5,
                        help="number of recorded training iterations")
    parser.add_argument("--input", default=os.path.join(relative_path, "training_data/input.txt"))

    # Defaults match the training configuration in train.py.
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-dblocks", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d-ff", type=int, default=2048)
    args = parser.parse_args()

    if args.warmup < 0 or args.steps < 1:
        parser.error("--warmup must be nonnegative and --steps must be positive")
    if args.d_model % args.num_heads:
        parser.error("--d-model must be divisible by --num-heads")

    device = resolve_device(args.device)
    print(f"Profiling transformer_owen training on {device}")
    profile_training(args, device)


if __name__ == "__main__":
    main()
