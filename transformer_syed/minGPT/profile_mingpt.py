"""Create PyTorch profiler traces for minGPT training or inference.

Examples:
  python profile_mingpt.py train --input input.txt --steps 12
  python profile_mingpt.py inference --pretrained gpt2 --steps 12

Open the resulting trace with `tensorboard --logdir traces` and select the
PyTorch Profiler tab. JSON trace files can also be opened in Perfetto.
"""

import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
from torch.utils.data import DataLoader, Dataset

from mingpt.bpe import BPETokenizer
from mingpt.model import GPT


class CharDataset(Dataset):
    """The character-level next-token dataset used by projects/chargpt."""

    def __init__(self, text, block_size):
        if len(text) <= block_size:
            raise ValueError("input text must be longer than --block-size")
        self.text = text
        self.block_size = block_size
        chars = sorted(set(text))
        self.stoi = {char: i for i, char in enumerate(chars)}
        self.vocab_size = len(chars)

    def __len__(self):
        return len(self.text) - self.block_size

    def __getitem__(self, index):
        chunk = self.text[index:index + self.block_size + 1]
        ids = [self.stoi[char] for char in chunk]
        return torch.tensor(ids[:-1]), torch.tensor(ids[1:])


def profiler_for(args, device):
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    run_name = f"{args.mode}-{args.model_type if args.mode == 'train' else args.pretrained}"
    trace_dir = Path(args.trace_dir) / run_name
    trace_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing traces to {trace_dir}")
    return profile(
        activities=activities,
        schedule=schedule(wait=1, warmup=args.warmup, active=args.steps, repeat=1),
        on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )


def profile_training(args, device):
    dataset = CharDataset(Path(args.input).read_text(), args.block_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    config = GPT.get_default_config()
    config.model_type = args.model_type
    config.vocab_size = dataset.vocab_size
    config.block_size = args.block_size
    model = GPT(config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    batches = iter(loader)

    with profiler_for(args, device) as prof:
        for _ in range(1 + args.warmup + args.steps):
            try:
                x, y = next(batches)
            except StopIteration:
                batches = iter(loader)
                x, y = next(batches)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            prof.step()
    print(f"last training loss: {loss.item():.4f}")


def profile_inference(args, device):
    tokenizer = BPETokenizer()
    model = GPT.from_pretrained(args.pretrained).to(device).eval()
    input_ids = tokenizer(args.prompt).to(device)
    with torch.inference_mode(), profiler_for(args, device) as prof:
        for _ in range(1 + args.warmup + args.steps):
            model.generate(input_ids, max_new_tokens=args.new_tokens, do_sample=False)
            prof.step()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "inference"))
    parser.add_argument("--trace-dir", default="traces")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5, help="number of recorded iterations")
    parser.add_argument("--input", default="input.txt")
    parser.add_argument("--model-type", default="gpt-mini")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--pretrained", choices=("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"), default="gpt2")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--new-tokens", type=int, default=16)
    args = parser.parse_args()
    if args.warmup < 0 or args.steps < 1:
        parser.error("--warmup must be nonnegative and --steps must be positive")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"profiling {args.mode} on {device}")
    if args.mode == "train":
        profile_training(args, device)
    else:
        profile_inference(args, device)


if __name__ == "__main__":
    main()
