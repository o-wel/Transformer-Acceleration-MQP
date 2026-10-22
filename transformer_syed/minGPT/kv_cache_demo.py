"""Compare cached and uncached generation using identical weights and a prompt."""
import argparse
import json
from pathlib import Path
import time
import torch
from transformers import GPT2Tokenizer
from mingpt.model import GPT
from sports_experiment import load_parameters


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['pretrained', 'full', 'lora'], default='lora')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--new-tokens', type=int, default=40)
    parser.add_argument('--prompt', default='Basketball match: Bulls vs Sharks. Halftime score: Bulls 51, Sharks 35. Final score: Bulls 96, Sharks 69.\nWho won the match, and what was the final score?')
    args = parser.parse_args()
    if args.new_tokens < 1:
        parser.error('--new-tokens must be positive')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(__file__).resolve().parent
    model = GPT.from_pretrained('gpt2').float()
    if args.model != 'pretrained':
        selection = json.loads((root/'experiments/sports/matched/selection.json').read_text())
        directory = root/selection['methods'][args.model]['runs']['42']
        checkpoint = torch.load(directory/('adapter.pt' if args.model=='lora' else 'model.pt'),
                                map_location='cpu', weights_only=True)
        if args.model == 'lora':
            model.enable_lora(checkpoint['rank'], checkpoint['alpha'])
        load_parameters(model, checkpoint)
        del checkpoint
    model.to(args.device).eval()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    x = torch.tensor([tokenizer.encode(f'User: {args.prompt}\nAssistant:')], device=args.device)
    print(f'Model: {args.model}; prompt tokens: {x.size(1)}; new tokens: {args.new_tokens}')
    print(args.prompt)
    outputs = []
    for cached in (False, True):
        # One warmup per path; these live timings are illustrative, not the repeated benchmark.
        model.generate(x, min(args.new_tokens, 4), use_cache=cached)
        if x.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        output = model.generate(x, args.new_tokens, use_cache=cached)
        if x.is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        tokens = output[0, x.size(1):].tolist()
        outputs.append(tokens)
        if tokenizer.eos_token_id in tokens:
            tokens = tokens[:tokens.index(tokenizer.eos_token_id)]
        text = tokenizer.decode(tokens, skip_special_tokens=True).split('\nUser:')[0].strip()
        print(f'\n{"CACHED" if cached else "UNCACHED"} ({elapsed:.3f}s): {text}')
    print('\nAll generated token IDs identical:', outputs[0] == outputs[1])
    print('For repeated GPU measurements, see experiments/kv_cache/results/report.json.')


if __name__ == '__main__':
    main()
