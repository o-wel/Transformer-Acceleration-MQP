import torch

from mingpt.bpe import BPETokenizer
from mingpt.model import GPT

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BPETokenizer()
model = GPT.from_pretrained("gpt2").to(device)
model.eval()

prompt = "Boston, MA is"
tokens = tokenizer(prompt).to(device)

with torch.no_grad():
    output = model.generate(
        tokens,
        max_new_tokens=50,
        do_sample=True,
        top_k=40,
    )

print(tokenizer.decode(output[0].cpu()))
