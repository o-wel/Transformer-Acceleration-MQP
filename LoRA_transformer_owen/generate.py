import os
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Subset
from tokenizer import get_vocab_info, decode, encode
from LoRA_transformer import build_transformer
from train import create_causal_mask

# set up path for logging and saving results
relative_path = os.path.dirname(os.path.relpath(__file__))

def generate(model, context, seq_length, max_new_tokens):
    token_list = context.tolist()
    context = context.unsqueeze(0)
    start = time.perf_counter()
    ttft = 0
    
    for i in range(max_new_tokens):
        model_context = context[:, -seq_length:]
        
        mask = create_causal_mask(model_context.shape[1], device)
        
        output = model(model_context, mask.unsqueeze(0))

        output = output[:, -1, :]
        probs = torch.softmax(output, dim=1)
        next = torch.multinomial(probs, num_samples=1)
        
        context = torch.cat((context, next), dim=-1)
        token_list.append(next.item())
        
        print(decoder([next.item()]), end="", flush=True)
        
        # time to first token
        if i == 0:
            ttft = time.perf_counter() - start
    
    elapsed = time.perf_counter() - start
    tps = max_new_tokens/elapsed
    print("\n-------------------")
    print(f"Generated {max_new_tokens} new tokens\n")
    print(f"Elapsed Time: {elapsed:.6f} sec")
    print(f"TTFT: {ttft:.6f} sec")
    print(f"TPS: {tps:.6f} tokens/sec")
    print(f"SPT: {1/tps:.6f} sec")
    
    return token_list


if __name__ == '__main__':
    
    with open(os.path.join(relative_path, "training_data/input.txt"), "r") as f:
         text = f.read()
            
    vocab_size, chars = get_vocab_info(text)
    encoder = encode(chars)
    decoder = decode(chars)
    
    relative_path = os.path.dirname(os.path.relpath(__file__))
    
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("Running on device " + str(device))
    
     # Transformer Hyperparameters
    seq_length = 128 # context window from dataset per batch
    num_heads = 8 # default is 8
    num_dblocks = 6 # decoder block layers, default is 6
    d_model = 512 # input embedding length, default is 512
    dropout = 0.1 # default of 0.1
    d_ff = 2048 # default of 2048
    lora_r = 1 # LoRA rank
    lora_a = 1 # LoRA alpha
    
    model = build_transformer(tgt_vocab_size=vocab_size, 
                                tgt_seq=seq_length,
                                d_model=d_model,
                                N=num_dblocks,
                                h=num_heads,
                                dropout=dropout,
                                d_ff=d_ff,
                                lora_a=lora_a,
                                lora_r=lora_r
                                ).to(device)
    
    state_dict = torch.load(os.path.join(relative_path, "models/LoRA_seq128_H8_N6.pth")) #transformer_seq128_H8_N6.pth
    model.load_state_dict(state_dict)
    model.eval()
    
    # get amount of parameters in the model
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    with torch.no_grad():
        while True:
            prompt = input("Prompt the model: ")
            print("-------------------")
            if(str.lower(prompt) == 'exit'):
                exit()
                
            val = encoder(prompt)
            val = torch.tensor(val).to(device)
                
            output = generate(model, val, seq_length, max_new_tokens=500)
            
            print("\n-------------------")
            