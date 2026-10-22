import os
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Subset
from tokenizer import get_vocab_info, decode, encode
from KV_transformer import build_transformer

def create_causal_mask(seq_len, device):
    """Create a causal mask for autoregressive attention."""
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
    return mask

# set up path for logging and saving results
relative_path = os.path.dirname(os.path.relpath(__file__))

def kv_generate(model, context, seq_length, max_new_tokens):
    token_list = context.tolist()
    context = context[-seq_length:].unsqueeze(0)
    torch.cuda.synchronize()
    start = time.perf_counter()
    ttft = 0
    
    # initial prompt
    model.reset_cache()
    mask = create_causal_mask(context.shape[1], device)
    output = model(context, mask.unsqueeze(0))
    
    for i in range(max_new_tokens):
        output = output[:, -1, :]
        probs = torch.softmax(output, dim=1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        token_list.append(next_token.item())
        print(decoder([next_token.item()]), end="", flush=True)
        
        context = torch.cat((context, next_token), dim=1)
        
        # time to first token
        if i == 0:
            ttft = time.perf_counter() - start
        
        # add only one new token
        output = model(next_token, mask=None, position=context.shape[1] - 1)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tps = max_new_tokens/elapsed
    # print("\n-------------------")
    # print(f"Generated {max_new_tokens} new tokens\n")
    # print(f"Elapsed Time: {elapsed:.6f} sec")
    # print(f"TTFT: {ttft:.6f} sec")
    # print(f"TPS: {tps:.6f} tokens/sec")
    # print(f"SPT: {1/tps:.6f} sec")
        
    # Track exact memory usage
    current_mem = torch.cuda.memory_allocated()       # Bytes currently occupied by live tensors
    peak_mem = torch.cuda.max_memory_allocated()      # Peak bytes occupied since the program/reset began

    # print(f"Current Tensor Allocation: {current_mem / 1024**2:.2f} MB")
    # print(f"Peak Tensor Allocation: {peak_mem / 1024**2:.2f} MB")
    
    return token_list, elapsed, ttft



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
    seq_length = 512 # context window from dataset per batch
    num_heads = 8 # default is 8
    num_dblocks = 6 # decoder block layers, default is 6
    d_model = 512 # input embedding length, default is 512
    dropout = 0.1 # default of 0.1
    d_ff = 2048 # default of 2048
    
    kv_model = build_transformer(tgt_vocab_size=vocab_size, 
                                tgt_seq=seq_length,
                                d_model=d_model,
                                N=num_dblocks,
                                h=num_heads,
                                dropout=dropout,
                                d_ff=d_ff,
                                use_kv=True
                                ).to(device)
    
    state_dict = torch.load(os.path.join(relative_path, "models/model.pth")) #transformer_seq128_H8_N6.pth
    kv_model.load_state_dict(state_dict, strict=False)
    kv_model.eval()
    
    with torch.no_grad():
        while True:
            prompt = input("Prompt the model: ")
            print("-------------------")
            if(str.lower(prompt) == 'exit'):
                exit()
                
            val = encoder(prompt)
            val = torch.tensor(val).to(device)
                
            output = kv_generate(kv_model, val, seq_length, max_new_tokens=500)
            
            print("\n-------------------")
    