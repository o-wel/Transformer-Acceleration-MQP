import os
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Subset
from tokenizer import get_vocab_info, decode, encode
from decoder_transformer import build_transformer


def generate(model, context, seq_length, max_new_tokens):
    token_list = context.tolist()
    context = context.unsqueeze(0)
    
    for _ in range(max_new_tokens):
        model_context = context[:, -seq_length:]
        
        output = model(model_context, None)

        output = output[:, -1, :]
        probs = torch.softmax(output, dim=1)
        next = torch.multinomial(probs, num_samples=1)
        
        context = torch.cat((context, next), dim=-1)
        token_list.append(next.item())
        
    return token_list


if __name__ == '__main__':
    
    with open("training_data/input.txt", "r") as f:
         text = f.read()
            
    vocab_size, chars = get_vocab_info(text)
    encoder = encode(chars)
    decoder = decode(chars)
    
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("Running on device " + str(device))
    
     # Transformer Hyperparameters
    seq_length = 128 # context window from dataset per batch
    num_heads = 8 # default is 8
    num_dblocks = 6 # decoder block layers, default is 6
    d_model = 512 # input embedding length, default is 512
    dropout = 0.1 # default of 0.1
    d_ff = 2048 # default of 2048
    
    model = build_transformer(tgt_vocab_size=vocab_size, 
                                tgt_seq=seq_length,
                                d_model=d_model,
                                N=num_dblocks,
                                h=num_heads,
                                dropout=dropout,
                                d_ff=d_ff
                                ).to(device)
    
    state_dict = torch.load("models/epoch10-turingWeights.pth")
    model.load_state_dict(state_dict)
    model.eval()
    
    with torch.no_grad():
        while True:
            prompt = input("Prompt the model: ")
            val = encoder(prompt)
            val = torch.tensor(val).to(device)
                
            output = generate(model, val, seq_length, max_new_tokens=300)
            
            text_output = decoder(output)
            print(text_output)