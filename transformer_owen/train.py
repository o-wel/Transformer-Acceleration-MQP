import os
import requests
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Subset
from decoder_transformer import build_transformer
from tokenizer import encode, get_vocab_info

# set up path for logging and saving results
relative_path = os.path.dirname(os.path.relpath(__file__))


def create_causal_mask(seq_len, device):
    """Create a causal mask for autoregressive attention."""
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
    return mask


class textDataset(Dataset):
    def __init__(self, text, encode, seq_len=512):
        super().__init__()
        self.seq_len = seq_len
        self.data = encode(text)
        
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, index):
        seq_chunk = self.data[index: index + self.seq_len + 1]
        x = torch.tensor(seq_chunk[:-1])
        y = torch.tensor(seq_chunk[1:])
        return x, y
    

if __name__ == '__main__':
    
    # open training data
    with open(os.path.join(relative_path, "training_data/input.txt"), "r") as f:
        text = f.read()
        
    vocab_size, chars = get_vocab_info(text)
    
    encoder = encode(chars)
    
    # set up cuda, give exit option if not available
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("Running on device " + str(device))
    cont = input("Continue? Y/N\n")
    if (cont.lower() != "y"):
        exit()
    
    # Training Hyperparameters
    batch_size = 16
    learning_rate = 0.001
    epochs = 1
    weight_decay = 1e-5
    
    # Transformer Hyperparameters
    seq_length = 4 # context window from dataset per batch
    num_heads = 2 # default is 8
    num_dblocks = 2 # decoder block layers, default is 6
    d_model = 128 # input embedding length, default is 512
    dropout = 0.1 # default of 0.1
    d_ff = 512 # default of 2048

    # create dataset
    dataset = textDataset(text, encoder, seq_len=seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Data loader length: {len(dataloader)}")
 
    model = build_transformer(tgt_vocab_size=vocab_size, 
                              tgt_seq=seq_length,
                              d_model=d_model,
                              N=num_dblocks,
                              h=num_heads,
                              dropout=dropout,
                              d_ff=d_ff
                              ).to(device)
    
    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # logging for getting model data
    log_path = os.path.join(relative_path, "training_logs")
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    with open(os.path.join(log_path, f"transformer_seq{seq_length}_H{num_heads}_N{num_dblocks}.txt"), "w+") as f:
        
        losses = []
        
        # main training loop
        start_time = time.perf_counter()
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            
            for batch_i, (x, y) in enumerate(dataloader):
                if batch_i % batch_size == 0:
                    print(f"Current batch: {batch_i}", end="\r")
                
                x = x.to(device)
                y = y.to(device)
                
                mask = create_causal_mask(seq_length, device)
                optimizer.zero_grad()
                outputs = model(x, mask.unsqueeze(0))
                
                # Compute loss
                loss = criterion(outputs.view(-1, outputs.shape[-1]), y.view(-1))
                # Backward pass
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # getting model time stats and loss
            epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch} loss: {epoch_loss}", file=f)
            print(f"Epoch {epoch} loss: {epoch_loss}")
            checkpoint_time = time.perf_counter() - start_time
            print(f"Elapsed time: {checkpoint_time:.6f} secs", file=f)
            print(f"Elapsed time: {checkpoint_time:.6f} secs")
            
            # add to list for graphing
            losses.append(epoch_loss)
            
            # save checkpoint every epoch
            checkpoint_path = os.path.join(relative_path, f"training_checkpoints/transformer_seq{seq_length}_H{num_heads}_N{num_dblocks}")
            if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_path, f"transformer_epoch{epoch}.pth"))

        final_time = time.perf_counter() - start_time
        print(f"\nFinal training time: {final_time:.6f} secs", file=f)
                
        plt.plot([i for i in range(0, epochs)], losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(log_path, f"transformer_seq{seq_length}_H{num_heads}_N{num_dblocks}.png"))
        
        save_path = os.path.join(relative_path, "models")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, f"transformer_seq{seq_length}_H{num_heads}_N{num_dblocks}.pth"))
        print(f"Model saved to: {save_path}", file=f)
    
    f.close()