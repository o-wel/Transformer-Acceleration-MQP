import os
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Subset
from decoder_transformer import build_transformer
from tokenizer import encode, get_vocab_info

def get_training_data(path):
    # download the tiny shakespeare dataset
    input_file_path = os.path.join(path, 'input.txt')
    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(requests.get(data_url).text)
        print("Wrote tiny shakespeare dataset to " + os.path.join(os.getcwd(), input_file_path))


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
    download_path = "training_data"
    get_training_data(download_path)
    
    with open("training_data/input.txt", "r") as f:
        text = f.read()
        
    vocab_size, chars = get_vocab_info(text)
    
    encoder = encode(chars)

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
    seq_length = 8 # context window from dataset per batch
    num_heads = 2 # default is 8
    num_dblocks = 2 # decoder block layers, default is 6
    d_model = 128 # input embedding length, default is 512
    dropout = 0.1 # default of 0.1
    d_ff = 512 # default of 2048

    # create dataset
    dataset = textDataset(text, encoder, seq_len=seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Data loader length: {dataloader.__len__()}")
 
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
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        training_i = 0
        
        for x, y in dataloader:
            if training_i % batch_size == 0:
                print(f"Current batch: {training_i//batch_size}", end="\r")
            
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
        
            training_i += 1
        
        epoch_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch} loss: {epoch_loss}")
    
    save_path = "models"
    # save = input(f"Save Model to '{save_path}'? Y/N\n")
    # if (save.lower() == "y"):
        
    torch.save(model.state_dict(), "models/customRunWeights.pth")
        