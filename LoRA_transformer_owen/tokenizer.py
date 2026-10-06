

def get_vocab_info(text):
    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    return vocab_size, chars

# very simple tokenizer
def encode(chars):
    stoi = { ch:i for i,ch in enumerate(chars) }
    return lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers


def decode(chars):
    itos = { i:ch for i,ch in enumerate(chars) }
    return lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


# testing
if __name__ == '__main__':
    import os
    relative_path = os.path.dirname(os.path.relpath(__file__))
    
    # open pretraining data for vocab info on the proper data
    with open(os.path.join(relative_path, "training_data/input.txt"), "r") as f:
        text = f.read()
    
    vocab_size, chars = get_vocab_info(text)
    print(chars)
    
     # open pretraining data for vocab info on the proper data
    with open(os.path.join(relative_path, "training_data/web_data_clean.txt"), "r") as f:
        text = f.read()
    
    vocab_size, chars = get_vocab_info(text)
    print(chars)