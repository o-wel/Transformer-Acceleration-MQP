

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