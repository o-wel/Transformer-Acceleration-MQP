""""
Following this guide : https://medium.com/@sayedebad.777/building-a-transformer-from-scratch-a-step-by-step-guide-a3df0aeb7c9a
"""

# Imports
import torch
import torch.nn as nn
import math


# returns the embeddings of the input sequence
# d_model = dimensions of the embeddings
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        # the sqrt root scaling is supposedly "common practice" to help stabilize gradients in training
        return self.embedding(x) * math.sqrt(self.d_model)
    

# returns positional encodings of the input sequence
# seq is max length of the input sequence
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq = seq
        self.dropout = nn.Dropout(dropout)
        
        encodings = torch.zeros(seq, d_model) # zeros of shape (seq, d_model)
        position = torch.arange(0, seq, dtype=torch.float).unsqueeze(1) # vals 0-seq, shape (seq, 1)
        
        # gives the exponential term in PE equation
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        
        encodings[:, 0::2] = torch.sin(position * div_term)
        encodings[:, 1::2] = torch.cos(position * div_term)
        
        encodings = encodings.unsqueeze(0) # adds dimension, (1, seq, d_model)
        
        self.register_buffer('encodings', encodings)
    
    def forward(self, x):
        # adds PE to input embeddings
        # PE encodings have same dimension d_model as the embeddings
        x = x + (self.encodings[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    

# essentially an implementation of pytorch LayerNorm
# https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html
class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float=1e-05):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        
    def forward(self, x):
        
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
        

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.f1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.f2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        # as in the paper -> max(0,xW1 + B1) W2 + B2
        return self.f2(self.dropout(torch.relu(self.f1(x))))
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h # number of heads, d_model should be divisible by h
        
        self.d_k = d_model // h # dim of vectors processed by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        # all of shape (batch, seq, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        # convert to shape (batch, h, seq, d_k)      
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        # calculate attention
        attention_scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores = attention_scores + mask
            
        attention_scores = attention_scores.softmax(dim=-1)
        
        if self.dropout is not None:
            attention_scores = self.dropout(attention_scores)
            
        attention_scores = torch.matmul(attention_scores, value)
        
        # combine heads (batch, seq, d_model)
        attention_scores = attention_scores.transpose(1, 2).contiguous().view(attention_scores.shape[0], -1, self.h * self.d_k)
        attention_scores = self.w_o(attention_scores)
        
        return attention_scores
    
    
# represents one block (sublayer) into the "add + norm" block in the diagram
class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
        
# excluding cross attention for decoder only
# defines two residual connections making the decoder block
class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block, feed_forward_block, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
    
    def forward(self, x, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

# class for looping over each DecoderBlock
# tgt_mask is to prevent the decoder from looking at future positions in the sequence
# we do not need a src_mask since there is no encoder
class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
        
    def forward(self, x, tgt_mask):
        for layer in self.layers:
            x = layer(x, tgt_mask)
        return self.norm(x)

# final layer for mapping decoder output to vocabulary
# (batch, seq, d_model) -> (batch, seq, vocab_size)
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        return self.proj(x)
    

class Transformer(nn.Module):
    def __init__(self, decoder, tgt_embed: InputEmbeddings, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
    def decode(self, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
    def forward(self, x, mask):
        decoder_output = self.decode(x, mask)
        
        return self.project(decoder_output)
    

def build_transformer(tgt_vocab_size: int,
                      tgt_seq: int,
                      d_model: int=512,
                      N: int=6,
                      h: int=8,
                      dropout: float=0.1,
                      d_ff: int=2048):
    
    embed = InputEmbeddings(d_model, tgt_vocab_size)
    pos = PositionalEncoding(d_model, tgt_seq, dropout)
    
    # defining each block in the decoder
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttention(d_model, h, dropout)
        ffn = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention, ffn, dropout)
        
        decoder_blocks.append(decoder_block)
    
    # define decoder structure using block list
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    proj_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    transformer = Transformer(decoder, embed, pos, proj_layer)
    
    return transformer

