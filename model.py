from typing import Any
import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    # Here Sentence is mapped to some numbers
    def __init__(self,d_model: int, vocab_size: int):
        super.__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    # Positional Encoding comes into picture after the sentence is embeded 
    # For we have matched sentence to a number but we don't know the position
    # of each word which requires an encoding it has two formulas 
    # 1). Sin(pos/1000**2i/d_model) which is for even poistions
    # 2). Cos(pos/1000**2i/d_model) which is for odd positions
    # seq len here refrenst the length of the sentence
    # dropout here represent the loss

    def __init__(self, d_model: int, seq_len: int, dropout:float ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # create a matrix of postions 
        pos_enc = torch.zeros(seq_len,d_model)
        # create a vector of 1 seq length
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/ d_model))
        # create the encoding for even and odd positions
        pos_enc[:,0::2] = torch.sin(position*div_term)
        pos_enc[:,1::2] = torch.cos(position*div_term)

        pos_enc = pos_enc.unsqueeze(0)

        self.register_buffer('pos_enc',pos_enc)

    def forward(self,x):
        x = x + (self.pos_enc[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    # Layer *
    def __init__(self, eps: float =10**-6, ) -> None:
        super().__init__()
        # Epsilon is used beacuse if mean is close to zero then our output will be higher 
        # and not maintable 
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) #Added

    def forward(self,x):
        # layer norm would be Xi = Xi - mean/squareroot of std+ eps
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x-mean)/(std+self.eps)+self.bias
    






