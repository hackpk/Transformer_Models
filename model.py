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
        self.dropout = dropout
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





