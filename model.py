from typing import Any
import torch
import torch.nn as nn
import math

class InputEmbedding():

    def __init__(self,d_model: int, vocab_size: int):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size,d_model)
    
    def __call__(self, x):
        return nn.Embedding(x) * math.sqrt(self.d_model)