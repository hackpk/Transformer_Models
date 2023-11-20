from typing import Any
import torch
import torch.nn as nn
import math

class InputEmbedding():
    # Here Sentence is mapped to some numbers
    def __init__(self,d_model: int, vocab_size: int):
        super.__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddind = nn.Embedding(vocab_size,d_model)
    
    def __call__(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)