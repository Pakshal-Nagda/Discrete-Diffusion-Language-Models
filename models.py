import torch
from torch import nn

class BaseModel(nn.Module):
    def __init__(self, d_model, vocab_size, n_layers, n_head):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.unembed = nn.Linear(d_model, vocab_size)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_head, 4 * d_model),
            n_layers,
        )

    def forward(self, X):
        X = self.embed(X)
        X = self.encoder(X)
        X = self.unembed(X)
        return X
