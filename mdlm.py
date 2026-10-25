import torch
from torch import nn

class MDLM(nn.Module):
    def __init__(self, base_model, noise, sampler, tokenizer):
        self.noise = noise
        self.sampler = sampler
        self.tokenizer = tokenizer
        pad_idx = self.tokenizer.encode(self.tokenizer.pad_token)
        assert pad_idx == len(self.tokenizer.vocab) - 1, 'Pad should be the last token of the vocabulary'
        embed = nn.Embedding(len(self.tokenizer.vocab), base_model.d_model, pad_idx)
        unembed = nn.Linear(base_model.d_model, len(self.tokenizer.vocab))
        self.model = nn.Sequential(embed, base_model, unembed)

    def forward(self, X):
        if not isinstance(X, torch.Tensor):
            X = self.tokenizer.encode(X, padding=True, return_tensors='pt')
        if len(X.shape) == 1:
            X.unsqueeze(0)                              # Shape: [b, n]
        Xt = self.noise.pi(X.shape)                     # Shape: [b, n, V]
        for s, t in self.sampler.steps():
            Xt = self.sampler.sample(Xt)                # Shape: [b, n]
            X0 = self.model(Xt).exp()                   # Shape: [b, n, V0]
            Xt = self.noise.infer(X0, Xt, s, t)         # Shape: [b, n, Vs]
        return self.model(self.sampler.sample(Xt))
