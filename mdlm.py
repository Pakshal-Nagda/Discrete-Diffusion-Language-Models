import torch
from torch import nn
import torch.nn.functional as F
from utils import batch_index

class MDLM(nn.Module):
    def __init__(self, base_model, noise, sampler):
        super().__init__()
        self.model = base_model
        self.noise = noise
        self.sampler = sampler

    def forward(self, X):
        if not isinstance(X, torch.Tensor):
            raise ValueError('The input should be a tensor of indices')
        if len(X.shape) == 1:
            X.unsqueeze(0)                                              # Shape: [b, n]
        if self.training:
            t = self.sampler.init(X)                                    # Shape: [b]
            Xt = batch_index(self.noise.prod(t), X)                     # Shape: [b, n, Vt]
            return t, self.model(self.sampler.sample(Xt))               # Shape: [b, n, V0]
        Xt = F.one_hot(X, self.model.vocab_size).type(torch.float32)    # Shape: [b, n, Vt]
        for s, t in self.sampler.steps(Xt):
            Xt = self.sampler.sample(Xt)                                # Shape: [b, n]
            X0 = self.model(Xt).softmax(dim=-1)                         # Shape: [b, n, V0]
            Xt = self.noise.infer(X0, Xt, s, t)                         # Shape: [b, n, Vs]
        return self.model(self.sampler.sample(Xt))                      # Shape: [b, n, V0]
