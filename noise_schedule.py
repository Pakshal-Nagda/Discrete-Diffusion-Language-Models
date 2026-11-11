import torch
import torch.nn.functional as F
from utils import batch_index

class Schedule:
    def __init__(self, V: int, alpha: Alpha):
        self.V = V
        self.alpha = alpha

    def pi(self, shape):
        pass

    def prod(self, s, t=None):
        pass

    def infer(self, X0, Xt, s, t):
        # WARNING: Very inefficient implementation with memory requirements O(BNV^2)
        # Redefine for all inherited classes by simplifying mathematically
        Qst = self.prod(s, t)                           # Shape: [b, Vs, Vt]
        Qs = self.prod(s)                               # Shape: [b, V0, Vs]
        Qt = self.prod(t)                               # Shape: [b, V0, Vt]
        term_1 = batch_index(Qst.mT, Xt)                # Shape: [b, n, Vs]
        term_2 = Qs.mT                                  # Shape: [b, Vs, V0]
        deno = batch_index(Qt.mT, Xt)                   # Shape: [b, n, V0]
        num = term_1[:, None] * term_2.unsqueeze(-3)    # Shape: [b, n, Vs, V0]
        Xs = torch.einsum('bnv,bnsv->bns', X0, (num / deno.unsqueeze(-2)))
        Xs /= Xs.sum(dim=-1, keepdim=True)              # Shape: [b, n, Vs]
        print('here')
        return Xs

class Uniform(Schedule):
    def pi(self, shape):
        return torch.ones(shape + (self.V,)) / self.V

    def prod(self, s, t=None):
        if t is None:
            s, t = 0, s
        alpha = (self.alpha(t) / self.alpha(s))[:, None, None]
        return alpha * torch.eye(self.V) + (1 - alpha) * self.pi((self.V,))

class Absorbing(Schedule):
    def pi(self, shape):
        pi_vec = torch.zeros(shape + (self.V,))
        pi_vec[..., -1] = 1
        return pi_vec

    def prod(self, s, t=None):
        if t is None:
            s, t = 0, s
        alpha = (self.alpha(t) / self.alpha(s))[:, None, None]
        return alpha * torch.eye(self.V) + (1 - alpha) * self.pi((self.V,))

    def infer(self, X0, Xt, s, t):
        mask = (Xt != self.V - 1)
        alpha_s, alpha_t = self.alpha(s)[:, None, None], self.alpha(t)[:, None, None]
        Xs = ((1 - alpha_s) * self.pi(Xt.shape) + (alpha_s - alpha_t) * X0) / (1 - alpha_t)
        Xs[mask] = 0
        Xs[mask, Xt[mask]] = 1
        return Xs

class Alpha:
    def __call__(self, t):
        pass

    def rate(self, t):
        pass

class Linear(Alpha):
    def __init__(self, M=1e8):
        self.M = M

    def __call__(self, t):
        return torch.exp(-self.M * t)

    def rate(self, t):
        return -self.M * torch.exp(-self.M * t)

class Loglinear(Alpha):
    def __call__(self, t):
        return 1 - t

    def rate(self, t):
        return -1

class Cosine(Alpha):
    def __call__(self, t):
        return torch.cos(torch.pi / 2 * (1 - t))

    def rate(self, t):
        return torch.pi / 2 * torch.sin(torch.pi / 2 * (1 - t))

class CosineSq(Alpha):
    def __call__(self, t):
        return torch.cos(torch.pi / 2 * (1 - t)) ** 2

    def rate(self, t):
        return torch.pi / 2 * torch.sin(torch.pi * (1 - t))
