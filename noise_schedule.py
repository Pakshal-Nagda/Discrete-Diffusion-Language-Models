import torch

class Schedule:
    def __init__(self, V, alpha):
        self.V = V
        self.alpha = alpha

    def pi(self, shape):
        pass

    def prod(self, s, t=None):
        pass

    def infer(self, X0, Xt, s, t):
        term_1 = self.noise.prod(s, t).T[Xt]        # Shape: [b, n, Vs]
        term_2 = self.noise.prod(s).T               # Shape: [Vs, V0]
        deno = self.noise.prod(t).T[Xt]             # Shape: [b, n, V0]
        num = term_1.unsqueeze(-1) * term_2         # Shape: [b, n, Vs, V0]
        Xs = torch.einsum('bnv,bnsv->bns', X0, (num / deno.unsqueeze(-2)))
        Xs /= Xs.sum(dim=-1, keepdim=True)          # Shape: [b, n, Vs]
        return Xs

class Uniform(Schedule):
    def pi(self, shape):
        return torch.ones(shape + (self.V,)) / self.V

    def prod(self, s, t=None):
        if t is None:
            s, t = 0, s
        alpha = self.alpha(t) / self.alpha(s)
        return alpha * torch.eye(self.V) + (1 - alpha) * self.pi((self.V,))

class Absorbing(Schedule):
    def pi(self, shape):
        pi_vec = torch.zeros(shape + (self.V,))
        pi_vec[:, :, -1] = 1
        return pi_vec

    def prod(self, s, t=None):
        if t is None:
            s, t = 0, s
        alpha = self.alpha(t) / self.alpha(s)
        return alpha * torch.eye(self.V) + (1 - alpha) * self.pi((self.V,))

    def infer(self, X0, Xt, s, t):
        mask = Xt == self.V - 1
        alpha_s, alpha_t = self.alpha(s), self.alpha(t)
        Xs = self.pi(Xt.shape)
        Xs[mask] *= (1 - alpha_s)
        Xs[mask] += (alpha_s - alpha_t) * X0
        Xs[mask] /= Xs[mask].sum(dim=1, keepdim=True)
        return Xs

# MDLM schedules

def linear(t, M=1e8):
    return torch.exp(-M * t)

def loglinear(t):
    return 1 - t

def cosine(t):
    return torch.cos(torch.pi / 2 * (1 - t))

def cosine_sq(t):
    return torch.cos(torch.pi / 2 * (1 - t)) ** 2
