import torch

class Uniform:
    def __init__(self, n_steps, seed=108):
        self.n_steps = n_steps
        self.rng = torch.Generator().manual_seed(seed)

    def steps(self):
        t = torch.linspace(0, 1, self.n_steps)
        t += torch.rand_like(t, generator=self.rng) / self.n_steps
        t[-1] = 1
        return zip(t[:-1], t[1:])
