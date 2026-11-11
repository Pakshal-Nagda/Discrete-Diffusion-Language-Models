import torch

class Sampler:
    def __init__(self, n_steps, seed=108):
        self.n_steps = n_steps
        self.rng = torch.Generator().manual_seed(seed)

    def init(self, X):
        return torch.rand(X.shape[:-1], generator=self.rng)

    def steps(self, X):
        t = torch.linspace(0, 1, self.n_steps,)[:, None] + \
            torch.rand((self.n_steps,) + X.shape[:1], generator=self.rng) / self.n_steps
        t[-1] = 1
        return zip(t[:-1], t[1:])

    def sample(self, probs):
        return torch.multinomial(
            probs.flatten(0, -2),
            num_samples=1,
            generator=self.rng
        ).reshape(probs.shape[:-1])
