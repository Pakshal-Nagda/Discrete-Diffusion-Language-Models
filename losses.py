from torch import nn
import torch.nn.functional as F

class DiffusionLoss(nn.Module):
    def __init__(self, dlm):
        self.dlm = dlm

    def forward(self, y_pred, y, t=None):
        pass

class MDLMLoss(DiffusionLoss):
    def forward(self, y_pred, y, t=None):
        if not t:
            raise ValueError('t is required for the computation of MDLMLoss')
        alpha = self.dlm.noise.alpha(t)
        rate = self.dlm.noise.alpha.rate(t)
        return (rate / (1 - alpha)) @ F.cross_entropy(y_pred, y, reduction='none').sum(dim=-1)
