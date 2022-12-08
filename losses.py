import torch.nn as nn
import torch.nn.functional as F


class BYOLLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def _normalize(self, x):
        return F.normalize(x, dim=-1, p=2)

    def forward(self, x_1, x_2):
        x_1 = self._normalize(x_1)
        x_2 = self._normalize(x_2)
        inner_prod = (x_1 * x_2).sum(dim=-1)

        return 2 - 2 * inner_prod
        