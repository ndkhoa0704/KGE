import torch.nn as nn
import torch
from logger_config import logger


class CustomTransSmth(nn.Module):
    def __init__(self, gamma, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        self.dim = dim
        self.w = nn.Parameter(torch.tensor(torch.zeros(self.dim, self.dim), requires_grad=True))

    def forward(self, h, t):
        return h@self.w
        