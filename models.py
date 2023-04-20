import torch.nn as nn
import torch
from logger_config import logger


class CustomTransSmth(nn.Module):
    def __init__(self, gamma, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.wr = nn.Parameter(torch.rand((self.dim, self.dim)), requires_grad=True)
        self.wh = nn.Parameter(torch.rand((self.dim, self.dim)), requires_grad=True)


    def forward(self, h, r):
        t = h@self.wh + r@self.wr
        return t
    

class CustomTransSmth(nn.Module):
    def __init__(self, gamma, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.wr = nn.Parameter(torch.rand((self.dim, self.dim)), requires_grad=True)
        self.wh = nn.Parameter(torch.rand((self.dim, self.dim)), requires_grad=True)


    def forward(self, h, r):
        t = h@self.wh * r@self.wr
        return t