import torch.nn as nn
import torch
from logger_config import logger


class CustomTransSmth(nn.Module):
    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.wr = nn.Parameter(torch.rand((self.dim, self.dim)), requires_grad=True)
        self.wh = nn.Parameter(torch.rand((self.dim, self.dim)), requires_grad=True)
        self.lamb = nn.Parameter(torch.rand((1,)), requires_grad=True)

    def forward(self, h, r):
        t = h@self.wh.expand(h.shape[0], -1, -1) + r@self.wr.expand(h.shape[0], -1, -1) + self.lamb
        return t
    

# class CustomTransSmth(nn.Module):
#     def __init__(self, gamma, dim, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.dim = dim
#         self.wr = nn.Parameter(torch.rand((self.dim, self.dim)), requires_grad=True)
#         self.wh = nn.Parameter(torch.rand((self.dim, self.dim)), requires_grad=True)


#     def forward(self, h, r):
#         t = h@self.wh * r@self.wr
#         return t