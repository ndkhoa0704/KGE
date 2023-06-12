import torch.nn as nn
import torch
from logger_config import logger
import torch.nn.functional as F


# class CustomTransSmth(nn.Module):
#     def __init__(self, dim, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.dim = dim
#         self.wr = nn.Parameter(torch.rand((self.dim, self.dim)), requires_grad=True)
#         self.wh = nn.Parameter(torch.rand((self.dim, self.dim)), requires_grad=True)
#         self.lamb = nn.Parameter(torch.rand((1,)), requires_grad=True)

#     def forward(self, h, r):
#         t = torch.sigmoid(h@self.wh.expand(h.shape[0], -1, -1) + r@self.wr.expand(h.shape[0], -1, -1) + self.lamb)
#         return t
    

class CustomTransSmth(nn.Module):
    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        # self.wr = nn.Parameter(torch.eye(self.dim), requires_grad=True)
        # self.wh = nn.Parameter(torch.eye(self.dim), requires_grad=True)
        self.wr = nn.Parameter(torch.ones((self.dim, self.dim)), requires_grad=True)
        self.wh = nn.Parameter(torch.ones((self.dim, self.dim)), requires_grad=True)
        self.bias = nn.Parameter(torch.ones((1,)), requires_grad=True)

        self.linearH = nn.Linear(self.dim, self.dim)
        self.linearR = nn.Linear(self.dim, self.dim)
        self.linearT = nn.Linear(self.dim, self.dim)



    def forward(self, h, r):
        h = self.linearH(h)
        r = self.linearR(r)
        t = torch.matmul(h, self.wh) + torch.matmul(r, self.wr) + self.bias
        t = self.linearT(t)
        t = F.sigmoid(t)
        return t