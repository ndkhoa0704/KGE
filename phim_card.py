import torch
from utils import move_to_cuda

if __name__=='__main__':
    a  = torch.rand((80000, 40, 768))
    a = move_to_cuda(a)

    while True:
        pass