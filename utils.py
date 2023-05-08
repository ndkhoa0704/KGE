import os
import glob
import torch
import shutil
from pathlib import Path

import numpy as np
import torch.nn as nn

from logger_config import logger


class AttrDict:
    pass


def save_model(epoch, model, optimizer, best=False):
    """Saves model checkpoint on given epoch with given data name.
    """
    checkpoint_folder = Path.cwd() / 'model_checkpoints'
    if not checkpoint_folder.is_dir():
        checkpoint_folder.mkdir()
    if not best:
        file = checkpoint_folder / f'epoch_{epoch}.pt'
    else: 
        file = checkpoint_folder / f'best.pt'
    if not file.is_file():
        file.touch()
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        file
                )
    return True

def load_model(epoch, model, optimizer):
    """Loads model state from file.
    """
    if epoch:
        file = Path.cwd() / 'model_checkpoints' / f'epoch_{epoch}.pt'
    else:
        file = Path.cwd() / 'model_checkpoints' / f'best.pt'
    checkpoint = torch.load(file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer


def delete_old_ckt(path_pattern: str, keep=5):
    files = sorted(glob.glob(path_pattern), key=os.path.getmtime, reverse=True)
    for f in files[keep:]:
        logger.info('Delete old checkpoint {}'.format(f)) 
        os.system('rm -f {}'.format(f))


def report_num_trainable_parameters(model: torch.nn.Module) -> int:
    assert isinstance(model, torch.nn.Module), 'Argument must be nn.Module'

    num_parameters = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            num_parameters += np.prod(list(p.size()))
            logger.info('{}: {}'.format(name, np.prod(list(p.size()))))

    logger.info('Number of parameters: {}M'.format(num_parameters // 10**6))
    return num_parameters


def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, "module") else model


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


def move_to_cpu(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cpu(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cpu()
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cpu(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cpu(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return [_move_to_cpu(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cpu(sample)


class HashTensorWrapper():
    def __init__(self, tensor):
        self.tensor = move_to_cpu(tensor)

    def __hash__(self):
        return hash(self.tensor.numpy().tobytes())

    def __eq__(self, other):
        return torch.all(self.tensor == other.tensor)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def all_axis(tensor):
    new_tensor = []
    for i in tensor:
        new_tensor.append(torch.all(i))
    new_tensor = torch.tensor(new_tensor)
    return new_tensor


def rowwise_in(a, b):
    shape1 = a.shape[0]
    shape2 = b.shape[0]
    c  = a.shape[1]
    assert c == b.shape[1] , "Tensors must have same number of columns"

    a_expand = a.unsqueeze(1).expand(-1,shape2,c)
    b_expand = b.unsqueeze(0).expand(shape1,-1,c)
    # element-wise equality
    mask = (a_expand == b_expand).all(-1).any(-1)
    return ~mask