import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, List
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from FNO_torch.helper.Func import DiceCELoss, FNOBlockNd, FNOBlockNd_NBF, get_timestep_embedding


class FNOnd(nn.Module):
    """
    N-dimensional FNO model.
    modes: list specifying the number of Fourier modes per dimension.
    """
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 modes: List[int],
                 width: int,
                 activation: Callable,
                 n_blocks: int = 4):
        super().__init__()
        self.ndim = len(modes)
        ConvNd = getattr(nn, f'Conv{self.ndim}d')
        self.lift = ConvNd(in_c, width, kernel_size=1)
        self.assemblies = nn.ModuleList([
            nn.ModuleList([
                FNOBlockNd(width, width, modes, activation)
                for _ in range(n_blocks)
            ])
            for _ in range(out_c)
        ])
        self.proj = ConvNd(width, 1, kernel_size=1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fnV2 = DiceCELoss()
        self.out_c = out_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.lift(x)
        outputs = []
        for assembly in self.assemblies:
            x_branch = x0
            for blk in assembly:
                x_branch = blk(x_branch)
            out = self.proj(x_branch)
            outputs.append(out)
        return torch.cat(outputs, dim=1)