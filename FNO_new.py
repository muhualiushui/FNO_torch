import torch
import torch.nn as nn
import torch.nn.functional as F
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
                 n_blocks: int = 4,
                 distributor: bool = False,
                 activation: Callable = nn.GELU()):
        super().__init__()
        self.distributor = distributor
        self.ndim = len(modes)
        ConvNd = getattr(nn, f'Conv{self.ndim}d')
        self.lift = ConvNd(in_c, width, kernel_size=1)
        if self.distributor:
            # Per-channel independent assemblies
            self.assemblies = nn.ModuleList([
                nn.ModuleList([
                    FNOBlockNd(width, width, modes, activation)
                    for _ in range(n_blocks)
                ])
                for _ in range(out_c)
            ])
            self.proj = ConvNd(width, 1, kernel_size=1)
            self.out_c = out_c
        else:
            # Shared blocks for all channels
            self.blocks = nn.ModuleList([
                FNOBlockNd(width, width, modes, activation)
                for _ in range(n_blocks)
            ])
            self.proj = ConvNd(width, out_c, kernel_size=1)
        # Loss functions remain the same
        self.loss_fn = nn.MSELoss()
        self.loss_fnV2 = DiceCELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.lift(x)
        if self.distributor:
            outputs = []
            for assembly in self.assemblies:
                x_branch = x0
                for blk in assembly:
                    x_branch = blk(x_branch)
                out = self.proj(x_branch)
                outputs.append(out)
            return torch.cat(outputs, dim=1)
        else:
            x_branch = x0
            for blk in self.blocks:
                x_branch = blk(x_branch)
            return self.proj(x_branch)

    def cal_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(x)
        return self.loss_fnV2(outputs, y)