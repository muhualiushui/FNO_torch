import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, List
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
# from Brats.module.medseg import MedSegDiff
from FNO_torch.Diffusion.diffusion_basic import Diffusion
from FNO_torch.helper.SS_Former import NBPFilter
from FNO_torch.helper.Func import DiceCELoss, FNOBlockNd, get_timestep_embedding


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
                 n_blocks: int = 4,
                 is_filter: bool = False):
        super().__init__()
        self.ndim = len(modes)
        ConvNd = getattr(nn, f'Conv{self.ndim}d')
        self.lift = ConvNd(in_c, width, kernel_size=1)
        self.blocks = nn.ModuleList([
            FNOBlockNd(width, width, modes, activation, is_filter)
            for _ in range(n_blocks)
        ])
        self.proj = ConvNd(width, out_c, kernel_size=1)
        self.loss_fn = nn.MSELoss()
        self.loss_fnV2 = DiceCELoss()
        self.out_c = out_c

        # time embedding modules
        self.time_embed_dim = width
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.GELU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

    def forward(self, x, t, image):
        # exactly the same logic you had in FNOnd.forward
        x = torch.cat([x, image], dim=1)
        x0 = self.lift(x)
        width = x0.shape[1]
        t_emb = get_timestep_embedding(width, t)  
        t_emb = self.time_mlp(t_emb)
        x_branch = x0
        for blk in self.blocks:
            x_branch = blk(x_branch, t_emb)
        return self.proj(x_branch)
    
    def cal_loss(self, image: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        return self.Diffusion.cal_loss(x0, image)