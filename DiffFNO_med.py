import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, List
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import math
from FNO_torch.Diffusion.diffusion_meg import Diffusion, ConditionModel
from FNO_torch.helper.SS_Former import SS_Former
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
                 heads: int = 1,
                 n_blocks: int = 4):
        super().__init__()
        self.ndim = len(modes)
        ConvNd = getattr(nn, f'Conv{self.ndim}d')
        self.lift = ConvNd(in_c, width, kernel_size=1)
        self.assemblies = nn.ModuleList([
            SS_Former(width, heads, fno_modes=modes, nbf_num_blocks=3, nbf_hidden_channels=32)
            for _ in range(n_blocks)
        ])
        self.proj = ConvNd(width, out_c, kernel_size=1)
        self.loss_fn = nn.MSELoss()

        # time embedding modules
        self.time_embed_dim = width
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.GELU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.cond_model = ConditionModel(in_c, out_c, width//2)

    def forward(self,x_t, t, cond_unet_out):
        # exactly the same logic you had in FNOnd.forward
        t_emb = get_timestep_embedding(self.time_embed_dim, t)  
        t_emb = self.time_mlp(t_emb)
        diff_t, cond_t = self.lift(x_t), cond_unet_out
        for assembly in self.assemblies:
            cond_t = assembly(diff_t, cond_t, t_emb)
        return self.proj(cond_t)
    
    def cal_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(x, y, torch.tensor([0.0], device=x.device))
        return self.loss_fn(outputs, y)