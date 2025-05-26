import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, List
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import math
from FNO_torch.Diffusion.diffusionV4 import Diffusion, ConditionModel
from FNO_torch.FNO_evolve.SS_Former import SS_Former



class Denoiser(nn.Module):
    def __init__(self, lift: nn.Module, assemblies: nn.ModuleList, proj: nn.Module,
                 get_timestep_embedding: Callable, time_mlp: nn.Module):
        super().__init__()

        self.lift = lift
        self.assemblies = assemblies
        self.proj = proj
        self.get_timestep_embedding = get_timestep_embedding
        self.time_mlp = time_mlp

    def forward(self,x_t, cond_unet_out, t):
        # exactly the same logic you had in FNOnd.forward
        t_emb = self.get_timestep_embedding(t)  
        t_emb = self.time_mlp(t_emb)
        diff_0, cond_0 = self.lift(x_t), cond_unet_out
        for assembly in self.assemblies:
            diff_b, cond_b = diff_0, cond_0
            for blk in assembly:
                cond_b = blk(diff_b, cond_b, t_emb)
        return self.proj(cond_b)


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
                 n_blocks: int = 4):
        super().__init__()
        self.ndim = len(modes)
        ConvNd = getattr(nn, f'Conv{self.ndim}d')
        self.lift = ConvNd(in_c, width, kernel_size=1)
        self.assemblies = nn.ModuleList([
            SS_Former(width, heads=1, dim_head=width, fno_modes=modes, nbf_num_blocks=3, nbf_hidden_channels=32)
            for _ in range(n_blocks)
        ])
        self.proj = ConvNd(width, out_c, kernel_size=1)
        self.loss_fn = nn.MSELoss()
        self.out_c = out_c

        # time embedding modules
        self.time_embed_dim = width
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.GELU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.denoiser = Denoiser(
            lift=self.lift,
            assemblies=self.assemblies,
            proj=self.proj,
            get_timestep_embedding=self.get_timestep_embedding,
            time_mlp=self.time_mlp
        )
        self.cond_model = ConditionModel(in_c, out_c, width//2)

        self.Diffusion = Diffusion(
            self.denoiser,
            self.cond_model,
            timesteps=1000
        )

    def get_timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        """
        half_dim = self.time_embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.time_embed_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, x, t, c):
        return self.denoiser(x, t, c)

    # keep the standard nn.Module.train(mode=True) behaviour
    def train(self, mode: bool = True):
        return super().train(mode)

    def train_epoch(self,
                    train_loader: DataLoader,
                    optimizer: Optimizer,
                    device: torch.device,
                    *,
                    x_name: str | None = None,
                    y_name: str | None = None) -> float:
        """
        Run one training epoch and return average loss.
        """
        super().train()
        running = 0.0
        total = 0
        pbar = tqdm(train_loader, desc='Train', position=1, leave=False, dynamic_ncols=True)
        for batch in pbar:
            # Support both tuple/list batches and dict batches keyed by x_name/y_name
            if isinstance(batch, dict):
                if x_name is None or y_name is None:
                    raise ValueError("When batches are dictionaries, x_name and y_name must be provided.")
                xb = batch[x_name].to(device)
                yb = batch[y_name].to(device)
            elif isinstance(batch, (list, tuple)):
                xb = batch[0].to(device)
                yb = batch[1].to(device)
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")
            optimizer.zero_grad()
            loss = self.Diffusion(yb, xb)
            # loss = self.loss_fn(self(xb), yb)
            # loss = self.loss_fnV2(self(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            total += xb.size(0)
        return running / total

    def valid_epoch(self,
                    test_loader: DataLoader,
                    device: torch.device,
                    *,
                    x_name: str | None = None,
                    y_name: str | None = None) -> float:
        """
        Run one validation epoch and return average loss.
        """
        super().eval()
        val_running = 0.0
        total = 0
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Valid', position=1, leave=False, dynamic_ncols=True)
            for batch in pbar:
                # Support both tuple/list batches and dict batches keyed by x_name/y_name
                if isinstance(batch, dict):
                    if x_name is None or y_name is None:
                        raise ValueError("When batches are dictionaries, x_name and y_name must be provided.")
                    xb = batch[x_name].to(device)
                    yb = batch[y_name].to(device)
                elif isinstance(batch, (list, tuple)):
                    xb = batch[0].to(device)
                    yb = batch[1].to(device)
                else:
                    raise TypeError(f"Unsupported batch type: {type(batch)}")
                loss = self.Diffusion(yb, xb)
                # loss = self.loss_fnV2(self(xb), yb)
                val_running += loss.item() * xb.size(0)
                total += xb.size(0)
        return val_running / total

    def train_model(self,
                    train_loader: DataLoader,
                    test_loader: DataLoader,
                    optimizer: Optimizer,
                    epochs: int,
                    device: torch.device,
                    x_name: str = None,
                    y_name: str = None):
        """
        Train the model with separate train() and valid() steps, plus early stopping.
        """
        self.to(device)
        history = {'train_loss': [], 'val_loss': []}
        pbar = tqdm(range(epochs), desc='Epoch', unit='epoch', leave=True, dynamic_ncols=True, position=0)
        for epoch in pbar:
            train_loss = self.train_epoch(train_loader, optimizer, device,
                                    x_name=x_name, y_name=y_name)
            val_loss   = self.valid_epoch(test_loader, device,
                                    x_name=x_name, y_name=y_name)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            pbar.set_postfix(train_loss=train_loss, val_loss=val_loss)
        return history

    def plot_loss(self, history: dict, save_path: str = None):
        """
        Plot training and validation loss curves.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()