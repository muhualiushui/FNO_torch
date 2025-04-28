import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, List
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Data loading and loss
# -----------------------------------------------------------------------------
def get_dataloader(x: torch.Tensor, y: torch.Tensor,
                   batch_size: int, shuffle: bool = True) -> DataLoader:
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def loss_fn(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_pred = model(x)
    return F.mse_loss(y_pred, y)

# -----------------------------------------------------------------------------
# 1D Spectral Convolution
# -----------------------------------------------------------------------------
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        # Create complex weights of shape (in, out, modes)
        self.weight = nn.Parameter(
            (torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
             * 2 * scale) - scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, length)
        x_fft = torch.fft.rfft(x)             # => (batch, in_channels, length_fft)
        x_fft = x_fft[:, :, :self.modes]      # trim to modes
        out_fft = torch.einsum("bim, iom -> bom", x_fft, self.weight)
        # back to time domain
        return torch.fft.irfft(out_fft, n=x.size(-1))

# -----------------------------------------------------------------------------
# 1D FNO block and model
# -----------------------------------------------------------------------------
class FNOBlock1d(nn.Module):
    def __init__(self, in_c: int, out_c: int, modes: int, activation: Callable):
        super().__init__()
        self.spectral = SpectralConv1d(in_c, out_c, modes)
        self.bypass   = nn.Conv1d(in_c, out_c, kernel_size=1)
        self.act      = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.bypass(x))

class FNO1d(nn.Module):
    def __init__(self, in_c: int, out_c: int, modes: int,
                 width: int, activation: Callable, n_blocks: int = 4):
        super().__init__()
        self.lift = nn.Conv1d(in_c, width, kernel_size=1)
        self.blocks = nn.ModuleList([
            FNOBlock1d(width, width, modes, activation)
            for _ in range(n_blocks)
        ])
        self.proj = nn.Conv1d(width, out_c, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        return self.proj(x)

    def train_model(self,
                    train_loader: DataLoader,
                    test_loader: DataLoader,
                    optimizer: Optimizer,
                    epochs: int,
                    device: torch.device,
                    x_name: str,
                    y_name: str,
                    ):
        self.to(device)
        history = {'train_loss': [], 'val_loss': []}
        pbar = tqdm(range(epochs), desc='Epoch', unit='epoch')
        for epoch in pbar:
            # -- train --
            self.train()
            running = 0.0
            for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False, unit="batch", position=0):
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = F.mse_loss(self(xb), yb)
                loss.backward()
                optimizer.step()
                running += loss.item() * xb.size(0)
            history['train_loss'].append(running / len(train_loader.dataset))

            # -- validate --
            self.eval()
            val_running = 0.0
            with torch.no_grad():
                for xb, yb in tqdm(test_loader, desc=f"Epoch {epoch} Val", leave=False, unit="batch", position=0):
                    xb, yb = xb.to(device), yb.to(device)
                    val_running += F.mse_loss(self(xb), yb).item() * xb.size(0)
            history['val_loss'].append(val_running / len(test_loader.dataset))
            pbar.set_postfix(train_loss=history['train_loss'][-1], val_loss=history['val_loss'][-1])

        return history

# -----------------------------------------------------------------------------
# 2D Spectral Convolution
# -----------------------------------------------------------------------------
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int, modes2: int):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            (torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
             * 2 * scale) - scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, H, W)
        x_fft = torch.fft.rfft2(x, norm='ortho')
        x_fft = x_fft[:, :, :self.modes1, :self.modes2]
        out_fft = torch.einsum("bikl, iokl -> bokl", x_fft, self.weight)
        return torch.fft.irfft2(out_fft,
                                s=(x.size(-2), x.size(-1)),
                                norm='ortho')

# -----------------------------------------------------------------------------
# 2D FNO block and model
# -----------------------------------------------------------------------------
class FNOBlock2d(nn.Module):
    def __init__(self, in_c: int, out_c: int,
                 modes1: int, modes2: int, activation: Callable):
        super().__init__()
        self.spectral = SpectralConv2d(in_c, out_c, modes1, modes2)
        self.bypass   = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.act      = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.bypass(x))

class FNO2d(nn.Module):
    def __init__(self, in_c: int, out_c: int, modes1: int,
                 modes2: int, width: int, activation: Callable,
                 n_blocks: int = 4, classification: bool = False):
        super().__init__()
        self.lift = nn.Conv2d(in_c, width, kernel_size=1)
        self.blocks = nn.ModuleList([
            FNOBlock2d(width, width, modes1, modes2, activation)
            for _ in range(n_blocks)
        ])
        self.proj = nn.Conv2d(width, out_c, kernel_size=1)

        if classification:
            # multi-label: 4 independent binary masks
            self.loss_fn = lambda pred, y: (
                F.binary_cross_entropy_with_logits(pred, y.float(), reduction='none')
                 # pred & y are (B, C, H, W) → BCE w/o reduction gives (B, C, H, W)
                 .mean(dim=(2, 3))            # collapse H,W → (B, C)
            )
        else:
            self.loss_fn = lambda pred, y: (
                F.mse_loss(pred, y, reduction='none')
                 .mean(dim=(2, 3))
            )




    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        return self.proj(x)


    def train_model(self,
                    train_loader: DataLoader,
                    test_loader: DataLoader,
                    optimizer: Optimizer,
                    epochs: int,
                    device: torch.device,
                    x_name: str,
                    y_name: str,
                    ):
        self.to(device)
        history = {'train_loss': [], 'val_loss': []}

        pbar = tqdm(range(epochs), desc='Epoch', unit='epoch')
        for epoch in pbar:
            self.train()
            running = []
            for batch in tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False, unit="batch", position=0):
                xb, yb = batch[x_name], batch[y_name]
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = self.loss_fn(self(xb), yb).sum(dim =1).mean()
                loss.backward()
                optimizer.step()
                running.append(loss.item())
            history['train_loss'].append(sum(running) / len(running))

            self.eval()
            val_running = []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Epoch {epoch} Val", leave=False, unit="batch", position=0):
                    xb, yb = batch[x_name], batch[y_name]
                    xb, yb = xb.to(device), yb.to(device)
                    loss = self.loss_fn(self(xb), yb).sum(dim =1).mean()
                    val_running.append(loss.item())
            history['val_loss'].append(sum(val_running) / len(val_running))
            pbar.set_postfix(train_loss=history['train_loss'][-1], val_loss=history['val_loss'][-1])

        return history
    
    def plot_loss(self, history: dict, save_path: str = None):
        """
        Plot the training and validation loss.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()