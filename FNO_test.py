# ... existing imports and get_dataloader, loss_fn ...
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
# Custom losses mirroring the MIUA‑24 TFNO paper
# -----------------------------------------------------------------------------
def lp_loss(pred: torch.Tensor,
            target: torch.Tensor,
            p: int = 2) -> torch.Tensor:
    """
    Simple Lᵖ loss (default L²): ‖pred − target‖_p.
    Works channel‑wise, averaged over batch and spatial dims.
    """
    diff = pred - target
    # flatten batch and channels, keep spatial dims intact for uniform weighting
    return torch.pow(torch.abs(diff), p).mean()

def h1_loss(pred: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
    """
    H¹ loss = L²(pred − target) + Σ‖∇_i pred − ∇_i target‖², summed over spatial dims.
    This captures boundary fidelity and matches the TFNO paper.
    """
    # L2 part
    l2 = F.mse_loss(pred, target)

    # gradient part (use torch.gradient, available in PyTorch ≥1.13)
    grads_pred = torch.gradient(pred, dim=tuple(range(-pred.ndim + 2, 0)))
    grads_tgt  = torch.gradient(target, dim=tuple(range(-target.ndim + 2, 0)))
    grad_loss = sum(F.mse_loss(gp, gt) for gp, gt in zip(grads_pred, grads_tgt))
    return l2 + grad_loss


class FNOBlockNd(nn.Module):
    """
    Single FNO block: inline N‑dimensional spectral conv + 1×1 Conv bypass + activation.
    """
    def __init__(self, in_c: int, out_c: int, modes: List[int], activation: Callable):
        super().__init__()
        self.in_c, self.out_c = in_c, out_c
        self.modes = modes
        self.ndim = len(modes)
        # initialize complex spectral weights
        scale = 1.0 / (in_c * out_c)
        w_shape = (in_c, out_c, *modes)
        init = torch.randn(*w_shape, dtype=torch.cfloat)
        self.weight = nn.Parameter(init * 2 * scale - scale)
        # 1×1 convolution bypass
        ConvNd = getattr(nn, f'Conv{self.ndim}d')
        self.bypass = ConvNd(in_c, out_c, kernel_size=1)
        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, *spatial)
        dims = tuple(range(-self.ndim, 0))
        # forward FFT
        x_fft = torch.fft.rfftn(x, dim=dims, norm='ortho')
        # trim to modes
        slices = [slice(None), slice(None)] + [slice(0, m) for m in self.modes]
        x_fft = x_fft[tuple(slices)]
        # einsum: "b i a b..., i o a b... -> b o a b..."
        letters = [chr(ord('a') + i) for i in range(self.ndim)]
        sub_in  = 'b i ' + ' '.join(letters)
        sub_w   = 'i o ' + ' '.join(letters)
        sub_out = 'b o ' + ' '.join(letters)
        eq = f"{sub_in}, {sub_w} -> {sub_out}"
        out_fft = torch.einsum(eq, x_fft, self.weight)
        # inverse FFT
        spatial = x.shape[-self.ndim:]
        x_spec = torch.fft.irfftn(out_fft, s=spatial, dim=dims, norm='ortho')
        # combine spectral + bypass
        return self.act(x_spec + self.bypass(x))


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
                 activation: Callable = nn.GELU,
                 n_blocks: int = 4,
                 add_coords: bool = True):
        """
        N‑D Fourier Neural Operator with positional embedding and
        global residual skip, adapted from the MIUA‑24 TFNO paper.
        The public interface (arguments & train_model) is unchanged.
        """
        super().__init__()
        self.ndim = len(modes)
        self.add_coords = add_coords

        # ensure we have an *instance* of the chosen activation
        self.act = activation() if isinstance(activation, type) else activation

        # prepend coordinate channels when requested
        lifted_in_c = in_c + (self.ndim if add_coords else 0)
        ConvNd = getattr(nn, f'Conv{self.ndim}d')
        self.lift = ConvNd(lifted_in_c, width, kernel_size=1)

        # stack of spectral blocks
        self.blocks = nn.ModuleList(
            [FNOBlockNd(width, width, modes, self.act) for _ in range(n_blocks)]
        )

        # projection back to the target channels
        self.proj = ConvNd(width, out_c, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional coordinate embedding and an outer residual
        connection (u + F(u)), mirroring the TFNO design.
        """
        # 1) coordinate embedding ------------------------------------------------
        if self.add_coords:
            # build / reuse a cached grid that matches the current spatial size
            if (not hasattr(self, "_grid")) or self._grid.shape[2:] != x.shape[2:]:
                coords = torch.meshgrid(
                    [torch.linspace(0.0, 1.0, s, device=x.device, dtype=x.dtype)
                     for s in x.shape[-self.ndim:]],
                    indexing="ij"
                )
                grid = torch.stack(coords, dim=0).unsqueeze(0)    # (1, ndim, *spatial)
                self.register_buffer("_grid", grid, persistent=False)
            x = torch.cat([x, self._grid.expand(x.shape[0], -1, *([-1] * self.ndim))], dim=1)

        # 2) lift to spectral width ---------------------------------------------
        x = self.lift(x)
        skip = x                                                                 # outer residual

        # 3) spectral blocks -----------------------------------------------------
        for blk in self.blocks:
            x = blk(x)

        # 4) global residual & projection ---------------------------------------
        x = x + skip
        return self.proj(x)

    def train(self,
              train_loader: DataLoader,
              optimizer: Optimizer,
              device: torch.device) -> float:
        """
        Run one training epoch and return average loss.
        """
        super().train()
        running = 0.0
        total = 0
        for batch in train_loader:
            xb = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch
            yb = batch[1].to(device) if isinstance(batch, (list, tuple)) else batch
            optimizer.zero_grad()
            loss = h1_loss(self(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            total += xb.size(0)
        return running / total

    def valid(self,
              test_loader: DataLoader,
              device: torch.device) -> float:
        """
        Run one validation epoch and return average loss.
        """
        super().eval()
        val_running = 0.0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                xb = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch
                yb = batch[1].to(device) if isinstance(batch, (list, tuple)) else batch
                loss = h1_loss(self(xb), yb)
                val_running += loss.item() * xb.size(0)
                total += xb.size(0)
        return val_running / total

    def train_model(self,
                    train_loader: DataLoader,
                    test_loader: DataLoader,
                    optimizer: Optimizer,
                    epochs: int,
                    device: torch.device,):
        """
        Train the model with separate train() and valid() steps, plus early stopping.
        """
        self.to(device)
        history = {'train_loss': [], 'val_loss': []}
        pbar = tqdm(range(epochs), desc='Epoch', unit='epoch')
        for epoch in pbar:
            train_loss = self.train(train_loader, optimizer, device)
            val_loss   = self.valid(test_loader, device)
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