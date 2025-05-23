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

class NBPFilter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_emb_dim: int,
        hidden_channels: int = 64,
        num_blocks: int = 6
    ):
        """
        Neural Band-Pass Filter that accepts an arbitrary number of input channels.
        
        Args:
            in_channels: number of channels in the input M (was 2 before for coords).
            time_emb_dim: dimensionality of the diffusion timestep embedding.
            hidden_channels: number of feature maps in each conv block.
            num_blocks: number of convolutional blocks (R in the paper).
        """
        super().__init__()
        # project from in_channels → hidden
        self.init_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        
        # R blocks of conv + LayerNorm + FiLM
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
            norm = nn.LayerNorm([hidden_channels, 1, 1])
            self.blocks.append(nn.ModuleDict({
                "conv": conv,
                "norm": norm,
                "mlp_scale": nn.Linear(time_emb_dim, hidden_channels),
                "mlp_shift": nn.Linear(time_emb_dim, hidden_channels),
            }))
        
        # final 1×1 conv to single-channel filter + sigmoid
        self.final_conv = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, M: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            M: input tensor (B, in_channels, H, W)
            t_emb: timestep embedding (B, time_emb_dim)
        Returns:
            M_filtered: same shape as M
        """
        B, C, H, W = M.shape

        # 1) initial projection
        x = self.init_conv(M)  # → (B, hidden, H, W)

        # 2) R blocks of conv + LayerNorm + FiLM
        for blk in self.blocks:
            x = blk["conv"](x)
            # LayerNorm over each channel:
            x = x.view(B, x.shape[1], -1)
            x = blk["norm"](x)
            x = x.view(B, x.shape[1], H, W)

            # FiLM
            scale = blk["mlp_scale"](t_emb).view(B, -1, 1, 1)
            shift = blk["mlp_shift"](t_emb).view(B, -1, 1, 1)
            x = F.relu(x * scale + shift, inplace=True)

        # 3) produce one filter per input channel and apply
        filt = self.act(self.final_conv(x))  # (B, in_channels, H, W)
        return M * filt

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        """
        Cross‐attention block.
        
        Args:
            dim: number of channels in each U-Net feature map
            heads: number of attention heads
            dim_head: dimensionality of each head
        """
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, Q_condidate: torch.Tensor, K_condidate: torch.Tensor):
        """
        Args:
            Q_condidate: shape (B, C, H, W)
            K_condidate: shape (B, C, H, W)
        Returns:
            out: attended features, shape (B, C, H, W)
            K: key tensor before splitting heads, shape (B, N, heads*dim_head)
            Q: query tensor before splitting heads, shape (B, N, heads*dim_head)
        """
        B, C, H, W = Q_condidate.shape
        N = H * W

        # flatten spatial dims → (B, N, C)
        q_in = Q_condidate.flatten(2).permute(0, 2, 1)
        k_in = K_condidate.flatten(2).permute(0, 2, 1)
        v_in = k_in

        # project
        Q = self.to_q(q_in)  # (B, N, heads*dim_head)
        K = self.to_k(k_in)  # (B, N, heads*dim_head)
        V = self.to_v(v_in)  # (B, N, heads*dim_head)

        # split heads
        Q_heads = Q.view(B, N, self.heads, -1).permute(0, 2, 1, 3)  # (B, heads, N, dim_head)
        K_heads = K.view(B, N, self.heads, -1).permute(0, 2, 1, 3)
        V_heads = V.view(B, N, self.heads, -1).permute(0, 2, 1, 3)

        # scaled dot‐product attention
        attn = torch.matmul(Q_heads, K_heads.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, V_heads)  # (B, heads, N, dim_head)

        # merge heads & project out
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.heads * (out.size(-1)))
        out = self.to_out(out)  # (B, N, C)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        # reshape Q and K back to spatial grids
        Q_spatial = Q.permute(0, 2, 1).view(B, C, H, W)
        K_spatial = K.permute(0, 2, 1).view(B, C, H, W)
        return out, K_spatial, Q_spatial



class FNOBlockNd(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        modes: List[int],
        time_emb_dim: int,
        nbf_hidden_channels: int = 64,
        nbf_num_blocks: int = 6
    ):
        super().__init__()
        self.in_c, self.out_c = in_c, out_c
        self.modes = modes
        self.ndim = len(modes)
        # initialize complex spectral weights
        scale = 1.0 / (in_c * out_c)
        w_shape = (in_c, out_c, *modes)
        init = torch.randn(*w_shape, dtype=torch.cfloat)
        self.weight = nn.Parameter(init * 2 * scale - scale)

        # NBP filter for combining two inputs
        self.nbp_filter = NBPFilter(
            in_channels=out_c,
            time_emb_dim=time_emb_dim,
            hidden_channels=nbf_hidden_channels,
            num_blocks=nbf_num_blocks
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x1, x2: (B, C, *spatial)
        dims = tuple(range(-self.ndim, 0))
        # Setup slicing for trimming to modes
        slices = [slice(None), slice(None)] + [slice(0, m) for m in self.modes]
        letters = [chr(ord('k') + i) for i in range(self.ndim)]
        sub_in  = 'bi' + ''.join(letters)
        sub_w   = 'io' + ''.join(letters)
        sub_out = 'bo' + ''.join(letters)
        eq = f"{sub_in}, {sub_w} -> {sub_out}"

        # 1) FFT, trim, filter for x1
        x1_fft = torch.fft.rfftn(x1, dim=dims, norm='ortho')
        x1_fft = x1_fft[tuple(slices)]
        out1_fft = torch.einsum(eq, x1_fft, self.weight)

        # 2) FFT, trim, filter for x2
        x2_fft = torch.fft.rfftn(x2, dim=dims, norm='ortho')
        x2_fft = x2_fft[tuple(slices)]
        out2_fft = torch.einsum(eq, x2_fft, self.weight)

        # 3) inverse FFT back to spatial for both
        spatial = x1.shape[-self.ndim:]
        x1_spec = torch.fft.irfftn(out1_fft, s=spatial, dim=dims, norm='ortho')
        x2_spec = torch.fft.irfftn(out2_fft, s=spatial, dim=dims, norm='ortho')

        # 4) combine via element-wise product
        x_combined = x1_spec * x2_spec

        # 5) apply Neural Band-Pass filter conditioned on t_emb
        x_filtered = self.nbp_filter(x_combined, t_emb)

        return x_filtered


# --- ATTFNOBlock: Fusion Attention + FNO block ---
class ATTFNOBlock(nn.Module):
    """
    Fusion Attention + FNO block.
    Takes Q_candidate, K_candidate, and timestep embedding.
    Applies cross-attention, feeds Q/K into FNOBlock, and combines with attention output.
    """
    def __init__(
        self,
        width: int,
        heads: int,
        dim_head: int,
        fno_modes: List[int],
        nbf_hidden_channels: int = 64,
        nbf_num_blocks: int = 6
    ):
        super().__init__()
        # Enforce that heads * dim_head matches model width for consistency
        assert heads * dim_head == width, "heads * dim_head must equal width for consistent channel size"
        # cross-attention module
        self.cross_attn = CrossAttention(dim=width, heads=heads, dim_head=dim_head)
        # FNO block input/output channels equal to width for consistency
        self.fno_block = FNOBlockNd(
            in_c=width,
            out_c=width,
            modes=fno_modes,
            time_emb_dim=width,
            nbf_hidden_channels=nbf_hidden_channels,
            nbf_num_blocks=nbf_num_blocks
        )

    def forward(
        self,
        Q_candidate: torch.Tensor,
        K_candidate: torch.Tensor,
        t_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            Q_candidate: tensor (B, width, H, W) for queries
            K_candidate: tensor (B, width, H, W) for keys/values
            t_emb: timestep embedding (B, width)
        Returns:
            fused output tensor (B, width, H, W)
        """
        # cross-attention to get attended out, and spatial Q/K
        attn_out, K_spatial, Q_spatial = self.cross_attn(Q_candidate, K_candidate)

        # process Q and K through FNO block
        fno_out = self.fno_block(Q_spatial, K_spatial, t_emb)  # (B, width, H, W)

        # element-wise fuse with attention output
        fused = fno_out * attn_out
        return fused


# --- SS_Former: Sequential Stacked ATTFNOBlocks ---
class SS_Former(nn.Module):
    """
    SS-Former module: applies two sequential ATTFNOBlocks.
    """
    def __init__(
        self,
        width: int,
        heads: int,
        dim_head: int,
        fno_modes: List[int],
        nbf_hidden_channels: int = 64,
        nbf_num_blocks: int = 6
    ):
        super().__init__()
        # first fusion block: condition then diffusion
        self.fatt1 = ATTFNOBlock(
            width=width,
            heads=heads,
            dim_head=dim_head,
            fno_modes=fno_modes,
            nbf_hidden_channels=nbf_hidden_channels,
            nbf_num_blocks=nbf_num_blocks
        )
        # second fusion block: diffusion then former output
        self.fatt2 = ATTFNOBlock(
            width=width,
            heads=heads,
            dim_head=dim_head,
            fno_modes=fno_modes,
            nbf_hidden_channels=nbf_hidden_channels,
            nbf_num_blocks=nbf_num_blocks
        )

    def forward(
        self,
        diff_unet_out: torch.Tensor,
        cond_unet_out: torch.Tensor,
        t_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            t_emb: timestep embedding, shape (B, time_emb_dim)
            diff_unet_out: diffusion U-Net output, shape (B, width, H, W)
            cond_unet_out: condition U-Net output, shape (B, width, H, W)
        Returns:
            output tensor, shape (B, width, H, W)
        """
        # first pass: condition as Q, diffusion as K
        former_output = self.fatt1(cond_unet_out, diff_unet_out, t_emb)
        # second pass: diffusion as Q, former_output as K
        later_output = self.fatt2(diff_unet_out, former_output, t_emb)
        return later_output



class Denoiser(nn.Module):
    def __init__(self, lift: nn.Module, assemblies: nn.ModuleList, proj: nn.Module,
                 get_timestep_embedding: Callable, time_mlp: nn.Module):
        super().__init__()

        self.lift = lift
        self.assemblies = assemblies
        self.proj = proj
        self.get_timestep_embedding = get_timestep_embedding
        self.time_mlp = time_mlp

    def forward(self,diff_unet_out, cond_unet_out, t):
        # exactly the same logic you had in FNOnd.forward
        t_emb = self.get_timestep_embedding(t)  
        t_emb = self.time_mlp(t_emb)
        diff_0, cond_0 = self.lift(diff_unet_out), self.lift(cond_unet_out)
        outputs = []
        for assembly in self.assemblies:
            diff_b, cond_b = diff_0, cond_0
            for blk in assembly:
                cond_b = blk(diff_b, cond_b, t_emb)
            outputs.append(self.proj(cond_b))
        return torch.cat(outputs, dim=1)


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
                SS_Former(width, heads=1, dim_head=width, fno_modes=modes, nbf_num_blocks=6, nbf_hidden_channels=64)
                for _ in range(n_blocks)
            ])
            for _ in range(out_c)
        ])
        self.proj = ConvNd(width, 1, kernel_size=1)
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
        self.cond_model = ConditionModel(in_c, out_c, width/2)

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