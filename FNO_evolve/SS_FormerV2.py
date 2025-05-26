import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List

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
            norm = nn.LayerNorm(hidden_channels)
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
            # apply LayerNorm over channels without flattening:
            x = x.permute(0, 2, 3, 1)         # B, H, W, C
            x = blk["norm"](x)               # LayerNorm over last dim C
            x = x.permute(0, 3, 1, 2)         # B, C, H, W

            # FiLM
            scale = blk["mlp_scale"](t_emb).view(B, -1, 1, 1)
            shift = blk["mlp_shift"](t_emb).view(B, -1, 1, 1)
            x = F.relu(x * scale + shift, inplace=True)

        # 3) produce one filter per input channel and apply
        filt = self.act(self.final_conv(x))  # (B, in_channels, H, W)
        return M * filt

# class CrossAttention(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64):
#         """
#         Cross‐attention block.
        
#         Args:
#             dim: number of channels in each U-Net feature map
#             heads: number of attention heads
#             dim_head: dimensionality of each head
#         """
#         super().__init__()
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#         inner_dim = heads * dim_head

#         self.to_q = nn.Linear(dim, inner_dim, bias=False)
#         self.to_k = nn.Linear(dim, inner_dim, bias=False)
#         self.to_v = nn.Linear(dim, inner_dim, bias=False)
#         self.to_out = nn.Linear(inner_dim, dim)

#     def forward(self, Q_condidate: torch.Tensor, K_condidate: torch.Tensor):
#         """
#         Args:
#             Q_condidate: shape (B, C, H, W)
#             K_condidate: shape (B, C, H, W)
#         Returns:
#             out: attended features, shape (B, C, H, W)
#             K: key tensor before splitting heads, shape (B, N, heads*dim_head)
#             Q: query tensor before splitting heads, shape (B, N, heads*dim_head)
#         """
#         B, C, H, W = Q_condidate.shape
#         N = H * W

#         # flatten spatial dims → (B, N, C)
#         q_in = Q_condidate.flatten(2).permute(0, 2, 1)
#         k_in = K_condidate.flatten(2).permute(0, 2, 1)
#         v_in = k_in

#         # project
#         Q = self.to_q(q_in)  # (B, N, heads*dim_head)
#         K = self.to_k(k_in)  # (B, N, heads*dim_head)
#         V = self.to_v(v_in)  # (B, N, heads*dim_head)

#         # split heads
#         Q_heads = Q.view(B, N, self.heads, -1).permute(0, 2, 1, 3)  # (B, heads, N, dim_head)
#         K_heads = K.view(B, N, self.heads, -1).permute(0, 2, 1, 3)
#         V_heads = V.view(B, N, self.heads, -1).permute(0, 2, 1, 3)

#         # offload attention computation to another GPU to save memory on the main device
#         device_main = Q_heads.device      # original device, e.g., 'cuda:0'
#         device_off = torch.device('cuda:1')  # choose another GPU, e.g., GPU 1

#         # move Q, K, V to the offload device
#         Q_off = Q_heads.to(device_off)
#         K_off = K_heads.to(device_off)
#         V_off = V_heads.to(device_off)

#         # compute attention off-device
#         attn_off = torch.matmul(Q_off, K_off.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
#         attn_off = attn_off.softmax(dim=-1)
#         out_off = torch.matmul(attn_off, V_off)  # (B, heads, N, dim_head)

#         # bring the result back to the main device
#         out = out_off.to(device_main)

#         # merge heads & project out
#         out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.heads * (out.size(-1)))
#         out = self.to_out(out)  # (B, N, C)
#         out = out.permute(0, 2, 1).view(B, C, H, W)
#         # reshape Q and K back to spatial grids
#         Q_spatial = Q.permute(0, 2, 1).view(B, C, H, W)
#         K_spatial = K.permute(0, 2, 1).view(B, C, H, W)
#         return out, K_spatial, Q_spatial

class FlashCrossAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.to_q = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_k = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_v = nn.Conv2d(channels, channels, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, Q, K):
        B, C, H, W = Q.shape
        # project and flatten → (B, N, C)
        q = self.to_q(Q).flatten(2).permute(0, 2, 1)
        k = self.to_k(K).flatten(2).permute(0, 2, 1)
        v = self.to_v(K).flatten(2).permute(0, 2, 1)

        # offload attention to another GPU
        device_main = q.device                           # e.g. 'cuda:0'
        device_off = torch.device('cuda:2')              # choose an idle GPU
        q_off = q.to(device_off)
        k_off = k.to(device_off)
        v_off = v.to(device_off)
        attn_out_off = F.scaled_dot_product_attention(
            q_off, k_off, v_off,
            attn_mask=None,
            dropout_p=0.1,
            is_causal=False,
        )  # → (B, N, C) on GPU 1
        attn_out = attn_out_off.to(device_main)          # move result back

        print('pass once')

        # reshape back → (B, C, H, W)
        out = attn_out.permute(0, 2, 1).view(B, C, H, W)

        # also return Q and K in spatial form
        Q_sp = q.permute(0, 2, 1).view(B, C, H, W)
        K_sp = k.permute(0, 2, 1).view(B, C, H, W)
        return Q_sp, K_sp, self.proj(out)



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
        # self.cross_attn = CrossAttention(dim=width, heads=heads, dim_head=dim_head)
        self.cross_attn = FlashCrossAttention(width)
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
        x_t: torch.Tensor,
        cond_unet_out: torch.Tensor,
        t_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            t_emb: timestep embedding, shape (B, time_emb_dim)
            x_t: noised label, shape (B, width, H, W)
            cond_unet_out: condition U-Net output, shape (B, width, H, W)
        Returns:
            output tensor, shape (B, width, H, W)
        """
        # first pass: condition as Q, diffusion as K
        former_output = self.fatt1(cond_unet_out, x_t, t_emb)
        # second pass: diffusion as Q, former_output as K
        later_output = self.fatt2(x_t, former_output, t_emb)
        return later_output
