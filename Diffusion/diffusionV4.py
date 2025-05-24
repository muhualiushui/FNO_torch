import torch
from torch import nn
import torch.nn.functional as F


def _get_gaussian_kernel(kernel_size: int, sigma: float, channels: int) -> torch.Tensor:
    # Create a 2D Gaussian kernel repeated for each channel
    ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='xy')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    return kernel.repeat(channels, 1, 1, 1)

class UncertainSpatialAttention(nn.Module):
    """
    Applies Uncertain Spatial Attention (U-SA) to integrate anchor features
    into diffusion features, per MedSegDiff-V2.
    """
    def __init__(self, num_anchors: int, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__()
        # Precompute Gaussian kernel for smoothing anchor maps
        gauss = _get_gaussian_kernel(kernel_size, sigma, num_anchors)
        self.register_buffer('gauss_kernel', gauss)
        # 1x1 conv to reduce anchor channels to single attention map
        self.conv1x1 = nn.Conv2d(num_anchors, 1, kernel_size=1)

    def forward(self, anchor: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """
        anchor: [B, num_anchors, H, W]
        feat:   [B, C_feat, H, W] diffusion encoder features at corresponding scale
        """
        # Smooth anchor with Gaussian kernel (grouped convolution)
        smoothed = F.conv2d(anchor, self.gauss_kernel, padding=self.gauss_kernel.shape[-1]//2, groups=anchor.shape[1])
        # Relax anchor by taking element-wise max with original
        relaxed = torch.max(smoothed, anchor)
        # Compute spatial attention map
        attn = torch.sigmoid(self.conv1x1(relaxed))  # [B, 1, H, W]
        # Apply attention to features
        return feat * attn + feat


class ConditionModel(nn.Module):
    """
    UNet-like Condition Model producing two outputs:
      - anchor: coarse segmentation map (B, num_classes, H, W)
      - semantic: bottleneck embedding vector (B, features*2)
    """
    def __init__(self, in_channels: int, num_classes: int, features: int = 64):
        super().__init__()
        # Encoder
        self.enc_conv = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Downsample
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Project encoder features to match bottleneck channels
        self.enc_proj = nn.Conv2d(features, features * 2, kernel_size=1)
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features, features * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features * 2, features * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Upsample bottleneck features
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Decoder
        self.dec_conv = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Output heads
        self.anchor_out = nn.Conv2d(features, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Encoder path
        enc = self.enc_conv(x)             # [B, features, H, W]
        # Project encoder map to 2*features channels
        enc_proj = self.enc_proj(enc)      # [B, features*2, H, W]
        down = self.pool(enc)              # [B, features, H/2, W/2]
        # Bottleneck path
        bott = self.bottleneck(down)       # [B, features*2, H/2, W/2]
        # Upsample and decoder
        up = self.up(bott)                 # [B, features*2, H, W]
        # Fuse upsampled bottleneck and projected encoder via addition
        dec = self.dec_conv(up + enc_proj)
        # Anchor output: coarse segmentation map
        anchor = self.anchor_out(dec)      # [B, num_classes, H, W]
        # Semantic output: global embedding by spatial averaging
        semantic = bott.flatten(2).mean(dim=2)  # [B, features*2]
        return anchor, semantic


class DiceCELoss(nn.Module):
    def __init__(self, ce_weight: float = 0.5, smooth: float = 1e-5):
        super().__init__()
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: [B, C, H, W] logits; target: [B, C, H, W] one-hot
        probs = torch.softmax(pred, dim=1)
        intersection = torch.sum(probs * target, dim=(2, 3))
        union = torch.sum(probs + target, dim=(2, 3))
        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: logits [B, C, H, W]
        target: one-hot [B, C, H, W]
        """
        # CrossEntropyLoss expects class indices
        target_indices = target.argmax(dim=1)  # [B, H, W]
        ce_loss = self.ce(pred, target_indices)
        dice = self.dice_loss(pred, target)
        return dice + self.ce_weight * ce_loss

class Diffusion(nn.Module):
    def __init__(self, model, condition_model, timesteps):
        super(Diffusion, self).__init__()
        self.model = model
        self.condition_model = condition_model
        # Uncertain Spatial Attention module
        # assumes condition_model.anchor_out outputs num_classes channels
        self.usa = UncertainSpatialAttention(num_anchors=condition_model.anchor_out.out_channels)
        self.timesteps = timesteps
        # linear schedule for betas, alphas, and their cumulative product
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        alpha_prod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_prod', alpha_prod)
        self.loss_fn = nn.MSELoss()
        self.dice_loss = DiceCELoss(ce_weight=0.5, smooth=1e-5)

    def forward(self, x0, image):
        """
        x0: original clean image tensor of shape (B, C, H, W)
        image: conditioning image tensor of shape (B, C, H, W)
        """
        # compute conditioning outputs
        cond_anchor, cond_semantic = self.condition_model(image)
        batch_size = x0.shape[0]

        # broadcast semantic embedding to spatial map matching x_t
        _, _, H, W = x0.shape
        cond_sem_map = cond_semantic.view(batch_size, -1, 1, 1).expand(batch_size, -1, H, W)

        # sample random timesteps for each example
        t = torch.randint(0, self.timesteps, (batch_size,), device=x0.device)
        print(t.shape)
        # add noise at those timesteps
        x_t, noise = self.noise(x0, t)

        # apply uncertain spatial attention on noisy features
        x_t = self.usa(cond_anchor, x_t)

        # predict the noise
        pred_noise = self.model(x_t, t, cond_sem_map)

        pred_x0 = self.pred_x0(x_t, t, pred_noise)

        return self.loss_fn(noise, pred_noise)*0.8 + self.dice_loss(pred_x0, x0)*0.2

    def noise(self, x0, t):
        """
        Adds noise to x0 at time step(s) t.
        Returns noisy image x_t and the noise added.
        """
        noise = torch.randn_like(x0)
        # get cumulative alpha for each t
        a_prod = self.alpha_prod[t].view(-1, 1, 1, 1)
        x_t = torch.sqrt(a_prod) * x0 + torch.sqrt(1 - a_prod) * noise
        return x_t, noise

    def Denoise(self, x_t, t, cond_anchor, cond_semantic):
        """
        Performs one reverse diffusion step from x_t to x_{t-1}.
        x_t: noisy input at time t
        t: timestep tensor of shape (B,)
        cond_semantic: conditioning semantic output
        """
        # apply U-SA to the noisy features before noise prediction
        x_t = self.usa(cond_anchor, x_t)

        # broadcast semantic embedding to spatial map
        _, _, H, W = x_t.shape
        cond_sem_map = cond_semantic.view(x_t.size(0), -1, 1, 1).expand(x_t.size(0), -1, H, W)

        # predict noise at this step
        pred_noise = self.model(x_t, t, cond_sem_map)
        # gather parameters
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_prod[t].view(-1, 1, 1, 1)
        # compute previous step (deterministic, no additional noise)
        x_prev = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise)
        return x_prev

    def pred_x0(self, x_t, t, pred_noise):
        """
        Predict the original clean x0 from noisy x_t at timestep t, given the predicted noise.
        """
        # retrieve the cumulative product of alphas for timestep t
        alpha_bar_t = self.alpha_prod[t].view(-1, 1, 1, 1)
        # compute predicted x0 according to the diffusion formulation
        return (x_t - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
    

    @torch.no_grad()
    def Inference(self, image):
        """
        Runs the full reverse diffusion chain to produce a clean x0 prediction.
        image: conditioning image tensor of shape (B, C, H, W)
        Returns: predicted x0 tensor of shape (B, C, H, W)
        """
        # start from pure noise
        # compute conditioning outputs once
        cond_anchor, cond_semantic = self.condition_model(image)
        device = image.device
        batch_size, C, H, W = image.shape
        x = torch.randn((batch_size, C, H, W), device=device)
        # iteratively denoise from T to 1
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.Denoise(x, t_tensor, cond_anchor, cond_semantic)
        return x