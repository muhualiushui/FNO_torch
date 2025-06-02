import torch
import warnings
from torch import nn
from FNO_torch.helper.Func import DiceCELoss

class Diffusion(nn.Module):
    def __init__(self, model, timesteps, loss_ratio=0.8, sanity_check: bool = False):
        super(Diffusion, self).__init__()
        self.model = model
        self.timesteps = timesteps
        self.sanity_check = sanity_check
        # linear schedule for betas, alphas, and their cumulative product
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        alpha_prod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_prod', alpha_prod)
        # Sanity check for noise schedule
        if self.sanity_check:
            # Print first, middle, and last beta and alpha_prod values
            t0_beta = betas[0].item()
            t_mid = timesteps // 2
            t_mid_beta = betas[t_mid].item()
            t_last_beta = betas[-1].item()
            a0 = alpha_prod[0].item()
            a_mid = alpha_prod[t_mid].item()
            a_last = alpha_prod[-1].item()
            print(f"[Diffusion Sanity] betas: t=0 -> {t0_beta:.6f}, t={t_mid} -> {t_mid_beta:.6f}, t={timesteps-1} -> {t_last_beta:.6f}")
            print(f"[Diffusion Sanity] alpha_prod: t=0 -> {a0:.6f}, t={t_mid} -> {a_mid:.6f}, t={timesteps-1} -> {a_last:.6f}")
            # Sanity assert: alpha_prod should decrease from 1.0 toward 0.0
            assert a0 > a_mid > a_last or torch.isclose(torch.tensor(a0), torch.tensor(a_mid)), "alpha_prod values do not decrease properly"
        self.loss_fn = nn.MSELoss()
        self.dice_loss = DiceCELoss(ce_weight=0.5, smooth=1e-5)
        self.loss_ratio = loss_ratio

    def forward(self, x0, image):
        """
        x0: original clean image tensor of shape (B, C, H, W)
        image: conditioning image tensor of shape (B, C, H, W)
        """
        batch_size = x0.shape[0]

        # sample random timesteps for each example
        t = torch.randint(0, self.timesteps, (batch_size,), device=x0.device)

        # add noise at those timesteps
        x_t, noise = self.noise(x0, t)

        # predict the noise
        pred_noise = self.model(x_t, t, image)

        pred_x0 = self.pred_x0(x_t, t, pred_noise)

        if self.sanity_check:
            assert torch.all((t >= 0) & (t < self.timesteps)), "Sampled t out of valid range"
            # Tracking checks in forward
            if torch.isnan(pred_noise).any():
                warnings.warn("NaN detected in predicted noise (pred_noise) during forward pass")
            if torch.isnan(pred_x0).any():
                warnings.warn("NaN detected in predicted x0 (pred_x0) during forward pass")
            # Check for low variance (possible collapse) in pred_x0
            std_x0 = pred_x0.detach().view(pred_x0.size(0), -1).std(dim=1)
            if torch.any(std_x0 < 1e-5):
                warnings.warn("Very low variance detected in pred_x0 (possible collapse) during forward pass")

        return pred_noise, pred_x0, noise
    
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

    def Denoise(self, x_t, t, image):
        """
        Performs one reverse diffusion step from x_t to x_{t-1}.
        x_t: noisy input at time t
        t: timestep tensor of shape (B,)
        image: conditioning image tensor
        """
        # predict noise at this step
        pred_noise = self.model(x_t,t, image)
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
        if self.sanity_check:
            assert self.timesteps > 0, "timesteps must be positive for inference"
        # start from pure noise
        device = image.device
        batch_size, C, H, W = image.shape
        x = torch.randn((batch_size, C, H, W), device=device)
        # iteratively denoise from T to 1
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.Denoise(x, t_tensor, image)
        return x
    
    def cal_loss(self, image: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        pred_noise, pred_x0, noise = self.forward(x0,image)
        return self.loss_fn(noise, pred_noise)*self.loss_ratio + self.dice_loss(pred_x0, x0)*(1-self.loss_ratio)