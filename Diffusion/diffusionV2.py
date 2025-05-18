import torch
from torch import nn

class Diffusion(nn.Module):
    def __init__(self, model, timesteps):
        super(Diffusion, self).__init__()
        self.model = model
        self.timesteps = timesteps
        # linear schedule for betas, alphas, and their cumulative product
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        alpha_prod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_prod', alpha_prod)
        self.loss_fn = nn.MSELoss()

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
        pred_noise = self.model(x_t, image, t)

        return self.loss_fn(noise, pred_noise)

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
        pred_noise = self.model(x_t, image)
        # gather parameters
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_prod[t].view(-1, 1, 1, 1)
        # compute previous step (deterministic, no additional noise)
        x_prev = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise)
        return x_prev

    def Inference(self, image):
        """
        Runs the full reverse diffusion chain to produce a clean x0 prediction.
        image: conditioning image tensor of shape (B, C, H, W)
        Returns: predicted x0 tensor of shape (B, C, H, W)
        """
        # start from pure noise
        device = image.device
        batch_size, C, H, W = image.shape
        x = torch.randn((batch_size, C, H, W), device=device)
        # iteratively denoise from T to 1
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.Denoise(x, t_tensor, image)
        return x