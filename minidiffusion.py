import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = (
        torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(alphas, 0.001, 1)


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


class Diffusion(nn.Module):
    def __init__(self, model, image_size, channels=3, timesteps=1000):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.model = model
        self.num_timesteps = int(timesteps)

        self.alpha = cosine_schedule(self.num_timesteps).to(DEVICE)
        self.alpha_bar = self.alpha.cumprod(dim=0).to(DEVICE)
        self.alpha_bar_prev = F.pad(self.alpha_bar, (1, 0), value=1)[:-1]
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.recip_sqrt_alpha_bar = 1 / torch.sqrt(self.alpha_bar)
        self.alpha_x_t = (
            torch.sqrt(self.alpha) * (1 - self.alpha_bar_prev)
        ) / (1 - self.alpha_bar)
        self.alpha_x_0 = (
            torch.sqrt(self.alpha_bar_prev)
            * (1 - self.alpha)
            / (1 - self.alpha_bar)
        )
        self.sigma = torch.sqrt(
            (1 - self.alpha_bar_prev)
            / (1 - self.alpha_bar)
            * (1 - self.alpha)
        )
        self.noise_coef = torch.sqrt(1 - self.alpha_bar)

    # backward diffusion
    @torch.no_grad()
    def p_sample(self, x, t):
        recip_sqrt_alpha_bar = extract(
            self.recip_sqrt_alpha_bar, t, x.shape
        )
        noise_coef = extract(self.noise_coef, t, x.shape)
        alpha_x_t = extract(self.alpha_x_t, t, x.shape)
        alpha_x_0 = extract(self.alpha_x_0, t, x.shape)
        sigma = extract(self.sigma, t, x.shape)

        pred_noise = self.model(x, t)
        pred_img = recip_sqrt_alpha_bar * (x - noise_coef * pred_noise)
        pred_img = torch.clamp(pred_img, -1, 1)
        pred_mu = alpha_x_t * x + alpha_x_0 * pred_img

        if t[0].item() == 0:
            return pred_mu

        z = torch.randn_like(x)
        x_prev = pred_mu + sigma * z
        return x_prev

    @torch.no_grad()
    def p_sample_loop(self, img):
        b = img.shape[0]
        for t_index in range(self.num_timesteps - 1, -1, -1):
            t = torch.full((b,), t_index, device=img.device)
            img = self.p_sample(img, t)
            img = torch.clamp(img, -1, 1)
            img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size):
        self.model.eval()
        noise = torch.randn(
            batch_size,
            self.channels,
            self.image_size,
            self.image_size,
            device=DEVICE,
        )
        img = self.p_sample_loop(noise)
        return img

    # forward diffusion
    def q_sample(self, x_0, t, noise=None):
        noise = torch.randn_like(x_0) if noise is None else noise
        sqrt_alpha_bar = extract(self.sqrt_alpha_bar, t, x_0.shape)
        x_t = (
            sqrt_alpha_bar * x_0
            + torch.sqrt(1 - extract(self.alpha_bar, t, x_0.shape)) * noise
        )
        return x_t

    def p_losses(self, x_0, t, noise):
        x_t = self.q_sample(x_0, t, noise)
        e_t = self.model(x_t, t)
        return F.l1_loss(e_t, noise)

    def forward(self, x_0, noise):
        b, c, h, w = x_0.shape
        device = x_0.device
        img_size = self.image_size
        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        return self.p_losses(x_0, t, noise)
