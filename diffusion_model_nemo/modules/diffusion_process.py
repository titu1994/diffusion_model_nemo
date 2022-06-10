import torch
import torch.nn.functional as F
from tqdm import tqdm
from abc import ABC, abstractmethod

from omegaconf import OmegaConf, DictConfig
from typing import Optional, List


def cosine_beta_schedule(timesteps, s=0.008, min_clip=0.0001, max_clip=0.999):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, min_clip, max_clip)


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    beta_start = beta_start
    beta_end = beta_end
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class AbstractSampler:

    def __init__(self, timesteps, schedule_name, schedule_cfg=None):
        self.timesteps = timesteps
        self.schedule_name = schedule_name
        self.schedule_cfg = schedule_cfg if schedule_cfg is not None else {}
        self.schedule_fn = None

    @abstractmethod
    def compute_constants(self, timesteps):
        raise NotImplementedError()

    @abstractmethod
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        raise NotImplementedError()

    @abstractmethod
    @torch.no_grad()
    def p_sample(self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError()

    @abstractmethod
    @torch.no_grad()
    def sample(self, model: torch.nn.Module, shape: List[int]):
        raise NotImplementedError()


class GaussianDiffusion(AbstractSampler):
    def __init__(self, timesteps: int, schedule_name: str, schedule_cfg: Optional[DictConfig] = None):
        super().__init__(timesteps=timesteps, schedule_name=schedule_name, schedule_cfg=schedule_cfg)

        assert schedule_name in [
            'linear',
            'quadratic',
            'sigmoid',
            'cosine',
        ], f"Invalid schedule `{schedule_name}` provided to sampler !"

        if schedule_name == 'linear':
            self.schedule_fn = linear_beta_schedule
        elif schedule_name == 'quadratic':
            self.schedule_fn = quadratic_beta_schedule
        elif schedule_name == 'sigmoid':
            self.schedule_fn = sigmoid_beta_schedule
        elif schedule_name == 'cosine':
            self.schedule_fn = cosine_beta_schedule

        self.compute_constants(timesteps)

    def compute_constants(self, timesteps):
        # define beta schedule
        scheduler_cfg = self.schedule_cfg.get(self.schedule_name, {})

        self.betas = self.schedule_fn(timesteps=timesteps, **scheduler_cfg)

        # define alphas
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.maximum(self.posterior_variance, torch.tensor(1e-20)))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    def extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_posterior(self, x_start: torch.Tensor, x: torch.Tensor, t: torch.Tensor):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        posterior_mean_coef1 = self.extract(self.posterior_mean_coef1, t, x.shape)
        posterior_mean_coef2 = self.extract(self.posterior_mean_coef2, t, x.shape)
        posterior_mean = posterior_mean_coef1 * x_start + posterior_mean_coef2 * x

        # posterior_variance = self.extract(self.posterior_variance, t, x.shape)
        posterior_log_variance_clipped = self.extract(self.posterior_log_variance_clipped, t, x.shape)
        return posterior_mean, posterior_log_variance_clipped

    # forward diffusion (using the nice property)
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def predict_start_from_noise(self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        assert x.shape == noise.shape
        sqrt_recip_alphas_cumprod = self.extract(self.sqrt_recip_alphas_cumprod, t, x.shape)
        sqrt_recipm1_alphas_cumprod = self.extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)

        return sqrt_recip_alphas_cumprod * x - sqrt_recipm1_alphas_cumprod * noise

    @torch.no_grad()
    def p_sample(self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor):
        # betas_t = self.extract(self.betas, t, x.shape)
        # sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        # sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

        B = x.size(0)
        x_recon = self.predict_start_from_noise(x=x, t=t, noise=model(x, t))
        x_recon = torch.clamp(x_recon, -1., 1.)

        # Equation 11 in the paper; reformulated to include clipping
        # Use our model (noise predictor) to predict the mean
        model_mean, model_log_variance = self.q_posterior(x_start=x_recon, x=x, t=t)

        # model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(B, *((1,) * (len(x.shape) - 1)))
        noise = torch.randn_like(x)
        # posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
        # Algorithm 2 line 4:
        # return model_mean + torch.sqrt(posterior_variance_t) * noise

        # Algorithm 2 line 4 (after clipping reformulation):
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def p_sample_loop(self, model, shape, use_tqdm=True):
        device = next(model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(
            reversed(range(0, self.timesteps)),
            desc='Sampling loop time step',
            total=self.timesteps,
            disable=not use_tqdm,
        ):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long))
            imgs.append(img.cpu())
        return imgs

    @torch.no_grad()
    def sample(self, model: torch.nn.Module, shape):
        return self.p_sample_loop(model, shape=shape)