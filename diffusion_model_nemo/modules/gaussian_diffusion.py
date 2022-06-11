from typing import Optional
from typing import Optional

import torch
from omegaconf import DictConfig
from torch.nn import functional as F
from tqdm import tqdm
from tqdm import tqdm

from diffusion_model_nemo import utils
from diffusion_model_nemo.modules import AbstractDiffusionProcess
from diffusion_model_nemo.modules import cosine_beta_schedule
from diffusion_model_nemo.modules import linear_beta_schedule
from diffusion_model_nemo.modules import quadratic_beta_schedule
from diffusion_model_nemo.modules import sigmoid_beta_schedule


class GaussianDiffusion(AbstractDiffusionProcess):
    def __init__(
        self,
        timesteps: int,
        schedule_name: str,
        schedule_cfg: Optional[DictConfig] = None,
        objective: str = "pred_noise",
    ):
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

        assert objective in ['pred_noise', 'pred_x0']
        self.objective = objective

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
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.maximum(self.posterior_variance, torch.tensor(1e-20)))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

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

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        assert x_t.shape == noise.shape, f"{x_t.shape} != {noise.shape}"
        sqrt_recip_alphas_cumprod = self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod = self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

        return sqrt_recip_alphas_cumprod * x_t - sqrt_recipm1_alphas_cumprod * noise

    def p_mean_variance(
        self,
        model,
        x: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor = None,
        return_pred_x_start: bool = False,
    ):
        model_output = utils.default(model_output, lambda: model(x, t))

        if self.objective == 'pred_noise':
            x_recon = self.predict_start_from_noise(x_t=x, t=t, noise=model_output)
        else:
            x_recon = model_output

        # Always clamp
        x_recon.clamp_(-1.0, 1.0)

        # Equation 11 in the paper; reformulated to include clipping
        # Use our model (noise predictor) to predict the mean
        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x=x, t=t)

        if return_pred_x_start:
            return model_mean, None, posterior_log_variance, x_recon
        else:
            return model_mean, None, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor):
        # betas_t = self.extract(self.betas, t, x.shape)
        # sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        # sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        B = x.size(0)

        # model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
        model_mean, _, model_log_variance = self.p_mean_variance(model, x=x, t=t)

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
            # unnormalize image
            img_cpu = (img.cpu() + 1) * 0.5
            imgs.append(img_cpu)
        return imgs

    @torch.no_grad()
    def sample(self, model: torch.nn.Module, shape):
        return self.p_sample_loop(model, shape=shape)

    @torch.no_grad()
    def interpolate(self, model, x1: torch.Tensor, x2: torch.Tensor, t: Optional[int] = None, lambd: float = 0.5):
        B = x1.size(0)
        device = x1.device
        t = utils.default(t, self.timesteps - 1)

        if t >= self.timesteps:
            raise ValueError(f"`t` must be < {self.timesteps} during interpolation")

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * B)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        imgs = []

        img = (1 - lambd) * xt1 + lambd * xt2
        for i in tqdm(reversed(range(0, t)), desc='Interpolation sample time step', total=t):
            img = self.p_sample(model, img, torch.full((B,), i, device=device, dtype=torch.long))
            # unnormalize image
            img_cpu = (img.cpu() + 1) * 0.5
            imgs.append(img_cpu)

        return imgs
