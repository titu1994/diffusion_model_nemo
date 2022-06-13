from typing import Optional
from typing import Optional

import torch
from omegaconf import DictConfig
from torch.nn import functional as F
from tqdm import tqdm
from tqdm import tqdm

from diffusion_model_nemo import utils
from diffusion_model_nemo.modules import GaussianDiffusion
from diffusion_model_nemo.modules import cosine_beta_schedule
from diffusion_model_nemo.modules import linear_beta_schedule
from diffusion_model_nemo.modules import quadratic_beta_schedule
from diffusion_model_nemo.modules import sigmoid_beta_schedule


# Ported from https://github.com/ermongroup/ddim/blob/51cb290f83049e5381b09a4cc0389f16a4a02cc9/functions/denoising.py#L10
class GeneralizedGaussianDiffusion(GaussianDiffusion):
    def __init__(
        self,
        timesteps: int,
        schedule_name: str,
        schedule_cfg: Optional[DictConfig] = None,
        objective: str = "pred_noise",
        eta: float = 0.0,
    ):
        super().__init__(
            timesteps=timesteps, schedule_name=schedule_name, schedule_cfg=schedule_cfg, objective=objective
        )

        if not (0.0 <= eta <= 1.0):
            raise ValueError(f"`eta` must be a value in [0, 1]. 0 = DDIM and 1 = DDPM mode")

        self.eta = eta
        # Recompute for all subclasses
        self.compute_constants(self.timesteps)

    def generalized_predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        assert x_t.shape == noise.shape, f"{x_t.shape} != {noise.shape}"
        alphas_extended_cumprod = self.extract(self.alphas_extended_cumprod, t + 1, x_t.shape)  # equivalent to `at_next`
        print(alphas_extended_cumprod, self.alphas_extended_cumprod.index_select(0, t.cpu() + 1).view(-1, 1, 1, 1))
        print(self.alphas_extended_cumprod)

        return (x_t - noise * (1. - alphas_extended_cumprod).sqrt()) / alphas_extended_cumprod.sqrt()

    def p_mean_variance(
        self,
        model,
        x: torch.Tensor,  # x_t
        t: torch.Tensor,
        model_output: torch.Tensor = None,
        return_pred_x_start: bool = False,
    ):
        model_output = utils.default(model_output, lambda: model(x, t))

        if self.objective == 'pred_noise':
            x_recon = self.generalized_predict_start_from_noise(x_t=x, t=t, noise=model_output)
        else:
            x_recon = model_output

        # x0_preds = x_recon
        print(x_recon.mean())

        # Equation 11 in the paper; reformulated to include clipping
        # Use our model (noise predictor) to predict the mean
        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x=x, t=t)

        if return_pred_x_start:
            return model_mean, None, posterior_log_variance, x_recon
        else:
            return model_mean, None, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor):
        model_output = model(x, t)

        # model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
        _, _, _, x0_t = self.p_mean_variance(
            model, x=x, t=t, model_output=model_output, return_pred_x_start=True
        )

        # Referring to https://github.com/ermongroup/ddim/blob/51cb290f83049e5381b09a4cc0389f16a4a02cc9/functions/denoising.py#L26
        alphas_cumprod = self.extract(self.alphas_cumprod, t + 1, x.shape)  # equivalent to `at_next`
        alphas_cumprod_prev = self.extract(self.alphas_cumprod_prev, t, x.shape)  # equivalent to `at`

        # Construct intermediates of equation (12) in https://arxiv.org/abs/2010.02502
        noise = torch.randn_like(x)
        c1 = self.eta * torch.sqrt(
            (1.0 - alphas_cumprod_prev / alphas_cumprod) * (1.0 - alphas_cumprod) / (1.0 - alphas_cumprod_prev)
        )  # predicted x0
        c2 = torch.sqrt((1.0 - alphas_cumprod) - c1 ** 2)  # direction pointing to x_t

        xt_next = alphas_cumprod.sqrt() * x0_t + c1 * noise + c2 * model_output
        return xt_next, x0_t

    # # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def p_sample_loop(self, model, shape, use_tqdm=True):
        device = next(model.parameters()).device

        b = shape[0]

        # prepare constants for sampling
        self.betas_extended = torch.cat([torch.zeros(1).to(self.betas.device), self.betas], dim=0)
        self.alphas_extended = (1. - self.betas_extended)
        self.alphas_extended_cumprod = self.alphas_extended.cumprod(dim=0)

        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []
        for i in tqdm(
            reversed(range(0, self.timesteps)),
            desc='Sampling loop time step',
            total=self.timesteps,
            disable=not use_tqdm,
        ):
            img, x0_t = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long))
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
            img, x0_start = self.p_sample(model, img, torch.full((B,), i, device=device, dtype=torch.long))
            # unnormalize image
            img_cpu = (img.cpu() + 1) * 0.5
            imgs.append(img_cpu)

        return imgs
