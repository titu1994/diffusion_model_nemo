from typing import Optional
from typing import Optional

import torch
import numpy as np

from omegaconf import DictConfig
from torch.nn import functional as F
from tqdm import tqdm

from diffusion_model_nemo import utils
from diffusion_model_nemo.modules import GaussianDiffusion


# Ported from https://github.com/ivanvovk/WaveGrad/blob/master/model/diffusion_process.py
class WaveGradDiffusion(GaussianDiffusion):
    def __init__(
        self,
        timesteps: int,
        schedule_name: str,
        schedule_cfg: Optional[DictConfig] = None,
        objective: str = "pred_noise",
    ):
        super().__init__(
            timesteps=timesteps, schedule_name=schedule_name, schedule_cfg=schedule_cfg, objective=objective
        )
        self.original_timesteps = timesteps

        # Recompute for all subclasses
        self.compute_constants(self.timesteps)

    def compute_constants(self, timesteps):
        super().compute_constants(timesteps)

        alphas_cumprod_prev_with_last = F.pad(self.alphas_cumprod, (1, 0), value=1.0)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(alphas_cumprod_prev_with_last)
        self.sqrt_alphas_cumprod_m1 = (1. - self.alphas_cumprod).sqrt() * self.sqrt_recip_alphas_cumprod

    def sample_continuous_noise_level(self, batch_size: int, device: torch.device):
        """
        Samples continuous noise level sqrt(alpha_cumprod).
        This is what makes WaveGrad different from other Denoising Diffusion Probabilistic Models.
        """
        s = torch.randint(1, self.timesteps + 1, size=batch_size)
        continuous_sqrt_alpha_cumprod = torch.tensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[s - 1],
                self.sqrt_alphas_cumprod_prev[s],
                size=batch_size
            )
        ).to(device)
        return continuous_sqrt_alpha_cumprod.unsqueeze(-1)

    def q_sample(self, x_start: torch.Tensor, continuous_sqrt_alpha_cumprod: torch.Tensor, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(x_start)

        continuous_sqrt_alpha_cumprod_t = self.sample_continuous_noise_level(x_start.size(0), device=x_start.device)
        sqrt_one_minus_alphas_cumprod_t = (1. - continuous_sqrt_alpha_cumprod_t ** 2).sqrt()

        return continuous_sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """
        Computes y_0 from given y_t and reconstructed noise.
        Is needed to reconstruct the reverse (denoising)
        process posterior q(y_{t-1}|y_0, y_t, x).
        """
        return self.sqrt_recip_alphas_cumprod[t] * x_t - self.sqrt_alphas_cumprod_m1[t] * noise

    def p_mean_variance(
        self,
        model,
        x: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor = None,
        return_pred_x_start: bool = False,
    ):
        batch_size = x.size(0)
        noise_level = torch.tensor([self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x)
        model_output = utils.default(model_output, lambda: model(x, noise_level))

        if self.objective == 'pred_noise':
            x_recon = self.predict_start_from_noise(x_t=x, t=t, noise=model_output)
        else:
            x_recon = model_output

        # Clamp if needed
        x_recon.clamp_(-1.0, 1.0)

        # Equation 11 in the paper; reformulated to include clipping
        # Use our model (noise predictor) to predict the mean
        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x=x, t=t)

        if return_pred_x_start:
            return model_mean, None, posterior_log_variance, x_recon
        else:
            return model_mean, None, posterior_log_variance

    # @torch.no_grad()
    # def sample(self, model: torch.nn.Module, shape, device: torch.device = None):
    #     with torch.inference_mode():
    #         return self.p_sample_loop(model, shape=shape, device=device)
    #
    # @torch.no_grad()
    # def interpolate(self, model, x: torch.Tensor, t: Optional[int] = None):
    #     return self.p_sample_loop(model, x.shape, img=x)
