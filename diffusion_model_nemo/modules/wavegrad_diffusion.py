from typing import Optional

import torch
import numpy as np

from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F
from tqdm import tqdm

from diffusion_model_nemo import utils
from diffusion_model_nemo.modules import GaussianDiffusion
from nemo.utils import logging


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
        self.sqrt_alphas_cumprod_m1 = (1.0 - self.alphas_cumprod).sqrt() * self.sqrt_recip_alphas_cumprod

        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_variance = torch.stack(
            [self.posterior_variance, torch.tensor([1e-20] * self.timesteps, dtype=torch.float32)]
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = self.posterior_variance.max(dim=0).values.log()

        logging.info(f"Changed time steps to {timesteps}")
        logging.info(f"Last few samples of `sqrt_alphas_cumprod_prev` : {self.sqrt_alphas_cumprod_prev[-10:]}")

    def change_noise_schedule(self, schedule_name: str = None, schedule_cfg: dict = None):
        if schedule_name is None:
            schedule_name = self.schedule_name

        if schedule_cfg is None:
            schedule_cfg = self.schedule_cfg

        # Set values
        self.schedule_name = schedule_name
        self.schedule_cfg = schedule_cfg

        logging.info(f"New scheduler name : {self.schedule_name}")
        logging.info(f"New scheduler config : {OmegaConf.to_yaml(self.schedule_cfg)}")

    def sample_continuous_noise_level(self, batch_size: int, device: torch.device):
        """
        Samples continuous noise level sqrt(alpha_cumprod).
        This is what makes WaveGrad different from other Denoising Diffusion Probabilistic Models.
        """
        s = np.random.randint(1, self.timesteps + 1, size=batch_size)
        continuous_sqrt_alpha_cumprod = torch.tensor(
            np.random.uniform(self.sqrt_alphas_cumprod_prev[s - 1], self.sqrt_alphas_cumprod_prev[s], size=batch_size),
            dtype=torch.float32,
        ).to(device)
        return continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1)

    def q_sample(self, x_start: torch.Tensor, continuous_sqrt_alpha_cumprod: torch.Tensor, noise: torch.Tensor = None):
        continuous_sqrt_alpha_cumprod_t = (
            self.sample_continuous_noise_level(x_start.size(0), device=x_start.device).to(x_start)
            if noise is None
            else continuous_sqrt_alpha_cumprod
        )

        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_one_minus_alphas_cumprod_t = (1.0 - continuous_sqrt_alpha_cumprod_t ** 2).sqrt()

        return continuous_sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """
        Computes y_0 from given y_t and reconstructed noise.
        Is needed to reconstruct the reverse (denoising)
        process posterior q(y_{t-1}|y_0, y_t, x).
        """
        sqrt_recip_alphas_cumprod = self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_alphas_cumprod_m1 = self.extract(self.sqrt_alphas_cumprod_m1, t, noise.shape)
        return sqrt_recip_alphas_cumprod * x_t - sqrt_alphas_cumprod_m1 * noise

    def p_mean_variance(
        self,
        model,
        x: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor = None,
        return_pred_x_start: bool = False,
    ):
        noise_level = self.extract(self.sqrt_alphas_cumprod_prev, t + 1, x.shape)
        model_output = model(x, noise_level)

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
