from typing import Optional
from typing import Optional

import torch
from omegaconf import DictConfig
from torch.nn import functional as F
from tqdm import tqdm
from tqdm import tqdm

from diffusion_model_nemo import utils
from diffusion_model_nemo.modules import GaussianDiffusion


class LearnedGaussianDiffusion(GaussianDiffusion):
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
        self.compute_constants(timesteps)

    def p_mean_variance(self, model, x, t, model_output=None):
        model_output = utils.default(model_output, lambda: model(x, t))
        pred_noise, var_interp_frac_unnormalized = model_output.chunk(2, dim=1)

        min_log = self.extract(self.posterior_log_variance_clipped, t, x.shape)
        max_log = self.extract(torch.log(self.betas), t, x.shape)
        var_interp_frac = (var_interp_frac_unnormalized + 1) * 0.5  # denormalization

        model_log_variance = var_interp_frac * max_log + (1 - var_interp_frac) * min_log
        model_variance = model_log_variance.exp()

        x_start = self.predict_start_from_noise(x, t, pred_noise)
        x_start.clamp_(-1., 1.)

        model_mean, _ = self.q_posterior(x_start, x, t)

        return model_mean, model_variance, model_log_variance

