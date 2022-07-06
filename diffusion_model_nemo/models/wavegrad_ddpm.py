import torch
import math

from typing import List, Dict, Optional, Union

import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only
from pathlib import Path
from omegaconf import OmegaConf, DictConfig, open_dict
from hydra.utils import instantiate
from functools import partial

from diffusion_model_nemo.models import DDPM
from diffusion_model_nemo.modules import WaveGradDiffusion
from diffusion_model_nemo import utils

from nemo.core import typecheck
from nemo.core.neural_types import NeuralType
from nemo.utils import logging


class WavegradDDPM(DDPM):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer=trainer)

        # self.diffusion_model = instantiate(self.cfg.diffusion_model)
        # self.sampler = instantiate(self.cfg.sampler)  # type: AbstractDiffusionProcess
        # self.loss = instantiate(self.cfg.loss)
        assert isinstance(self.sampler, WaveGradDiffusion), "This class must implement WaveGradDiffusion as its sampler"
        # Add type info
        self.sampler = self.sampler  # type: WaveGradDiffusion

    @typecheck()
    def forward(self, x_t: torch.Tensor, sqrt_alpha_cumprod: torch.Tensor, classes: torch.Tensor = None):
        return self.diffusion_model(x_t, sqrt_alpha_cumprod)

    def get_diffusion_model(self, batch: Dict):
        diffusion_model_fn = self.forward
        return diffusion_model_fn

    def training_step(self, batch, batch_nb):
        device = next(self.parameters()).device
        batch_size = batch["pixel_values"].shape[0]
        samples = batch["pixel_values"]  # x_start

        diffusion_model_fn = self.get_diffusion_model(batch)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        # t = torch.randint(0, self.timesteps, size=(batch_size,), device=device, dtype=torch.long)
        continuous_sqrt_alpha_cumprod = self.sampler.sample_continuous_noise_level(batch_size, device=device)
        noise = torch.randn_like(samples)

        x_t = self.sampler.q_sample(
            x_start=samples, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod, noise=noise
        )
        model_output = diffusion_model_fn(x_t=x_t, sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod)

        loss = self.loss(input=model_output, target=noise)

        # Compute log dict
        self.log('train_loss', loss.detach())
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])
        self.log('global_step', self.trainer.global_step)

        # save generated images
        if self.trainer.global_step != 0 and self.trainer.global_step % self.save_and_sample_every == 0:
            self.eval()
            # Do faster inference
            self.sampler.compute_constants(timesteps=50)

            self._save_image_step(batch_size=batch_size, step=self.trainer.global_step)

            if self.cfg.get('compute_bpd', False):
                log_dict = self.calculate_bits_per_dimension(x_start=samples, diffusion_model_fn=self.forward)
                for key in log_dict.keys():
                    log_dict[key] = log_dict[key].mean()

                self.log('total_bits_per_dimension', log_dict.pop('total_bpd'), prog_bar=True)
                self.log_dict(log_dict)

            # Return to train mode
            self.sampler.compute_constants(timesteps=self.sampler.original_timesteps)

        return loss

    def test_step(self, batch, batch_nb):
        batch_size = batch["pixel_values"].shape[0]
        samples = batch["pixel_values"]  # x_start

        diffusion_model_fn = self.get_diffusion_model(batch)

        log_dict = self.calculate_bits_per_dimension(
            x_start=samples, diffusion_model_fn=diffusion_model_fn, max_batch_size=-1
        )
        for key in log_dict.keys():
            log_dict[key] = log_dict[key].sum()

        log_dict['num_samples'] = torch.tensor(batch_size, dtype=torch.long)
        return log_dict

    def sample(self, batch_size: int, image_size: int, device: torch.device = None):
        with torch.inference_mode():
            self.eval()
            shape = [batch_size, self.channels, image_size, image_size]
            return self.sampler.sample(self.diffusion_model, shape=shape, device=device)

    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, t: Optional[int] = None, lambd: float = 0.5, **kwargs):
        raise NotImplementedError()
