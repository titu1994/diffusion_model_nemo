import math

import torch
from typing import List, Dict, Optional, Union
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only
from pathlib import Path
from omegaconf import OmegaConf, DictConfig, open_dict
from hydra.utils import instantiate

from diffusion_model_nemo.models import DDPM
from diffusion_model_nemo.modules import AbstractDiffusionProcess
from diffusion_model_nemo.data.hf_vision_data import get_transform, get_reverse_transform
from diffusion_model_nemo import utils

from nemo.core import typecheck
from nemo.core.neural_types import NeuralType
from nemo.utils import logging


class ImprovedDDPM(DDPM):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer=trainer)

        self.vb_loss = instantiate(self.cfg.vb_loss)

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    # @typecheck()
    # def forward(self, batch_size, image_size):
    #     shape = [batch_size, self.channels, image_size, image_size]
    #     samples = self.sampler.sample(self.diffusion_model, shape)
    #     return samples

    def training_step(self, batch, batch_nb):
        device = next(self.parameters()).device
        batch_size = batch["pixel_values"].shape[0]
        samples = batch["pixel_values"]  # x_start

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, self.timesteps, size=(batch_size,), device=device, dtype=torch.long)
        noise = torch.randn_like(samples)

        x_t = self.sampler.q_sample(x_start=samples, t=t, noise=noise)
        model_output = self.diffusion_model(x=x_t, time=t)

        # simple loss - predicting noise, x0, or x_prev
        pred_noise, _ = model_output.chunk(2, dim=1)
        simple_losses = self.loss(input=pred_noise, target=noise)

        # calculating kl loss for learned variance (interpolation)
        true_mean, true_log_variance_clipped = self.sampler.q_posterior(x_start=samples, x=x_t, t=t)
        model_mean, _, model_log_variance = self.sampler.p_mean_variance(
            self.diffusion_model, x=x_t, t=t, model_output=model_output
        )
        vb_losses, decoder_nll = self.vb_loss(
            samples=samples,
            model_mean=model_mean,
            model_log_variance=model_log_variance,
            true_mean=true_mean,
            true_log_variance_clipped=true_log_variance_clipped,
            t=t,
        )

        # calculate total loss
        total_loss = simple_losses + vb_losses

        self.log('train_loss', total_loss.detach())
        self.log('simple_loss', simple_losses.detach())
        self.log('vb_losses', vb_losses.detach())
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])
        self.log('global_step', self.trainer.global_step)

        # save generated images
        if self.trainer.global_step != 0 and self.trainer.global_step % self.save_and_sample_every == 0:
            self.eval()
            self._save_image_step(batch_size=batch_size, step=self.trainer.global_step)

            if self.cfg.get('compute_bpd', False):
                log_dict = self.calculate_bits_per_dimension(x_start=samples, diffusion_model_fn=diffusion_model_fn)
                for key in log_dict.keys():
                    log_dict[key] = log_dict[key].mean()

                self.log('total_bits_per_dimension', log_dict.pop('total_bpd'), prog_bar=True)
                self.log_dict(log_dict)

        return total_loss
