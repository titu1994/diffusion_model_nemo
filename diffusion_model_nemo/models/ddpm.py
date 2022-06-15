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

from diffusion_model_nemo.models import AbstractDiffusionModel
from diffusion_model_nemo.modules import AbstractDiffusionProcess
from diffusion_model_nemo.loss.variational_bound_loss import VariationalBoundLoss
from diffusion_model_nemo import utils

from nemo.core import typecheck
from nemo.core.neural_types import NeuralType
from nemo.utils import logging


class DDPM(AbstractDiffusionModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer=trainer)

        self.diffusion_model = instantiate(self.cfg.diffusion_model)
        self.sampler = instantiate(self.cfg.sampler)  # type: AbstractDiffusionProcess
        self.loss = instantiate(self.cfg.loss)

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    @typecheck()
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, classes: torch.Tensor = None):
        return self.diffusion_model(x_t, t)

    def training_step(self, batch, batch_nb):
        device = next(self.parameters()).device
        batch_size = batch["pixel_values"].shape[0]
        samples = batch["pixel_values"]  # x_start

        if self.sampler.use_class_conditioning:
            labels = batch["labels"]
            diffusion_model_fn = partial(self.diffusion_model, classes=labels)
        else:
            diffusion_model_fn = self.diffusion_model

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, self.timesteps, size=(batch_size,), device=device, dtype=torch.long)
        noise = torch.randn_like(samples)

        x_t = self.sampler.q_sample(x_start=samples, t=t, noise=noise)
        model_output = diffusion_model_fn(x=x_t, time=t)

        loss = self.loss(input=model_output, target=noise)

        # Compute log dict
        self.log('train_loss', loss.detach())
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])

        # save generated images
        if self.trainer.global_step != 0 and self.trainer.global_step % self.save_and_sample_every == 0:
            self._save_image_step(batch_size=batch_size, step=self.trainer.global_step)

            log_dict = self.calculate_bits_per_dimension(x_start=samples, diffusion_model_fn=diffusion_model_fn)
            self.log('total_bits_per_dimension', log_dict.pop('total_bpd'), prog_bar=True)
            self.log_dict(log_dict)

        return loss

    def sample(self, batch_size: int, image_size: int):
        with torch.inference_mode():
            shape = [batch_size, self.channels, image_size, image_size]
            return self.sampler.sample(self.diffusion_model, shape=shape)

    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, t: Optional[int] = None, lambd: float = 0.5, **kwargs):
        with torch.inference_mode():
            assert x1.ndim == 4, f"x1 is not a batch of tensors ! Given shape {x1.shape}"
            assert x2.ndim == 4, f"x2 is not a batch of tensors ! Given shape {x2.shape}"
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)

            imgs = self.sampler.interpolate(self.diffusion_model, x1=x1, x2=x2, t=t, lambd=lambd)
        return imgs

    def calculate_bits_per_dimension(self, x_start: torch.Tensor, diffusion_model_fn):
        # Implemented from https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L313
        B, C, H, W = x_start.size()
        T = self.timesteps
        B = min(32, B)

        with torch.inference_mode():
            x_start = x_start[:B]

            terms_buffer = torch.zeros(B, T, device=x_start.device)
            T_range = torch.arange(T, device=x_start.device).unsqueeze(0)
            t_b = torch.full([B], fill_value=T, device=x_start.device, dtype=torch.long)
            zero_tensor = torch.tensor(0.0, device=x_start.device)

            for t in tqdm.tqdm(range(T - 1, 0, -1), desc='Computing bits per dimension', total=T):
                t_b = t_b * 0 + t

                x_t = self.sampler.q_sample(x_start=x_start, t=t_b)
                # calculating kl loss for calculating NLL in bits per dimension
                true_mean, true_log_variance_clipped = self.sampler.q_posterior(x_start=x_start, x=x_t, t=t_b)
                model_mean, _, model_log_variance, pred_x_start = self.sampler.p_mean_variance(
                    diffusion_model_fn, x=x_t, t=t_b, return_pred_x_start=True
                )
                if model_log_variance.shape != model_mean.shape:
                    model_log_variance = model_log_variance.expand(-1, *model_mean.size()[1:])

                # Calculate VLB term at the current timestep
                new_vals_b = VariationalBoundLoss.compute_variation_loss_terms(
                    samples=x_start,
                    model_mean=model_mean,
                    model_log_variance=model_log_variance,
                    true_mean=true_mean,
                    true_log_variance_clipped=true_log_variance_clipped,
                    t=t_b,
                )

                # MSE for progressive prediction loss
                mask_bt = ((t_b.unsqueeze(-1)) == (T_range)).to(torch.float32)
                terms_buffer = terms_buffer * (1.0 - mask_bt) + new_vals_b.unsqueeze(-1) * mask_bt

                assert mask_bt.shape == terms_buffer.shape == torch.Size([B, T])

            t_prior = torch.full([B], fill_value=T - 1, device=x_start.device, dtype=torch.long)
            qt_mean, _, qt_log_variance = self.sampler.q_mean_variance(x_start=x_start, t=t_prior)
            kl_prior = utils.normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=zero_tensor, logvar2=zero_tensor)
            prior_bpd_b = utils.mean_flattened(kl_prior) / math.log(2.0)
            total_bpd_b = torch.sum(terms_buffer, dim=1) + prior_bpd_b

        # assert terms_buffer.shape == mse_buffer.shape == [B, T] and total_bpd_b.shape == prior_bpd_b.shape == [B]
        result = {
            'total_bpd': total_bpd_b.mean(),
            'terms_bpd': terms_buffer.mean(),
            'prior_bpd': prior_bpd_b.mean(),
        }

        return result
