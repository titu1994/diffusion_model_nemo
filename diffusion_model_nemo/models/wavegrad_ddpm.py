import torch
import math

from typing import List, Dict, Optional, Union

from tqdm import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only
from pathlib import Path
from omegaconf import OmegaConf, DictConfig, open_dict
from hydra.utils import instantiate
from functools import partial

from diffusion_model_nemo.models import DDPM
from diffusion_model_nemo.modules import WaveGradDiffusion
from diffusion_model_nemo.loss import VariationalBoundLoss
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
    def forward(self, x_t: torch.Tensor, noise_level: torch.Tensor, classes: torch.Tensor = None):
        return self.diffusion_model(x_t, noise_level)

    def get_diffusion_model(self, batch: Dict):
        diffusion_model_fn = self.forward
        return diffusion_model_fn

    def training_step(self, batch, batch_nb):
        device = next(self.parameters()).device
        batch_size = batch["pixel_values"].shape[0]
        samples = batch["pixel_values"]  # x_start

        diffusion_model_fn = self.get_diffusion_model(batch)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        continuous_sqrt_alpha_cumprod = self.sampler.sample_continuous_noise_level(batch_size, device=device)
        noise = torch.randn_like(samples)

        x_t = self.sampler.q_sample(
            x_start=samples, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod, noise=noise
        )
        model_output = diffusion_model_fn(x_t=x_t, noise_level=continuous_sqrt_alpha_cumprod)

        loss = self.loss(input=model_output, target=noise)

        # Compute log dict
        self.log('train_loss', loss.detach())
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])
        self.log('global_step', float(self.trainer.global_step))

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

    def calculate_bits_per_dimension(self, x_start: torch.Tensor, diffusion_model_fn, max_batch_size: int = 32):
        # Implemented from https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L313
        B, C, H, W = x_start.size()
        T = self.sampler.timesteps

        if max_batch_size > 0:
            B = min(max_batch_size, B)

        with torch.inference_mode():
            x_start = x_start[:B]

            terms_buffer = torch.zeros(B, T, device=x_start.device)
            T_range = torch.arange(T, device=x_start.device).unsqueeze(0)
            t_b = torch.full([B], fill_value=T, device=x_start.device, dtype=torch.long)
            zero_tensor = torch.tensor(0.0, device=x_start.device)

            for t in tqdm(range(T - 1, -1, -1), desc='Computing bits per dimension', total=T):
                t_b = t_b * 0 + t

                x_t = self.sampler.q_sample(x_start=x_start, continuous_sqrt_alpha_cumprod=None)
                # calculating kl loss for calculating NLL in bits per dimension
                true_mean, true_log_variance_clipped = self.sampler.q_posterior(x_start=x_start, x=x_t, t=t_b)
                model_mean, _, model_log_variance, pred_x_start = self.sampler.p_mean_variance(
                    diffusion_model_fn, x=x_t, t=t_b, return_pred_x_start=True,
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

            # Compute prior
            t_prior = torch.full([B], fill_value=T - 1, device=x_start.device, dtype=torch.long)
            qt_mean, _, qt_log_variance = self.sampler.q_mean_variance(x_start=x_start, t=t_prior)
            kl_prior = utils.normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=zero_tensor, logvar2=zero_tensor)
            prior_bpd_b = utils.mean_flattened(kl_prior) / math.log(2.0)

            # Compute total bpd
            total_bpd_b = torch.sum(terms_buffer, dim=1) + prior_bpd_b


        # assert terms_buffer.shape == mse_buffer.shape == [B, T] and total_bpd_b.shape == prior_bpd_b.shape == [B]
        result = {
            'total_bpd': total_bpd_b,
            'terms_bpd': terms_buffer,
            'prior_bpd': prior_bpd_b,
        }

        return result
