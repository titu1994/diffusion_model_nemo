import torch
from typing import List, Dict, Optional, Union
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only
from pathlib import Path
from omegaconf import OmegaConf, DictConfig, open_dict
from hydra.utils import instantiate

from diffusion_model_nemo.models import AbstractDiffusionModel
from diffusion_model_nemo.modules import AbstractDiffusionProcess
from diffusion_model_nemo.data.hf_vision_data import get_transform, get_reverse_transform

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
    def forward(self, batch_size, image_size):
        shape = [batch_size, self.channels, image_size, image_size]
        samples = self.sampler.sample(self.diffusion_model, shape)
        return samples

    def training_step(self, batch, batch_nb):
        device = next(self.parameters()).device
        batch_size = batch["pixel_values"].shape[0]
        samples = batch["pixel_values"]

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, self.timesteps, size=(batch_size,), device=device, dtype=torch.long)
        noise = torch.randn_like(samples)

        x_noisy = self.sampler.q_sample(x_start=samples, t=t, noise=noise)
        predicted_noise = self.diffusion_model(x=x_noisy, time=t)

        loss = self.loss(input=predicted_noise, target=noise)

        self.log('train_loss', loss.detach())
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])

        # save generated images
        if self.trainer.global_step != 0 and self.trainer.global_step % self.save_and_sample_every == 0:
            self._save_image_step(batch_size=batch_size, step=self.trainer.global_step)

        return loss

    def sample(self, batch_size: int, image_size: int):
        return self.forward(batch_size=batch_size, image_size=image_size)

    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, t: Optional[int] = None, lambd: float = 0.5, **kwargs):
        assert x1.ndim == 4, f"x1 is not a batch of tensors ! Given shape {x1.shape}"
        assert x2.ndim == 4, f"x2 is not a batch of tensors ! Given shape {x2.shape}"
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)

        imgs = self.sampler.interpolate(self.diffusion_model, x1=x1, x2=x2, t=t, lambd=lambd)
        return imgs
