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
from diffusion_model_nemo.modules import AbstractDiffusionProcess
from diffusion_model_nemo import utils

from nemo.core import typecheck
from nemo.core.neural_types import NeuralType
from nemo.utils import logging


class ConditionalDDPM(DDPM):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer=trainer)

        # Required argument
        if 'num_classes' not in self.cfg or self.cfg.num_classes is None:
            raise RuntimeWarning("Conditional ddpm must have the `num_classes` value inside cfg.model !")

        self.num_classes = self.cfg.num_classes
        self.random_class_index = self.num_classes
        self.sampler.use_class_conditioning = True  # force set

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    @typecheck()
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, classes: torch.Tensor = None):
        if self.sampler.use_class_conditioning:
            if classes is None:
                classes = torch.ones(x_t.size(0), device=x_t.device, dtype=torch.long) * self.random_class_index

        return self.diffusion_model(x_t, t, classes)

    def get_diffusion_model(self, batch: Dict):
        if self.sampler.use_class_conditioning:
            device = next(self.parameters()).device
            label = batch["label"]
            label = label.to(device=device)

            # Randomly set some of the conditioning labels to zero vector
            # This enables the model to jointly model both conditional and uncoditional generation
            if self.training:
                sample_mask = torch.randint(0, 2, size=[label.size(0)], dtype=torch.bool, device=label.device)
                label = torch.masked_fill(label, sample_mask, value=self.random_class_index)

            diffusion_model_fn = partial(self.forward, classes=label)
        else:
            diffusion_model_fn = self.forward

        return diffusion_model_fn

    def sample(self, batch_size: int, image_size: int, label=None):
        with torch.inference_mode():
            self.eval()
            device = next(self.parameters()).device
            shape = [batch_size, self.channels, image_size, image_size]

            if label is None:
                label = torch.ones(batch_size, dtype=torch.long, device=device) * self.random_class_index
            else:
                label = torch.full([batch_size], fill_value=int(label), dtype=torch.long, device=device)

            batch = {'label': label}
            diffusion_model_fn = self.get_diffusion_model(batch)

            return self.sampler.sample(diffusion_model_fn, shape=shape, device=device)

    def interpolate(
        self, x1: torch.Tensor, x2: torch.Tensor, t: Optional[int] = None, lambd: float = 0.5, label=None, **kwargs
    ):
        with torch.inference_mode():
            self.eval()
            assert x1.ndim == 4, f"x1 is not a batch of tensors ! Given shape {x1.shape}"
            assert x2.ndim == 4, f"x2 is not a batch of tensors ! Given shape {x2.shape}"
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            batch_size = x1.shape[0]

            if label is None:
                label = torch.ones(batch_size, dtype=torch.long, device=device) * self.random_class_index
            else:
                label = torch.full([batch_size], fill_value=int(label), dtype=torch.long, device=device)

            batch = {'label': label}

            diffusion_model_fn = self.get_diffusion_model(batch)
            imgs = self.sampler.interpolate(diffusion_model_fn, x1=x1, x2=x2, t=t, lambd=lambd)
        return imgs
