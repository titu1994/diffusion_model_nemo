import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from typing import List, Dict, Optional, Union
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only
from pathlib import Path
import datetime
from omegaconf import OmegaConf, DictConfig, open_dict
from hydra.utils import instantiate
from tqdm.auto import tqdm

from diffusion_model_nemo.models import AbstractDiffusionModel
from diffusion_model_nemo.modules import sde_lib, PredictorCorrectorSampler
from diffusion_model_nemo.loss import SDEScoreFunctionLoss
from diffusion_model_nemo import utils

from nemo.core import ModelPT, PretrainedModelInfo, typecheck
from nemo.core.neural_types import NeuralType
from nemo.utils import logging


class ScoreSDE(AbstractDiffusionModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer=trainer)

        # get global constants
        self.continuous = self.cfg.get('continuous', True)
        self.likelihood_weighting = self.cfg.get('likelihood_weighting', False)

        self.diffusion_model = instantiate(self.cfg.diffusion_model)

        # initialize SDE library
        sde_type = self.cfg.sde.get('sde_type').lower()
        sde_cfg = self.cfg.sde.get(sde_type)
        self.sde = instantiate(sde_cfg)  # type: sde_lib.SDE

        # initialize the sampler
        self.sampler = instantiate(self.cfg.sampler)  # type: PredictorCorrectorSampler
        self.sampler.update_sde(self.sde)

        self.loss = instantiate(self.cfg.loss)  # type: SDEScoreFunctionLoss
        self.loss.update_sde(self.sde)

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    @typecheck()
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, classes: torch.Tensor = None):
        return self.diffusion_model(x_t, t)

    def get_diffusion_model(self, batch: Dict):
        diffusion_model_fn = self.forward
        return diffusion_model_fn

    def training_step(self, batch, batch_nb):
        device = next(self.parameters()).device
        batch_size = batch["pixel_values"].shape[0]
        samples = batch["pixel_values"]  # x_start

        diffusion_model_fn = self.get_diffusion_model(batch)

        t = torch.rand(batch_size, device=samples.device)  # scaled by (self.sde.T - eps) + eps inside loss
        noise = torch.randn_like(samples)

        loss = self.loss(diffusion_model_fn, x_start=samples, t=t, noise=noise)

        # Compute log dict
        self.log('train_loss', loss.detach())
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])
        self.log('global_step', float(self.trainer.global_step))

        # save generated images
        if self.trainer.global_step != 0 and self.trainer.global_step % self.save_and_sample_every == 0:
            self._save_image_step(batch_size=batch_size, step=self.trainer.global_step)

            # if self.cfg.get('compute_bpd', False):
            #     log_dict = self.calculate_bits_per_dimension(x_start=samples, diffusion_model_fn=self.forward)
            #     for key in log_dict.keys():
            #         log_dict[key] = log_dict[key].mean()
            #
            #     self.log('total_bits_per_dimension', log_dict.pop('total_bpd'), prog_bar=True)
            #     self.log_dict(log_dict)

        return loss

    # def test_step(self, batch, batch_nb):
    #     batch_size = batch["pixel_values"].shape[0]
    #     samples = batch["pixel_values"]  # x_start
    #
    #     diffusion_model_fn = self.get_diffusion_model(batch)
    #
    #     log_dict = self.calculate_bits_per_dimension(x_start=samples, diffusion_model_fn=diffusion_model_fn, max_batch_size=-1)
    #     for key in log_dict.keys():
    #         log_dict[key] = log_dict[key].sum()
    #
    #     log_dict['num_samples'] = torch.tensor(batch_size, dtype=torch.long)
    #     return log_dict
    #
    # def test_epoch_end(
    #     self, outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]
    # ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
    #     total_samples = torch.sum(torch.stack([x['num_samples'] for x in outputs])).float()
    #     total_bpd = torch.sum(torch.stack([x['total_bpd'] for x in outputs])) / total_samples
    #     terms_bpd = torch.sum(torch.stack([x['terms_bpd'] for x in outputs])) / total_samples
    #     prior_bpd = torch.sum(torch.stack([x['prior_bpd'] for x in outputs])) / total_samples
    #
    #     result = {'test_total_bpd': total_bpd, 'test_terms_bpd': terms_bpd, 'test_prior_bpd': prior_bpd}
    #     self.log_dict(result)
    #
    #     return result

    def sample(self, batch_size: int, image_size: int, device: torch.device = None):
        with torch.inference_mode():
            self.eval()

            if device is None:
                device = next(self.parameters()).device

            shape = [batch_size, self.channels, image_size, image_size]
            return self.sampler.sample(self.diffusion_model, shape=shape, device=device)

    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, t: Optional[int] = None, lambd: float = 0.5, **kwargs):
        raise NotImplementedError()

    # Sampler modification
    def change_sampler(self, sampler_cfg):
        self.sampler = instantiate(sampler_cfg)  # type: PredictorCorrectorSampler
        self.sampler.update_sde(self.sde)

        self.cfg.sampler = sampler_cfg
        self.cfg = self.cfg  # Update internal config

        logging.info(f"Sampler config changed to : \n"
                     f"{OmegaConf.to_yaml(sampler_cfg)}")
