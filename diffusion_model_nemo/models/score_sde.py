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
from diffusion_model_nemo.modules import LikelihoodEstimate
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

        # initialize the loss function
        self.loss = instantiate(self.cfg.loss)  # type: SDEScoreFunctionLoss
        self.loss.update_sde(self.sde)

        # initialize the likelihood evaluator
        likelihood_cfg = self.cfg.get('likelihood_estimate', None)
        if likelihood_cfg is None:
            self.likelihood_estimator = LikelihoodEstimate()
        else:
            self.likelihood_estimator = instantiate(self.cfg.likelihood_estimate)  # type: LikelihoodEstimate
        self.likelihood_estimator.update_sde(self.sde)

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

        loss = self.loss.forward(diffusion_model_fn, x_start=samples, t=t, noise=noise)

        # Compute log dict
        self.log('train_loss', loss.detach())
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])
        self.log('global_step', float(self.trainer.global_step))

        # save generated images
        if self.trainer.global_step != 0 and self.trainer.global_step % self.save_and_sample_every == 0:
            self.eval()
            self._save_image_step(batch_size=batch_size, step=self.trainer.global_step)

            if self.cfg.get('compute_bpd', False):
                bpds, z, nfe = self.likelihood_estimator.likelihood(diffusion_model_fn, samples)
                avg_bpd = bpds.detach().mean()
                nfe = torch.tensor(nfe, dtype=torch.float32)
                self.log('total_bits_per_dimension', avg_bpd, prog_bar=True)
                self.log('num_forward_evaluations', nfe)

        return loss

    def test_step(self, batch, batch_nb):
        batch_size = batch["pixel_values"].shape[0]
        samples = batch["pixel_values"]  # x_start

        diffusion_model_fn = self.get_diffusion_model(batch)

        bpds, z, nfe = self.likelihood_estimator.likelihood(diffusion_model_fn, samples)
        bpds = bpds.detach().cpu()
        nfe = torch.tensor(nfe, dtype=torch.float32)

        log_dict = {'bpds': bpds, 'nfe': nfe}
        log_dict['num_samples'] = torch.tensor(batch_size, dtype=torch.long)
        return log_dict

    def test_epoch_end(
        self, outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        total_samples = torch.sum(torch.stack([x['num_samples'] for x in outputs])).float()
        total_bpd = torch.sum(torch.stack([x['bpds'] for x in outputs])) / total_samples
        nfes = torch.sum(torch.stack([x['nfe'] for x in outputs])) / total_samples

        result = {'test_total_bpd': total_bpd, 'avg_num_forward_evaluations': nfes}
        self.log_dict(result)

        return result

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
