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

from diffusion_model_nemo.data.hf_vision_data import HFVisionDataset
from diffusion_model_nemo.modules.diffusion_process import AbstractDiffusionProcess
from diffusion_model_nemo.loss.variational_bound_loss import VariationalBoundLoss
from diffusion_model_nemo import utils

from nemo.core import ModelPT, PretrainedModelInfo, typecheck
from nemo.core.neural_types import NeuralType
from nemo.utils import logging


class AbstractDiffusionModel(ModelPT):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer=trainer)

        # Timesteps must be defined at the global level
        self.image_size = cfg.image_size
        self.timesteps = cfg.timesteps
        self.channels = cfg.channels

        # setup output dirs
        self.save_and_sample_every = cfg.get('save_every', 1000)
        self._result_dir = None

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    @typecheck()
    def forward(self, batch_size, image_size):
        raise NotImplementedError()

    def validation_step(self, batch, batch_nb):
        return None

    def _setup_dataloader(self, cfg):
        if cfg.name is not None:
            dataset = HFVisionDataset(name=cfg.name, split=cfg.split, cache_dir=cfg.get('cache_dir', None))

            dataloader = DataLoader(
                dataset,
                batch_size=cfg.batch_size,
                shuffle=cfg.get('shuffle', False),
                num_workers=cfg.get('num_workers', 2),
                pin_memory=cfg.get('pin_memory', True),
            )
            return dataloader

        else:
            return None

    def sample(self, batch_size: int, image_size: int):
        raise NotImplementedError()

    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, t: Optional[int] = None, lambd: float = 0.5, **kwargs):
        raise NotImplementedError()

    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        self._update_dataset_config('train', train_data_config)
        if 'shuffle' in train_data_config:
            train_data_config['shuffle'] = True

        self._train_dl = self._setup_dataloader(train_data_config)

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        self._update_dataset_config('validation', val_data_config)

        if 'shuffle' in val_data_config:
            val_data_config['shuffle'] = False

    @rank_zero_only
    def _prepare_output_dir(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")

        results_dir = self.cfg.get('results_dir', f'./results/{timestamp}/')
        results_folder = Path(results_dir).absolute()
        results_folder.mkdir(exist_ok=True, parents=True)
        self._result_dir = results_folder

    @rank_zero_only
    def _save_image_step(self, batch_size, step):
        if self._result_dir is None:
            self._prepare_output_dir()

        img_size = self.image_size
        milestone = step // self.save_and_sample_every
        batches = utils.num_to_groups(4, batch_size)
        all_images_list = list(
            map(
                lambda n: self.sampler.sample(self.diffusion_model, shape=[n, self.channels, img_size, img_size]),
                batches,
            )
        )
        for idx, image_list in enumerate(all_images_list):
            all_images = torch.cat(image_list, dim=0)
            save_path = str(self._result_dir / f'sample-{milestone}-{idx + 1}.png')
            save_image(all_images, save_path, nrow=6)
            logging.info(f"Images saved at path : {save_path}")

    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return None

    def change_sampler(self, sampler_cfg: DictConfig):
        self.sampler = instantiate(sampler_cfg)  # type: AbstractDiffusionProcess
        self.cfg.sampler = sampler_cfg
        self.cfg = self.cfg  # update PTL config

        logging.info(f"Sampler changed to : \n{OmegaConf.to_yaml(sampler_cfg)}")

    def calculate_bits_per_dimension(self, x_start: torch.Tensor, diffusion_model_fn, max_batch_size: int = 32):
        # Implemented from https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L313
        B, C, H, W = x_start.size()
        T = self.timesteps

        if max_batch_size > 0:
            B = min(max_batch_size, B)

        with torch.inference_mode():
            x_start = x_start[:B]

            terms_buffer = torch.zeros(B, T, device=x_start.device)
            T_range = torch.arange(T, device=x_start.device).unsqueeze(0)
            t_b = torch.full([B], fill_value=T, device=x_start.device, dtype=torch.long)
            zero_tensor = torch.tensor(0.0, device=x_start.device)

            for t in tqdm(range(T - 1, 0, -1), desc='Computing bits per dimension', total=T):
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
            'total_bpd': total_bpd_b,
            'terms_bpd': terms_buffer,
            'prior_bpd': prior_bpd_b,
        }

        return result
