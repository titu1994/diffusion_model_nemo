import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from typing import List, Dict, Optional, Union
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only
from pathlib import Path
import datetime
from omegaconf import OmegaConf, DictConfig, open_dict
from hydra.utils import instantiate

from diffusion_model_nemo.data.hf_vision_data import HFVisionDataset
from diffusion_model_nemo.modules.diffusion_process import AbstractDiffusionProcess
from diffusion_model_nemo.utils import num_to_groups

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
        batches = num_to_groups(4, batch_size)
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
