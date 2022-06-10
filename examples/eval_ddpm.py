import pytorch_lightning as pl
import torchvision.utils
from omegaconf import OmegaConf, open_dict
from dataclasses import dataclass, is_dataclass
from typing import Optional
import datetime
from pathlib import Path

from diffusion_model_nemo.models import DDPM
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

"""

python train_ddpm.py ^
    --config-path="configs/unet" ^
    --config-name="unet_small.yaml" ^
    model.image_size=28 ^
    model.timesteps=200 ^
    model.channels=1 ^
    model.save_every=400 ^
    model.diffusion_model.resnet_block_groups=8 ^
    model.diffusion_model.dim_mults=[1,2,4] ^
    model.train_ds.name="fashion_mnist" ^
    model.train_ds.split="train" ^
    trainer.max_epochs=3 ^
    trainer.strategy=null ^
    exp_manager.name="DDPM" ^
    exp_manager.exp_dir="Experiments" ^
    exp_manager.create_wandb_logger=True ^
    exp_manager.wandb_logger_kwargs.name="DDPM" ^
    exp_manager.wandb_logger_kwargs.project="DDPM" ^
    exp_manager.wandb_logger_kwargs.entity="smajumdar"


"""


@dataclass
class EvalConfig:
    model_path: str = "DDPM.nemo"
    batch_size: int = 32
    image_size: int = -1

    output_dir: str = "samples"
    add_timestamp: bool = True


@hydra_runner(config_path=None, config_name="EvalConfig", schema=EvalConfig)
def main(cfg: EvalConfig):
    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    model = DDPM.restore_from(cfg.model_path)  # type: DDPM

    if cfg.image_size < 0:
        cfg.image_size = model.image_size

    samples = model.sample(batch_size=cfg.batch_size, image_size=cfg.image_size)

    # import matplotlib.pyplot as plt
    # plt.imshow(samples[-1][0].transpose(0, 1).transpose(1, 2), cmap='gray')
    # plt.show()

    results_dir = cfg.get('output_dir')
    results_folder = Path(results_dir).absolute()

    if cfg.add_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        results_folder = results_folder / timestamp

    results_folder.mkdir(exist_ok=True, parents=True)

    for result_idx in range(cfg.batch_size):
        result_path = str(results_folder / f"sample_{result_idx + 1}.png")
        result = samples[-1][result_idx]
        result = (result + 1) * 0.5
        torchvision.utils.save_image(result, result_path)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
