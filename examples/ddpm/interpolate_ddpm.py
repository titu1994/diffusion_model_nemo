import torch
import torchvision
import torchvision.utils
from omegaconf import OmegaConf, open_dict
from dataclasses import dataclass, is_dataclass
from typing import Optional
import datetime
from pathlib import Path
from pytorch_lightning import seed_everything

from diffusion_model_nemo.models import DDPM
from diffusion_model_nemo.data.hf_vision_data import get_transform
from nemo.core.config import hydra_runner
from nemo.utils import logging
from PIL import Image


"""

python eval_ddpm.py ^


"""


@dataclass
class InterpolateConfig:
    dir_1: str = "dir1/"
    dir_2: str = "dir2/"
    model_path: str = "DDPM.nemo"

    # data arguments
    timesteps: int = -1
    image_size: int = -1
    lambd: float = 0.5

    # additional args:
    center_crop: bool = False

    output_dir: str = "interpolations"
    add_timestamp: bool = True
    seed: Optional[int] = None


def read_image_dir(path: str, channels: int, cfg: InterpolateConfig):
    path = Path(path).absolute()
    paths = list(path.glob("*.png")) + list(path.glob("*.jpg")) + list(path.glob("*.jpeg"))

    transform = get_transform(image_size=cfg.image_size, scale=True, center_crop=cfg.get('center_crop', False))
    images = []

    for filepath in paths:
        img = Image.open(str(filepath))

        if channels == 1:
            img = img.convert('L')

        img = transform(img)
        images.append(img)

    images = torch.stack(images, dim=0)
    return images


@hydra_runner(config_path=None, config_name="InterpolateConfig", schema=InterpolateConfig)
def main(cfg: InterpolateConfig):
    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    model = DDPM.restore_from(cfg.model_path)  # type: DDPM

    if cfg.timesteps <= 0:
        cfg.timesteps = model.timesteps - 1

    if cfg.image_size < 0:
        cfg.image_size = model.image_size

    x1_images = read_image_dir(cfg.dir_1, channels=model.channels, cfg=cfg)
    x2_images = read_image_dir(cfg.dir_2, channels=model.channels, cfg=cfg)

    if len(x1_images) != len(x2_images):
        raise ValueError(f"Number of images in the directories must match exactly ! "
                         f"Found {len(x1_images)} images in dir : {cfg.dir_1} and "
                         f"Found {len(x2_images)} images in dir : {cfg.dir_2}")

    # Seed everything if provided
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    # Compute the interpolation
    samples = model.interpolate(x1=x1_images, x2=x2_images, t=cfg.timesteps, lambd=cfg.lambd)

    results_dir = cfg.get('output_dir')
    results_folder = Path(results_dir).absolute()

    if cfg.add_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        results_folder = results_folder / timestamp

    results_folder.mkdir(exist_ok=True, parents=True)

    for result_idx in range(len(x1_images)):
        result_path = str(results_folder / f"interpolation_{result_idx + 1}_lambda_{cfg.lambd}.png")

        sample_ = [samples[t_][result_idx] for t_ in range(cfg.timesteps)]
        result = torch.stack(sample_, dim=0)
        torchvision.utils.save_image(result, result_path, nrows=max(32, (cfg.timesteps + 1) // 16))


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
