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

python interpolate_ddim.py ^


"""


@dataclass
class InterpolateConfig:
    model_path: str = "DDPM.nemo"

    # DDIM Interpolation Config
    interpolation_step_size: float = 0.05  # step size between [0.0, 1.0]
    ddim_timesteps: int = 100  # DDIM requires much smaller number of steps than DDPM; -1 uses original timesteps

    # data arguments
    batch_size: int = 32

    # additional arguments
    output_dir: str = "interpolations"
    add_timestamp: bool = True
    seed: Optional[int] = None


def use_ddim_sampler(model: DDPM, cfg: InterpolateConfig):
    # Change sampler
    sampler_cfg = model.cfg.sampler
    with open_dict(sampler_cfg):
        sampler_cfg._target_ = "diffusion_model_nemo.modules.GeneralizedGaussianDiffusion"
        sampler_cfg.eta = 0.0
        sampler_cfg.ddim_timesteps = cfg.ddim_timesteps

        model.change_sampler(sampler_cfg)

    return model


def slerp(z1, z2, alpha):
    theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
    return torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1 + torch.sin(alpha * theta) / torch.sin(theta) * z2


def compute_interpolation(model: DDPM, cfg: InterpolateConfig):
    # Compute the interpolation
    z1 = torch.randn(
        1,
        model.cfg.channels,
        model.cfg.image_size,
        model.cfg.image_size,
        device=model.device,
    )
    z2 = torch.randn_like(z1)
    alpha = torch.arange(0.0, 1.01, cfg.interpolation_step_size).to(z1.device)

    z_ = []
    for i in range(alpha.size(0)):
        z_.append(slerp(z1, z2, alpha[i]))

    x = torch.cat(z_, dim=0)
    xs = []

    # Hard coded here, modify to your preferences
    with torch.no_grad():
        for i in range(0, x.size(0), cfg.batch_size):
            samples = model.sampler.interpolate(model.diffusion_model, x=x[i: i + cfg.batch_size])
            samples = samples[-1]  # last timestep values
            xs.append(samples)

    x = torch.cat(xs, dim=0)
    return x


@hydra_runner(config_path=None, config_name="InterpolateConfig", schema=InterpolateConfig)
def main(cfg: InterpolateConfig):
    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    model = DDPM.restore_from(cfg.model_path)  # type: DDPM
    model = model.eval()

    # Seed everything if provided
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    # Setup DDIM sampler
    use_ddim_sampler(model, cfg)

    # Compute interpolations
    x = compute_interpolation(model, cfg)

    results_dir = cfg.get('output_dir')
    results_folder = Path(results_dir).absolute()

    if cfg.add_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        results_folder = results_folder / timestamp

    results_folder.mkdir(exist_ok=True, parents=True)

    for result_idx in range(len(x)):
        result_path = str(results_folder / f"interpolation_{result_idx + 1}.png")

        result = x[result_idx]
        torchvision.utils.save_image(result, result_path)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
