import torchvision.utils
from omegaconf import OmegaConf, open_dict
from dataclasses import dataclass, is_dataclass
from typing import Optional
import datetime
from pathlib import Path
from pytorch_lightning import seed_everything

from diffusion_model_nemo.models import DDPM
from nemo.core.config import hydra_runner
from nemo.utils import logging

"""

python eval_ddpm.py ^
    

"""


@dataclass
class EvalConfig:
    model_path: str = "DDPM.nemo"
    batch_size: int = 32
    image_size: int = -1

    output_dir: str = "samples"
    add_timestamp: bool = True
    show_diffusion: bool = False

    seed: Optional[int] = None


@hydra_runner(config_path=None, config_name="EvalConfig", schema=EvalConfig)
def main(cfg: EvalConfig):
    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    model = DDPM.restore_from(cfg.model_path)  # type: DDPM

    if cfg.image_size < 0:
        cfg.image_size = model.image_size

    # Seed everything if provided
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    # Compute samples
    samples = model.sample(batch_size=cfg.batch_size, image_size=cfg.image_size)

    results_dir = cfg.get('output_dir')
    results_folder = Path(results_dir).absolute()

    if cfg.add_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        results_folder = results_folder / timestamp

    results_folder.mkdir(exist_ok=True, parents=True)

    for result_idx in range(cfg.batch_size):
        if not cfg.show_diffusion:
            result_path = str(results_folder / f"sample_{result_idx + 1}.png")
            result = samples[-1][result_idx]
            torchvision.utils.save_image(result, result_path)
        else:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation

            result_path = str(results_folder / f"sample_{result_idx + 1}.gif")

            fig = plt.figure()
            ims = []
            num_channels = samples[-1][result_idx].size(1)
            cmap = 'gray' if num_channels == 1 else None
            for i in range(len(samples)):
                im = plt.imshow(samples[i][result_idx].transpose(0, 1).transpose(1, 2), cmap=cmap, animated=True)
                ims.append([im])

            animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=5000)
            animate.save(result_path)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
