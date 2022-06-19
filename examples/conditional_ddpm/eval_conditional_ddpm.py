import torchvision.utils
from omegaconf import OmegaConf, open_dict
from dataclasses import dataclass, is_dataclass
from typing import Optional
import datetime
from pathlib import Path
from pytorch_lightning import seed_everything

from diffusion_model_nemo.models import ConditionalDDPM
from nemo.core.config import hydra_runner
from nemo.utils import logging

"""

python eval_ddpm.py ^
    

"""


@dataclass
class EvalConfig:
    # ConditionalDDPM Config
    model_path: str = "ConditionalDDPM.nemo"
    batch_size: int = 32
    image_size: int = -1
    label_id: Optional[int] = 0

    # DDIM Config
    use_ddim_sampler: bool = False
    ddim_eta: float = 0.0  # 0 = DDIM mode, 1 = DDPM mode
    ddim_timesteps: int = 100  # DDIM requires much smaller number of steps than DDPM; -1 uses original timesteps

    output_dir: str = "samples"
    add_timestamp: bool = True

    # animation settings
    show_diffusion: bool = False
    frame_step: int = 1  # interval of timesteps to plot
    animation_format: str = "mp4"  # [gif, mp4]
    fps: int = 30

    seed: Optional[int] = None


def maybe_use_ddim_sampler(model: ConditionalDDPM, cfg: EvalConfig):

    # Check if user wants DDIM sampling
    if cfg.use_ddim_sampler:
        # Change sampler
        sampler_cfg = model.cfg.sampler
        with open_dict(sampler_cfg):
            sampler_cfg._target_ = "diffusion_model_nemo.modules.GeneralizedGaussianDiffusion"
            sampler_cfg.eta = cfg.ddim_eta
            sampler_cfg.ddim_timesteps = cfg.ddim_timesteps

            model.change_sampler(sampler_cfg)
    return


@hydra_runner(config_path=None, config_name="EvalConfig", schema=EvalConfig)
def main(cfg: EvalConfig):
    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    model = ConditionalDDPM.restore_from(cfg.model_path)  # type: ConditionalDDPM
    model = model.eval()

    if cfg.image_size < 0:
        cfg.image_size = model.image_size

    # Seed everything if provided
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    # Maybe use DDIM sampler
    maybe_use_ddim_sampler(model, cfg)

    # Compute samples
    if cfg.label_id is not None:
        logging.info(f'Selected class for sample generation: {cfg.label_id} ')
    samples = model.sample(batch_size=cfg.batch_size, image_size=cfg.image_size, label=cfg.label_id)

    results_dir = cfg.get('output_dir')
    results_folder = Path(results_dir).absolute()

    if cfg.add_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        results_folder = results_folder / timestamp

    results_folder.mkdir(exist_ok=True, parents=True)

    for result_idx in range(cfg.batch_size):
        if not cfg.show_diffusion:
            if cfg.use_ddim_sampler:
                result_path = str(results_folder / f"sample_{result_idx + 1}_ddim_timesteps_{cfg.ddim_timesteps}")
            else:
                result_path = str(results_folder / f"sample_{result_idx + 1}")

            if cfg.label_id is not None:
                result_path = result_path + f'_class_{cfg.label_id}'
            else:
                result_path = result_path + f'_unconditional'

            result_path = result_path + '.png'
            result = samples[-1][result_idx]
            torchvision.utils.save_image(result, result_path)
        else:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation

            result_path = str(results_folder / f"sample_{result_idx + 1}.{cfg.animation_format}")

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ims = []
            num_channels = samples[-1][result_idx].size(0)
            cmap = 'gray' if num_channels == 1 else None
            for i in range(0, len(samples), cfg.frame_step):
                ttl = plt.text(
                    0.5,
                    1.01,
                    f"T = {i + 1:4d} / {model.timesteps}",
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    transform=ax.transAxes,
                )
                im = plt.imshow(
                    samples[i][result_idx].transpose(0, 1).transpose(1, 2),
                    cmap=cmap,
                    animated=True,
                )
                ims.append([im, ttl])

            print(f"Creating animation for {str(result_path)}")
            interval = max(1, round(10000.0 / model.timesteps))
            # repeat_delay=5000
            animate = animation.ArtistAnimation(fig, ims, repeat=False, interval=interval, blit=True, )
            animate.save(result_path, fps=cfg.fps)
            print()


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter