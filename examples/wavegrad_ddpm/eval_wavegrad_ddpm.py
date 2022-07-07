import torch
import torchvision.utils
from omegaconf import OmegaConf, open_dict
from dataclasses import dataclass, is_dataclass
from typing import Optional
import datetime
from pathlib import Path
from pytorch_lightning import seed_everything

from diffusion_model_nemo.modules import diffusion_process as DP
from diffusion_model_nemo.models import WavegradDDPM
from nemo.core.config import hydra_runner
from nemo.utils import logging

"""

python eval_ddpm.py ^
    

"""


@dataclass
class Schedules:
    cosine: DP.CosineSchedule = DP.CosineSchedule()
    linear: DP.LinearSchedule = DP.LinearSchedule(beta_start=1e-6, beta_end=0.9)
    quadratic: DP.QuadraticSchedule = DP.QuadraticSchedule(beta_start=1e-6, beta_end=0.01)
    sigmoid: DP.SigmoidSchedule = DP.SigmoidSchedule(beta_start=1e-6, beta_end=0.01)


@dataclass
class ScheduleConfig:
    schedule_name: Optional[str] = None
    schedule_cfg: Schedules = Schedules()


@dataclass
class EvalConfig:
    # DDPM Config
    model_path: str = "WaveGrad-DDPM.nemo"
    batch_size: int = 32
    image_size: int = -1
    timesteps: int = 5

    # Schedule config
    override_schedule: bool = True
    schedule: ScheduleConfig = ScheduleConfig()

    # Output config
    output_dir: str = "samples"
    add_timestamp: bool = True
    grid_plot: bool = True

    # animation settings
    show_diffusion: bool = False
    frame_step: int = 1  # interval of timesteps to plot
    animation_format: str = "mp4"  # [gif, mp4]
    fps: int = 30

    seed: Optional[int] = None


def maybe_change_sampler_schedule(model: WavegradDDPM, cfg: EvalConfig):

    # Check if user wants DDIM sampling
    if cfg.override_schedule:
        # Change sampler
        model.sampler.change_noise_schedule(
            schedule_name=cfg.schedule.schedule_name, schedule_cfg=cfg.schedule.schedule_cfg
        )

    if cfg.timesteps > 0:
        model.sampler.compute_constants(cfg.timesteps)

    return model


@hydra_runner(config_path=None, config_name="EvalConfig", schema=EvalConfig)
def main(cfg: EvalConfig):
    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    model = WavegradDDPM.restore_from(cfg.model_path)  # type: WavegradDDPM
    model.eval()

    if cfg.image_size < 0:
        cfg.image_size = model.image_size

    # Seed everything if provided
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    if cfg.timesteps > 0:
        model = maybe_change_sampler_schedule(model, cfg)

    cfg.timesteps = model.sampler.timesteps

    # Compute samples
    samples = model.sample(batch_size=cfg.batch_size, image_size=cfg.image_size)

    results_dir = cfg.get('output_dir')
    results_folder = Path(results_dir).absolute()

    if cfg.add_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        results_folder = results_folder / timestamp

    results_folder.mkdir(exist_ok=True, parents=True)

    images = []
    for result_idx in range(cfg.batch_size):
        if not cfg.show_diffusion:
            result_path = str(results_folder / f"sample_{cfg.timesteps}_timesteps_{result_idx + 1}.png")
            result = samples[-1][result_idx]

            if cfg.grid_plot:
                images.append(result)
            else:
                torchvision.utils.save_image(result, result_path)
        else:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation

            result_path = str(
                results_folder / f"sample_{cfg.timesteps}_timesteps_{result_idx + 1}.{cfg.animation_format}"
            )

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
            animate = animation.ArtistAnimation(
                fig,
                ims,
                repeat=False,
                interval=interval,
                blit=True,
            )
            animate.save(result_path, fps=cfg.fps)
            print()

    if len(images) > 0 and cfg.grid_plot:
        result_path = str(results_folder / f"sample_grid_{cfg.timesteps}_timesteps.png")

        n_rows = int(torch.tensor(len(images), dtype=torch.float32).sqrt().round())
        torchvision.utils.save_image(images, result_path, nrow=n_rows)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
