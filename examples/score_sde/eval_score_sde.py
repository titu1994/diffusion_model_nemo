import torchvision.utils
from omegaconf import OmegaConf, open_dict
from dataclasses import dataclass, is_dataclass
from typing import Optional
import datetime
from pathlib import Path
from pytorch_lightning import seed_everything

from diffusion_model_nemo.models import ScoreSDE
from nemo.core.config import hydra_runner
from nemo.utils import logging

"""

python eval_ddpm.py ^
    

"""


@dataclass
class EvalConfig:
    # DDPM Config
    model_path: str = "VPSDE-DDPM.nemo"
    batch_size: int = 32
    image_size: int = -1

    # Predictor-Corrector Sampler modifications
    pc_snr: Optional[float] = None
    pc_predictor: Optional[str] = None  # [null, euler_maruyama, reverse_diffusion, ancestral_sampling]
    pc_corrector: Optional[str] = None  # [null, langevin, ald]

    output_dir: str = "samples"
    add_timestamp: bool = True

    # animation settings
    show_diffusion: bool = False
    frame_step: int = 1  # interval of timesteps to plot
    animation_format: str = "mp4"  # [gif, mp4]
    fps: int = 30

    seed: Optional[int] = None


def update_predictor_corrector_sampler_cfg(model: ScoreSDE, cfg: EvalConfig):
    sampler_cfg = model.cfg.sampler

    with open_dict(sampler_cfg):
        if cfg.pc_snr is not None:
            sampler_cfg.snr = cfg.pc_snr
        if cfg.pc_predictor is not None:
            sampler_cfg.predictor = cfg.pc_predictor
        if cfg.pc_corrector is not None:
            sampler_cfg.corrector = cfg.pc_corrector

    # Updating sampler config
    model.change_sampler(sampler_cfg)


@hydra_runner(config_path=None, config_name="EvalConfig", schema=EvalConfig)
def main(cfg: EvalConfig):
    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    model = ScoreSDE.restore_from(cfg.model_path)  # type: ScoreSDE
    model.eval()

    if cfg.image_size < 0:
        cfg.image_size = model.image_size

    # Seed everything if provided
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    # Update PC sampler
    update_predictor_corrector_sampler_cfg(model, cfg)

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
