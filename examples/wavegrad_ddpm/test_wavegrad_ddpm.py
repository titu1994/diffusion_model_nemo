import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf, MISSING, open_dict
from dataclasses import dataclass, is_dataclass
from typing import Optional, List

from diffusion_model_nemo.modules import diffusion_process as DP
from diffusion_model_nemo.models import WavegradDDPM
from nemo.core.config import hydra_runner
from nemo.utils import logging

"""
# Test script

# Fashion MNIST
python test_wavegrad_ddpm.py ^
    nemo_model=WaveGrad-DDPM.nemo ^
    pretrained_model=null ^
    timesteps=-1 ^
    cuda=-1 ^
    test_ds.name=fashion_mnist ^
    test_ds.split='test[:8]' ^
    test_ds.batch_size=128
    
    
# CIFAR 10

python test_wavegrad_ddpm.py ^
    nemo_model=final_models/WaveGrad-DDPM.nemo ^
    pretrained_model=null ^
    timesteps=-1 ^
    cuda=-1 ^
    test_ds.name=cifar10 ^
    test_ds.split='test' ^
    test_ds.batch_size=256
    
"""


@dataclass
class Schedules:
    cosine: DP.CosineSchedule = DP.CosineSchedule()
    linear: DP.LinearSchedule = DP.LinearSchedule(beta_start=1e-6, beta_end=0.01)
    quadratic: DP.QuadraticSchedule = DP.QuadraticSchedule(beta_start=1e-6, beta_end=0.01)
    sigmoid: DP.SigmoidSchedule = DP.SigmoidSchedule(beta_start=1e-6, beta_end=0.01)


@dataclass
class ScheduleConfig:
    schedule_name: Optional[str] = None
    schedule_cfg: Schedules = Schedules()


@dataclass()
class TestDatasetConfig:
    name: str = MISSING
    split: Optional[str] = None
    cache_dir: Optional[str] = None
    # dataloader params
    batch_size: int = 32
    shuffle: bool = False
    num_workers: int = 4
    pin_memory: bool = True


@dataclass()
class TestConfig:
    nemo_model: Optional[str] = None
    pretrained_model: Optional[str] = None

    test_ds: TestDatasetConfig = TestDatasetConfig()

    timesteps: int = -1
    cuda: int = -1

    # Schedule config
    override_schedule: bool = True
    search_schedule_iters: int = 1000  # set to 0 to disable searching of schedule
    schedule: ScheduleConfig = (
        ScheduleConfig()
    )  # used only for manual selection of schedule; recommended to use automatic search

    # seed
    seed: Optional[int] = None


def maybe_change_sampler_schedule(model: WavegradDDPM, cfg: TestConfig):

    # Check if user wants DDIM sampling
    if cfg.override_schedule:
        if cfg.search_schedule_iters > 0:
            # Change sampler via search
            model.sampler.search_noise_schedule_coefficients(
                timesteps=cfg.timesteps, iters=cfg.search_schedule_iters, seed=cfg.seed
            )
            model.sampler.change_noise_schedule()

        else:
            # Change sampler manually
            model.sampler.change_noise_schedule(
                schedule_name=cfg.schedule.schedule_name, schedule_cfg=cfg.schedule.schedule_cfg
            )

    if cfg.timesteps > 0:
        model.sampler.compute_constants(cfg.timesteps)

    return model


@hydra_runner(config_path=None, config_name="TestConfig", schema=TestConfig)
def main(cfg: TestConfig):
    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if cfg.nemo_model is not None and cfg.pretrained_model is not None:
        raise ValueError("Only one of `nemo_model` or `pretrained_model` should be passed")
    elif cfg.nemo_model is None and cfg.pretrained_model is None:
        raise ValueError("At least one of `nemo_model` or `pretrained_model` should be passed.")

    if cfg.cuda < 0:
        device_id = [0] if torch.cuda.is_available() else 0
    else:
        device_id = cfg.cuda if torch.cuda.is_available() else 0

    if cfg.seed is not None:
        pl.seed_everything(cfg.seed)

    trainer = pl.Trainer(devices=device_id, accelerator='auto')
    if cfg.nemo_model:
        model = WavegradDDPM.restore_from(cfg.nemo_model, trainer=trainer, map_location='cpu')  # type: WavegradDDPM
    else:
        model = WavegradDDPM.from_pretrained(
            cfg.pretrained_model, trainer=trainer, map_location='cpu'
        )  # type: WavegradDDPM

    if cfg.timesteps > 0:
        model = maybe_change_sampler_schedule(model, cfg)

    model.eval()
    model.freeze()

    model.setup_multiple_test_data(cfg.test_ds)

    trainer.test(model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
