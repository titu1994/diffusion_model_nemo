import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf, MISSING, open_dict
from dataclasses import dataclass, is_dataclass
from typing import Optional, List

from diffusion_model_nemo.models import ImprovedDDPM
from nemo.core.config import hydra_runner
from nemo.utils import logging

"""
# Test script

# Fashion MNIST
python test_ddpm.py ^
    nemo_model=null ^
    pretrained_model=null ^
    cuda=-1 ^
    test_ds.name=fashion_mnist ^
    test_ds.split=test ^
    test_ds.batch_size=128
    
    
# CIFAR 10

python test_ddpm.py ^
    nemo_model='final/Improved-DDPM.nemo' ^
    pretrained_model=null ^
    cuda=-1 ^
    test_ds.name=cifar10 ^
    test_ds.split='test' ^
    test_ds.batch_size=256
    
"""

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
    cuda: int = -1

    # seed
    seed: Optional[int] = None


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
        model = ImprovedDDPM.restore_from(cfg.nemo_model, trainer=trainer, map_location='cpu')
    else:
        model = ImprovedDDPM.from_pretrained(cfg.pretrained_model, trainer=trainer, map_location='cpu')

    model.eval()
    model.freeze()

    model.setup_multiple_test_data(cfg.test_ds)

    trainer.test(model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
