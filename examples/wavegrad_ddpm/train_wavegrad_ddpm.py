import pytorch_lightning as pl
from omegaconf import OmegaConf

from diffusion_model_nemo.models import WavegradDDPM
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

"""
# Train script

# Fashion MNIST
python train_wavegrad_ddpm.py ^
    --config-path="../configs/wavegrad_ddpm" ^
    --config-name="unet_small.yaml" ^
    model.image_size=28 ^
    model.timesteps=1000 ^
    model.channels=1 ^
    model.save_every=500 ^
    model.diffusion_model.resnet_block_groups=8 ^
    model.diffusion_model.dim_mults=[1,2,4] ^
    model.train_ds.name="fashion_mnist" ^
    model.train_ds.split="train" ^
    trainer.max_epochs=5 ^
    trainer.strategy=null ^
    exp_manager.name="WaveGrad-DDPM" ^
    exp_manager.exp_dir="Experiments" ^
    exp_manager.create_wandb_logger=True ^
    exp_manager.wandb_logger_kwargs.name="Wavegrad-DDPM" ^
    exp_manager.wandb_logger_kwargs.project="Wavegrad-DDPM" ^
    exp_manager.wandb_logger_kwargs.entity="smajumdar"
    
    
# CIFAR 10
    
    +init_from_nemo_model="final_models/DDPM.nemo" ^

python train_ddpm.py ^
    --config-path="../configs/ddpm" ^
    --config-name="unet_small.yaml" ^
    model.image_size=32 ^
    model.timesteps=1000 ^
    model.save_every=20 ^
    model.diffusion_model.dim=32 ^
    model.diffusion_model.dim_mults=[1,2,2,2] ^
    model.sampler.class_conditional=True ^
    model.train_ds.name="cifar10" ^
    model.train_ds.split="train" ^
    model.train_ds.batch_size=128 ^
    model.optim.lr=0.0002 ^
    trainer.max_epochs=5 ^
    trainer.strategy=null ^
    exp_manager.name="DDPM" ^
    exp_manager.exp_dir="CIFAR-Experiments" ^
    exp_manager.create_wandb_logger=True ^
    exp_manager.wandb_logger_kwargs.name="Wavegrad-DDPM" ^
    exp_manager.wandb_logger_kwargs.project="Wavegrad-CIFAR-DDPM" ^
    exp_manager.wandb_logger_kwargs.entity="smajumdar"
    
    
"""


@hydra_runner(config_path="../configs/unet", config_name="unet_small.yaml")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = WavegradDDPM(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    model.maybe_init_from_pretrained_checkpoint(cfg)

    trainer.fit(model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
