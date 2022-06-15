import torch
from abc import ABC, abstractmethod

from typing import Optional, List


def cosine_beta_schedule(timesteps, s=0.008, min_clip=0.0001, max_clip=0.999):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, min_clip, max_clip)


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    beta_start = beta_start
    beta_end = beta_end
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class AbstractDiffusionProcess(ABC, torch.nn.Module):
    """ Abstract Diffusion Process which provides common interface to common implementations """
    use_class_conditioning: bool = False

    def __init__(self, timesteps, schedule_name, schedule_cfg=None):
        super().__init__()
        self.timesteps = timesteps
        self.schedule_name = schedule_name
        self.schedule_cfg = schedule_cfg if schedule_cfg is not None else {}
        self.schedule_fn = None

    @abstractmethod
    def compute_constants(self, timesteps):
        raise NotImplementedError()

    @abstractmethod
    def q_mean_variance(self, x_start: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError()

    @abstractmethod
    def q_posterior(self, x_start: torch.Tensor, x: torch.Tensor, t: torch.Tensor):
        return NotImplementedError()

    @abstractmethod
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        raise NotImplementedError()

    @abstractmethod
    def p_mean_variance(self, model, x: torch.Tensor, t: torch.Tensor, model_output: torch.Tensor = None):
        raise NotImplementedError()

    @abstractmethod
    @torch.no_grad()
    def p_sample(self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError()

    @abstractmethod
    @torch.no_grad()
    def sample(self, model: torch.nn.Module, shape: List[int]):
        raise NotImplementedError()

    @torch.no_grad()
    def interpolate(self, model, x1: torch.Tensor, x2: torch.Tensor, t: Optional[int] = None, lambd: float = 0.0):
        raise NotImplementedError()

    def extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward(self, *args, **kwargs):
        raise RuntimeWarning(f"{self.__class__.__name__} should not be used with forward(), please explicitly call "
                             f"the methods of this module.")
