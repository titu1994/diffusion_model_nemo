from abc import ABC, abstractmethod
import torch

from diffusion_model_nemo.modules import sde_lib
from diffusion_model_nemo.loss.sde_loss import SDEScoreFunctionLoss

PREDICTOR_REGISTRY = {}


def register_predictor(cls: 'Predictor', name=None):
    global PREDICTOR_REGISTRY

    if name is None:
        name = cls.__name__

    if name in PREDICTOR_REGISTRY:
        raise ValueError(f"Predictor {name} has already been registered !")

    PREDICTOR_REGISTRY[name] = cls


def get_predictor(name: str):
    global PREDICTOR_REGISTRY

    if name in PREDICTOR_REGISTRY:
        return PREDICTOR_REGISTRY[name]

    else:
        return None


class Predictor(ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abstractmethod
    def update_fn(self, x, t):
        """One update of the predictor.
        Args:
          x: A PyTorch tensor representing the current state
          t: A Pytorch tensor representing the current time step.
        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass

    @classmethod
    def register_predictor(cls, name=None):
        if name is None:
            name = cls.__name__

        if get_predictor(name) is None:
            register_predictor(cls, name=name)


class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t):
        return x, x


NonePredictor.register_predictor('none')
NonePredictor.register_predictor('null')
