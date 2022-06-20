from abc import ABC, abstractmethod
import torch

from diffusion_model_nemo.loss.sde_loss import SDEScoreFunctionLoss

CORRECTOR_REGISTRY = {}


def register_corrector(cls: 'Corrector', name=None):
    global CORRECTOR_REGISTRY

    if name is None:
        name = cls.__name__

    if name in CORRECTOR_REGISTRY:
        raise ValueError(f"Corrector {name} has already been registered !")

    CORRECTOR_REGISTRY[name] = cls


def get_corrector(name: str):
    global CORRECTOR_REGISTRY

    if name in CORRECTOR_REGISTRY:
        return CORRECTOR_REGISTRY[name]

    else:
        return None


class Corrector(ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abstractmethod
    def update_fn(self, x, t):
        """One update of the corrector.
        Args:
          x: A PyTorch tensor representing the current state
          t: A PyTorch tensor representing the current time step.
        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass

    @classmethod
    def register_corector(cls, name: str = None):
        if name is None:
            name = cls.__name__

        if get_corrector(name) is None:
            register_corrector(cls, name=name)


class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, t):
        return x, x


NoneCorrector.register_corector('none')
NoneCorrector.register_corector('null')
