import torch
import functools
from typing import List, Optional

from diffusion_model_nemo.modules import sde_lib
from diffusion_model_nemo.modules import Predictor, NonePredictor, Corrector, NoneCorrector
from diffusion_model_nemo.loss import SDEScoreFunctionLoss


class PredictorCorrectorSampler(torch.nn.Module):
    def __init__(
        self,
        predictor: str,
        corrector: str,
        snr: float,
        n_steps: int = 1,
        probability_flow: bool = False,
        continuous: bool = True,
        denoise: bool = True,
        eps: float = None,
    ):
        """

        Args:
            predictor: str or 'null' or 'none' or None. A str name of a Predictor object that has been registered.
            collector: str or 'null' or 'none' or None. A str name of a Collector object that has been registered.
            snr: A `float` number. The signal-to-noise ratio for configuring correctors.
            n_steps: An integer. The number of corrector steps per predictor update.
            probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
            continuous: `True` indicates that the score model was continuously trained.
            denoise: If `True`, add one-step denoising to the final samples.
            eps: A `float` number or None. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
                If None, will use the default value set by the SDE class.
        """
        super().__init__()
        self.predictor = predictor
        self.corrector = corrector
        self.snr = snr
        self.n_steps = n_steps
        self.probability_flow = probability_flow
        self.continuous = continuous
        self.denoise = denoise
        self.eps = eps

        self.sde = None  # type: Optional[sde_lib.SDE]

    def update_sde(self, sde: sde_lib.SDE):
        self.sde = sde  # type: sde_lib.SDE

    def forward(self, model: torch.nn.Module, shape: List[int], device: torch.device) -> (torch.Tensor, int):
        if self.sde is None:
            raise ValueError(f"Must explicitly set `update_sde(sde)` for this module prior to calling forward()")

        if self.eps is None:
            eps = self.sde.sampling_epsilon
        else:
            eps = self.eps

        # Create predictor & corrector update functions
        predictor_update_fn = functools.partial(self.prepare_predictor_fn,
                                                sde=self.sde,
                                                predictor=self.predictor,
                                                probability_flow=self.probability_flow,
                                                continuous=self.continuous)
        corrector_update_fn = functools.partial(self.prepare_corrector_fn,
                                                sde=self.sde,
                                                corrector=self.corrector,
                                                continuous=self.continuous,
                                                snr=self.snr,
                                                n_steps=self.n_steps)

        with torch.inference_mode():
            # Initial sample
            x = self.sde.prior_sampling(shape).to(device)
            timesteps = torch.linspace(self.sde.T, eps, self.sde.N, device=device)

            for i in range(self.sde.N):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                x, x_mean = corrector_update_fn(x, vec_t, model=model)
                x, x_mean = predictor_update_fn(x, vec_t, model=model)

        # denormalize image
        x = (x.cpu() + 1.) * 0.5  # [-1, 1] -> [0, 1]

        nfe = self.sde.N * (self.n_steps + 1)
        if self.denoise:
            return x_mean, nfe
        else:
            return x, nfe

    def sample(self, model: torch.nn.Module, shape: List[int], device: torch.device = None) -> (torch.Tensor, int):
        if device is None:
            device = next(model.parameters()).device

        return self.forward(model=model, shape=shape, device=device)

    @staticmethod
    def prepare_predictor_fn(
            x: torch.Tensor,
            t: torch.Tensor,
            model: torch.nn.Module,
            sde: 'sde_lib.SDE',
            predictor: 'Predictor',
            continuous: bool,
            probability_flow: bool,
    ):
        """A wrapper tha configures and returns the update function of correctors."""
        score_fn = SDEScoreFunctionLoss.resolve_score_function(model=model, sde=sde, continuous=continuous)

        if predictor is None:
            # Predictor-only sampler
            predictor_obj = NonePredictor(sde=sde, score_fn=score_fn, probability_flow=probability_flow)
        else:
            predictor_obj = predictor(sde=sde, score_fn=score_fn, probability_flow=probability_flow)

        return predictor_obj.update_fn(x, t)

    @staticmethod
    def prepare_corrector_fn(
            x: torch.Tensor,
            t: torch.Tensor,
            model: torch.nn.Module,
            sde: 'sde_lib.SDE',
            corrector: 'Corrector',
            continuous: bool,
            snr: float,
            n_steps: int,
    ):
        """A wrapper tha configures and returns the update function of correctors."""
        score_fn = SDEScoreFunctionLoss.resolve_score_function(model=model, sde=sde, continuous=continuous)

        if corrector is None:
            # Predictor-only sampler
            corrector_obj = NoneCorrector(sde=sde, score_fn=score_fn, snr=snr, n_steps=n_steps)
        else:
            corrector_obj = corrector(sde=sde, score_fn=score_fn, snr=snr, n_steps=n_steps)

        return corrector_obj.update_fn(x, t)
