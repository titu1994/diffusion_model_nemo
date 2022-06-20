import torch
import numpy as np

import functools
from typing import List, Optional

from diffusion_model_nemo.modules import sde_lib
from diffusion_model_nemo.modules import ReverseDiffusionPredictor
from diffusion_model_nemo.loss import SDEScoreFunctionLoss

from scipy import integrate


class ProbabilityFlowSampler(torch.nn.Module):
    def __init__(
        self,
        method: str = 'RK45',
        rtol: float = 1e-5,
        atol: float = 1e-5,
        denoise: bool = False,
        eps: float = None,
    ):
        """

        Args:
            method: A `str`. The algorithm used for the black-box ODE solver.
                See the documentation of `scipy.integrate.solve_ivp`.
            rtol: A `float` number. The relative tolerance level of the ODE solver.
            atol: A `float` number. The absolute tolerance level of the ODE solver.
            denoise: If `True`, add one-step denoising to the final samples.
            eps: A `float` number or None. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
                If None, will use the default value set by the SDE class.
        """
        super().__init__()
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.denoise = denoise
        self.eps = eps

        self.sde = None  # type: Optional[sde_lib.SDE]

    def update_sde(self, sde: sde_lib.SDE):
        self.sde = sde  # type: sde_lib.SDE

    def forward(
        self, model: torch.nn.Module, shape: List[int], device: torch.device, noise: Optional[torch.Tensor] = None
    ) -> (torch.Tensor, int):
        """
        The probability flow ODE sampler with black-box ODE solver.
        Args:
            model: A score model.
            device: torch device.
            z: If present, generate samples from latent code `z`.
        Returns:
            samples, number of function evaluations.
        """
        if self.sde is None:
            raise ValueError(f"Must explicitly set `update_sde(sde)` for this module prior to calling forward()")

        if self.eps is None:
            eps = self.sde.sampling_epsilon
        else:
            eps = self.eps

        with torch.inference_mode():
            # Initial sample
            if noise is None:
                # If not represent, sample the latent code from the prior distibution of the SDE.
                x = self.sde.prior_sampling(shape).to(device)
            else:
                x = noise

            def ode_func(t, x: np.ndarray):
                x = torch.from_numpy(x.reshape(shape)).to(device=device, dtype=torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = self.drift_fn(model, self.sde, x, vec_t)
                return drift.detach().cpu().numpy().reshape((-1,))

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(ode_func, (self.sde.T, self.eps), x.detach().cpu().numpy().reshape((-1,)),
                                           rtol=self.rtol, atol=self.atol, method=self.method)

            nfe = solution.nfe
            x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device=device, dtype=torch.float32)

        # Denoising is equivalent to running one predictor step without adding noise
        if self.denoise:
            x = self.denoise_update_fn(model, self.sde, x, eps)

        return x, nfe

    def sample(self, model: torch.nn.Module, shape: List[int], device: torch.device) -> (torch.Tensor, int):
        return self.forward(model=model, shape=shape, device=device)

    @staticmethod
    def denoise_update_fn(model: torch.nn.Module, sde: sde_lib.SDE, x: torch.Tensor, eps: float):
        score_fn = SDEScoreFunctionLoss.resolve_score_function(model, sde, continuous=True)
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor_obj.update_fn(x, vec_eps)
        return x

    @staticmethod
    def drift_fn(model: torch.nn.Module, sde: sde_lib.SDE, x: torch.Tensor, t: torch.Tensor):
        """Get the drift function of the reverse-time SDE."""
        score_fn = SDEScoreFunctionLoss.resolve_score_function(model, sde, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)
        drift, diffusion = rsde.sde(x, t)
        return drift
