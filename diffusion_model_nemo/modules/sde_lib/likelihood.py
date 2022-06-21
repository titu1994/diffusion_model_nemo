import torch
import numpy as np
from scipy import integrate

from diffusion_model_nemo.modules import sde_lib
from diffusion_model_nemo.loss import SDEScoreFunctionLoss


def get_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x, t, eps):
        with torch.enable_grad():
            x.requires_grad_(True)
            fn_eps = torch.sum(fn(x, t) * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
        x.requires_grad_(False)
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

    return div_fn


class LikelihoodEstimate:
    def __init__(
        self,
        hutchinson_type: str = 'rademacher',
        method: str = 'RK45',
        rtol: float = 1e-5,
        atol: float = 1e-5,
        eps: float = 1e-5,
    ):
        """Create a function to compute the unbiased log-likelihood estimate of a given data point.

        Args:
            hutchinson_type: "rademacher" or "gaussian". The type of noise for Hutchinson-Skilling trace estimator.
            method: A `str`. The algorithm for the black-box ODE solver.
                See documentation for `scipy.integrate.solve_ivp`.
            rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
            atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
            eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

        Returns:
            A function that a batch of data points and returns the log-likelihoods in bits/dim,
            the latent code, and the number of function evaluations cost by computation.
        """
        self.hutchinson_type = hutchinson_type.lower()
        if hutchinson_type not in ['rademacher', 'gaussian']:
            raise ValueError(f"`hutchinson_type` must be one of `rademacher` or `gaussian`")

        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.eps = eps

    def update_sde(self, sde: sde_lib.SDE):
        self.sde = sde  # type: sde_lib.SDE

    def likelihood(self, model: torch.nn.Module, data: torch.Tensor):
        with torch.no_grad():
            shape = data.shape
            if self.hutchinson_type == 'gaussian':
                epsilon = torch.randn_like(data)
            elif self.hutchinson_type == 'rademacher':
                epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.0
            else:
                raise NotImplementedError()

            def ode_func(t, x):
                sample = torch.from_numpy(x[: -shape[0]].reshape(shape)).to(device=data.device, dtype=torch.float32)
                vec_t = torch.ones(sample.shape[0], device=sample.device) * t
                drift = self.drift_fn(model, self.sde, x=sample, t=vec_t).detach().cpu().numpy().reshape((-1,))
                logp_grad = (
                    self.divergence_fn(model, self.sde, x=sample, t=vec_t, noise=epsilon)
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape((-1,))
                )
                return np.concatenate([drift, logp_grad], axis=0)

            init = np.concatenate([data.detach().cpu().numpy().reshape((-1,)), np.zeros((shape[0],))], axis=0)
            solution = integrate.solve_ivp(
                ode_func, (self.eps, self.sde.T), init, rtol=self.rtol, atol=self.atol, method=self.method
            )
            nfe = solution.nfev
            zp = solution.y[:, -1]
            z = torch.from_numpy(zp[: -shape[0]].reshape(shape)).to(device=data.device, dtype=torch.float32)
            delta_logp = torch.from_numpy(zp[-shape[0] :].reshape((shape[0],))).to(
                device=data.device, dtype=torch.float32
            )
            prior_logp = self.sde.prior_logp(z)
            bpd = -(prior_logp + delta_logp) / np.log(2)
            N = np.prod(shape[1:])
            bpd = bpd / N
            # A hack to convert log-likelihoods to bits/dim
            # The 7 here comes from ln(128) / ln(2). 128 is because data is scaled down to range [-1, 1] from original [0, 256].
            # If data was scaled to range [0, 1], then inverse_scale(-1) = -1, and offset = 8. This is equivalent to ln(256) / ln(2).
            # See https://stats.stackexchange.com/questions/423120/what-is-bits-per-dimension-bits-dim-exactly-in-pixel-cnn-papers for
            # further details about why this needs to be done.
            offset = 7.0  # inverse_scaler(-1.0) := (-1 + 1) / 2. = 0
            bpd = bpd + offset
            return bpd, z, nfe

    @staticmethod
    def drift_fn(model: torch.nn.Module, sde: sde_lib.SDE, x: torch.Tensor, t: torch.Tensor):
        """The drift function of the reverse-time SDE."""
        score_fn = SDEScoreFunctionLoss.resolve_score_function(model, sde, continuous=True)
        # Probability flow ODE is a special case of Reverse SDE
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    @staticmethod
    def divergence_fn(model: torch.nn.Module, sde: sde_lib.SDE, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""
        fn = get_div_fn(lambda xx, tt: LikelihoodEstimate.drift_fn(model, sde, x=xx, t=tt))
        return fn(x, t, noise)
