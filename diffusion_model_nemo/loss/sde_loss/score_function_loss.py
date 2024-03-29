import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from nemo.core import Loss, typecheck
from nemo.core.neural_types import NeuralType, LossType

from diffusion_model_nemo.modules import sde_lib


class SDEScoreFunctionLoss(Loss):
    def __init__(self, continuous: bool = True, likelihood_weighting: bool = True, eps: float = 1e-5, reduction='mean'):
        """Create a loss function for training with arbirary SDEs.
        Args:
          continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
            ad-hoc interpolation to take continuous time steps.
          likelihood_weighting: If `True`, weight the mixture of score matching losses
            according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
          eps: A `float` number. The smallest time step to sample from.
          reduction: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
        Returns:
          A loss value.
        """
        self._reduction = reduction
        if reduction == 'batch_mean':
            reduction = 'none'
        super().__init__(reduction=reduction)

        self.continuous = continuous
        self.likelihood_weighting = likelihood_weighting
        self.eps = eps

        self.sde = None  # type: Optional[sde_lib.SDE]

    def update_sde(self, sde: sde_lib.SDE):
        self.sde = sde

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"loss": NeuralType(None, LossType())}

    @classmethod
    def resolve_score_function(cls, model: torch.nn.Module, sde: 'sde_lib.SDE', continuous: bool = True):
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):

            def score_fn(x, t):
                # Scale neural network output by standard deviation and flip sign
                if continuous or isinstance(sde, sde_lib.subVPSDE):
                    # For VP-trained models, t=0 corresponds to the lowest noise level
                    # The maximum value of time embedding is assumed to 999 for
                    # continuously-trained models.
                    labels = t * (sde.N - 1)  # t * 999
                    score = model(x, labels)
                    _, std = sde.marginal_prob(torch.zeros_like(x), t)
                else:
                    # For VP-trained models, t=0 corresponds to the lowest noise level
                    labels = t * (sde.N - 1)
                    score = model(x, labels)

                    if not hasattr(sde, 'sqrt_1m_alphas_cumprod'):
                        sqrt_1m_alphas_cumprod = torch.sqrt(1. - sde.alphas_cumprod)
                        sde.sqrt_1m_alphas_cumprod = sqrt_1m_alphas_cumprod

                    std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

                score = -score / std[:, None, None, None]
                return score

        elif isinstance(sde, sde_lib.VESDE):

            def score_fn(x, t):
                if continuous:
                    labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
                else:
                    # For VE-trained models, t=0 corresponds to the highest noise level
                    labels = sde.T - t
                    labels *= sde.N - 1
                    labels = torch.round(labels).long()

                score = model(x, labels)
                return score

        else:
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

        return score_fn

    @typecheck()
    def forward(self, model: torch.nn.Module, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        if self.sde is None:
            raise RuntimeWarning("Must set the SDE solver via `update_sde()` !")

        batch_size = x_start.size(0)
        t = t * (self.sde.T - self.eps) + self.eps
        z = noise

        if self._reduction == 'batch_mean':
            reduce_op = lambda x, *args, **kwargs: x.view(batch_size, -1).sum(-1)
        elif self.reduction == 'mean':
            reduce_op = lambda x, *args, **kwargs: torch.mean(x, *args, **kwargs)
        elif self.reduction == 'sum':
            reduce_op = lambda x, *args, **kwargs: 0.5 * torch.sum(x, *args, **kwargs)
        else:
            reduce_op = lambda x, *args, **kwargs: x

        score_fn = self.resolve_score_function(model, sde=self.sde, continuous=self.continuous)
        mean, std = self.sde.marginal_prob(x_start, t)
        perturbed_data = mean + std[:, None, None, None] * noise
        score = score_fn(perturbed_data, t)

        if not self.likelihood_weighting:
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = self.sde.sde(torch.zeros_like(x_start), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        loss = losses.mean()

        return loss
