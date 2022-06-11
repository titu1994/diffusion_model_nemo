import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict

from nemo.core import Loss, typecheck
from nemo.core.neural_types import NeuralType, LossType

from diffusion_model_nemo import utils


class VariationalBoundLoss(Loss):
    def __init__(self, weight: 0.001, reduction='mean'):
        self._reduction = reduction
        if reduction == 'batch_mean':
            reduction = 'none'

        super().__init__(reduction=reduction)
        self.loss_weight = weight

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"loss": NeuralType(None, LossType())}

    @typecheck()
    def forward(
        self,
        samples: torch.Tensor,
        model_mean: torch.Tensor,
        model_log_variance: torch.Tensor,
        true_mean: torch.Tensor,
        true_log_variance_clipped: torch.Tensor,
        t: torch.Tensor,
    ):
        # Model output must contain both the predictions of predicted noise + learned variance

        # kl loss with detached model predicted mean, for stability reasons as in paper
        detached_model_mean = model_mean.detach()

        kl = utils.normal_kl(true_mean, true_log_variance_clipped, detached_model_mean, model_log_variance)
        kl = utils.mean_flattened(kl) * (1.0 / math.log(2.0))

        decoder_nll = -utils.discretized_gaussian_log_likelihood(
            samples, means=detached_model_mean, log_scales=0.5 * model_log_variance
        )
        decoder_nll = utils.mean_flattened(decoder_nll) * (1.0 / math.log(2.0))

        # at the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        vb_losses = torch.where(t == 0, decoder_nll, kl)

        vb_losses = self.loss_weight * vb_losses

        if self._reduction == 'batch_mean':
            vb_losses = vb_losses.view(samples.size(0), -1).sum(-1)
            vb_losses = vb_losses.mean()
        elif self.reduction == 'mean':
            vb_losses = vb_losses.mean()
        elif self.reduction == 'sum':
            vb_losses = vb_losses.sum()

        return vb_losses
