import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from nemo.core import Loss, typecheck
from nemo.core.neural_types import NeuralType, LossType

from diffusion_model_nemo.modules import diffusion_process


class DiffusionLoss(Loss):
    def __init__(self, loss_type: str, reduction='mean'):
        self._reduction = reduction
        if reduction == 'batch_mean':
            reduction = 'none'

        super(DiffusionLoss, self).__init__(reduction=reduction)
        assert loss_type in ['l1', 'l2', 'huber'], f"Loss type {loss_type} is not implemented !"
        self.loss_type = loss_type

        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        elif loss_type == "huber":
            self.loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError()

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"loss": NeuralType(None, LossType())}

    @typecheck()
    def forward(self, input, target):
        loss = self.loss_fn(input, target, reduction=self.reduction)

        if self._reduction == 'batch_mean':
            loss = loss.view(input.size(0), -1).sum(-1)
            loss = loss.mean()

        return loss


