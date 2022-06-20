import torch
import numpy as np

from diffusion_model_nemo.modules.sde_predictors import Predictor


class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        f, G = self.rsde.discretize(x, t)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean


ReverseDiffusionPredictor.register_predictor('reverse_diffusion')
