import torch
import torch.nn as nn

from diffusion_model_nemo import utils
from einops import rearrange


LINEAR_SCALE = 5000


class PositionalEncoding(nn.Module):
    def __init__(self, n_channels):
        super(PositionalEncoding, self).__init__()
        self.n_channels = n_channels

    def forward(self, noise_level):
        if len(noise_level.shape) > 1:
            noise_level = noise_level.squeeze(-1)
        half_dim = self.n_channels // 2
        exponents = torch.arange(half_dim, dtype=torch.float32).to(noise_level) / float(half_dim)
        exponents = 1e-4 ** exponents
        exponents = LINEAR_SCALE * noise_level.unsqueeze(1) * exponents.unsqueeze(0)
        out = torch.cat([exponents.sin(), exponents.cos()], dim=-1)
        out = out.transpose(1, 3)  # [B, H, W, C] -> [B, C, H, W]
        return out


class FeatureWiseLinearModulation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureWiseLinearModulation, self).__init__()
        self.signal_conv = torch.nn.Sequential(*[
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.LeakyReLU(0.2)
        ])
        self.positional_encoding = PositionalEncoding(in_channels)
        self.scale_conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.shift_conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x, noise_level):
        outputs = self.signal_conv(x)
        outputs = outputs + self.positional_encoding(noise_level)
        scale, shift = self.scale_conv(outputs), self.shift_conv(outputs)
        return scale, shift


# class FeatureWiseAffine(nn.Module):
#     def __init__(self):
#         super(FeatureWiseAffine, self).__init__()
#
#     def forward(self, x, scale, shift):
#         outputs = scale * x + shift
#         return outputs
