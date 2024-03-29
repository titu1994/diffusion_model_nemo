import torch
import torch.nn as nn

from diffusion_model_nemo import utils
from einops import rearrange


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, dropout=None, order='bn_act_conv'):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

        if dropout is None:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(dropout)

        valid_orders = ['conv_bn_act', 'bn_act_conv']
        if order not in valid_orders:
            raise ValueError(f"Valid ordering for block are : {valid_orders}")
        self.order = order

    def forward(self, x, scale_shift=None):
        if self.order == 'conv_bn_act':
            return self.forward_conv_bn_relu(x, scale_shift=scale_shift)
        elif self.order == 'bn_act_conv':
            return self.forward_conv_bn_relu(x, scale_shift=scale_shift)
        else:
            raise ValueError(f"Wrong ordering provided : {self.order}")

    def forward_conv_bn_relu(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if utils.exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)

        if self.dropout is not None:
            x = self.dropout(x)
        return x

    def forward_bn_act_conv(self, x, scale_shift=None):
        x = self.norm(x)

        if utils.exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.proj(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8, order='bn_act_conv', dropout=None):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if utils.exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups, order=order)
        self.block2 = Block(dim_out, dim_out, groups=groups, order=order, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if utils.exists(self.mlp) and utils.exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)


class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True, dropout=None):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if utils.exists(time_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, kernel_size=3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if utils.exists(self.mlp) and utils.exists(time_emb):
            assert utils.exists(time_emb), "time embedding must be passed in"
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)

        if self.dropout is not None:
            h = self.dropout(h)

        return h + self.res_conv(x)
