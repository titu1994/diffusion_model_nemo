import torch
import torch.nn as nn
from functools import partial
from typing import Optional, Dict

from nemo.core import NeuralModule, typecheck
from nemo.core.neural_types import NeuralType

from diffusion_model_nemo.parts import convnext, mha, positional_encoding
from diffusion_model_nemo import utils


class Unet(NeuralModule):
    def __init__(
            self,
            input_dim,
            dim=None,
            out_dim=None,
            dim_mults=None,
            channels=3,
            with_time_emb=True,
            resnet_block_groups=8,
            use_convnext=True,
            convnext_mult=2,
            learned_variance: bool = False
    ):
        super().__init__()

        if dim_mults is None:
            dim_mults = (1, 2, 4, 8)

        # determine dimensions
        self.channels = channels
        self.learned_variance = learned_variance

        init_dim = utils.default(dim, dim // 3 * 2)
        self.dim = init_dim
        self.init_conv = nn.Conv2d(channels, init_dim, kernel_size=7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if use_convnext:
            block_klass = partial(convnext.ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(convnext.ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                positional_encoding.SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        utils.Residual(utils.PreNorm(dim_out, mha.LinearAttention(dim_out))),
                        utils.Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = utils.Residual(utils.PreNorm(mid_dim, mha.Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        utils.Residual(utils.PreNorm(dim_in, mha.LinearAttention(dim_in))),
                        utils.Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        default_out_dim = channels * (1 if not learned_variance else 2)
        out_dim = utils.default(out_dim, default_out_dim)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, kernel_size=1)
        )

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    @typecheck()
    def forward(self, x, time):
        x = self.init_conv(x)

        t = self.time_mlp(time) if utils.exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)
