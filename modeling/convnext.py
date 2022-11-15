import torch
from torch import nn
import torch.nn.functional as F

from torchvision.ops import stochastic_depth


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight,
                                self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Permute(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)


class ConvNextBlock(nn.Module):

    def __init__(self, filter_dim, kernel_size=7,
                 m=4, layer_scale=1e-6):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(*[
            nn.Conv2d(filter_dim,
                      filter_dim,
                      kernel_size=kernel_size,
                      padding=padding,
                      groups=filter_dim),
            Permute([0, 2, 3, 1]),
            LayerNorm(filter_dim, eps=1e-6),
            nn.Linear(filter_dim, filter_dim * m),
            nn.GELU(),
            nn.Linear(filter_dim * m, filter_dim),
            Permute([0, 3, 1, 2])
        ])
        self.gamma = nn.Parameter(torch.ones(filter_dim, 1, 1) * layer_scale)

    def forward(self, x):
        return self.block(x) * self.gamma


class ConvNextLayer(nn.Module):

    def __init__(self, filter_dim, depth, drop_rates):
        super().__init__()
        self.blocks = nn.ModuleList([])

        for _ in range(depth):
            self.blocks.append(ConvNextBlock(filter_dim=filter_dim))

        self.drop_rates = drop_rates

    def forward(self, x):
        for idx, block in enumerate(self.blocks):
            x = x + stochastic_depth(block(x),
                                     self.drop_rates[idx],
                                     mode="batch",
                                     training=self.training)
        return x


class ConvNext(nn.Module):

    def __init__(self,
                 num_channels=3,
                 patch_size=4,
                 layer_dims=[96, 192, 384, 768],
                 depths=[3, 3, 9, 3],
                 drop_rate=0.):
        super().__init__()

        # init downsample layers with stem
        self.downsample_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(num_channels, layer_dims[0], kernel_size=patch_size, stride=patch_size),
                LayerNorm(layer_dims[0],
                              eps=1e-6,
                              data_format="channels_first")
            )])
        for idx in range(len(layer_dims) - 1):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(layer_dims[idx],
                              eps=1e-6,
                              data_format="channels_first"),
                    nn.Conv2d(layer_dims[idx],
                              layer_dims[idx + 1],
                              kernel_size=2,
                              stride=2),
                ))

        drop_rates=[x.item() for x in torch.linspace(0, drop_rate, sum(depths))] 
        self.stage_layers = nn.ModuleList([])
        for idx, layer_dim in enumerate(layer_dims):
            layer_dr = drop_rates[sum(depths[:idx]): sum(depths[:idx]) + depths[idx]]
            self.stage_layers.append(
                ConvNextLayer(filter_dim=layer_dim, depth=depths[idx], drop_rates=layer_dr))


    def forward(self, x):
        outputs = []
        all_layers = list(zip(self.downsample_layers, self.stage_layers))
        for downsample_layer, stage_layer in all_layers:
            x = downsample_layer(x)
            x = stage_layer(x)
            outputs.append(x)
        # we want only last three feature maps (C3, C4, C5)
        return outputs[1:]
