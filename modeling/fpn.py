import torch
import torch.nn.functional as F
from torch import nn


def _group_norm(out_channels, groups=32, affine=True, epsilon=1e-5):
    return torch.nn.GroupNorm(groups, out_channels, epsilon, affine)


def _conv_block(in_channels, out_channels, kernel_size, stride=1, dilation=1):
    conv = nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=dilation * (kernel_size - 1) // 2,
                     dilation=dilation,
                     bias=False)
    return nn.Sequential(conv, _group_norm(out_channels), nn.ReLU(inplace=True))


class FeaturePyramidNetwork(nn.Module):

    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            inner_block_module = _conv_block(in_channels,
                                             out_channels,
                                             kernel_size=1,
                                             stride=1)
            layer_block_module = _conv_block(out_channels,
                                             out_channels,
                                             kernel_size=3,
                                             stride=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        self.top_blocks = LastLevelP6P7(out_channels, out_channels)

    def forward(self, x):
        last_inner = self.inner_blocks[-1](x[-1])
        results = [self.layer_blocks[-1](last_inner)]
        for i in range(len(x) - 2, -1, -1):
            inner_lateral = self.inner_blocks[i](x[i])
            inner_top_down = F.interpolate(last_inner,
                                           size=(int(inner_lateral.shape[-2]),
                                                 int(inner_lateral.shape[-1])),
                                           mode='nearest')
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[i](last_inner))

        last_results = self.top_blocks(x[-1], results[-1])
        results.extend(last_results)

        return tuple(results)


class LastLevelP6P7(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]
