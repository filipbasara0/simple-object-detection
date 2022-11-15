import torch
import torch.nn.functional as F
from torch import nn


class Scale(nn.Module):

    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class Head(nn.Module):

    def __init__(self, in_channels, n_classes, n_share_convs=4):
        super().__init__()

        tower = []
        for _ in range(n_share_convs):
            tower.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=True))
            tower.append(nn.GroupNorm(32, in_channels))
            tower.append(nn.ReLU())
        self.shared_layers = nn.Sequential(*tower)

        self.cls_logits = nn.Conv2d(in_channels,
                                    n_classes,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        self.bbox_pred = nn.Conv2d(in_channels,
                                   4,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.ctrness = nn.Conv2d(in_channels,
                                 1,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        cls_logits = []
        bbox_preds = []
        cness_preds = []
        for l, features in enumerate(x):
            features = self.shared_layers(features)

            cls_logits.append(self.cls_logits(features))
            cness_preds.append(self.ctrness(features))
            reg = self.bbox_pred(features)
            reg = self.scales[l](reg)
            bbox_preds.append(F.relu(reg))
        return cls_logits, bbox_preds, cness_preds
