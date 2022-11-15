import torch
from torch import nn

from modeling.fpn import FeaturePyramidNetwork
from modeling.convnext import ConvNext
from modeling.head import Head


class FCOS(torch.nn.Module):

    def __init__(self,
                 in_channels=[192, 384, 768],
                 out_channels=192,
                 num_classes=19,
                 backbone_layer_dims=[96, 192, 384, 768],
                 backbone_depths=[3, 9, 3, 3]):
        super(FCOS, self).__init__()

        backbone = ConvNext(num_channels=3,
                            patch_size=4,
                            layer_dims=backbone_layer_dims,
                            depths=backbone_depths,
                            drop_rate=0.0)

        fpn = FeaturePyramidNetwork(in_channels_list=in_channels,
                                    out_channels=out_channels)
        self.feature_extractor = nn.Sequential(backbone, fpn)
        self.head = Head(out_channels, num_classes)

    def forward(self, images):
        features = self.feature_extractor(images)
        box_cls, box_regression, centerness = self.head(features)
        return features, box_cls, box_regression, centerness
