"""Base BEV backbone: multi-scale CNN encoder with deconvolutional fusion."""

import torch
import torch.nn as nn


class BaseBEVBackbone(nn.Module):
    """Two-branch CNN encoder over the BEV feature map.

    The architecture mirrors the OpenPCDet ``BaseBEVBackbone`` used by the
    MVMM baseline. Each downsample branch is followed by a transposed
    convolution back to a common resolution; the upsampled features are
    concatenated along the channel dim before being passed to the head.
    """

    def __init__(self, cfg, in_channels):
        """Builds the down/up sample blocks declared in ``cfg``.

        Args:
            cfg: ``bev_backbone`` sub-dictionary of the YAML config; must
                define ``num_layers``, ``downsample_strides``,
                ``downsample_filters``, ``upsample_strides`` and
                ``upsample_filters``.
            in_channels: Input channel count of the BEV tensor produced by L2.
        """
        super().__init__()
        self.in_channels = in_channels

        self.num_layers = cfg['num_layers']
        self.downsample_strides = cfg['downsample_strides']
        self.downsample_filters = cfg['downsample_filters']
        self.upsample_strides = cfg['upsample_strides']
        self.upsample_filters = cfg['upsample_filters']

        num_levels = len(self.num_layers)
        c_in_list = [self.in_channels, *self.downsample_filters[:-1]]

        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        for idx in range(num_levels):
            cur_layers = [
                nn.Conv2d(c_in_list[idx], self.downsample_filters[idx],
                          kernel_size=3, stride=self.downsample_strides[idx],
                          padding=1, bias=False),
                nn.BatchNorm2d(self.downsample_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]
            for _ in range(self.num_layers[idx]):
                cur_layers.extend([
                    nn.Conv2d(self.downsample_filters[idx],
                              self.downsample_filters[idx],
                              kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(self.downsample_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                ])
            self.blocks.append(nn.Sequential(*cur_layers))

            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(
                    self.downsample_filters[idx],
                    self.upsample_filters[idx],
                    kernel_size=self.upsample_strides[idx],
                    stride=self.upsample_strides[idx],
                    bias=False,
                ),
                nn.BatchNorm2d(self.upsample_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = sum(self.upsample_filters)

    def forward(self, batch_dict):
        """Reads ``pv_features``, writes ``bev_features`` to ``batch_dict``."""
        x = batch_dict['pv_features']
        batch_bev_features = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            batch_bev_features.append(self.deblocks[i](x))
        batch_dict['bev_features'] = torch.cat(batch_bev_features, dim=1)
        return batch_dict
