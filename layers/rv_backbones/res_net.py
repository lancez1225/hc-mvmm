"""ResNet-based range-view backbone.

The backbone consumes a front-cropped range image stacked from
``(x, y, z, intensity, r, g, b)`` projections and returns per-point
range-view features (and the point-level appearance confidence from L1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleNeck(nn.Module):
    """Standard ResNet bottleneck block (1x1 - 3x3 - 1x1)."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class Encoder(nn.Module):
    """Standard ResNet-50 style encoder over a front-view range image."""

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.num_blocks = [3, 4, 6, 3]
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layers = nn.ModuleList()
        self.src_channels = []
        self._make_layers(BottleNeck, 64, self.num_blocks[0], stride=1)
        self._make_layers(BottleNeck, 128, self.num_blocks[1], stride=2)
        self._make_layers(BottleNeck, 256, self.num_blocks[2], stride=2)
        self._make_layers(BottleNeck, 512, self.num_blocks[3], stride=2)

    def _make_layers(self, block, planes, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
        self.layers.append(nn.Sequential(*layers))
        self.src_channels.append(planes * block.expansion)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        out = []
        for layer in self.layers:
            x = layer(x)
            out.append(x)
        return out


class Decoder(nn.Module):
    """Top-down FPN-like decoder fusing multi-scale features back to stride 2."""

    def __init__(self, src_channels):
        super().__init__()
        self.out_channels = 64
        self.lat_layers = nn.ModuleList(
            nn.Conv2d(c, self.out_channels, kernel_size=1)
            for c in reversed(src_channels)
        )

    def forward(self, src_features):
        x = self.lat_layers[0](src_features[-1])
        num = len(src_features)
        for i in range(1, num):
            lat_features = self.lat_layers[i](src_features[-i - 1])
            x = F.interpolate(x, scale_factor=2, mode='bilinear') + lat_features
        return F.interpolate(x, scale_factor=2, mode='bilinear')


class ResNet(nn.Module):
    """Range-view backbone that yields per-point RV features.

    The forward pass also surfaces ``point_confidence``, the per-point
    appearance confidence from L1, since downstream modules (the VFE and
    BEV backbone) consume it together with the RV features.
    """

    def __init__(self, cfg, range_convertor):
        """Initialises the encoder/decoder and the final 1x1 projection.

        Args:
            cfg: ``rv_backbone`` sub-dictionary of the YAML config. Only
                ``filters`` is read; its single entry is the output channel
                count of the RV features handed to L2.
            range_convertor: A :class:`utils.range_image_utils.RangeConvertor`
                used to scatter and gather per-point features.
        """
        super().__init__()
        # Seven input channels: normalised xyz, intensity and RGB.
        self.in_channels = 7
        self.range_convertor = range_convertor

        self.encoder = Encoder(self.in_channels)
        self.decoder = Decoder(self.encoder.src_channels)

        assert len(cfg['filters']) == 1
        self.num_rv_features = cfg['filters'][-1]
        self.conv_1x1 = nn.Conv2d(
            self.decoder.out_channels, self.num_rv_features, kernel_size=1
        )

    def forward(self, batch_dict):
        """Forward pass.

        Args:
            batch_dict: Must contain ``colored_points`` of shape
                ``[N_total, 9]`` (``batch_id, x, y, z, i, r, g, b, conf``),
                ``range_image`` of shape ``[B, 7, H, W]`` and ``batch_size``.

        Returns:
            ``batch_dict`` populated with:
              * ``rv_features``: ``[N_total, num_rv_features]``
              * ``point_confidence``: ``[N_total, 1]`` (passed through from L1)
        """
        batch_points = batch_dict['colored_points']
        batch_range_image = batch_dict['range_image']
        batch_size = batch_dict['batch_size']

        decoder_out = self.decoder(self.encoder(batch_range_image))
        x = self.conv_1x1(decoder_out)

        batch_rv_features = []
        batch_confidence = []
        for batch_idx in range(batch_size):
            mask = batch_points[:, 0] == batch_idx
            points = batch_points[mask]
            range_image = x[batch_idx, ...]
            range_features = self.range_convertor.get_range_features(
                points[:, 1:4], range_image
            )
            batch_rv_features.append(range_features)
            batch_confidence.append(points[:, 8:9])

        batch_dict['rv_features'] = torch.cat(batch_rv_features, dim=0)
        batch_dict['point_confidence'] = torch.cat(batch_confidence, dim=0)
        return batch_dict
