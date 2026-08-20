"""Voxel feature extractor (L2 of HC-MVMM).

Performs three jobs:

  1. Generates a sparse-3D voxel feature tensor by mean-pooling per-point
     features (xyz, intensity, range-view feature, and optionally the
     per-point appearance confidence).
  2. Computes a per-voxel reliability score ``r_v`` (Eqs. 13-17 of the
     paper) as ``r_v = clip(r_cnt * r_pt * r_con, r_min, 1)``. The point-
     level term ``r_pt`` supports two formulations -- see ``pt_conf_method``.
  3. Builds a BEV-resolution confidence map ``r_bev`` by max-pooling
     ``r_v`` along the z-axis and applying stride-aware average pooling.

The output ``pv_features`` (``[B, C, H, W]``) and ``bev_confidence``
(``[B, 1, H', W']``) feed the BEV backbone (L3).
"""

import numpy as np
import spconv.pytorch as spconv
import torch
import torch.nn as nn
from spconv.pytorch.utils import PointToVoxel


def _spconv_block(in_channels, out_channels, kernel_size=1, stride=1,
                  padding=0, indice_key=None, conv_type='subm'):
    """Returns a Conv3d -> BN -> ReLU sparse-conv stack."""
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(
            in_channels, out_channels, kernel_size=kernel_size,
            padding=padding, bias=False, indice_key=indice_key,
        )
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False, indice_key=indice_key,
        )
    else:
        raise NotImplementedError(f'Unknown sparse conv type: {conv_type}')
    return spconv.SparseSequential(
        conv,
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


class VFE(nn.Module):
    """Voxel feature extractor with the L2 hierarchical-confidence path.

    Configuration entries (``pv_bridge`` block of the YAML):

      * ``voxel_size``: ``(vx, vy, vz)``.
      * ``filters``: Six-element list of channel widths for the sparse 3-D conv stack.
      * ``use_hierarchical_conf``: Enable the L2 confidence pipeline (default True).
      * ``pt_conf_method``: How to aggregate the point-level confidence into
        ``r_pt_v``. Two values are accepted:

          - ``'proxy_avg'`` (default, HC-MVMM full): a single scalar equal to
            the mean of all per-point confidences in the current frame
            (i.e. the whole colored-points proxy); the same scalar is used
            for every voxel in that frame.
          - ``'per_voxel'``: original Eq. 13 of the paper; the mean is taken
            inside each voxel separately.

      * ``append_hierarchical_conf_to_voxel_features``: When True (default),
        ``r_v`` is concatenated as an extra feature channel before sparse
        3-D encoding (Eq. 18).
    """

    def __init__(self, cfg, in_channels, point_cloud_range):
        """Builds the sparse 3-D encoder and reads the L2 configuration."""
        super().__init__()

        self.voxel_size = cfg['voxel_size']
        self.point_cloud_range = point_cloud_range
        self.grid_size = np.round(
            (self.point_cloud_range[3:6] - self.point_cloud_range[0:3])
            / np.array(self.voxel_size)
        ).astype(np.int64)
        self.spatial_shape = self.grid_size[::-1] + [1, 0, 0]

        # The four base channels are (x, y, z, intensity).
        self.base_channels = 4
        self.in_channels = in_channels

        self.use_hierarchical_conf = cfg.get('use_hierarchical_conf', False)
        self.append_hierarchical_conf_to_voxel_features = cfg.get(
            'append_hierarchical_conf_to_voxel_features', True
        )
        # 'proxy_avg' (default) or 'per_voxel'. See class docstring.
        self.pt_conf_method = cfg.get('pt_conf_method', 'proxy_avg')
        assert self.pt_conf_method in ('proxy_avg', 'per_voxel'), (
            f"pt_conf_method must be 'proxy_avg' or 'per_voxel', "
            f"got {self.pt_conf_method!r}"
        )

        assert len(cfg['filters']) == 6
        filters = cfg['filters']

        # Channel count entering ``conv0``.
        conv0_in_channels = self.base_channels + self.in_channels
        if self.use_hierarchical_conf and self.append_hierarchical_conf_to_voxel_features:
            conv0_in_channels += 1

        self.conv0 = spconv.SparseSequential(
            _spconv_block(conv0_in_channels, filters[0],
                          kernel_size=3, padding=1, indice_key='subm0'),
        )
        self.conv1 = spconv.SparseSequential(
            _spconv_block(filters[0], filters[1],
                          kernel_size=3, padding=1, indice_key='subm1'),
        )
        self.conv2 = spconv.SparseSequential(
            _spconv_block(filters[1], filters[2], kernel_size=3, stride=2,
                          padding=1, indice_key='spconv2', conv_type='spconv'),
            _spconv_block(filters[2], filters[2],
                          kernel_size=3, padding=1, indice_key='subm2'),
            _spconv_block(filters[2], filters[2],
                          kernel_size=3, padding=1, indice_key='subm2'),
        )
        self.conv3 = spconv.SparseSequential(
            _spconv_block(filters[2], filters[3], kernel_size=3, stride=2,
                          padding=1, indice_key='spconv3', conv_type='spconv'),
            _spconv_block(filters[3], filters[3],
                          kernel_size=3, padding=1, indice_key='subm3'),
            _spconv_block(filters[3], filters[3],
                          kernel_size=3, padding=1, indice_key='subm3'),
        )
        self.conv4 = spconv.SparseSequential(
            _spconv_block(filters[3], filters[4], kernel_size=3, stride=2,
                          padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            _spconv_block(filters[4], filters[4],
                          kernel_size=3, padding=1, indice_key='subm4'),
            _spconv_block(filters[4], filters[4],
                          kernel_size=3, padding=1, indice_key='subm4'),
        )
        self.conv5 = spconv.SparseSequential(
            _spconv_block(filters[4], filters[5], kernel_size=(3, 1, 1),
                          stride=(2, 1, 1), padding=0, indice_key='spconv5',
                          conv_type='spconv'),
        )

        # After collapsing the z-axis the BEV channel count is ``filters[5] * 2``.
        self.num_pv_features = filters[5] * 2

    # ------------------------------------------------------------------ #
    #  Forward pass                                                      #
    # ------------------------------------------------------------------ #

    def forward(self, batch_dict):
        """Runs voxelisation, L2 confidence aggregation and sparse 3-D conv."""
        batch_points = batch_dict['colored_points'][:, 0:5]
        batch_size = batch_dict['batch_size']
        device = batch_points.device

        # Concatenate range-view features per point if available.
        if self.in_channels > 0:
            batch_points = torch.cat(
                [batch_points, batch_dict['rv_features']], dim=-1
            )

        # When hierarchical confidence is enabled we also append the per-point
        # confidence (will be averaged inside each voxel below).
        batch_conf = None
        if self.use_hierarchical_conf:
            batch_conf = batch_dict['colored_points'][:, 8:9]
            batch_points = torch.cat([batch_points, batch_conf], dim=-1)
        num_feat = (
            self.base_channels + self.in_channels
            + (1 if self.use_hierarchical_conf else 0)
        )

        voxel_generator = PointToVoxel(
            vsize_xyz=self.voxel_size,
            coors_range_xyz=self.point_cloud_range,
            num_point_features=num_feat,
            max_num_voxels=16000 if self.training else 40000,
            max_num_points_per_voxel=5,
            device=device,
        )

        batch_voxels = []
        batch_coords = []
        batch_voxel_confs = [] if self.use_hierarchical_conf else None

        for batch_idx in range(batch_size):
            mask = batch_points[:, 0] == batch_idx
            points = batch_points[mask]
            voxels, coords, num_points_per_voxel = voxel_generator(
                points[:, 1:].contiguous()
            )

            # Mean pooling over the points that fell inside each voxel.
            feat_voxels = voxels[:, :, :-1] if self.use_hierarchical_conf else voxels
            voxel_features = feat_voxels.sum(dim=1, keepdim=False)
            normalizer = torch.clamp_min(
                num_points_per_voxel.view(-1, 1), min=1.0
            ).type_as(voxel_features)
            voxel_features = voxel_features / normalizer

            # L2: hierarchical voxel confidence (Eqs. 13-17).
            if self.use_hierarchical_conf:
                point_mask_hc = (
                    num_points_per_voxel.view(-1, 1, 1)
                    > torch.arange(voxels.shape[1], device=device).view(1, -1, 1)
                ).float()

                # r_cnt: density confidence in Eq. 14.
                count_conf = torch.sigmoid(
                    torch.log(num_points_per_voxel.float().clamp(min=1.0))
                    - np.log(3.0)
                ).unsqueeze(1)

                # r_pt: point-level appearance confidence aggregator. Either
                # the paper's per-voxel mean (Eq. 13) or the proxy-wide mean.
                if batch_conf is None:
                    avg_color_conf = torch.full_like(count_conf, 0.5)
                elif self.pt_conf_method == 'proxy_avg':
                    # Single scalar for the whole frame's coloured points.
                    frame_conf = batch_dict['colored_points'][mask, 8]
                    if frame_conf.numel() == 0:
                        avg_color_conf = torch.full_like(count_conf, 0.5)
                    else:
                        scalar = frame_conf.mean()
                        avg_color_conf = scalar.expand_as(count_conf).clone()
                else:
                    conf_voxels = voxels[:, :, -1:] * point_mask_hc
                    avg_color_conf = (
                        conf_voxels.sum(dim=1)
                        / num_points_per_voxel.view(-1, 1).clamp(min=1).float()
                    )
                avg_color_conf = avg_color_conf.clamp(min=0.1, max=1.0)

                # r_con: intra-voxel feature consistency (Eqs. 15-16). Only the
                # base xyz/intensity channels feed the variance to avoid
                # scale mismatch with the learned RV features.
                feat_for_var = voxels[:, :, :self.base_channels] * point_mask_hc
                feat_mean = (
                    feat_for_var.sum(dim=1)
                    / num_points_per_voxel.view(-1, 1).clamp(min=1).float()
                )
                feat_diff = ((feat_for_var - feat_mean.unsqueeze(1)) ** 2) * point_mask_hc
                feat_var = (
                    feat_diff.sum(dim=1).mean(dim=1, keepdim=True)
                    / num_points_per_voxel.view(-1, 1).clamp(min=1).float()
                )
                consistency_conf = torch.exp(-feat_var * 10.0)

                voxel_conf = count_conf * avg_color_conf * consistency_conf
                voxel_conf = voxel_conf.clamp(0.01, 1.0)
                batch_voxel_confs.append(voxel_conf)
                if self.append_hierarchical_conf_to_voxel_features:
                    voxel_features = torch.cat([voxel_features, voxel_conf], dim=1)

            batch_voxels.append(voxel_features)
            coords = torch.cat(
                [torch.ones((coords.shape[0], 1), dtype=coords.dtype,
                            device=device) * batch_idx, coords],
                dim=-1,
            )
            batch_coords.append(coords)

        batch_voxels = torch.cat(batch_voxels, dim=0)
        batch_coords = torch.cat(batch_coords, dim=0)
        if self.use_hierarchical_conf:
            batch_voxel_confs = torch.cat(batch_voxel_confs, dim=0)

        x = spconv.SparseConvTensor(
            features=batch_voxels,
            indices=batch_coords.int(),
            spatial_shape=self.spatial_shape,
            batch_size=batch_size,
        )

        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Build the BEV confidence map ``r_bev`` (Eq. 19) before the dense
        # collapse, while voxel-level confidence still carries spatial info.
        if self.use_hierarchical_conf:
            voxel_conf_channel = batch_voxel_confs[:, 0]
            conf_sparse = spconv.SparseConvTensor(
                features=voxel_conf_channel.unsqueeze(1),
                indices=batch_coords.int(),
                spatial_shape=self.spatial_shape,
                batch_size=batch_size,
            )
            conf_dense = conf_sparse.dense()
            bev_conf = conf_dense.max(dim=2)[0]
            # Match the BEV feature stride used by the detection head (8x).
            bev_conf = torch.nn.functional.avg_pool2d(
                bev_conf, kernel_size=8, stride=8
            )
            batch_dict['bev_confidence'] = bev_conf

        x = x.dense()
        B, C, D, H, W = x.shape
        batch_dict['pv_features'] = x.view(B, C * D, H, W)
        return batch_dict
