"""Top-level HC-MVMM detector composition.

Stitches the four functional stages described in the paper:

  * Range-view backbone  (``layers/rv_backbones``)
  * Point-to-Voxel bridge with L2 hierarchical confidence
    (``layers/pv_bridges/voxel_feature_extractor.py``)
  * BEV backbone + L3 deformable enhancement
    (``layers/bev_backbones/enhanced_bev_backbone.py``)
  * CenterPoint head with IoU branch
    (``layers/heads/centerpoint_head.py``)
"""

import torch
import torch.nn as nn

from layers import bev_backbones
from layers import heads
from layers import pv_bridges
from layers import rv_backbones


def build_model(cfg, dataset):
    """Factory entry point for :class:`MVMM` (kept for backward compatibility).

    Args:
        cfg: ``model`` sub-dictionary of the YAML config.
        dataset: A :class:`data.kitti_dataset.KittiDataset` instance whose
            attributes are used to discover the range convertor, point-cloud
            range and class names.

    Returns:
        An :class:`MVMM` instance.

    Raises:
        NotImplementedError: If ``cfg['type']`` is unsupported.
    """
    if cfg['type'] == 'MVMM':
        return MVMM(cfg=cfg, dataset=dataset)
    raise NotImplementedError(f"Unknown model type: {cfg['type']}")


class MVMM(nn.Module):
    """End-to-end HC-MVMM detector (RV -> VFE -> BEV -> CenterPoint head)."""

    def __init__(self, cfg, dataset):
        """Wires the four sub-modules listed in ``cfg``.

        Args:
            cfg: ``model`` sub-dictionary of the YAML config.
            dataset: Dataset object used to surface ``range_convertor``,
                ``point_cloud_range`` and ``class_names``.
        """
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.module_list = []

        rv_backbone = rv_backbones.__all__[cfg['rv_backbone']['type']](
            cfg=cfg['rv_backbone'],
            range_convertor=self.dataset.range_convertor,
        )
        self.num_rv_features = rv_backbone.num_rv_features
        self.add_module('rv_backbone', rv_backbone)
        self.module_list.append(rv_backbone)

        pv_bridge = pv_bridges.__all__[cfg['pv_bridge']['type']](
            cfg=cfg['pv_bridge'],
            in_channels=self.num_rv_features,
            point_cloud_range=self.dataset.point_cloud_range,
        )
        self.num_pv_features = pv_bridge.num_pv_features
        self.add_module('pv_bridge', pv_bridge)
        self.module_list.append(pv_bridge)

        bev_backbone = bev_backbones.__all__[cfg['bev_backbone']['type']](
            cfg=cfg['bev_backbone'],
            in_channels=self.num_pv_features,
        )
        self.num_bev_features = bev_backbone.num_bev_features
        self.add_module('bev_backbone', bev_backbone)
        self.module_list.append(bev_backbone)

        head = heads.__all__[cfg['head']['type']](
            cfg=cfg['head'],
            in_channels=self.num_bev_features,
            class_names=self.dataset.class_names,
            grid_size=self.pv_bridge.grid_size,
            point_cloud_range=self.dataset.point_cloud_range,
        )
        self.add_module('head', head)
        self.module_list.append(head)

    def forward(self, batch_dict, score_thresh=0.1, nms_thresh=0.1):
        """Runs the full detector and returns either losses or predictions.

        Args:
            batch_dict: Batch dict produced by ``KittiDataset.collate_batch``.
            score_thresh: Detection score threshold (currently unused by the
                CenterPoint head, which has its own threshold).
            nms_thresh: NMS threshold (likewise unused by CenterPoint).

        Returns:
            When training, a tuple ``(total_loss_tensor, stats_dict)``.
            Otherwise the updated ``batch_dict`` with ``pred_boxes``,
            ``pred_scores`` and ``pred_classes``.
        """
        del score_thresh, nms_thresh  # CenterPoint decoder uses its own thresh.
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            detailed_losses = self.head.get_all_losses(batch_dict)
            total_loss = detailed_losses['total_tensor']
            stats = {
                'heatmap': detailed_losses['heatmap'],
                'offset': detailed_losses['offset'],
                'z': detailed_losses['z'],
                'dim': detailed_losses['dim'],
                'rot': detailed_losses['rot'],
            }
            if 'iou' in detailed_losses:
                stats['iou'] = detailed_losses['iou']
            return total_loss, stats

        # The CenterPoint head decodes and applies its own peak-NMS internally.
        return batch_dict
