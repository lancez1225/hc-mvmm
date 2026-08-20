"""CenterPoint detection head with the IoU-aware branch enabled."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterPointHead(nn.Module):
    """Anchor-free CenterPoint head with five regression sub-branches.

    Predicts a per-class heatmap, the centre offset, height ``z``, log box
    dimensions, ``(sin, cos)`` rotation, and (optionally) an IoU score used
    to rectify the final classification confidence at inference time.
    """

    def __init__(self, cfg, in_channels, class_names, grid_size, point_cloud_range):
        """Builds the convolutional sub-heads.

        Args:
            cfg: ``head`` sub-dictionary of the YAML config.
            in_channels: Channel count of the BEV feature returned by L3.
            class_names: List of detection categories.
            grid_size: ``[grid_x, grid_y, grid_z]`` voxel grid used by the L2 VFE.
            point_cloud_range: ``(xmin, ymin, zmin, xmax, ymax, zmax)``.
        """
        super().__init__()
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range

        pc_range = np.array(point_cloud_range)
        self.voxel_size = (pc_range[3:6] - pc_range[0:3]) / grid_size
        self.feature_map_stride = cfg.get('feature_map_stride', 8)

        self.top_k = cfg.get('top_k', 100)
        self.score_threshold = cfg['score_threshold']

        loss_weights = cfg.get('loss_weights', {})
        self.heatmap_weight = loss_weights.get('heatmap', 1.0)
        self.bbox_weight = loss_weights.get('bbox_weight', 2.0)
        self.code_weights = {
            'offset': loss_weights.get('offset', 1.0),
            'z': loss_weights.get('z', 1.0),
            'dim': loss_weights.get('dim', 1.0),
            'rotation': loss_weights.get('rotation', 1.0),
        }

        iou_cfg = cfg.get('iou_branch', {})
        self.use_iou_branch = iou_cfg.get('enabled', False)
        self.iou_weight = iou_cfg.get('loss_weight', 1.0)
        self.iou_score_alpha = iou_cfg.get('score_alpha', 0.5)

        common_channels = cfg.get('common_channels', 64)
        self.common_conv = nn.Sequential(
            nn.Conv2d(in_channels, common_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(common_channels),
            nn.ReLU(inplace=True),
        )

        self.heatmap_head = self._make_branch(common_channels, self.num_classes)
        self.offset_head = self._make_branch(common_channels, 2)
        self.z_head = self._make_branch(common_channels, 1)
        self.dim_head = self._make_branch(common_channels, 3)
        self.rot_head = self._make_branch(common_channels, 2)
        if self.use_iou_branch:
            self.iou_head = self._make_branch(common_channels, 1)

        self._init_weights()

    @staticmethod
    def _make_branch(in_channels, out_channels):
        """Creates a (3x3 BN ReLU) + 1x1 convolution branch."""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1),
        )

    def _init_weights(self):
        """Initialises the heatmap bias to focus on positives and reg heads small."""
        nn.init.constant_(self.heatmap_head[-1].bias, -2.19)
        reg_heads = [self.offset_head, self.z_head, self.dim_head, self.rot_head]
        if self.use_iou_branch:
            reg_heads.append(self.iou_head)
        for head in reg_heads:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0, std=0.001)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------ #
    #  Forward + decoding                                                #
    # ------------------------------------------------------------------ #

    def forward(self, batch_dict):
        """Runs the head and either stores logits or decodes detections."""
        x = self.common_conv(batch_dict['bev_features'])

        heatmap = self.heatmap_head(x).sigmoid()
        offset = self.offset_head(x)
        z = self.z_head(x)
        dim = self.dim_head(x)
        rot = self.rot_head(x)

        iou_pred = self.iou_head(x).sigmoid() if self.use_iou_branch else None

        if self.training:
            batch_dict['pred_heatmap'] = heatmap
            batch_dict['pred_offset'] = offset
            batch_dict['pred_z'] = z
            batch_dict['pred_dim'] = dim
            batch_dict['pred_rot'] = rot
            if iou_pred is not None:
                batch_dict['pred_iou'] = iou_pred
            return batch_dict

        batch_boxes, batch_scores, batch_classes = self.decode_predictions(
            heatmap, offset, z, dim, rot, iou_pred
        )
        batch_dict['pred_boxes'] = batch_boxes
        batch_dict['pred_scores'] = batch_scores
        batch_dict['pred_classes'] = batch_classes
        return batch_dict

    def decode_predictions(self, heatmap, offset, z, dim, rot, iou_pred=None):
        """Decodes per-frame top-k peaks into 7-DoF boxes and scores."""
        B, C, H, W = heatmap.shape
        device = heatmap.device

        batch_boxes, batch_scores, batch_classes = [], [], []

        for b in range(B):
            boxes, scores, classes = [], [], []
            for cls_id in range(C):
                heat = heatmap[b, cls_id]
                heat_max = F.max_pool2d(
                    heat.unsqueeze(0).unsqueeze(0),
                    kernel_size=3, stride=1, padding=1,
                ).squeeze()
                peak_mask = (heat == heat_max).float()
                heat = heat * peak_mask

                heat_flat = heat.view(-1)
                topk_scores, topk_inds = torch.topk(
                    heat_flat, k=min(self.top_k, heat_flat.numel())
                )
                mask = topk_scores > self.score_threshold
                topk_scores = topk_scores[mask]
                topk_inds = topk_inds[mask]
                if len(topk_inds) == 0:
                    continue

                topk_ys = torch.div(topk_inds, W, rounding_mode='floor')
                topk_xs = topk_inds % W

                center_x = topk_xs.float() + offset[b, 0, topk_ys, topk_xs]
                center_y = topk_ys.float() + offset[b, 1, topk_ys, topk_xs]
                center_z = z[b, 0, topk_ys, topk_xs]

                center_x = (center_x * self.voxel_size[0] * self.feature_map_stride
                            + self.point_cloud_range[0])
                center_y = (center_y * self.voxel_size[1] * self.feature_map_stride
                            + self.point_cloud_range[1])

                l = dim[b, 0, topk_ys, topk_xs].exp()
                w = dim[b, 1, topk_ys, topk_xs].exp()
                h = dim[b, 2, topk_ys, topk_xs].exp()

                rot_sin = rot[b, 0, topk_ys, topk_xs]
                rot_cos = rot[b, 1, topk_ys, topk_xs]
                theta = torch.atan2(rot_sin, rot_cos)

                cls_boxes = torch.stack(
                    [center_x, center_y, center_z, l, w, h, theta], dim=1
                )

                if iou_pred is not None:
                    pred_iou = iou_pred[b, 0, topk_ys, topk_xs].clamp(min=0.01)
                    topk_scores = topk_scores * pred_iou.pow(self.iou_score_alpha)

                boxes.append(cls_boxes)
                scores.append(topk_scores)
                classes.append(torch.full_like(topk_scores, cls_id + 1, dtype=torch.long))

            if boxes:
                batch_boxes.append(torch.cat(boxes, dim=0))
                batch_scores.append(torch.cat(scores, dim=0))
                batch_classes.append(torch.cat(classes, dim=0))
            else:
                batch_boxes.append(torch.zeros((0, 7), device=device))
                batch_scores.append(torch.zeros((0,), device=device))
                batch_classes.append(torch.zeros((0,), dtype=torch.long, device=device))

        return batch_boxes, batch_scores, batch_classes

    # ------------------------------------------------------------------ #
    #  Loss computation                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _transpose_and_gather_feat(feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), feat.size(2))
        return feat.gather(1, ind)

    def get_cls_loss(self, batch_dict):
        """Modified focal loss for the heatmap branch."""
        pred_heatmap = batch_dict['pred_heatmap']
        gt_heatmap = batch_dict['gt_heatmap']

        pos_mask = gt_heatmap.eq(1).float()
        neg_mask = gt_heatmap.lt(1).float()
        neg_weights = torch.pow(1 - gt_heatmap, 4)

        pos_loss = (
            -torch.log(pred_heatmap + 1e-12)
            * torch.pow(1 - pred_heatmap, 2) * pos_mask
        )
        neg_loss = (
            -torch.log(1 - pred_heatmap + 1e-12)
            * torch.pow(pred_heatmap, 2) * neg_weights * neg_mask
        )

        num_pos = pos_mask.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        loss = neg_loss if num_pos == 0 else (pos_loss + neg_loss) / num_pos
        return loss * self.heatmap_weight

    def get_reg_loss(self, batch_dict, return_dict=False):
        """L1 regression losses on offset/z/dim/rot at the GT slots."""
        gt_anno_box = batch_dict['gt_anno_box']
        gt_ind = batch_dict['gt_ind'].long()
        gt_mask = batch_dict['gt_mask']
        num_pos = gt_mask.sum().float().clamp(min=1.0)

        pred_offset = self._transpose_and_gather_feat(batch_dict['pred_offset'], gt_ind)
        pred_z = self._transpose_and_gather_feat(batch_dict['pred_z'], gt_ind)
        pred_dim = self._transpose_and_gather_feat(batch_dict['pred_dim'], gt_ind)
        pred_rot = self._transpose_and_gather_feat(batch_dict['pred_rot'], gt_ind)

        gt_mask_float = gt_mask.unsqueeze(2).float()
        offset_loss = (torch.abs(pred_offset - gt_anno_box[:, :, 0:2]) * gt_mask_float).sum()
        offset_loss = offset_loss / num_pos * self.code_weights['offset']
        z_loss = (torch.abs(pred_z - gt_anno_box[:, :, 2:3]) * gt_mask_float).sum()
        z_loss = z_loss / num_pos * self.code_weights['z']
        dim_loss = (torch.abs(pred_dim - gt_anno_box[:, :, 3:6]) * gt_mask_float).sum()
        dim_loss = dim_loss / num_pos * self.code_weights['dim']
        rot_loss = (torch.abs(pred_rot - gt_anno_box[:, :, 6:8]) * gt_mask_float).sum()
        rot_loss = rot_loss / num_pos * self.code_weights['rotation']

        total_reg_loss = (offset_loss + z_loss + dim_loss + rot_loss) * self.bbox_weight
        if return_dict:
            return {
                'offset': offset_loss,
                'z': z_loss,
                'dim': dim_loss,
                'rot': rot_loss,
                'total': total_reg_loss,
            }
        return total_reg_loss

    def _compute_iou_targets(self, batch_dict):
        """Axis-aligned 3-D IoU between predicted and GT boxes at GT slots."""
        gt_anno_box = batch_dict['gt_anno_box']
        gt_ind = batch_dict['gt_ind'].long()

        pred_dim = self._transpose_and_gather_feat(batch_dict['pred_dim'], gt_ind)
        pred_z = self._transpose_and_gather_feat(batch_dict['pred_z'], gt_ind)
        pred_offset = self._transpose_and_gather_feat(batch_dict['pred_offset'], gt_ind)

        W = batch_dict['pred_offset'].shape[3]
        gt_ys = torch.div(gt_ind, W, rounding_mode='floor').float()
        gt_xs = (gt_ind % W).float()

        stride_x = self.voxel_size[0] * self.feature_map_stride
        stride_y = self.voxel_size[1] * self.feature_map_stride

        pred_cx = (gt_xs + pred_offset[:, :, 0]) * stride_x + self.point_cloud_range[0]
        pred_cy = (gt_ys + pred_offset[:, :, 1]) * stride_y + self.point_cloud_range[1]
        pred_cz = pred_z[:, :, 0]
        pred_l = pred_dim[:, :, 0].exp()
        pred_w = pred_dim[:, :, 1].exp()
        pred_h = pred_dim[:, :, 2].exp()

        gt_offset = gt_anno_box[:, :, 0:2]
        gt_cx = (gt_xs + gt_offset[:, :, 0]) * stride_x + self.point_cloud_range[0]
        gt_cy = (gt_ys + gt_offset[:, :, 1]) * stride_y + self.point_cloud_range[1]
        gt_cz = gt_anno_box[:, :, 2]
        gt_l = gt_anno_box[:, :, 3].exp()
        gt_w = gt_anno_box[:, :, 4].exp()
        gt_h = gt_anno_box[:, :, 5].exp()

        ix1 = torch.max(pred_cx - pred_l / 2, gt_cx - gt_l / 2)
        ix2 = torch.min(pred_cx + pred_l / 2, gt_cx + gt_l / 2)
        iy1 = torch.max(pred_cy - pred_w / 2, gt_cy - gt_w / 2)
        iy2 = torch.min(pred_cy + pred_w / 2, gt_cy + gt_w / 2)
        iz1 = torch.max(pred_cz - pred_h / 2, gt_cz - gt_h / 2)
        iz2 = torch.min(pred_cz + pred_h / 2, gt_cz + gt_h / 2)

        inter = (
            (ix2 - ix1).clamp(min=0)
            * (iy2 - iy1).clamp(min=0)
            * (iz2 - iz1).clamp(min=0)
        )
        pred_vol = pred_l * pred_w * pred_h
        gt_vol = gt_l * gt_w * gt_h
        iou = inter / (pred_vol + gt_vol - inter + 1e-8)
        return iou.clamp(0, 1).detach()

    def get_all_losses(self, batch_dict):
        """Returns the aggregated training loss and a per-term scalar dict."""
        heatmap_loss = self.get_cls_loss(batch_dict)
        reg_losses = self.get_reg_loss(batch_dict, return_dict=True)
        total_loss_tensor = heatmap_loss + reg_losses['total']

        result = {
            'heatmap': heatmap_loss.item(),
            'offset': reg_losses['offset'].item(),
            'z': reg_losses['z'].item(),
            'dim': reg_losses['dim'].item(),
            'rot': reg_losses['rot'].item(),
            'total': total_loss_tensor.item(),
            'total_tensor': total_loss_tensor,
        }

        if self.use_iou_branch:
            gt_ind = batch_dict['gt_ind'].long()
            gt_mask = batch_dict['gt_mask']
            num_pos = gt_mask.sum().float().clamp(min=1.0)

            iou_targets = self._compute_iou_targets(batch_dict)
            pred_iou = self._transpose_and_gather_feat(
                batch_dict['pred_iou'], gt_ind
            ).squeeze(-1)

            iou_loss = F.smooth_l1_loss(pred_iou, iou_targets, reduction='none')
            iou_loss = (iou_loss * gt_mask.float()).sum() / num_pos * self.iou_weight
            total_loss_tensor = total_loss_tensor + iou_loss

            result['iou'] = iou_loss.item()
            result['iou_target_mean'] = (
                (iou_targets * gt_mask.float()).sum().item()
                / max(num_pos.item(), 1.0)
            )
            result['total'] = total_loss_tensor.item()
            result['total_tensor'] = total_loss_tensor

        return result
