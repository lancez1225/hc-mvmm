"""Target generators for the CenterPoint detection head.

Builds Gaussian heatmaps and (offset, z, dim, rotation) regression targets
using the ``ind + slot`` indexing scheme also used by the official
CenterPoint implementation.
"""

import numpy as np
import torch


# Average dimensions (l, w, h) per KITTI class used when ``use_gprr`` is True.
KITTI_CLASS_MEAN_SIZE = {
    'Car': np.array([3.9, 1.6, 1.56], dtype=np.float32),
    'Pedestrian': np.array([0.8, 0.6, 1.73], dtype=np.float32),
    'Cyclist': np.array([1.76, 0.6, 1.73], dtype=np.float32),
}


def gaussian_2d(shape, sigma=1):
    """Returns a 2-D Gaussian kernel of given shape and sigma.

    Args:
        shape: Tuple ``(H, W)`` (typically the diameter of the target Gaussian).
        sigma: Standard deviation.

    Returns:
        Array of shape ``shape`` with values in ``[0, 1]``.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian_to_heatmap(heatmap, center, radius, k=1):
    """In-place ``max``-blends a 2-D Gaussian into ``heatmap``.

    Args:
        heatmap: 2-D numpy array to receive the Gaussian (modified in place).
        center: ``(x, y)`` pixel centre.
        radius: Integer radius (``diameter = 2 * radius + 1``).
        k: Optional scalar multiplier applied to the Gaussian.

    Returns:
        ``heatmap`` (modified in place).
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_radius(det_size, min_overlap=0.1):
    """Computes the radius such that any Gaussian-shifted box still has IoU>=min.

    Follows the mmdetection3d convention with ``min_overlap=0.1``.

    Args:
        det_size: Tuple ``(box_h, box_w)`` in pixel units of the BEV feature map.
        min_overlap: Minimum overlap that must be preserved.

    Returns:
        The smallest radius satisfying the three quadratic constraints.
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + np.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + np.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + np.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2

    return min(r1, r2, r3)


def generate_centerpoint_targets(gt_boxes, class_names, feature_map_size,
                                 point_cloud_range, voxel_size,
                                 num_classes=3, max_objs=200, use_gprr=False):
    """Builds CenterPoint training targets for one sample.

    Args:
        gt_boxes: Array of shape ``[N, 7]`` (``x, y, z, l, w, h, theta``).
        class_names: List of class names parallel to ``gt_boxes``.
        feature_map_size: Tuple ``(H, W)`` of the BEV feature map (after stride).
        point_cloud_range: ``(x_min, y_min, z_min, x_max, y_max, z_max)``.
        voxel_size: ``(voxel_x, voxel_y, voxel_z)``.
        num_classes: Number of object classes.
        max_objs: Maximum number of objects packed per sample.
        use_gprr: If True, regress ``log(dim / class_mean_size)`` instead of
            ``log(dim)`` (Geometric Prior Residual Regression).

    Returns:
        Dict of torch tensors with keys ``gt_heatmap`` ``gt_anno_box``
        ``gt_ind`` ``gt_mask`` ``gt_cls_id``.
    """
    class_name_to_id = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

    H, W = feature_map_size
    num_objs = gt_boxes.shape[0]

    heatmap = np.zeros((num_classes, H, W), dtype=np.float32)
    anno_box = np.zeros((max_objs, 8), dtype=np.float32)
    ind = np.zeros((max_objs,), dtype=np.int64)
    mask = np.zeros((max_objs,), dtype=np.uint8)
    cls_ids = np.zeros((max_objs,), dtype=np.int64)

    pc_range = np.array(point_cloud_range)
    voxel_sz = np.array(voxel_size)

    grid_size_x = (pc_range[3] - pc_range[0]) / voxel_sz[0]
    grid_size_y = (pc_range[4] - pc_range[1]) / voxel_sz[1]

    downsample_x = grid_size_x / W
    downsample_y = grid_size_y / H

    obj_idx = 0
    for k in range(num_objs):
        if obj_idx >= max_objs:
            break

        box = gt_boxes[k]
        cls_name = class_names[k]
        if cls_name not in class_name_to_id:
            continue
        cls_id = class_name_to_id[cls_name]

        x, y, z, l, w, h, ry = box
        grid_x = (x - pc_range[0]) / voxel_sz[0]
        grid_y = (y - pc_range[1]) / voxel_sz[1]

        ct_x = grid_x / downsample_x
        ct_y = grid_y / downsample_y
        if ct_x < 0 or ct_x >= W or ct_y < 0 or ct_y >= H:
            continue

        ct_int = np.array([int(ct_x), int(ct_y)])
        ct_y_int, ct_x_int = ct_int[1], ct_int[0]
        ct_offset = np.array([ct_x - ct_int[0], ct_y - ct_int[1]])

        box_h = l / voxel_sz[1] / downsample_y
        box_w = w / voxel_sz[0] / downsample_x

        radius = gaussian_radius((box_h, box_w))
        radius = max(2, int(radius))
        draw_gaussian_to_heatmap(heatmap[cls_id], ct_int, radius)

        ind[obj_idx] = ct_y_int * W + ct_x_int
        mask[obj_idx] = 1
        cls_ids[obj_idx] = cls_id

        anno_box[obj_idx, 0] = ct_offset[0]
        anno_box[obj_idx, 1] = ct_offset[1]
        anno_box[obj_idx, 2] = z

        if use_gprr:
            mean_size = KITTI_CLASS_MEAN_SIZE[cls_name]
            anno_box[obj_idx, 3] = np.log(l / mean_size[0])
            anno_box[obj_idx, 4] = np.log(w / mean_size[1])
            anno_box[obj_idx, 5] = np.log(h / mean_size[2])
        else:
            anno_box[obj_idx, 3] = np.log(l)
            anno_box[obj_idx, 4] = np.log(w)
            anno_box[obj_idx, 5] = np.log(h)

        anno_box[obj_idx, 6] = np.sin(ry)
        anno_box[obj_idx, 7] = np.cos(ry)

        obj_idx += 1

    return {
        'gt_heatmap': torch.from_numpy(heatmap),
        'gt_anno_box': torch.from_numpy(anno_box),
        'gt_ind': torch.from_numpy(ind),
        'gt_mask': torch.from_numpy(mask),
        'gt_cls_id': torch.from_numpy(cls_ids),
    }
