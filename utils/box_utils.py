"""3-D bounding-box geometric helpers (LiDAR/camera/image projections)."""

import numpy as np
import scipy
from scipy.spatial import Delaunay
import torch

from ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
from utils.common_utils import check_numpy_to_torch
from utils.point_cloud_utils import rotate_points_along_z


def in_hull(p, hull):
    """Tests whether points lie inside a 3-D convex hull.

    Args:
        p: Array of shape ``[N, 3]`` of query points.
        hull: Either an array of shape ``[M, 3]`` (corner points used to build
            a Delaunay triangulation) or a pre-built :class:`scipy.spatial.Delaunay`.

    Returns:
        Boolean array of shape ``[N]`` that is ``True`` for points inside.
    """
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        flag = np.zeros(p.shape[0], dtype=bool)
    return flag


def boxes3d_to_corners3d(boxes3d):
    """Converts 7-DoF boxes to their 8-corner representation.

    Corner indexing convention::

            7 -------- 4
           /|         /|
          6 -------- 5 .
          | |        | |
          . 3 -------- 0
          |/         |/
          2 -------- 1

    Args:
        boxes3d: Array or tensor of shape ``[N, 7]`` with rows
            ``(x, y, z, dx, dy, dz, heading)`` and ``(x, y, z)`` as the centre.

    Returns:
        Array or tensor of shape ``[N, 8, 3]``.
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(
        corners3d.view(-1, 8, 3), boxes3d[:, 6]
    ).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def mask_boxes3d_by_range(boxes3d, limit_range, min_num_corners=1):
    """Keeps only boxes with enough corners inside the cube ``limit_range``.

    Args:
        boxes3d: Array of shape ``[N, 7+]``; first 7 columns are
            ``(x, y, z, dx, dy, dz, heading)``.
        limit_range: ``(xmin, ymin, zmin, xmax, ymax, zmax)``.
        min_num_corners: Minimum number of corners that must fall inside.

    Returns:
        Filtered array of boxes.
    """
    corners = boxes3d_to_corners3d(boxes3d[:, 0:7])
    mask = ((corners >= limit_range[0:3]) & (corners <= limit_range[3:6])).all(axis=2)
    mask = mask.sum(axis=1) >= min_num_corners
    return boxes3d[mask]


def remove_points_in_boxes3d(points, boxes3d):
    """Removes points that fall inside any of the given 3-D boxes.

    Args:
        points: Array or tensor of shape ``[num_points, 3 + C]``.
        boxes3d: Array or tensor of shape ``[N, 7]``
            (``x, y, z, dx, dy, dz, heading``).

    Returns:
        Array or tensor of points that lie outside every box.
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)
    points, _ = check_numpy_to_torch(points)
    point_masks = points_in_boxes_cpu(points[:, 0:3], boxes3d)
    points = points[point_masks.sum(dim=0) == 0]
    return points.numpy() if is_numpy else points


def boxes3d_lidar_to_camera(boxes3d_lidar, calib):
    """Transforms 7-DoF LiDAR boxes to the rectified camera frame.

    Args:
        boxes3d_lidar: Array of shape ``[N, 7]``
            (``x, y, z, dx, dy, dz, heading``) in LiDAR coordinates.
        calib: A :class:`utils.kitti_calibration_utils.Calibration` instance.

    Returns:
        Array of shape ``[N, 7]`` whose rows are
        ``(x, y, z, h, w, l, r)`` in rectified camera coordinates.
    """
    xyz_lidar = boxes3d_lidar[:, 0:3]
    l = boxes3d_lidar[:, 3:4]
    w = boxes3d_lidar[:, 4:5]
    h = boxes3d_lidar[:, 5:6]
    r = boxes3d_lidar[:, 6:7]

    xyz_camera = calib.lidar_to_rect(xyz_lidar)
    r = -r - np.pi / 2
    return np.concatenate([xyz_camera, h, w, l, r], axis=-1)


def boxes3d_lidar_to_image(boxes3d_lidar, calib, image_shape=None):
    """Projects LiDAR boxes to axis-aligned 2-D image rectangles.

    Args:
        boxes3d_lidar: Array of shape ``[N, 7]`` in LiDAR coordinates.
        calib: A :class:`utils.kitti_calibration_utils.Calibration` instance.
        image_shape: Optional ``(H, W)`` for clipping the rectangles.

    Returns:
        Array of shape ``[N, 4]`` rows ``(u1, v1, u2, v2)``.
    """
    corners = boxes3d_to_corners3d(boxes3d_lidar[:, 0:7])
    boxes2d = []
    for i in range(corners.shape[0]):
        pts_img, _ = calib.lidar_to_img(corners[i])
        min_u, min_v = pts_img[:, 0].min(), pts_img[:, 1].min()
        max_u, max_v = pts_img[:, 0].max(), pts_img[:, 1].max()
        boxes2d.append([min_u, min_v, max_u, max_v])

    boxes2d = np.array(boxes2d).reshape(-1, 4)

    if image_shape is not None:
        boxes2d[:, 0] = np.clip(boxes2d[:, 0], a_min=0, a_max=image_shape[1] - 1)
        boxes2d[:, 1] = np.clip(boxes2d[:, 1], a_min=0, a_max=image_shape[0] - 1)
        boxes2d[:, 2] = np.clip(boxes2d[:, 2], a_min=0, a_max=image_shape[1] - 1)
        boxes2d[:, 3] = np.clip(boxes2d[:, 3], a_min=0, a_max=image_shape[0] - 1)

    return boxes2d
