"""Point-cloud helpers used by data augmentation and FOV filtering."""

import numpy as np
import torch

from utils.common_utils import check_numpy_to_torch


def rotate_points_along_z(points, angle):
    """Rotates ``points`` around the z-axis by ``angle`` (per-batch).

    Args:
        points: Array or tensor of shape ``[B, N, 3 + C]``.
        angle: Array or tensor of shape ``[B]``. Positive rotates x toward y.

    Returns:
        Rotated points with the original backend (numpy or torch).
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones,
    ), dim=1).view(-1, 3, 3).float()

    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)

    return points_rot.numpy() if is_numpy else points_rot


def mask_points_by_range(points, limit_range):
    """Filters points whose xy coordinates fall inside ``limit_range``.

    Args:
        points: Array of shape ``[N, 3 + C]``.
        limit_range: Six-element iterable ``(xmin, ymin, zmin, xmax, ymax, zmax)``.

    Returns:
        The subset of ``points`` falling inside the xy bounds of ``limit_range``.
    """
    mask = ((points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3])
            & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4]))
    return points[mask]


def get_fov_flag(points, image_shape, calib):
    """Returns a boolean mask of LiDAR points that project inside the image.

    Args:
        points: Array of shape ``[N, 3 + C]`` in LiDAR coordinates.
        image_shape: Iterable ``(H, W)`` of the corresponding image.
        calib: A :class:`utils.kitti_calibration_utils.Calibration` object.

    Returns:
        Boolean array of shape ``[N]`` indicating whether each point projects
        into the image and has positive rectified depth.
    """
    pts_rect = calib.lidar_to_rect(points[:, 0:3])
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)

    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < image_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < image_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)

    return np.logical_and(val_flag_merge, pts_rect_depth >= 0)
