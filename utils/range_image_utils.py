"""Range-image projection helpers for the range-view backbone."""

import numpy as np
import torch

from utils.box_utils import boxes3d_to_corners3d


class RangeConvertor:
    """Converts between LiDAR points and (front-cropped) spherical range images.

    Attributes:
        full_size: Range image size of the full 360-deg projection ``(W, H)``.
        front_size: Cropped front-view size ``(W, H)`` actually fed to the
            range-view backbone.
        fov_up: Upper FoV bound in radians.
        fov_down: Lower FoV bound in radians.
        fov: Vertical FoV span (``fov_up + fov_down``).
    """

    def __init__(self, cfg):
        """Initialises with full and front image sizes from ``cfg``."""
        self.full_size = cfg['full_size']
        self.front_size = cfg['front_size']
        # KITTI-like Velodyne HDL-64 vertical field of view.
        self.fov_up = 2.0 * 3.14159 / 180
        self.fov_down = 24.8 * 3.14159 / 180
        self.fov = self.fov_up + self.fov_down

    def get_pixel_coords_torch(self, points):
        """Computes range-image pixel coordinates for torch points.

        Tracking the center column makes it easy to crop the front view even
        after random z-axis rotation.

        Args:
            points: Tensor of shape ``[N, 3 + C]``.

        Returns:
            Tuple ``(us, vs, center_u, center_v)``.
        """
        xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
        rs = torch.sqrt(xs ** 2 + ys ** 2 + zs ** 2)

        us = 0.5 * (1 - torch.atan2(ys, xs) / torch.pi) * self.full_size[0]
        vs = (1 - (torch.arcsin(zs / rs) + self.fov_down) / self.fov) * self.full_size[1]
        us = torch.clip(us, min=0, max=self.full_size[0] - 1).long()
        vs = torch.clip(vs, min=0, max=self.full_size[1] - 1).long()

        center_u = int((us.max() + us.min()) / 2)
        center_v = int((vs.max() + vs.min()) / 2)
        return us, vs, center_u, center_v

    def get_pixel_coords_numpy(self, points):
        """Numpy counterpart of :meth:`get_pixel_coords_torch`.

        Args:
            points: Array of shape ``[N, 3 + C]``.

        Returns:
            Tuple ``(us, vs, center_u, center_v)``.
        """
        xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
        rs = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)

        us = 0.5 * (1 - np.arctan2(ys, xs) / np.pi) * self.full_size[0]
        vs = (1 - (np.arcsin(zs / rs) + self.fov_down) / self.fov) * self.full_size[1]
        us = np.clip(us, a_min=0, a_max=self.full_size[0] - 1).astype(np.int32)
        vs = np.clip(vs, a_min=0, a_max=self.full_size[1] - 1).astype(np.int32)

        center_u = int((us.max() + us.min()) / 2)
        center_v = int((vs.max() + vs.min()) / 2)
        return us, vs, center_u, center_v

    def get_front_image_origin(self, center_u, center_v):
        """Computes the top-left corner of the front crop inside ``full_size``.

        Args:
            center_u: Centre column of the active points in the full image.
            center_v: Centre row of the active points in the full image.

        Returns:
            Tuple ``(u0, v0)``.
        """
        u0 = center_u - self.front_size[0] // 2
        u0 = min(max(0, u0), self.full_size[0] - self.front_size[0])

        v0 = center_v - self.front_size[1] // 2
        v0 = min(max(0, v0), self.full_size[1] - self.front_size[1])
        return u0, v0

    def get_range_image(self, points, features):
        """Builds a CHW front-view range image from points and their features.

        Args:
            points: Array of shape ``[N, 3 + C]``.
            features: Array of shape ``[N, F]`` to scatter as image channels.

        Returns:
            Array of shape ``[F, front_H, front_W]``.
        """
        assert len(points.shape) == 2 and len(features.shape) == 2
        assert points.shape[0] == features.shape[0]

        image_shape = (features.shape[1], self.full_size[1], self.full_size[0])
        full_image = np.zeros(image_shape, dtype=np.float32)

        us, vs, cu, cv = self.get_pixel_coords_numpy(points)
        full_image[:, vs, us] = features.transpose()

        u0, v0 = self.get_front_image_origin(cu, cv)
        return full_image[:, v0:v0 + self.front_size[1], u0:u0 + self.front_size[0]]

    def get_range_features(self, points, front_image):
        """Reads per-point features back from a front-view feature map.

        Args:
            points: Tensor of shape ``[N, 3 + C]`` (LiDAR coords).
            front_image: Tensor of shape ``[F, front_H, front_W]``.

        Returns:
            Tensor of shape ``[N, F]``.
        """
        image_shape = (front_image.shape[0], self.full_size[1], self.full_size[0])
        full_image = torch.zeros(
            image_shape, dtype=front_image.dtype, device=front_image.device
        )

        us, vs, cu, cv = self.get_pixel_coords_torch(points)
        u0, v0 = self.get_front_image_origin(cu, cv)

        full_image[:, v0:v0 + self.front_size[1], u0:u0 + self.front_size[0]] = front_image
        return full_image[:, vs, us].t()

    def get_range_boxes_in_full_image(self, boxes_lidar):
        """Projects LiDAR boxes to 2-D rectangles in the full range image.

        Args:
            boxes_lidar: Array of shape ``[N, 7]``.

        Returns:
            Array of shape ``[N, 4]`` (``u1, v1, u2, v2``) in full-image coords.
        """
        boxes = []
        corners = boxes3d_to_corners3d(boxes_lidar)
        for i in range(boxes_lidar.shape[0]):
            us, vs, _, _ = self.get_pixel_coords_numpy(corners[i])
            boxes.append([us.min(), vs.min(), us.max(), vs.max()])
        return np.array(boxes).reshape(-1, 4)

    def get_range_boxes_in_front_image(self, points, boxes_lidar):
        """Projects LiDAR boxes to rectangles in the cropped front image.

        Args:
            points: Array of shape ``[N, 3+C]`` used to locate the crop origin.
            boxes_lidar: Array of shape ``[M, 7]``.

        Returns:
            Array of shape ``[M, 4]`` (``u1, v1, u2, v2``) in front-image coords.
        """
        _, _, cu, cv = self.get_pixel_coords_numpy(points)
        u0, v0 = self.get_front_image_origin(cu, cv)

        boxes = []
        corners = boxes3d_to_corners3d(boxes_lidar)
        for i in range(boxes_lidar.shape[0]):
            us, vs, _, _ = self.get_pixel_coords_numpy(corners[i])
            us -= u0
            vs -= v0
            boxes.append([us.min(), vs.min(), us.max(), vs.max()])
        return np.array(boxes).reshape(-1, 4)

    def get_range_indicator(self):
        """Returns an empty HxW indicator array for occupancy bookkeeping."""
        return np.zeros((self.full_size[1], self.full_size[0]))
