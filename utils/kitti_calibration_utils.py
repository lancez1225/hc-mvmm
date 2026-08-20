"""KITTI camera/LiDAR calibration utilities.

Reads a KITTI ``calib`` text file and exposes the standard frame
transforms used throughout the project:

    {lidar, (x, y, z)} --V2C--> {rect, (x, y, z)} --P2--> {img, (u, v)}.
"""

import numpy as np


def parse_calib(calib_file):
    """Creates a :class:`Calibration` object from a KITTI calib file.

    Args:
        calib_file: Path to a KITTI-format ``calib`` text file.

    Returns:
        A :class:`Calibration` instance.
    """
    return Calibration(calib_file)


def get_calib_from_file(calib_file):
    """Parses the four 3x4 / 3x3 matrices stored in a KITTI calib file.

    Args:
        calib_file: Path to a KITTI-format ``calib`` text file.

    Returns:
        A dictionary with keys ``P2``, ``P3``, ``R0`` and ``Tr_velo2cam``.
    """
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {
        'P2': P2.reshape(3, 4),
        'P3': P3.reshape(3, 4),
        'R0': R0.reshape(3, 3),
        'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4),
    }


class Calibration(object):
    """Frame transforms between LiDAR, rectified camera, and image planes."""

    def __init__(self, calib_file):
        """Loads matrices from a KITTI calib file.

        Args:
            calib_file: Path to a KITTI-format ``calib`` text file.
        """
        param = get_calib_from_file(calib_file)
        self.P2 = param['P2']               # [3, 4] left color camera projection.
        self.R0 = param['R0']               # [3, 3] rectification rotation.
        self.V2C = param['Tr_velo2cam']     # [3, 4] velo -> rect-camera transform.

        # Cached camera intrinsics and the corresponding optical-center offsets.
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def cart_to_hom(self, pts):
        """Appends a trailing 1 to every row to produce homogeneous coords.

        Args:
            pts: Array of shape ``[N, 3]`` or ``[N, 2]``.

        Returns:
            Array of shape ``[N, 4]`` or ``[N, 3]``.
        """
        return np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))

    def rect_to_lidar(self, pts_rect):
        """Converts points from rectified camera to LiDAR coordinates.

        Args:
            pts_rect: Array of shape ``[N, 3]`` in rectified camera frame.

        Returns:
            Array of shape ``[N, 3]`` in LiDAR frame.
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        R0_ext = np.hstack((self.R0, np.zeros((3, 1), dtype=np.float32)))
        R0_ext = np.vstack((R0_ext, np.zeros((1, 4), dtype=np.float32)))
        R0_ext[3, 3] = 1
        V2C_ext = np.vstack((self.V2C, np.zeros((1, 4), dtype=np.float32)))
        V2C_ext[3, 3] = 1
        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(np.dot(R0_ext, V2C_ext).T))
        return pts_lidar[:, 0:3]

    def lidar_to_rect(self, pts_lidar):
        """Converts points from LiDAR to rectified camera coordinates.

        Args:
            pts_lidar: Array of shape ``[N, 3]`` in LiDAR frame.

        Returns:
            Array of shape ``[N, 3]`` in rectified camera frame.
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        return np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))

    def rect_to_img(self, pts_rect):
        """Projects rectified-camera points to the left-color image plane.

        Args:
            pts_rect: Array of shape ``[N, 3]``.

        Returns:
            A tuple ``(pts_img, pts_rect_depth)`` where ``pts_img`` has shape
            ``[N, 2]`` and ``pts_rect_depth`` has shape ``[N]``.
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """Projects LiDAR points directly to the left-color image plane.

        Args:
            pts_lidar: Array of shape ``[N, 3]`` in LiDAR frame.

        Returns:
            A tuple ``(pts_img, pts_depth)`` of shapes ``[N, 2]`` and ``[N]``.
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        return self.rect_to_img(pts_rect)

    def img_to_rect(self, u, v, depth_rect):
        """Back-projects image pixels with rectified depth to camera coords.

        Args:
            u: Array of shape ``[N]`` containing pixel x coordinates.
            v: Array of shape ``[N]`` containing pixel y coordinates.
            depth_rect: Array of shape ``[N]`` with rectified depth values.

        Returns:
            Array of shape ``[N, 3]`` in the rectified camera frame.
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        return np.concatenate(
            (x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)),
            axis=1,
        )
