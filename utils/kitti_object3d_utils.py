"""KITTI label-file parsing utilities.

Each KITTI label line follows the official 15- (or 16-) field format:

    type, truncation, occlusion, alpha,
    bbox(4), dimensions h w l, location x y z, rotation_y, [score].
"""

import numpy as np


def parse_objects(label_file):
    """Parses a KITTI ``label_2`` text file into a list of :class:`Object3d`.

    Args:
        label_file: Path to a KITTI ``label_2`` text file.

    Returns:
        List of :class:`Object3d` instances, one per non-empty line.
    """
    with open(label_file, 'r') as f:
        lines = f.readlines()
    return [Object3d(line) for line in lines]


class Object3d(object):
    """A single labelled 3-D object parsed from a KITTI label line."""

    def __init__(self, line):
        """Parses one KITTI label line.

        Args:
            line: Raw label string (15 or 16 whitespace-separated fields).
        """
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.truncation = float(label[1])
        # 0: fully visible, 1: partly occluded, 2: largely occluded, 3: unknown.
        self.occlusion = float(label[2])
        self.alpha = float(label[3])
        self.box2d = np.array(
            (float(label[4]), float(label[5]), float(label[6]), float(label[7])),
            dtype=np.float32,
        )
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.loc = np.array(
            (float(label[11]), float(label[12]), float(label[13])),
            dtype=np.float32,
        )
        self.dis_to_cam = np.linalg.norm(self.loc)
        self.ry = float(label[14])
        self.score = float(label[15]) if len(label) == 16 else -1.0
        self.level_str = None
        self.level = self.get_kitti_obj_level()

    def get_kitti_obj_level(self):
        """Computes the official KITTI difficulty level for the object.

        Returns:
            ``0`` for Easy, ``1`` for Moderate, ``2`` for Hard, ``-1`` otherwise.
            The corresponding human-readable label is stored in ``level_str``.
        """
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 0
        if height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 1
        if height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 2
        self.level_str = 'UnKnown'
        return -1
