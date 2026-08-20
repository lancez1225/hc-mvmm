"""Data-augmentation utilities for LiDAR / multi-modal 3-D detection.

Includes ground-truth sampling (the database sampler used by KITTI-style
methods such as VoxelNet/SECOND) plus the standard random rotation,
scaling and flip augmentations operating on the eight-channel
``colored_points`` tensor.
"""

import pickle
from functools import partial
from pathlib import Path

import numpy as np
import torch

from ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu
from ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
from utils.box_utils import remove_points_in_boxes3d
from utils.point_cloud_utils import rotate_points_along_z


def _random_rotation(gt_boxes, points, rotation_range):
    """Rotates boxes and points around z by a uniform random angle."""
    angle = np.random.uniform(rotation_range[0], rotation_range[1])
    points = rotate_points_along_z(points[np.newaxis, :, :], np.array([angle]))[0]
    gt_boxes[:, 0:3] = rotate_points_along_z(
        gt_boxes[np.newaxis, :, 0:3], np.array([angle])
    )[0]
    gt_boxes[:, 6] += angle
    return gt_boxes, points, angle


def _random_scaling(gt_boxes, points, scaling_range):
    """Multiplies positions and box sizes by a uniform random scalar."""
    scale = np.random.uniform(scaling_range[0], scaling_range[1])
    points[:, :3] *= scale
    gt_boxes[:, :6] *= scale
    return gt_boxes, points, scale


def _random_flip(gt_boxes, points):
    """Mirrors the scene across the y=0 plane with probability 0.5."""
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]
    return gt_boxes, points, enable


class DatabaseSampler:
    """Samples cached ground-truth objects into the current scene.

    The sampler ensures: BEV-IoU non-overlap with existing boxes, sufficient
    in-box points after pasting, and respect for class-specific minimums and
    excluded difficulty labels.
    """

    def __init__(self, cfg, root_dir, class_names, range_convertor):
        """Builds the in-memory ground-truth database from a pickle file.

        Args:
            cfg: Sampler config dictionary parsed from the YAML file.
            root_dir: Dataset root that contains the ``db_info_path`` and
                ``gt_database_*`` directory.
            class_names: List of class names the detector cares about.
            range_convertor: A :class:`utils.range_image_utils.RangeConvertor`
                used during occlusion suppression in range-image space.
        """
        self.root_dir = root_dir
        self.class_names = class_names
        self.range_convertor = range_convertor

        self.db_infos = {name: [] for name in class_names}
        db_info_path = Path(self.root_dir) / cfg['db_info_path']
        with open(str(db_info_path), 'rb') as f:
            infos = pickle.load(f)
            for cur_class in class_names:
                self.db_infos[cur_class].extend(infos[cur_class])

        self.gt_points_thresholds = {}
        for entry in cfg['filter_by_min_points']:
            class_name, min_points = entry.split(':')
            if class_name not in class_names:
                continue
            self.gt_points_thresholds[class_name] = int(min_points)
        self.db_infos = self.filter_by_min_points(
            self.db_infos, self.gt_points_thresholds
        )

        self.removed_difficulty = cfg['filter_by_difficulty']
        self.db_infos = self.filter_by_difficulty(
            self.db_infos, self.removed_difficulty
        )

        self.sample_groups = {}
        for entry in cfg['sample_groups']:
            class_name, sample_num = entry.split(':')
            if class_name not in class_names:
                continue
            self.sample_groups[class_name] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name])),
            }
        self.random_sample = cfg['random_sample']

    @staticmethod
    def filter_by_min_points(db_infos, gt_points_thresholds):
        """Removes objects with too few interior points."""
        for class_name, thresh in gt_points_thresholds.items():
            if thresh > 0 and class_name in db_infos.keys():
                db_infos[class_name] = [
                    info for info in db_infos[class_name]
                    if info['num_points_in_gt'] >= thresh
                ]
        return db_infos

    @staticmethod
    def filter_by_difficulty(db_infos, removed_difficulty):
        """Removes objects whose KITTI difficulty is in ``removed_difficulty``."""
        return {
            key: [info for info in dinfos
                  if info['difficulty'] not in removed_difficulty]
            for key, dinfos in db_infos.items()
        }

    def sample_with_fixed_number(self, class_name, sample_group, random_sample=True):
        """Draws a deterministic chunk of objects of one class from the DB."""
        sample_num = int(sample_group['sample_num'])
        pointer = sample_group['pointer']
        indices = sample_group['indices']
        if pointer + sample_num > len(self.db_infos[class_name]):
            if random_sample:
                pointer = 0
                indices = np.random.permutation(len(self.db_infos[class_name]))
            else:
                pointer = 0
                indices = np.arange(len(self.db_infos[class_name]))
        sampled_dicts = [
            self.db_infos[class_name][idx]
            for idx in indices[pointer:pointer + sample_num]
        ]
        sample_group['pointer'] = pointer + sample_num
        sample_group['indices'] = indices
        return sampled_dicts

    @staticmethod
    def put_boxes_on_road_plane(sampled_boxes, road_plane, calib):
        """Snaps sampled box bottoms onto the supplied road-plane equation."""
        a, b, c, d = road_plane
        center_cam = calib.lidar_to_rect(sampled_boxes[:, 0:3])
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]
        mv_height = sampled_boxes[:, 2] - sampled_boxes[:, 5] / 2 - cur_lidar_height
        sampled_boxes[:, 2] -= mv_height
        return sampled_boxes, mv_height

    def remove_occluded_points(self, data_dict, sampled_boxes,
                               sampled_names, sampled_obj_points):
        """Removes scene points hidden by pasted objects in range-image space."""
        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names']
        all_boxes = np.concatenate([sampled_boxes, gt_boxes], axis=0)
        all_names = np.concatenate([sampled_names, gt_names], axis=0)

        points = data_dict['colored_points']
        point_indices = points_in_boxes_cpu(
            torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
        ).numpy()

        all_obj_points = sampled_obj_points
        for i in range(gt_boxes.shape[0]):
            all_obj_points.append(points[point_indices[i] > 0])
        bg_points = remove_points_in_boxes3d(points, gt_boxes)
        bg_us, bg_vs, _, _ = self.range_convertor.get_pixel_coords_numpy(bg_points)

        all_rv_boxes = self.range_convertor.get_range_boxes_in_full_image(all_boxes)
        all_distances = np.sqrt(all_boxes[:, 0] ** 2 + all_boxes[:, 1] ** 2)

        keep_indices = []
        for i in reversed(np.argsort(all_distances)):
            x1, y1, x2, y2 = all_rv_boxes[i].squeeze()
            mask = (bg_us > x1) & (bg_vs > y1) & (bg_us < x2) & (bg_vs < y2)
            pp = bg_points[mask]
            d = np.sqrt(pp[:, 0] ** 2 + pp[:, 1] ** 2).mean() if pp.shape[0] > 0 else np.inf
            if all_distances[i] < d:
                keep_indices.append(i)
        all_boxes = all_boxes[keep_indices]
        all_names = all_names[keep_indices]
        all_obj_points = [all_obj_points[i] for i in keep_indices]

        rv_indicator = self.range_convertor.get_range_indicator()
        for i in range(all_boxes.shape[0]):
            if all_obj_points[i].shape[0] > 0:
                us, vs, _, _ = self.range_convertor.get_pixel_coords_numpy(all_obj_points[i])
                rv_indicator[vs, us] = i + 1

        bg_points = bg_points[rv_indicator[bg_vs, bg_us] == 0]

        for i in range(all_boxes.shape[0]):
            if all_obj_points[i].shape[0] > 0:
                us, vs, _, _ = self.range_convertor.get_pixel_coords_numpy(all_obj_points[i])
                all_obj_points[i] = all_obj_points[i][rv_indicator[vs, us] == i + 1]

        keep_indices = []
        for i in range(all_boxes.shape[0]):
            thresh = self.gt_points_thresholds[all_names[i]]
            if all_obj_points[i].shape[0] >= thresh:
                keep_indices.append(i)
        all_boxes = all_boxes[keep_indices]
        all_names = all_names[keep_indices]
        all_obj_points = [all_obj_points[i] for i in keep_indices]

        data_dict['gt_boxes'] = all_boxes
        data_dict['gt_names'] = all_names
        data_dict['colored_points'] = np.concatenate([bg_points, *all_obj_points], axis=0)
        return data_dict

    def __call__(self, data_dict):
        """Augments ``data_dict`` with sampled ground-truth objects.

        ``data_dict['colored_points']`` must already be the eight-channel
        tensor ``(x, y, z, intensity, r, g, b, confidence)``. Sampled object
        points carry a default confidence of ``1.0``.
        """
        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names']

        all_boxes = gt_boxes
        all_names = gt_names
        total_sampled_obj_points = []

        for class_name, sample_group in self.sample_groups.items():
            if int(sample_group['sample_num']) <= 0:
                continue
            sampled_dicts = self.sample_with_fixed_number(
                class_name, sample_group, self.random_sample
            )
            sampled_boxes = np.stack(
                [x['gt_box_lidar'] for x in sampled_dicts], axis=0
            )
            sampled_names = np.array([x['name'] for x in sampled_dicts])
            sampled_obj_points = []

            for info in sampled_dicts:
                file_path = Path(self.root_dir) / info['path']
                pts = np.fromfile(str(file_path), dtype=np.float32).reshape(-1, 7)
                pts[:, :3] += info['gt_box_lidar'][:3]
                # Append a default confidence column so the sampled points stay
                # shape-compatible with the eight-channel colored_points tensor.
                confidence = np.ones((pts.shape[0], 1), dtype=np.float32)
                pts = np.concatenate([pts, confidence], axis=1)
                sampled_obj_points.append(pts)

            iou1 = boxes_bev_iou_cpu(sampled_boxes[:, 0:7], all_boxes[:, 0:7])
            iou2 = boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
            iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
            iou1 = iou1 if iou1.shape[1] > 0 else iou2

            valid_indices = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0).nonzero()[0]
            sampled_boxes = sampled_boxes[valid_indices]
            sampled_names = sampled_names[valid_indices]
            sampled_obj_points = [sampled_obj_points[x] for x in valid_indices]

            all_boxes = np.concatenate((all_boxes, sampled_boxes), axis=0)
            all_names = np.concatenate((all_names, sampled_names), axis=0)
            total_sampled_obj_points.extend(sampled_obj_points)

        total_sampled_boxes = all_boxes[gt_boxes.shape[0]:, :]
        total_sampled_names = all_names[gt_names.shape[0]:]

        if len(total_sampled_obj_points) > 0:
            total_sampled_boxes, mv_height = self.put_boxes_on_road_plane(
                total_sampled_boxes, data_dict['road_plane'], data_dict['calib']
            )
            for i in range(len(total_sampled_obj_points)):
                total_sampled_obj_points[i][:, 2] -= mv_height[i]
            data_dict = self.remove_occluded_points(
                data_dict, total_sampled_boxes,
                total_sampled_names, total_sampled_obj_points,
            )

        return data_dict


class DataAugmentor:
    """Sequential dispatcher for the augmentations defined in the YAML."""

    def __init__(self, augmentor_list, root_dir, class_names, range_convertor):
        """Builds the queue of augmentation callables.

        Args:
            augmentor_list: List of dicts read from the config; each dict
                must define a ``type`` matching one of the methods below.
            root_dir: Dataset root for sampler operations.
            class_names: List of class names.
            range_convertor: Shared :class:`RangeConvertor` instance.
        """
        self.root_dir = root_dir
        self.class_names = class_names
        self.range_convertor = range_convertor

        self.data_augmentor_queue = []
        if augmentor_list is not None:
            for cfg in augmentor_list:
                cur_augmentor = getattr(self, cfg['type'])(cfg=cfg)
                self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, cfg=None):
        """Returns a :class:`DatabaseSampler` instance configured from ``cfg``."""
        return DatabaseSampler(
            cfg, self.root_dir, self.class_names, self.range_convertor
        )

    def random_rotation(self, data_dict=None, cfg=None):
        """Random rotation around the z-axis."""
        if data_dict is None:
            return partial(self.random_rotation, cfg=cfg)
        data_dict['gt_boxes'], data_dict['colored_points'], _ = _random_rotation(
            data_dict['gt_boxes'], data_dict['colored_points'], cfg['rotation_range']
        )
        return data_dict

    def random_scaling(self, data_dict=None, cfg=None):
        """Random isotropic scaling."""
        if data_dict is None:
            return partial(self.random_scaling, cfg=cfg)
        data_dict['gt_boxes'], data_dict['colored_points'], _ = _random_scaling(
            data_dict['gt_boxes'], data_dict['colored_points'], cfg['scaling_range']
        )
        return data_dict

    def random_flip(self, data_dict=None, cfg=None):
        """Random horizontal flip across the y=0 plane."""
        if data_dict is None:
            return partial(self.random_flip, cfg=cfg)
        data_dict['gt_boxes'], data_dict['colored_points'], _ = _random_flip(
            data_dict['gt_boxes'], data_dict['colored_points']
        )
        return data_dict

    def forward(self, data_dict):
        """Runs each augmentation in the queue."""
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)
        return data_dict
