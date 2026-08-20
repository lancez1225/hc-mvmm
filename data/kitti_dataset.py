"""KITTI dataset wrapper for HC-MVMM.

Loads the eight-channel ``colored_points`` tensor produced by the
:class:`data.occlusion_aware_colorizer.OcclusionAwareColorizer`, range-view
features, and the CenterPoint training targets for one sample.
"""

import copy
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from skimage import io

from data.kitti_object_eval_python.kitti_common import get_label_annos
from ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
from utils.augmentor_utils import DataAugmentor
from utils.box_utils import boxes3d_to_corners3d
from utils.box_utils import in_hull
from utils.box_utils import mask_boxes3d_by_range
from utils.centerpoint_utils import generate_centerpoint_targets
from utils.common_utils import limit_period
from utils.kitti_calibration_utils import parse_calib
from utils.kitti_object3d_utils import parse_objects
from utils.point_cloud_utils import get_fov_flag
from utils.point_cloud_utils import mask_points_by_range
from utils.range_image_utils import RangeConvertor


class KittiDataset(torch.utils.data.Dataset):
    """PyTorch ``Dataset`` returning HC-MVMM-ready training samples."""

    def __init__(self, cfg, split, is_training=True, augment_data=True,
                 create_kitti_infos=False):
        """Initialises paths, augmentations and the L1 colorizer.

        Args:
            cfg: ``dataset`` sub-dictionary of the YAML config.
            split: One of ``train``, ``val``, ``trainval``, ``test``.
            is_training: When False, returns inference-only samples.
            augment_data: Enables data augmentation (only meaningful when
                ``split`` is ``train`` or ``trainval``).
            create_kitti_infos: When True, skips loading info pickles -- used
                by the ``create_kitti_infos`` CLI mode.
        """
        self.cfg = cfg
        self.root_dir = 'data/kitti'
        self.split = split
        assert self.split in ['train', 'val', 'trainval', 'test']

        self.split_file = os.path.join(self.root_dir, 'ImageSets', self.split + '.txt')
        self.id_list = [x.strip() for x in open(self.split_file).readlines()]

        self.data_dir = os.path.join(
            self.root_dir, 'testing' if self.split == 'test' else 'training'
        )
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.velodyne_dir = os.path.join(self.data_dir, 'velodyne')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_2')
        self.plane_dir = os.path.join(self.data_dir, 'planes')

        self.info_file = os.path.join(self.root_dir, f'kitti_infos_{self.split}.pkl')
        if create_kitti_infos:
            return

        self.kitti_infos = []
        with open(self.info_file, 'rb') as f:
            self.kitti_infos.extend(pickle.load(f))

        self.class_names = cfg['class_names']
        self.write_list = cfg['write_list']
        self.point_cloud_range = np.array(cfg['point_cloud_range'], dtype=np.float32)

        self.is_training = is_training
        self.augment_data = augment_data
        if self.split not in ['train', 'trainval']:
            self.augment_data = False

        self.range_convertor = RangeConvertor(cfg['range_image'])
        if self.augment_data and cfg.get('augmentor_list') is not None:
            self.data_augmentor = DataAugmentor(
                cfg['augmentor_list'], self.root_dir,
                self.class_names, self.range_convertor,
            )
        else:
            self.data_augmentor = None

        # L1 colorizer (optional, defaults to enabled in the full config).
        self.colorizer = None
        colorizer_cfg = cfg.get('colorizer', None)
        if colorizer_cfg is not None and colorizer_cfg.get('enabled', False):
            from data.occlusion_aware_colorizer import OcclusionAwareColorizer
            self.colorizer = OcclusionAwareColorizer(colorizer_cfg)

    # ------------------------------------------------------------------ #
    #  Raw file readers                                                  #
    # ------------------------------------------------------------------ #

    def get_image(self, idx):
        """Returns the left-color RGB image as a float32 array in ``[0, 1]``."""
        img_file = os.path.join(self.image_dir, '%06d.png' % int(idx))
        assert os.path.exists(img_file)
        return io.imread(img_file).astype(np.float32) / 255.0

    def get_image_shape(self, idx):
        """Returns the ``(H, W)`` of the corresponding RGB image."""
        img_file = os.path.join(self.image_dir, '%06d.png' % int(idx))
        assert os.path.exists(img_file)
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_points(self, idx):
        """Reads the Velodyne ``[N, 4]`` point cloud ``(x, y, z, i)``."""
        pts_file = os.path.join(self.velodyne_dir, '%06d.bin' % int(idx))
        assert os.path.exists(pts_file)
        return np.fromfile(pts_file, dtype=np.float32).reshape(-1, 4)

    def get_calib(self, idx):
        """Returns a :class:`Calibration` parsed from the KITTI calib file."""
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % int(idx))
        assert os.path.exists(calib_file)
        return parse_calib(calib_file)

    def get_label(self, idx):
        """Returns a list of :class:`Object3d` parsed from a ``label_2`` file."""
        label_file = os.path.join(self.label_dir, '%06d.txt' % int(idx))
        assert os.path.exists(label_file)
        return parse_objects(label_file)

    def get_road_plane(self, idx):
        """Returns the unit-normal road-plane vector ``(a, b, c, d)``."""
        plane_file = os.path.join(self.plane_dir, '%06d.txt' % int(idx))
        assert os.path.exists(plane_file)
        with open(plane_file, 'r') as f:
            lines = f.readlines()
        plane = np.asarray([float(i) for i in lines[3].split()])
        if plane[1] > 0:
            plane = -plane
        return plane / np.linalg.norm(plane[0:3])

    # ------------------------------------------------------------------ #
    #  Colored-point generation (L1 entry point on the dataset side)     #
    # ------------------------------------------------------------------ #

    def get_colored_points_in_fov(self, idx):
        """Returns the eight-channel coloured point cloud for one frame.

        When ``cfg.use_extended_fov`` or ``cfg.fov_extend_ratio > 0`` is set,
        the colorization is performed on a region wider than the camera FoV
        and out-of-FoV points are filled by L1 propagation.

        Args:
            idx: KITTI sample index (string id).

        Returns:
            Array of shape ``[N, 8]`` ``(x, y, z, i, r, g, b, c)``.
        """
        use_ext = self.cfg.get('use_extended_fov', False) if self.colorizer is not None else False
        fov_extend_ratio = self.cfg.get('fov_extend_ratio', 0.0)

        points = self.get_points(idx)
        image_shape = self.get_image_shape(idx)
        calib = self.get_calib(idx)
        fov_flag = get_fov_flag(points, image_shape, calib)

        if not use_ext and fov_extend_ratio <= 0:
            points = points[fov_flag]
            img = self.get_image(idx)
            if self.colorizer is not None:
                return self.colorizer.colorize(points, img, calib)

            pts_img, _ = calib.lidar_to_img(points[:, 0:3])
            pts_img = pts_img.astype(np.int32)
            rgb = img[pts_img[:, 1], pts_img[:, 0], :]
            confidence = np.ones((len(points), 1), dtype=np.float32)
            return np.concatenate([points, rgb, confidence], axis=1)

        # Extended-FoV branch: directly colorize the FoV slice and propagate
        # colour/confidence to points outside the FoV via the L1 colorizer.
        img = self.get_image(idx)
        H, W = image_shape[0], image_shape[1]
        pc_range = self.point_cloud_range

        if fov_extend_ratio > 0:
            margin_x = int(W * fov_extend_ratio)
            margin_y = int(H * fov_extend_ratio * 0.5)
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            pts_img_ext, pts_depth_ext = calib.rect_to_img(pts_rect)
            ext_flag = (
                (pts_img_ext[:, 0] >= -margin_x) & (pts_img_ext[:, 0] < W + margin_x)
                & (pts_img_ext[:, 1] >= -margin_y) & (pts_img_ext[:, 1] < H + margin_y)
                & (pts_depth_ext >= 0)
            )
            range_mask = (
                ext_flag
                & (points[:, 0] >= pc_range[0]) & (points[:, 0] <= pc_range[3])
                & (points[:, 1] >= pc_range[1]) & (points[:, 1] <= pc_range[4])
                & (points[:, 2] >= pc_range[2]) & (points[:, 2] <= pc_range[5])
            )
        else:
            range_mask = (
                (points[:, 0] >= pc_range[0]) & (points[:, 0] <= pc_range[3])
                & (points[:, 1] >= pc_range[1]) & (points[:, 1] <= pc_range[4])
                & (points[:, 2] >= pc_range[2]) & (points[:, 2] <= pc_range[5])
            )

        all_points = points[range_mask]
        all_fov = fov_flag[range_mask]
        N = len(all_points)
        xyz = all_points[:, :3]

        colors = np.tile(self.colorizer.default_color, (N, 1))
        is_direct = np.zeros(N, dtype=bool)
        confidence = np.zeros(N, dtype=np.float32)
        uv_all = np.zeros((N, 2), dtype=np.float32)

        fov_indices = np.where(all_fov)[0]
        fov_xyz = xyz[all_fov]
        fov_colors, fov_is_direct, fov_uv, fov_zbuffer = (
            self.colorizer._direct_colorization_zbuffer(fov_xyz, img, calib)
        )

        colors[fov_indices] = fov_colors
        is_direct[fov_indices[fov_is_direct]] = True
        uv_all[fov_indices] = fov_uv
        if is_direct.sum() > 0:
            confidence[is_direct] = self.colorizer._compute_confidence(
                xyz[is_direct], uv_all[is_direct], fov_zbuffer, img
            )

        if self.colorizer.use_depth_constraint and self.colorizer.seg_enabled:
            object_ids = self.colorizer._segment_objects(
                xyz, is_direct=is_direct
            )
            colors, prop_conf = self.colorizer._propagate_segmented_depth_constrained(
                xyz, colors, confidence, is_direct, object_ids
            )
        elif self.colorizer.use_depth_constraint:
            colors, prop_conf = self.colorizer._propagate_depth_constrained(
                xyz, colors, confidence, is_direct
            )
        elif self.colorizer.seg_enabled:
            object_ids = self.colorizer._segment_objects(
                xyz, is_direct=is_direct
            )
            colors, prop_conf = self.colorizer._propagate_colors_with_confidence(
                xyz, colors, confidence, is_direct, object_ids
            )
        else:
            object_ids = np.zeros(N, dtype=np.int32)
            colors, prop_conf = self.colorizer._propagate_colors_with_confidence(
                xyz, colors, confidence, is_direct, object_ids
            )

        final_conf = np.where(is_direct, confidence, prop_conf)
        return np.concatenate(
            [all_points, colors, final_conf[:, np.newaxis]], axis=1
        )

    # ------------------------------------------------------------------ #
    #  Info-pickle generation (offline preprocessing)                    #
    # ------------------------------------------------------------------ #

    def get_infos(self, has_label=True, count_inside_pts=True, num_workers=4):
        """Builds the per-sample metadata list saved as ``kitti_infos_*.pkl``.

        Args:
            has_label: If True, also reads and packages ``label_2`` annotations.
            count_inside_pts: If True, also counts the LiDAR points inside each GT box.
            num_workers: Number of threads used by the executor.

        Returns:
            List of dictionaries (one per sample).
        """
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print(f'sample_idx: {sample_idx} in {self.split}.txt')
            info = {
                'sample_idx': sample_idx,
                'image_shape': self.get_image_shape(sample_idx),
            }

            if has_label:
                obj_list = self.get_label(sample_idx)
                annos = {
                    'name': np.array([obj.cls_type for obj in obj_list]),
                    'truncated': np.array([obj.truncation for obj in obj_list]),
                    'occluded': np.array([obj.occlusion for obj in obj_list]),
                    'alpha': np.array([obj.alpha for obj in obj_list]),
                    'bbox': np.concatenate(
                        [obj.box2d.reshape(1, 4) for obj in obj_list], axis=0
                    ),
                    'dimensions': np.array(
                        [[obj.h, obj.w, obj.l] for obj in obj_list]
                    ),
                    'location': np.concatenate(
                        [obj.loc.reshape(1, 3) for obj in obj_list], axis=0
                    ),
                    'rotation_y': np.array([obj.ry for obj in obj_list]),
                    'score': np.array([obj.score for obj in obj_list]),
                    'difficulty': np.array(
                        [obj.level for obj in obj_list], dtype=np.int32
                    ),
                }

                num_objects = len(
                    [obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare']
                )
                xyz = annos['location'][:num_objects]
                hwl = annos['dimensions'][:num_objects]
                rot_y = annos['rotation_y'][:num_objects]

                calib = self.get_calib(sample_idx)
                xyz_lidar = calib.rect_to_lidar(xyz)
                h, w, l = hwl[:, 0:1], hwl[:, 1:2], hwl[:, 2:3]
                xyz_lidar[:, 2] += h[:, 0] / 2
                annos['gt_box_lidar'] = np.concatenate(
                    [xyz_lidar, l, w, h, -(np.pi / 2 + rot_y[..., np.newaxis])],
                    axis=1,
                )

                info['annos'] = annos
                if count_inside_pts:
                    colored_points = self.get_colored_points_in_fov(sample_idx)
                    corners = boxes3d_to_corners3d(annos['gt_box_lidar'])
                    num_points_in_gt = np.zeros(num_objects, dtype=np.int32)
                    for k in range(num_objects):
                        flag = in_hull(colored_points[:, 0:3], corners[k])
                        num_points_in_gt[k] = flag.sum()
                    annos['num_points_in_gt'] = num_points_in_gt
            return info

        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, self.id_list)
        return list(infos)

    def create_gt_database(self):
        """Builds the GT-sampling database under ``gt_database_<split>``."""
        database_dir = Path(self.root_dir) / f'gt_database_{self.split}'
        db_info_file = Path(self.root_dir) / f'kitti_dbinfos_{self.split}.pkl'

        database_dir.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(self.info_file, 'rb') as f:
            infos = pickle.load(f)

        for k, info in enumerate(infos):
            print(f'gt_database sample in {self.split}.txt: {k + 1}/{len(infos)}')
            sample_idx = info['sample_idx']
            annos = info['annos']

            colored_points = self.get_colored_points_in_fov(sample_idx)
            names = annos['name']
            gt_boxes = annos['gt_box_lidar']
            difficulties = annos['difficulty']
            bboxes = annos['bbox']

            num_objects = gt_boxes.shape[0]
            point_indices = points_in_boxes_cpu(
                torch.from_numpy(colored_points[:, 0:3]),
                torch.from_numpy(gt_boxes),
            ).numpy()

            for i in range(num_objects):
                pts_file = database_dir / f'{sample_idx}_{names[i]}_{i}.bin'
                gt_points = colored_points[point_indices[i] > 0]
                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(pts_file, 'w') as f:
                    gt_points.tofile(f)

                db_info = {
                    'path': str(pts_file.relative_to(self.root_dir)),
                    'name': names[i],
                    'gt_box_lidar': gt_boxes[i],
                    'num_points_in_gt': gt_points.shape[0],
                    'difficulty': difficulties[i],
                    'bbox': bboxes[i],
                }
                all_db_infos.setdefault(names[i], []).append(db_info)

        for cls_name, infos_for_class in all_db_infos.items():
            print(f'Number of ground truths in the {cls_name} class: {len(infos_for_class)}')
        with open(db_info_file, 'wb') as f:
            pickle.dump(all_db_infos, f)

    # ------------------------------------------------------------------ #
    #  KITTI evaluation                                                  #
    # ------------------------------------------------------------------ #

    def eval(self, result_dir, logger):
        """Runs the official KITTI evaluation over ``result_dir``."""
        # Lazy import to avoid loading the numba CUDA kernels at startup.
        from data.kitti_object_eval_python.eval import get_official_eval_result

        logger.info('==> Loading detections and ground truths...')
        img_ids = [int(idx) for idx in self.id_list]
        dt_annos = get_label_annos(result_dir)
        gt_annos = get_label_annos(self.label_dir, img_ids)
        logger.info('==> Done.')

        logger.info('==> Evaluating...')
        test_id = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        for category in self.write_list:
            result_str = get_official_eval_result(
                gt_annos, dt_annos, test_id[category],
                use_ldf_eval=False, print_info=False,
            )
            logger.info(result_str)

    # ------------------------------------------------------------------ #
    #  Dataset interface                                                 #
    # ------------------------------------------------------------------ #

    def __len__(self):
        return len(self.kitti_infos)

    def __getitem__(self, idx):
        """Returns one fully prepared training/evaluation sample.

        The dictionary contains the eight-channel ``colored_points``, calib,
        image shape, optional ``gt_boxes``, a front-view range image, and
        CenterPoint targets when training labels are available.
        """
        info = copy.deepcopy(self.kitti_infos[idx])
        sample_idx = info['sample_idx']
        data_dict = {
            'frame_id': sample_idx,
            'colored_points': self.get_colored_points_in_fov(sample_idx),
            'calib': self.get_calib(sample_idx),
            'image_shape': info['image_shape'],
        }
        data_dict['colored_points'] = mask_points_by_range(
            data_dict['colored_points'], self.point_cloud_range
        )

        if 'annos' in info:
            annos = info['annos']
            keep_indices = [
                i for i, x in enumerate(annos['name']) if x in self.class_names
            ]
            keep_annos = {key: annos[key][keep_indices] for key in annos.keys()}

            data_dict.update({
                'gt_boxes': keep_annos['gt_box_lidar'],
                'gt_names': keep_annos['name'],
                'road_plane': self.get_road_plane(sample_idx),
            })

            if self.data_augmentor is not None:
                data_dict = self.data_augmentor.forward(data_dict)

            gt_cls_ids = [self.class_names.index(n) + 1 for n in data_dict['gt_names']]
            data_dict['gt_boxes'] = np.concatenate(
                [data_dict['gt_boxes'],
                 np.array(gt_cls_ids).reshape(-1, 1).astype(np.float32)],
                axis=1,
            )

            data_dict.pop('gt_names', None)
            data_dict.pop('road_plane', None)
            data_dict['gt_boxes'][:, 6] = limit_period(
                data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
            )
            data_dict['gt_boxes'] = mask_boxes3d_by_range(
                data_dict['gt_boxes'], self.point_cloud_range
            )
            data_dict['colored_points'] = mask_points_by_range(
                data_dict['colored_points'], self.point_cloud_range
            )

        if self.is_training:
            if len(data_dict['gt_boxes']) == 0:
                return self.__getitem__(np.random.randint(self.__len__()))
            shuffle_indices = np.random.permutation(
                data_dict['colored_points'].shape[0]
            )
            data_dict['colored_points'] = data_dict['colored_points'][shuffle_indices]

        points = data_dict['colored_points']
        xs = points[:, 0:1]
        ys = points[:, 1:2]
        zs = points[:, 2:3]
        intensities = points[:, 3:4]
        colors = points[:, 4:7]

        xmin, ymin, zmin, xmax, ymax, zmax = self.point_cloud_range
        xs = (xs - xmin) / (xmax - xmin)
        ys = (ys - ymin) / (ymax - ymin)
        zs = (zs - zmin) / (zmax - zmin)

        point_features = np.concatenate([xs, ys, zs, intensities, colors], axis=1)
        data_dict['range_image'] = self.range_convertor.get_range_image(
            points, point_features
        )

        # CenterPoint targets when training labels are available.
        if (self.is_training and 'gt_boxes' in data_dict
                and len(data_dict['gt_boxes']) > 0):
            pc_range = np.array(self.point_cloud_range)
            voxel_size = 0.05
            downsample_stride = 8
            grid_size_x = int((pc_range[3] - pc_range[0]) / voxel_size / downsample_stride)
            grid_size_y = int((pc_range[4] - pc_range[1]) / voxel_size / downsample_stride)
            feature_map_size = (grid_size_y, grid_size_x)

            gt_boxes_lidar = data_dict['gt_boxes'][:, :7]
            gt_class_ids = data_dict['gt_boxes'][:, 7].astype(np.int32)
            gt_class_names = [self.class_names[int(c) - 1] for c in gt_class_ids]

            use_gprr = self.cfg.get('use_gprr', False) if hasattr(self, 'cfg') else False
            targets = generate_centerpoint_targets(
                gt_boxes=gt_boxes_lidar,
                class_names=gt_class_names,
                feature_map_size=feature_map_size,
                point_cloud_range=self.point_cloud_range,
                voxel_size=[0.05, 0.05, 0.1],
                num_classes=len(self.class_names),
                max_objs=200,
                use_gprr=use_gprr,
            )
            data_dict.update(targets)

        return data_dict

    # ------------------------------------------------------------------ #
    #  Collate / GPU transfer                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def collate_batch(batch_list):
        """Concatenates a list of per-sample dicts into a single batch dict.

        Notes:
            * ``colored_points`` are concatenated with a batch-id column inserted.
            * ``gt_boxes`` are padded to the largest object count in the batch.
            * CenterPoint target tensors are stacked along the batch dim.
        """
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)

        batch_size = len(batch_list)
        batch_dict = {'batch_size': batch_size}

        for key, val in data_dict.items():
            if key == 'colored_points':
                coors = []
                for i, coor in enumerate(val):
                    coor_pad = np.pad(
                        coor, ((0, 0), (1, 0)),
                        mode='constant', constant_values=i,
                    )
                    coors.append(coor_pad)
                batch_dict[key] = np.concatenate(coors, axis=0)
            elif key == 'gt_boxes':
                max_gt = max(len(x) for x in val)
                batch_gt = np.zeros(
                    (batch_size, max_gt, val[0].shape[-1]), dtype=np.float32
                )
                for k in range(batch_size):
                    batch_gt[k, :len(val[k]), :] = val[k]
                batch_dict[key] = batch_gt
            elif key in ('gt_heatmap', 'gt_anno_box', 'gt_ind', 'gt_mask', 'gt_cls_id'):
                batch_dict[key] = torch.stack(val, dim=0)
            else:
                batch_dict[key] = np.stack(val, axis=0)
        return batch_dict

    @staticmethod
    def load_data_to_gpu(batch_dict, device):
        """Moves all float tensors in ``batch_dict`` onto ``device``."""
        for key, val in batch_dict.items():
            if key in ('batch_size', 'frame_id', 'calib', 'image_shape'):
                continue
            if key in ('gt_heatmap', 'gt_anno_box', 'gt_ind', 'gt_mask', 'gt_cls_id'):
                batch_dict[key] = val.float().to(device)
            else:
                batch_dict[key] = torch.from_numpy(val).float().to(device)
        return batch_dict


# --------------------------------------------------------------------------- #
#  Offline info / database generation entry point                             #
# --------------------------------------------------------------------------- #


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'create_kitti_infos':
        print('==> Generating data info files...')

        for split in ('train', 'val', 'test'):
            dataset = KittiDataset(cfg={}, split=split, create_kitti_infos=True)
            has_label = split != 'test'
            infos = dataset.get_infos(
                has_label=has_label, count_inside_pts=has_label
            )
            with open(dataset.info_file, 'wb') as f:
                pickle.dump(infos, f)
            print(f'==> The info file for `{split}.txt` is saved to: {dataset.info_file}')

        dataset = KittiDataset(cfg={}, split='trainval', create_kitti_infos=True)
        train = pickle.load(open(os.path.join(dataset.root_dir, 'kitti_infos_train.pkl'), 'rb'))
        val = pickle.load(open(os.path.join(dataset.root_dir, 'kitti_infos_val.pkl'), 'rb'))
        with open(dataset.info_file, 'wb') as f:
            pickle.dump(train + val, f)
        print(f'==> The info file for `trainval.txt` is saved to: {dataset.info_file}')

        print('==> Generating ground truth databases...')
        for split in ('train', 'trainval'):
            dataset = KittiDataset(cfg={}, split=split, create_kitti_infos=True)
            dataset.create_gt_database()
