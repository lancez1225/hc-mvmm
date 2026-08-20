"""Occlusion-aware point-cloud colorizer (L1 of HC-MVMM).

Pipeline (see Sec. 3.2 of the paper):

  1. Z-buffer first-hit direct colorization: only the nearest LiDAR return
     at each occupied pixel receives an RGB sample (Eqs. 1-3).
  2. Direct-point confidence ``r_dir`` is the product of three terms:
     reprojection stability, depth-edge consistency, and image-gradient
     attenuation (Eqs. 4-7).
  3. Optional depth-stratified DBSCAN segmentation of the point cloud.
  4. Depth-constrained K-NN propagation of colour/confidence from direct
     seeds to non-first-hit points (Eqs. 8-10), restricted to either the
     same DBSCAN cluster, the same depth slab, or globally with depth gating.

The output is an eight-channel array ``(x, y, z, intensity, r, g, b, c)``
where ``c`` is the per-point appearance confidence used by L2/L3.
"""

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


class OcclusionAwareColorizer:
    """First-hit colorization + depth-constrained confidence propagation."""

    def __init__(self, config=None):
        """Initialises hyper-parameters from a config dictionary.

        Args:
            config: Nested ``colorizer`` config dictionary, typically the one
                stored under ``dataset.colorizer`` in the YAML. ``None`` is
                accepted for backward compatibility and uses defaults.
        """
        if config is None:
            config = {}

        self.enabled = config.get('enabled', True)

        # Z-buffer config.
        zbuffer_cfg = config.get('zbuffer', {})
        self.depth_threshold = zbuffer_cfg.get('depth_threshold', 0.5)

        # Per-point confidence config (sigma_reproj, depth_edge, image_grad).
        conf_cfg = config.get('confidence', {})
        self.sigma_reproj = conf_cfg.get('sigma_reproj', 2.0)
        self.depth_edge_threshold = conf_cfg.get('depth_edge_threshold', 1.0)
        self.use_img_grad = conf_cfg.get('use_img_grad', True)
        self.img_grad_threshold = conf_cfg.get('img_grad_threshold', 0.1)

        # Depth-stratified DBSCAN clustering for non-rigid object grouping.
        seg_cfg = config.get('object_segmentation', {})
        self.seg_enabled = seg_cfg.get('enabled', True)
        self.layer_thickness = seg_cfg.get('layer_thickness', 2.0)
        self.dbscan_eps = seg_cfg.get('dbscan_eps', 0.5)
        self.dbscan_min_samples = seg_cfg.get('dbscan_min_samples', 5)
        self.dbscan_algorithm = seg_cfg.get('dbscan_algorithm', 'ball_tree')
        self.unambiguous_component_eps = seg_cfg.get(
            'unambiguous_component_eps', self.dbscan_eps
        )
        self.unambiguous_weight_ratio = seg_cfg.get(
            'unambiguous_weight_ratio', 0.85
        )
        self.unambiguous_min_weight = seg_cfg.get(
            'unambiguous_min_weight', 1e-3
        )
        self.unambiguous_brute_force_knn_threshold = seg_cfg.get(
            'unambiguous_brute_force_knn_threshold', 65536
        )

        # Propagation config.
        prop_cfg = config.get('propagation', {})
        self.prop_method = prop_cfg.get('method', 'point')
        self.sigma_spatial = prop_cfg.get('sigma_spatial', 1.0)
        self.max_iterations = prop_cfg.get('max_iterations', 3)
        self.k_neighbors = prop_cfg.get('k_neighbors', 20)
        self.decay_factor = prop_cfg.get('decay_factor', 0.9)
        self.brute_force_knn_threshold = prop_cfg.get(
            'brute_force_knn_threshold', 4096
        )

        # Depth-gated propagation parameters (tau_depth gating in Eq. 8).
        self.use_depth_constraint = prop_cfg.get('use_depth_constraint', False)
        self.tau_depth = prop_cfg.get('tau_depth', 1.0)

        # Propagated-confidence formulation: 'ratio' is Eq. 10; 'variance' is
        # an alternative that maps colour-variance to confidence.
        self.prop_conf_method = prop_cfg.get('confidence_method', 'ratio')
        self.variance_scale = prop_cfg.get('variance_scale', 0.1)

        # Default colour assigned to points that fail every fallback.
        self.default_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  Public entry points                                               #
    # ------------------------------------------------------------------ #

    def colorize(self, points, image, calib):
        """Runs the full L1 colorization pipeline for one scene.

        Args:
            points: Array of shape ``[N, 4]`` ``(x, y, z, intensity)``.
            image: Array of shape ``[H, W, 3]`` (float32 in ``[0, 1]``).
            calib: A :class:`utils.kitti_calibration_utils.Calibration` object.

        Returns:
            Array of shape ``[N, 8]`` ``(x, y, z, i, r, g, b, c)``.
        """
        if not self.enabled:
            return self._fallback_projection(points, image, calib)

        N = len(points)
        xyz = points[:, :3]

        colors, is_direct, uv_coords, zbuffer = self._direct_colorization_zbuffer(
            xyz, image, calib
        )

        confidence = np.zeros(N, dtype=np.float32)
        if is_direct.sum() > 0:
            confidence[is_direct] = self._compute_confidence(
                xyz[is_direct], uv_coords[is_direct], zbuffer, image
            )

        if self.seg_enabled:
            object_ids = self._segment_objects(xyz, is_direct=is_direct)
        else:
            object_ids = np.zeros(N, dtype=np.int32)

        if self.use_depth_constraint and not self.seg_enabled:
            colors, propagated_confidence = self._propagate_depth_constrained(
                xyz, colors, confidence, is_direct
            )
        elif self.use_depth_constraint and self.seg_enabled:
            colors, propagated_confidence = self._propagate_segmented_depth_constrained(
                xyz, colors, confidence, is_direct, object_ids
            )
        else:
            colors, propagated_confidence = self._propagate_colors_with_confidence(
                xyz, colors, confidence, is_direct, object_ids
            )

        final_confidence = np.where(is_direct, confidence, propagated_confidence)
        return np.concatenate(
            [points, colors, final_confidence[:, np.newaxis]], axis=1
        )

    # ------------------------------------------------------------------ #
    #  Stage 1: z-buffer first-hit direct colorization                   #
    # ------------------------------------------------------------------ #

    def _direct_colorization_zbuffer(self, xyz, image, calib):
        """First-hit z-buffer direct colorization (Eqs. 1-3).

        Args:
            xyz: Array of shape ``[N, 3]`` in LiDAR coordinates.
            image: Array of shape ``[H, W, 3]``.
            calib: A :class:`Calibration` instance.

        Returns:
            Tuple ``(colors, is_direct, uv_coords, zbuffer)``.
        """
        N = len(xyz)
        H, W = image.shape[:2]

        colors = np.tile(self.default_color, (N, 1))
        is_direct = np.zeros(N, dtype=bool)
        uv_coords = np.zeros((N, 2), dtype=np.float32)

        pts_img, pts_depth = calib.lidar_to_img(xyz)
        uv_coords = pts_img.astype(np.float32)

        valid_mask = (
            (pts_depth > 0)
            & (pts_img[:, 0] >= 0) & (pts_img[:, 0] < W)
            & (pts_img[:, 1] >= 0) & (pts_img[:, 1] < H)
        )

        zbuffer = np.full((H, W), np.inf, dtype=np.float32)

        valid_indices = np.where(valid_mask)[0]
        u_coords = pts_img[valid_mask, 0].astype(np.int32)
        v_coords = pts_img[valid_mask, 1].astype(np.int32)
        depths = pts_depth[valid_mask]

        if len(valid_indices) > 0:
            pixel_ids = v_coords.astype(np.int64) * W + u_coords.astype(np.int64)
            order = np.lexsort((depths, pixel_ids))
            sorted_pixels = pixel_ids[order]
            first_in_pixel = np.ones(len(order), dtype=bool)
            first_in_pixel[1:] = sorted_pixels[1:] != sorted_pixels[:-1]
            winners = order[first_in_pixel]

            win_u = u_coords[winners]
            win_v = v_coords[winners]
            direct_point_ids = valid_indices[winners]

            zbuffer[win_v, win_u] = depths[winners]
            colors[direct_point_ids] = image[win_v, win_u]
            is_direct[direct_point_ids] = True

        return colors, is_direct, uv_coords, zbuffer

    # ------------------------------------------------------------------ #
    #  Stage 2: per-point appearance confidence (Eqs. 4-7)               #
    # ------------------------------------------------------------------ #

    def _compute_confidence(self, xyz_direct, uv_direct, zbuffer, image):
        """Computes ``r_dir`` for directly colorized points (Eqs. 4-7).

        Args:
            xyz_direct: Array of shape ``[M, 3]`` of direct-point coordinates.
            uv_direct: Array of shape ``[M, 2]`` of their pixel coordinates.
            zbuffer: Array of shape ``[H, W]`` with first-hit depths.
            image: RGB image of shape ``[H, W, 3]``.

        Returns:
            Confidence array of shape ``[M]`` in ``[0.01, 1.0]``.
        """
        M = len(xyz_direct)
        H, W = zbuffer.shape

        # Term 1: reprojection stability (Eq. 5).
        u_center = np.floor(uv_direct[:, 0]) + 0.5
        v_center = np.floor(uv_direct[:, 1]) + 0.5
        delta = np.sqrt(
            (uv_direct[:, 0] - u_center) ** 2 + (uv_direct[:, 1] - v_center) ** 2
        )
        c_reproj = np.exp(-delta ** 2 / (2 * self.sigma_reproj ** 2))

        # Term 2: depth-edge consistency (Eq. 6) using central differences on
        # the sparse z-buffer. Missing neighbours are skipped per-axis.
        u_int = np.clip(uv_direct[:, 0].astype(np.int32), 1, W - 2)
        v_int = np.clip(uv_direct[:, 1].astype(np.int32), 1, H - 2)
        point_depths = np.linalg.norm(xyz_direct, axis=1)

        depth_left = zbuffer[v_int, u_int - 1]
        depth_right = zbuffer[v_int, u_int + 1]
        depth_up = zbuffer[v_int - 1, u_int]
        depth_down = zbuffer[v_int + 1, u_int]

        c_depth_edge = np.ones(M, dtype=np.float32)
        valid_h = np.isfinite(depth_left) & np.isfinite(depth_right)
        grad_x = np.zeros(M)
        grad_x[valid_h] = np.abs(depth_right[valid_h] - depth_left[valid_h]) / 2

        valid_v = np.isfinite(depth_up) & np.isfinite(depth_down)
        grad_y = np.zeros(M)
        grad_y[valid_v] = np.abs(depth_down[valid_v] - depth_up[valid_v]) / 2

        valid_any = valid_h | valid_v
        if valid_any.sum() > 0:
            depth_grad = np.maximum(grad_x, grad_y)
            depth_grad_normalized = depth_grad / (point_depths + 1e-6)
            c_depth_edge[valid_any] = np.exp(
                -depth_grad_normalized[valid_any] / self.depth_edge_threshold
            )

        # Term 3: image-gradient attenuation (Eq. 7).
        if self.use_img_grad:
            img_gray = image.mean(axis=2)
            img_grad_x = np.abs(img_gray[v_int, u_int + 1] - img_gray[v_int, u_int - 1]) / 2
            img_grad_y = np.abs(img_gray[v_int + 1, u_int] - img_gray[v_int - 1, u_int]) / 2
            img_grad = np.maximum(img_grad_x, img_grad_y)
            c_img_grad = np.exp(-img_grad / self.img_grad_threshold)
        else:
            c_img_grad = np.ones(M, dtype=np.float32)

        confidence = c_reproj * c_depth_edge * c_img_grad
        return np.clip(confidence, 0.01, 1.0).astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Stage 3: object segmentation (depth-stratified DBSCAN)            #
    # ------------------------------------------------------------------ #

    def _segment_objects(self, xyz, is_direct=None):
        """Depth-stratified DBSCAN segmentation of the LiDAR point cloud.

        Args:
            xyz: Array of shape ``[N, 3]`` of LiDAR points.
            is_direct: Optional boolean array indicating direct color seeds. If
                provided, layers that cannot affect propagation are skipped.

        Returns:
            Integer array of shape ``[N]`` with per-point object ids.
        """
        num_points = len(xyz)
        ranges = np.linalg.norm(xyz, axis=1)
        layer_ids = (ranges / self.layer_thickness).astype(np.int32)

        object_ids = np.full(num_points, -1, dtype=np.int32)
        global_object_id = 0

        if is_direct is not None:
            source_mask = is_direct
            query_mask = ~is_direct
            if not np.any(source_mask) or not np.any(query_mask):
                return object_ids

            active_layers = np.intersect1d(
                layer_ids[source_mask],
                layer_ids[query_mask],
                assume_unique=False,
            )
            if len(active_layers) == 0:
                return object_ids

            active_indices = np.flatnonzero(np.isin(layer_ids, active_layers))
            layer_order = active_indices[
                np.argsort(layer_ids[active_indices], kind='stable')
            ]
        else:
            layer_order = np.argsort(layer_ids, kind='stable')

        sorted_layer_ids = layer_ids[layer_order]
        unique_layers, starts = np.unique(sorted_layer_ids, return_index=True)
        ends = np.r_[starts[1:], len(layer_order)]

        for _, start, end in zip(unique_layers, starts, ends):
            layer_indices = layer_order[start:end]

            if is_direct is not None:
                layer_direct = is_direct[layer_indices]
                has_query = np.any(~layer_direct)
                has_source = np.any(layer_direct)
                if not has_query or not has_source:
                    continue

                if self.use_depth_constraint:
                    source_ranges = ranges[layer_indices[layer_direct]]
                    query_ranges = ranges[layer_indices[~layer_direct]]
                    if not self._has_depth_feasible_pair(
                        source_ranges, query_ranges
                    ):
                        continue

            if len(layer_indices) < self.dbscan_min_samples:
                object_ids[layer_indices] = global_object_id
                global_object_id += 1
                continue

            layer_points = xyz[layer_indices]
            if (
                is_direct is not None
                and self.use_depth_constraint
                and self._queries_are_locally_unambiguous(
                    layer_points, layer_direct, ranges[layer_indices]
                )
            ):
                object_ids[layer_indices] = global_object_id
                global_object_id += 1
                continue

            labels = self._full_dbscan_labels(layer_points)
            for label in np.unique(labels):
                cluster_mask = labels == label
                object_ids[layer_indices[cluster_mask]] = global_object_id
                global_object_id += 1

        return object_ids

    def _full_dbscan_labels(self, points):
        """Returns DBSCAN labels for one depth layer."""
        clustering = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            algorithm=self.dbscan_algorithm,
        ).fit(points)
        return clustering.labels_

    def _queries_are_locally_unambiguous(self, points, is_direct, ranges):
        """Checks whether DBSCAN can be skipped for this depth layer."""
        source_points = points[is_direct]
        query_points = points[~is_direct]
        if len(source_points) == 0 or len(query_points) == 0:
            return True

        source_ranges = ranges[is_direct]
        query_ranges = ranges[~is_direct]
        k = min(self.k_neighbors, len(source_points))
        distances, indices = self._query_knn(
            source_points,
            query_points,
            k,
            brute_force_threshold=self.unambiguous_brute_force_knn_threshold,
        )
        if k == 1:
            distances = distances[:, np.newaxis]
            indices = indices[:, np.newaxis]

        for row in range(len(query_points)):
            candidate_idx = indices[row]
            candidate_dist = distances[row]
            depth_valid = (
                np.abs(source_ranges[candidate_idx] - query_ranges[row])
                < self.tau_depth
            )
            if not np.any(depth_valid):
                continue

            valid_idx = candidate_idx[depth_valid]
            valid_dist = candidate_dist[depth_valid]
            weights = np.exp(
                -(valid_dist ** 2) / (2 * self.sigma_spatial ** 2)
            )
            total_weight = float(weights.sum())
            if total_weight < self.unambiguous_min_weight or len(valid_idx) <= 1:
                continue

            valid_points = source_points[valid_idx]
            distances_to_first = np.linalg.norm(
                valid_points - valid_points[0], axis=1
            )
            if np.all(distances_to_first <= self.unambiguous_component_eps):
                continue

            component_ids = self._candidate_components(valid_points)
            component_weights = np.bincount(component_ids, weights=weights)
            dominant_weight = (
                float(component_weights.max()) if len(component_weights) else 0.0
            )
            if dominant_weight / (total_weight + 1e-8) < (
                self.unambiguous_weight_ratio
            ):
                return False

        return True

    def _candidate_components(self, points):
        """Computes connected components among a small candidate set."""
        num_points = len(points)
        if num_points <= 1:
            return np.zeros(num_points, dtype=np.int32)

        distances = np.linalg.norm(
            points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=-1
        )
        adjacency = distances <= self.unambiguous_component_eps
        component_ids = np.full(num_points, -1, dtype=np.int32)
        component_id = 0
        for point_idx in range(num_points):
            if component_ids[point_idx] >= 0:
                continue

            component_ids[point_idx] = component_id
            stack = [point_idx]
            while stack:
                current = stack.pop()
                for neighbor in np.flatnonzero(adjacency[current]):
                    if component_ids[neighbor] < 0:
                        component_ids[neighbor] = component_id
                        stack.append(int(neighbor))
            component_id += 1
        return component_ids

    def _has_depth_feasible_pair(self, source_ranges, query_ranges):
        """Returns True if any source/query pair can pass the depth gate."""
        if len(source_ranges) == 0 or len(query_ranges) == 0:
            return False

        source_ranges = np.sort(source_ranges)
        insert_pos = np.searchsorted(source_ranges, query_ranges)
        left_pos = np.clip(insert_pos - 1, 0, len(source_ranges) - 1)
        right_pos = np.clip(insert_pos, 0, len(source_ranges) - 1)
        nearest_diff = np.minimum(
            np.abs(query_ranges - source_ranges[left_pos]),
            np.abs(query_ranges - source_ranges[right_pos]),
        )
        return bool(np.any(nearest_diff < self.tau_depth))

    @staticmethod
    def _group_indices_by_object(object_ids, selection_mask):
        """Groups selected point indices by object id."""
        selected = np.flatnonzero(selection_mask)
        if len(selected) == 0:
            return {}

        selected_obj_ids = object_ids[selected]
        valid = selected_obj_ids >= 0
        selected = selected[valid]
        selected_obj_ids = selected_obj_ids[valid]
        if len(selected) == 0:
            return {}

        order = np.argsort(selected_obj_ids, kind='stable')
        sorted_obj_ids = selected_obj_ids[order]
        sorted_indices = selected[order]
        unique_obj_ids, starts = np.unique(sorted_obj_ids, return_index=True)
        ends = np.r_[starts[1:], len(sorted_indices)]
        return {
            int(obj_id): sorted_indices[start:end]
            for obj_id, start, end in zip(unique_obj_ids, starts, ends)
        }

    def _query_knn(self, source_xyz, query_xyz, k, brute_force_threshold=None):
        """Queries nearest source points for each query point."""
        if len(source_xyz) == 0 or len(query_xyz) == 0 or k == 0:
            return (
                np.zeros((len(query_xyz), k), dtype=np.float64),
                np.zeros((len(query_xyz), k), dtype=np.int64),
            )

        if brute_force_threshold is None:
            brute_force_threshold = self.brute_force_knn_threshold

        if len(source_xyz) * len(query_xyz) <= brute_force_threshold:
            diff = (
                query_xyz.astype(np.float64)[:, np.newaxis, :]
                - source_xyz.astype(np.float64)[np.newaxis, :, :]
            )
            dist_sq = np.einsum('qkd,qkd->qk', diff, diff)
            if k == len(source_xyz):
                nearest = np.argsort(dist_sq, axis=1, kind='stable')[:, :k]
            else:
                nearest = np.argpartition(dist_sq, kth=k - 1, axis=1)[:, :k]
                nearest_dist = np.take_along_axis(dist_sq, nearest, axis=1)
                order = np.argsort(nearest_dist, axis=1, kind='stable')
                nearest = np.take_along_axis(nearest, order, axis=1)
            distances = np.sqrt(np.take_along_axis(dist_sq, nearest, axis=1))
            if k == 1:
                return distances[:, 0], nearest[:, 0]
            return distances, nearest

        tree = cKDTree(source_xyz)
        return tree.query(query_xyz, k=k)

    # ------------------------------------------------------------------ #
    #  Stage 4: depth-constrained colour propagation (Eqs. 8-10)         #
    # ------------------------------------------------------------------ #

    def _propagate_segmented_depth_constrained(self, xyz, colors, confidence,
                                               is_direct, object_ids):
        """Per-DBSCAN-cluster propagation with an additional depth gate.

        This is the configuration used by the full HC-MVMM model.
        """
        colors = colors.copy()
        propagated_confidence = np.zeros(len(xyz), dtype=np.float32)

        source_groups = self._group_indices_by_object(object_ids, is_direct)
        query_groups = self._group_indices_by_object(object_ids, ~is_direct)
        for obj_id, query_indices in query_groups.items():
            source_indices = source_groups.get(obj_id)
            if source_indices is None or len(source_indices) == 0:
                continue

            colors, propagated_confidence = self._propagate_indices(
                xyz, colors, confidence, source_indices, query_indices,
                propagated_confidence, use_depth_gate=True
            )

        return colors, propagated_confidence

    def _propagate_depth_constrained(self, xyz, colors, confidence, is_direct):
        """Global depth-gated propagation without DBSCAN segmentation."""
        colors = colors.copy()
        propagated_confidence = np.zeros(len(xyz), dtype=np.float32)

        source_indices = np.flatnonzero(is_direct)
        query_indices = np.flatnonzero(~is_direct)
        if len(source_indices) == 0 or len(query_indices) == 0:
            return colors, propagated_confidence

        return self._propagate_indices(
            xyz, colors, confidence, source_indices, query_indices,
            propagated_confidence, use_depth_gate=True
        )

    def _propagate_colors_with_confidence(self, xyz, colors, confidence,
                                          is_direct, object_ids):
        """Per-object propagation without depth gating (Eqs. 8-10).

        Returns:
            Tuple ``(colors, propagated_confidence)``.
        """
        colors = colors.copy()
        propagated_confidence = np.zeros(len(xyz), dtype=np.float32)

        source_groups = self._group_indices_by_object(object_ids, is_direct)
        query_groups = self._group_indices_by_object(object_ids, ~is_direct)
        for obj_id, query_indices in query_groups.items():
            source_indices = source_groups.get(obj_id)
            if source_indices is None or len(source_indices) == 0:
                continue

            colors, propagated_confidence = self._propagate_indices(
                xyz, colors, confidence, source_indices, query_indices,
                propagated_confidence, use_depth_gate=False
            )

        return colors, propagated_confidence

    def _propagate_indices(self, xyz, colors, confidence, source_indices,
                           query_indices, propagated_confidence,
                           use_depth_gate):
        """KNN propagation from source indices to query indices.

        Args:
            xyz: Array of shape ``[N, 3]``.
            colors: Array of shape ``[N, 3]`` updated in place for query points.
            confidence: Array of shape ``[N]`` of source confidences.
            source_indices: Integer array selecting direct seed points.
            query_indices: Integer array selecting points to fill.
            propagated_confidence: Output buffer of shape ``[N]`` updated in place.
            use_depth_gate: If True, only neighbours with ``|d_i - d_j| < tau``
                contribute (the ``1_{...}`` indicator in Eq. 8).

        Returns:
            Tuple ``(colors, propagated_confidence)`` (same buffers).
        """
        if len(source_indices) == 0 or len(query_indices) == 0:
            return colors, propagated_confidence

        source_xyz = xyz[source_indices]
        source_colors = colors[source_indices]
        source_confidence = confidence[source_indices]
        query_xyz = xyz[query_indices]

        k = min(self.k_neighbors, len(source_xyz))
        if k == 0:
            return colors, propagated_confidence

        distances, indices = self._query_knn(source_xyz, query_xyz, k)
        if k == 1:
            distances = distances[:, np.newaxis]
            indices = indices[:, np.newaxis]

        # GaussianLSS-style weights: w_ij = r_j * exp(-d_ij^2 / 2*sigma^2).
        density = np.exp(-distances ** 2 / (2 * self.sigma_spatial ** 2))
        weights = source_confidence[indices] * density

        if use_depth_gate:
            source_depth = np.linalg.norm(source_xyz, axis=1)
            query_depth = np.linalg.norm(query_xyz, axis=1)
            neighbor_depth = source_depth[indices]
            depth_diff = np.abs(neighbor_depth - query_depth[:, np.newaxis])
            depth_valid = (depth_diff < self.tau_depth).astype(np.float32)
            weights = weights * depth_valid
        else:
            depth_valid = np.ones_like(density)

        neighbor_colors = source_colors[indices]
        weight_sum = weights.sum(axis=1, keepdims=True) + 1e-8
        propagated_colors = (
            (weights[:, :, np.newaxis] * neighbor_colors).sum(axis=1) / weight_sum
        )

        # Propagated confidence (Eq. 10) or variance-based alternative.
        if self.prop_conf_method == 'variance':
            mean_color = propagated_colors
            color_diff_sq = ((neighbor_colors - mean_color[:, np.newaxis, :]) ** 2).sum(axis=-1)
            weighted_variance = (
                (weights * color_diff_sq).sum(axis=1) / (weight_sum.squeeze() + 1e-6)
            )
            uncertainty = np.sqrt(weighted_variance)
            propagated_conf = np.exp(-uncertainty / self.variance_scale)
        else:
            conf_sum = (source_confidence[indices] * depth_valid).sum(axis=1)
            weight_sum_local = weights.sum(axis=1)
            propagated_conf = np.clip(weight_sum_local / (conf_sum + 1e-6), 0, 1)

        if use_depth_gate:
            has_valid = weights.sum(axis=1) > 1e-6
            propagated_conf = np.where(has_valid, propagated_conf, 0.0)
        else:
            has_valid = np.ones(weights.shape[0], dtype=bool)

        colors[query_indices[has_valid]] = propagated_colors[has_valid]
        propagated_confidence[query_indices] = propagated_conf
        return colors, propagated_confidence

    # ------------------------------------------------------------------ #
    #  Fallback: plain projection (used when ``enabled=False``)          #
    # ------------------------------------------------------------------ #

    def _fallback_projection(self, points, image, calib):
        """Plain projective colorization without occlusion handling.

        Args:
            points: Array of shape ``[N, 4]`` ``(x, y, z, intensity)``.
            image: Array of shape ``[H, W, 3]``.
            calib: A :class:`Calibration` instance.

        Returns:
            Array of shape ``[N, 8]`` with confidence 1.0 for in-image points
            and 0.5 elsewhere.
        """
        xyz = points[:, :3]
        H, W = image.shape[:2]

        pts_img, pts_depth = calib.lidar_to_img(xyz)
        pts_img_int = pts_img.astype(np.int32)
        valid = (
            (pts_depth > 0)
            & (pts_img_int[:, 0] >= 0) & (pts_img_int[:, 0] < W)
            & (pts_img_int[:, 1] >= 0) & (pts_img_int[:, 1] < H)
        )

        N = len(points)
        rgb = np.tile(self.default_color, (N, 1))
        rgb[valid] = image[pts_img_int[valid, 1], pts_img_int[valid, 0]]

        confidence = np.full(N, 0.5, dtype=np.float32)
        confidence[valid] = 1.0
        return np.concatenate([points, rgb, confidence[:, np.newaxis]], axis=1)
