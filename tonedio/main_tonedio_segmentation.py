from __future__ import annotations

from pathlib import Path
import argparse
import math
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for import_path in (REPO_ROOT, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import cv2
import numpy as np
import torch
from pymavlink import mavutil

from example.timesync import TimeSync
from tonedio.main_tonedio import (
    DepthGateRacer,
    build_args as build_base_args,
    normalize,
    quaternion_to_rotation_matrix,
    sim_to_normal,
    sim_to_normal_rotation,
    stop_and_join,
)
from tonedio.mavlink_rx import MAVLinkRX
from tonedio.utils import airsim_to_normal_vector
from tonedio.vision_rx import VisionRX


REAL_GATE_WIDTH_M = 2.7
REAL_GATE_HEIGHT_M = 2.7
REAL_GATE_DEPTH_M = 0.26
GATE_EDGE_TYPES = (("TL", "TR"), ("TR", "BR"), ("BR", "BL"), ("BL", "TL"))


def build_args():
    extras = argparse.ArgumentParser(add_help=False)
    extras.add_argument("--seg_gate_crop_width", type=int, default=640)
    extras.add_argument("--seg_gate_crop_height", type=int, default=360)
    extras.add_argument("--seg_gate_input_width", type=int, default=640)
    extras.add_argument("--seg_gate_input_height", type=int, default=360)
    extras.add_argument("--seg_gate_real_width_m", type=float, default=REAL_GATE_WIDTH_M)
    extras.add_argument("--seg_gate_real_height_m", type=float, default=REAL_GATE_HEIGHT_M)
    extras.add_argument("--seg_gate_depth_scale", type=float, default=1)
    extras.add_argument("--seg_cached_gate_switch_distance_m", type=float, default=2.0)
    extras.add_argument("--save_segmentation_dir", type=str, default="")
    extras.add_argument("--save_segmentation_every", type=int, default=1)
    extra_args, remaining = extras.parse_known_args()
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining]
        args = build_base_args()
    finally:
        sys.argv = original_argv

    args.seg_gate_crop_width = int(extra_args.seg_gate_crop_width)
    args.seg_gate_crop_height = int(extra_args.seg_gate_crop_height)
    args.seg_gate_input_width = extra_args.seg_gate_input_width
    args.seg_gate_input_height = extra_args.seg_gate_input_height
    args.seg_gate_real_width_m = float(extra_args.seg_gate_real_width_m)
    args.seg_gate_real_height_m = float(extra_args.seg_gate_real_height_m)
    args.seg_gate_depth_scale = float(extra_args.seg_gate_depth_scale)
    args.seg_cached_gate_switch_distance_m = float(extra_args.seg_cached_gate_switch_distance_m)
    args.save_segmentation_dir = str(extra_args.save_segmentation_dir)
    args.save_segmentation_every = int(extra_args.save_segmentation_every)
    return args


class SegmentationGateDetector:
    def __init__(
        self,
        *,
        sat_min=35,
        val_min=20,
        hue_low_max=10,
        hue_high_min=160,
        close_kernel=5,
        open_kernel=3,
        min_area=20.0,
        square_ratio_tol=0.3,
        min_edge_coverage=0.8,
        edge_samples=30,
        edge_thickness=2,
        max_gate_depth_m=25.0,
        free_select_depth_m=4.0,
        depth_consistency_tol_m=1.5,
        real_gate_width_m=REAL_GATE_WIDTH_M,
        real_gate_height_m=REAL_GATE_HEIGHT_M,
        depth_scale=1.0,
        camera_pose=None,
    ):
        self.sat_min = int(sat_min)
        self.val_min = int(val_min)
        self.hue_low_max = int(hue_low_max)
        self.hue_high_min = int(hue_high_min)
        self.close_kernel = int(close_kernel)
        self.open_kernel = int(open_kernel)
        self.min_area = float(min_area)
        self.square_ratio_tol = float(square_ratio_tol)
        self.min_edge_coverage = float(min_edge_coverage)
        self.edge_samples = int(edge_samples)
        self.edge_thickness = int(edge_thickness)
        self.max_gate_depth_m = float(max_gate_depth_m)
        self.free_select_depth_m = float(free_select_depth_m)
        self.depth_consistency_tol_m = float(depth_consistency_tol_m)
        self.real_gate_width_m = float(real_gate_width_m)
        self.real_gate_height_m = float(real_gate_height_m)
        self.depth_scale = float(depth_scale)
        self.last_selected_gate_depth_m = None
        self.last_target_rel_drone = None
        self.last_selected_gate = None
        self.last_rejection_summary = {}
        self.last_geometry_rejected_candidates = []
        self.fixed_camera_intrinsics = None
        self.camera_intrinsics = None
        self.camera_pose = {
            "X": 0.0,
            "Y": 0.0,
            "Z": 0.0,
            "Roll": 0.0,
            "Pitch": 20.0,
            "Yaw": 0.0,
        }
        if camera_pose:
            self.camera_pose.update(camera_pose)

    def segment_red(self, image_rgb):
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        low = cv2.inRange(
            hsv,
            np.array([0, self.sat_min, self.val_min], dtype=np.uint8),
            np.array([self.hue_low_max, 255, 255], dtype=np.uint8),
        )
        high = cv2.inRange(
            hsv,
            np.array([self.hue_high_min, self.sat_min, self.val_min], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        mask = cv2.bitwise_or(low, high)
        if self.open_kernel > 0:
            kernel = np.ones((self.open_kernel, self.open_kernel), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        if self.close_kernel > 0:
            kernel = np.ones((self.close_kernel, self.close_kernel), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    @staticmethod
    def order_corners(corners):
        corners = np.asarray(corners, dtype=np.float32)
        center = np.mean(corners, axis=0)
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        ordered = corners[np.argsort(angles)]
        start = int(np.argmin(np.sum(ordered, axis=1)))
        return np.roll(ordered, -start, axis=0)

    @staticmethod
    def euler_to_rotation_matrix(roll, pitch, yaw, degrees=False):
        if degrees:
            roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
        return rz @ ry @ rx

    def sample_edge_coverage(self, mask, p0, p1):
        p0 = np.asarray(p0, dtype=np.float32)
        p1 = np.asarray(p1, dtype=np.float32)
        samples = max(2, self.edge_samples)
        radius = max(0, self.edge_thickness)
        height, width = mask.shape[:2]
        hits = 0
        for i in range(samples):
            t = float(i) / float(samples - 1)
            p = p0 * (1.0 - t) + p1 * t
            x = int(round(float(p[0])))
            y = int(round(float(p[1])))
            x0 = max(0, x - radius)
            x1 = min(width, x + radius + 1)
            y0 = max(0, y - radius)
            y1 = min(height, y + radius + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            if np.any(mask[y0:y1, x0:x1] > 0):
                hits += 1
        return float(hits) / float(samples)

    @staticmethod
    def _mask_hit(mask, point):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        height, width = mask.shape[:2]
        if x < 0 or x >= width or y < 0 or y >= height:
            return False
        return bool(mask[y, x] > 0)

    def estimate_inner_offsets(self, mask, corners):
        corners = np.asarray(corners, dtype=np.float32)
        center = np.mean(corners, axis=0)
        edge_lengths = [
            float(np.linalg.norm(corners[(i + 1) % 4] - corners[i]))
            for i in range(4)
        ]
        max_search = max(2, int(round(0.45 * min(edge_lengths))))
        offsets = []
        for i in range(4):
            p0 = corners[i]
            p1 = corners[(i + 1) % 4]
            midpoint = (p0 + p1) * 0.5
            inward = center - midpoint
            inward_norm = float(np.linalg.norm(inward))
            if inward_norm < 1e-6:
                offsets.append(0.0)
                continue
            inward = inward / inward_norm

            distances = []
            for t in np.linspace(0.15, 0.85, 9, dtype=np.float32):
                start = p0 * (1.0 - float(t)) + p1 * float(t)
                saw_frame = False
                for distance in range(max_search + 1):
                    point = start + inward * float(distance)
                    if self._mask_hit(mask, point):
                        saw_frame = True
                    elif saw_frame:
                        distances.append(float(distance))
                        break
            if distances:
                offsets.append(float(np.median(distances)))
            else:
                offsets.append(0.0)
        return offsets

    def build_inner_corners(self, mask, outer_corners):
        outer_corners = np.asarray(outer_corners, dtype=np.float32)
        center = np.mean(outer_corners, axis=0)
        offsets = self.estimate_inner_offsets(mask, outer_corners)
        if max(offsets) <= 0.0:
            return outer_corners.copy(), offsets

        inner_edges = []
        for i in range(4):
            p0 = outer_corners[i]
            p1 = outer_corners[(i + 1) % 4]
            edge = p1 - p0
            midpoint = (p0 + p1) * 0.5
            inward = center - midpoint
            inward_norm = float(np.linalg.norm(inward))
            if inward_norm < 1e-6:
                shifted_p0 = p0
                shifted_p1 = p1
            else:
                inward = inward / inward_norm
                shifted_p0 = p0 + inward * float(offsets[i])
                shifted_p1 = p1 + inward * float(offsets[i])
            inner_edges.append((shifted_p0, edge))

        inner_corners = []
        for i in range(4):
            p_a, d_a = inner_edges[i]
            p_b, d_b = inner_edges[(i + 1) % 4]
            matrix = np.array([d_a, -d_b], dtype=np.float32).T
            rhs = p_b - p_a
            det = float(np.linalg.det(matrix))
            if abs(det) < 1e-6:
                inner_corners.append(outer_corners[(i + 1) % 4].copy())
                continue
            t, _ = np.linalg.solve(matrix, rhs)
            inner_corners.append((p_a + d_a * float(t)).astype(np.float32))

        inner_corners = self.order_corners(np.asarray(inner_corners, dtype=np.float32))
        inner_center = np.mean(inner_corners, axis=0)
        if not np.all(np.isfinite(inner_corners)):
            return outer_corners.copy(), offsets
        if cv2.contourArea(inner_corners) <= 1e-6:
            return outer_corners.copy(), offsets
        if np.max(np.linalg.norm(inner_corners - inner_center, axis=1)) >= np.max(
            np.linalg.norm(outer_corners - center, axis=1)
        ):
            return outer_corners.copy(), offsets
        return inner_corners, offsets

    def split_touching_contours(self, mask):
        binary = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        split_contours = []
        split_count = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            shifted = contour - np.array([[[x, y]]], dtype=contour.dtype)
            cv2.drawContours(roi_mask, [shifted], -1, 255, thickness=-1)

            area = float(cv2.contourArea(contour))
            if area < max(1.0, self.min_area * 2.0):
                split_contours.append(contour)
                continue

            distance = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
            max_distance = float(distance.max())
            if max_distance <= 1e-6:
                split_contours.append(contour)
                continue

            peaks = np.zeros_like(roi_mask)
            peaks[distance >= max_distance * 0.45] = 255
            peak_kernel = np.ones((3, 3), dtype=np.uint8)
            peaks = cv2.morphologyEx(peaks, cv2.MORPH_OPEN, peak_kernel)
            marker_count, markers = cv2.connectedComponents(peaks)
            if marker_count <= 2:
                split_contours.append(contour)
                continue

            markers = markers.astype(np.int32)
            unknown = cv2.subtract(roi_mask, peaks)
            markers[unknown > 0] = 0
            watershed_image = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
            cv2.watershed(watershed_image, markers)

            local_parts = []
            for label in range(1, marker_count):
                part_mask = np.zeros_like(roi_mask)
                part_mask[markers == label] = 255
                part_contours, _ = cv2.findContours(
                    part_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for part in part_contours:
                    if cv2.contourArea(part) < self.min_area:
                        continue
                    local_parts.append(part + np.array([[[x, y]]], dtype=part.dtype))

            if len(local_parts) >= 2:
                split_contours.extend(local_parts)
                split_count += len(local_parts) - 1
            else:
                split_contours.append(contour)

        return split_contours, split_count

    def detect_biggest_gate(self, mask):
        contours, split_count = self.split_touching_contours(mask)
        candidates = []
        geometry_rejected_candidates = []
        rejection_counts = {
            "small_area_or_few_points": 0,
            "invalid_rect": 0,
            "invalid_size": 0,
            "not_square": 0,
            "incomplete_edges": 0,
        }
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area or contour.shape[0] < 4:
                rejection_counts["small_area_or_few_points"] += 1
                continue
            rect = cv2.minAreaRect(contour)
            corners = self.order_corners(cv2.boxPoints(rect))
            rect_area = float(cv2.contourArea(corners))
            if rect_area <= 1e-6:
                rejection_counts["invalid_rect"] += 1
                continue
            center = np.mean(corners, axis=0)
            rejected_candidate = {
                "points": {
                    "TL": corners[0].copy(),
                    "TR": corners[1].copy(),
                    "BR": corners[2].copy(),
                    "BL": corners[3].copy(),
                },
                "center": center.astype(np.float32),
                "rect_area": rect_area,
                "mask_area": area,
            }
            edge_lengths = [
                float(np.linalg.norm(corners[(i + 1) % 4] - corners[i]))
                for i in range(4)
            ]
            width_px = max(edge_lengths[0], edge_lengths[2])
            height_px = max(edge_lengths[1], edge_lengths[3])
            rejected_candidate["size"] = np.array([width_px, height_px], dtype=np.float32)
            if width_px <= 1e-6 or height_px <= 1e-6:
                rejection_counts["invalid_size"] += 1
                rejected_candidate["geometry_rejection_reason"] = "invalid_size"
                geometry_rejected_candidates.append(rejected_candidate)
                continue
            square_ratio = min(width_px, height_px) / max(width_px, height_px)
            rejected_candidate["square_ratio"] = square_ratio
            min_square_ratio = max(0.0, min(1.0, 1.0 - self.square_ratio_tol))
            if square_ratio < min_square_ratio:
                rejection_counts["not_square"] += 1
                rejected_candidate["geometry_rejection_reason"] = "not_square"
                geometry_rejected_candidates.append(rejected_candidate)
                continue
            edge_coverages = [
                self.sample_edge_coverage(mask, corners[i], corners[(i + 1) % 4])
                for i in range(4)
            ]
            rejected_candidate["edge_coverages"] = edge_coverages
            if min(edge_coverages) < self.min_edge_coverage:
                rejection_counts["incomplete_edges"] += 1
                rejected_candidate["geometry_rejection_reason"] = "incomplete_edges"
                geometry_rejected_candidates.append(rejected_candidate)
                continue
            inner_corners, inner_offsets = self.build_inner_corners(mask, corners)
            inner_center = np.mean(inner_corners, axis=0)
            inner_edge_lengths = [
                float(np.linalg.norm(inner_corners[(i + 1) % 4] - inner_corners[i]))
                for i in range(4)
            ]
            inner_width_px = max(inner_edge_lengths[0], inner_edge_lengths[2])
            inner_height_px = max(inner_edge_lengths[1], inner_edge_lengths[3])
            inner_rect_area = float(cv2.contourArea(inner_corners))
            candidates.append(
                {
                    "points": {
                        "TL": inner_corners[0].copy(),
                        "TR": inner_corners[1].copy(),
                        "BR": inner_corners[2].copy(),
                        "BL": inner_corners[3].copy(),
                    },
                    "center": inner_center.astype(np.float32),
                    "size": np.array([inner_width_px, inner_height_px], dtype=np.float32),
                    "outer_points": {
                        "TL": corners[0].copy(),
                        "TR": corners[1].copy(),
                        "BR": corners[2].copy(),
                        "BL": corners[3].copy(),
                    },
                    "outer_center": center.astype(np.float32),
                    "outer_size": np.array([width_px, height_px], dtype=np.float32),
                    "confidence": area / rect_area,
                    "gate_score": inner_rect_area,
                    "rect_area": inner_rect_area,
                    "outer_rect_area": rect_area,
                    "mask_area": area,
                    "square_ratio": square_ratio,
                    "edge_coverages": edge_coverages,
                    "inner_offsets_px": np.array(inner_offsets, dtype=np.float32),
                }
            )
        candidates.sort(key=lambda item: float(item["rect_area"]), reverse=True)
        self.last_rejection_summary = {
            "contours": len(contours),
            "geometry_rejections": rejection_counts,
            "geometry_candidates": len(candidates),
            "split_contours_added": split_count,
        }
        self.last_geometry_rejected_candidates = geometry_rejected_candidates[:10]
        return candidates[0] if candidates else None, candidates

    def filter_gates_by_depth(self, candidates, intr):
        filtered = []
        depth_rejections = {
            "too_far": 0,
            "depth_jump_rejected": 0,
        }
        rejected_candidates = []
        for candidate in candidates:
            depth_m = self.estimate_depth_from_gate_size(candidate, intr)
            candidate["estimated_depth_m"] = depth_m
            if not np.isfinite(depth_m) or depth_m > self.max_gate_depth_m:
                candidate["depth_filter_reason"] = "too_far"
                depth_rejections["too_far"] += 1
                rejected_candidates.append(candidate)
                continue

            if (
                self.last_selected_gate_depth_m is not None
                and float(self.last_selected_gate_depth_m) < self.free_select_depth_m
            ):
                candidate["previous_gate_depth_m"] = float(self.last_selected_gate_depth_m)
                candidate["depth_filter_reason"] = "previous_close_free_select"
                filtered.append(candidate)
                continue

            if self.last_selected_gate_depth_m is None:
                candidate["depth_filter_reason"] = "no_previous_depth"
                filtered.append(candidate)
                continue

            depth_diff = abs(depth_m - float(self.last_selected_gate_depth_m))
            candidate["previous_gate_depth_m"] = float(self.last_selected_gate_depth_m)
            candidate["depth_difference_from_previous_m"] = depth_diff
            if depth_diff <= self.depth_consistency_tol_m:
                candidate["depth_filter_reason"] = "consistent_with_previous"
                filtered.append(candidate)
            else:
                candidate["depth_filter_reason"] = "depth_jump_rejected"
                depth_rejections["depth_jump_rejected"] += 1
                rejected_candidates.append(candidate)
        filtered.sort(key=lambda item: float(item["rect_area"]), reverse=True)
        self.last_rejection_summary.update(
            {
                "depth_rejections": depth_rejections,
                "depth_candidates": len(filtered),
                "depth_rejected_candidates": [
                    {
                        "reason": candidate.get("depth_filter_reason"),
                        "depth_m": candidate.get("estimated_depth_m"),
                        "previous_depth_m": candidate.get("previous_gate_depth_m"),
                        "depth_difference_m": candidate.get("depth_difference_from_previous_m"),
                        "max_depth_m": self.max_gate_depth_m,
                        "depth_consistency_tol_m": self.depth_consistency_tol_m,
                    }
                    for candidate in rejected_candidates[:3]
                ],
            }
        )
        return filtered

    def estimate_depth_from_gate_size(self, gate, intr):
        # Use the inner opening box for depth so the projection matches the
        # same geometry used for the target center.
        depth_size = gate.get("size")
        if depth_size is None:
            depth_size = gate.get("outer_size")
        width_px = float(depth_size[0])
        height_px = float(depth_size[1])
        depth_candidates = []
        raw_width_depth_m = np.inf
        raw_height_depth_m = np.inf
        if width_px > 1e-6:
            raw_width_depth_m = float(intr["fx"]) * self.real_gate_width_m / width_px
            depth_candidates.append(raw_width_depth_m)
        if height_px > 1e-6:
            raw_height_depth_m = float(intr["fy"]) * self.real_gate_height_m / height_px
            depth_candidates.append(raw_height_depth_m)
        gate["raw_depth_from_width_m"] = raw_width_depth_m
        gate["raw_depth_from_height_m"] = raw_height_depth_m
        gate["depth_from_width_m"] = raw_width_depth_m * self.depth_scale
        gate["depth_from_height_m"] = raw_height_depth_m * self.depth_scale
        gate["gate_depth_scale"] = self.depth_scale
        if not depth_candidates:
            return np.inf
        return float(np.mean(depth_candidates) * self.depth_scale)

    def cached_gate_response(self, mask, reason):
        if self.last_target_rel_drone is None:
            return None, {
                "segmentation_mask": mask,
                "segmentation_blob_selection": reason,
                "segmentation_blob_count": 0,
                "gate_detection_target_cache_used": False,
                "gate_rejection_summary": self.last_rejection_summary,
            }

        aux = {
            "segmentation_mask": mask,
            "segmentation_blob_selection": reason,
            "segmentation_blob_count": 0,
            "gate_detection_target_cache_used": True,
            "gate_detection_target_cache_reason": reason,
            "gate_detection_target_rel_drone": self.last_target_rel_drone.astype(np.float32),
            "gate_depth_m": self.last_selected_gate_depth_m,
            "gate_depth_source": "cached_segmentation_gate",
            "gate_rejection_summary": self.last_rejection_summary,
            "geometry_rejected_candidates": self.last_geometry_rejected_candidates,
        }
        if self.last_selected_gate is not None:
            aux.update(
                {
                    "segmentation_rect": self.last_selected_gate,
                    "segmentation_primary_rect": self.last_selected_gate,
                    "segmentation_backup_rect": self.last_selected_gate,
                    "segmentation_rect_size_px": self.last_selected_gate.get("size"),
                    "gate_center_px": self.last_selected_gate.get("center"),
                    "gate_confidence": self.last_selected_gate.get("confidence"),
                    "gate_square_ratio": self.last_selected_gate.get("square_ratio"),
                    "gate_edge_coverages": self.last_selected_gate.get("edge_coverages"),
                    "gate_corner_points_px": self.last_selected_gate.get("points"),
                    "corner_gate_candidates": [self.last_selected_gate],
                    "corner_gate_candidates_all": [self.last_selected_gate],
                    "geometry_rejected_candidates": self.last_geometry_rejected_candidates,
                    "corner_gate_count": 1,
                    "corner_gate_detected_count": 1,
                }
            )
        return self.last_target_rel_drone.astype(np.float32), aux

    def estimate_target_point_airsim(self, rgb, depth=None):
        image_rgb = np.asarray(rgb, dtype=np.uint8)
        height, width = image_rgb.shape[:2]
        mask = self.segment_red(image_rgb)
        selected, candidates = self.detect_biggest_gate(mask)
        geometry_candidates = list(candidates)
        if selected is None:
            return self.cached_gate_response(mask, "cached_no_gate_detected")

        intr = self.fixed_camera_intrinsics or self.camera_intrinsics
        if intr is None:
            f = float(width) / (2.0 * math.tan(math.radians(90.0) * 0.5))
            intr = {
                "fx": f,
                "fy": f,
                "cx": float(width) * 0.5,
                "cy": float(height) * 0.5,
            }

        max_depth_candidates = []
        for candidate in geometry_candidates:
            depth_m_for_backup = self.estimate_depth_from_gate_size(candidate, intr)
            candidate["estimated_depth_m"] = depth_m_for_backup
            if np.isfinite(depth_m_for_backup) and depth_m_for_backup <= self.max_gate_depth_m:
                max_depth_candidates.append(candidate)
        max_depth_candidates.sort(key=lambda item: float(item["rect_area"]), reverse=True)
        backup = max_depth_candidates[1] if len(max_depth_candidates) > 1 else None

        candidates = self.filter_gates_by_depth(candidates, intr)
        selected = candidates[0] if candidates else None
        if selected is None:
            target_rel_drone, aux = self.cached_gate_response(mask, "cached_all_candidates_rejected")
            aux.update(
                {
                    "corner_gate_candidates_all": geometry_candidates,
                    "geometry_rejected_candidates": self.last_geometry_rejected_candidates,
                    "corner_gate_detected_count": len(geometry_candidates),
                    "gate_depth_source": "cached_after_rejected_far_gate",
                    "gate_max_depth_m": self.max_gate_depth_m,
                    "gate_free_select_depth_m": self.free_select_depth_m,
                    "gate_depth_consistency_tol_m": self.depth_consistency_tol_m,
                    "previous_gate_depth_m": self.last_selected_gate_depth_m,
                }
            )
            return target_rel_drone, aux

        def project_candidate(candidate):
            candidate_depth_m = self.estimate_depth_from_gate_size(candidate, intr)
            center_u, center_v = map(float, candidate["center"])
            x_off = (center_u - float(intr["cx"])) * candidate_depth_m / float(intr["fx"])
            y_off = (center_v - float(intr["cy"])) * candidate_depth_m / float(intr["fy"])
            target_rel_camera = np.array([candidate_depth_m, x_off, y_off], dtype=np.float32)

            camera = self.camera_pose
            p_camera_drone = np.array([camera["X"], camera["Y"], camera["Z"]], dtype=np.float32)
            rot_camera_drone = self.euler_to_rotation_matrix(
                camera["Roll"],
                camera["Pitch"],
                camera["Yaw"],
                degrees=True,
            )
            target_rel_drone = p_camera_drone + rot_camera_drone @ target_rel_camera
            return candidate_depth_m, target_rel_camera, target_rel_drone.astype(np.float32)

        depth_m, target_rel_camera, target_rel_drone = project_candidate(selected)
        backup_depth_m = None
        backup_target_rel_camera = None
        backup_target_rel_drone = None
        if backup is not None:
            backup_depth_m, backup_target_rel_camera, backup_target_rel_drone = project_candidate(backup)

        center_u, center_v = map(float, selected["center"])
        self.last_selected_gate_depth_m = depth_m
        self.last_target_rel_drone = target_rel_drone.astype(np.float32)
        self.last_selected_gate = selected

        aux = {
            "segmentation_mask": mask,
            "segmentation_rect": selected,
            "segmentation_primary_rect": selected,
            "segmentation_backup_rect": backup,
            "segmentation_rect_size_px": selected["size"],
            "segmentation_depth_size_px": selected["size"],
            "segmentation_blob_count": len(candidates),
            "segmentation_blob_selection": "red_mask_geometry_biggest",
            "segmentation_blob_depth_m": depth_m,
            "segmentation_backup_depth_m": backup_depth_m,
            "segmentation_primary_depth_m": depth_m,
            "gate_depth_m": depth_m,
            "gate_depth_source": "segmentation_gate_size_configured_real_size",
            "gate_depth_from_width_m": selected.get("depth_from_width_m"),
            "gate_depth_from_height_m": selected.get("depth_from_height_m"),
            "raw_gate_depth_from_width_m": selected.get("raw_depth_from_width_m"),
            "raw_gate_depth_from_height_m": selected.get("raw_depth_from_height_m"),
            "gate_depth_scale": self.depth_scale,
            "gate_center_px": selected["center"],
            "gate_confidence": selected["confidence"],
            "gate_square_ratio": selected.get("square_ratio"),
            "gate_edge_coverages": selected.get("edge_coverages"),
            "gate_max_depth_m": self.max_gate_depth_m,
            "gate_free_select_depth_m": self.free_select_depth_m,
            "gate_depth_consistency_tol_m": self.depth_consistency_tol_m,
            "previous_gate_depth_m": selected.get("previous_gate_depth_m"),
            "gate_depth_difference_from_previous_m": selected.get("depth_difference_from_previous_m"),
            "gate_depth_filter_reason": selected.get("depth_filter_reason"),
            "gate_rejection_summary": self.last_rejection_summary,
            "gate_corner_points_px": selected["points"],
            "corner_gate_candidates": candidates[:5],
            "corner_gate_candidates_all": geometry_candidates,
            "geometry_rejected_candidates": self.last_geometry_rejected_candidates,
            "corner_gate_count": len(candidates),
            "corner_gate_detected_count": len(geometry_candidates),
            "gate_real_size_m": {
                "width": self.real_gate_width_m,
                "height": self.real_gate_height_m,
                "depth": REAL_GATE_DEPTH_M,
            },
            "camera_intrinsics": intr,
            "segmentation_target_rel_camera": target_rel_camera,
            "segmentation_backup_target_rel_camera": backup_target_rel_camera,
            "gate_detection_target_rel_drone": target_rel_drone.astype(np.float32),
            "segmentation_backup_target_rel_drone": backup_target_rel_drone,
        }
        return target_rel_drone.astype(np.float32), aux


class SegmentationDepthGateRacer(DepthGateRacer):
    def __init__(self, mavlink_conn, shared_data, system_boot_ms, args):
        args.disable_depth_estimation = True
        args.disable_learned_gate_detector = True
        args.control_fake_infinite_depth = True
        super().__init__(mavlink_conn, shared_data, system_boot_ms, args)
        self.seg_debug_print = bool(getattr(args, "debug_print", False))
        self.save_segmentation_dir = None
        if getattr(args, "save_segmentation_dir", ""):
            self.save_segmentation_dir = Path(args.save_segmentation_dir)
            self.save_segmentation_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving segmentation masks to: {self.save_segmentation_dir.resolve()}", flush=True)
        self.segmentation_save_counter = 0
        self.gate_detector = SegmentationGateDetector(
            sat_min=getattr(args, "seg_sat_min", 35),
            val_min=getattr(args, "seg_val_min", 20),
            hue_low_max=getattr(args, "seg_hue_low_max", 10),
            hue_high_min=getattr(args, "seg_hue_high_min", 160),
            open_kernel=getattr(args, "seg_open_kernel", 3),
            close_kernel=getattr(args, "seg_close_kernel", 5),
            min_area=getattr(args, "seg_min_area", 20.0),
            square_ratio_tol=getattr(args, "seg_square_ratio_tol", 0.3),
            min_edge_coverage=getattr(args, "seg_min_edge_coverage", 0.5),
            edge_samples=getattr(args, "seg_edge_samples", 30),
            edge_thickness=getattr(args, "seg_edge_thickness", 2),
            max_gate_depth_m=getattr(args, "seg_max_gate_depth_m", 25.0),
            free_select_depth_m=getattr(args, "seg_free_select_depth_m", 4.0),
            depth_consistency_tol_m=getattr(args, "seg_depth_consistency_tol_m", 1.5),
            real_gate_width_m=getattr(args, "seg_gate_real_width_m", REAL_GATE_WIDTH_M),
            real_gate_height_m=getattr(args, "seg_gate_real_height_m", REAL_GATE_HEIGHT_M),
            depth_scale=getattr(args, "seg_gate_depth_scale", 1.0),
            camera_pose={
                "X": 0.0,
                "Y": 0.0,
                "Z": 0.0,
                "Roll": 0.0,
                "Pitch": 20.0,
                "Yaw": 0.0,
            },
        )

    def depth_callback(self):
        frame = self.data.get("latest_frame")
        if frame is None:
            self.debug_idle("missing_frame")
            return

        frame_id = frame.get("frame_id")
        frame_sim_time_ns = frame.get("sim_time_ns")
        if frame_id == self.last_depth_frame_id:
            return
        self.last_depth_frame_id = frame_id

        image_bgr = frame["image_bgr"]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        depth_rgb, _ = self.fit_rgb_for_depth(rgb)
        gate_rgb, gate_transform = self.fit_rgb_for_gate(rgb)
        self.save_rgb_image(gate_rgb, frame_id=frame_id)

        depth = np.full(depth_rgb.shape[:2], np.inf, dtype=np.float32)

        with self._sensor_cond:
            self.current_depth_id += 1
            depth_id = self.current_depth_id
            self.depth_buffer.append(
                {
                    "depth_id": depth_id,
                    "frame_id": frame_id,
                    "frame_sim_time_ns": frame_sim_time_ns,
                    "image_bgr": image_bgr.copy(),
                    "depth_rgb": depth_rgb.copy(),
                    "gate_rgb": gate_rgb.copy(),
                    "gate_transform": gate_transform,
                    "depth": depth.copy(),
                }
            )
            self._sensor_cond.notify_all()

        if self.args.debug_print:
            print(
                "\n[frame]",
                "depth_estimation=disabled",
                "depth_id=", depth_id,
                "frame=", frame_id,
                "frame_sim_time_ns=", frame_sim_time_ns,
                flush=True,
            )

    def draw_gate_candidate(self, overlay, gate, color, thickness=1, marker_size=7):
        if not isinstance(gate, dict):
            return
        points = gate.get("points") or {}
        for a, b in GATE_EDGE_TYPES:
            p0 = self._pixel_xy(points.get(a))
            p1 = self._pixel_xy(points.get(b))
            if p0 is not None and p1 is not None:
                cv2.line(overlay, p0, p1, color, thickness, lineType=cv2.LINE_AA)
        for point in points.values():
            xy = self._pixel_xy(point)
            if xy is not None:
                cv2.circle(overlay, xy, max(1, thickness + 1), color, -1, lineType=cv2.LINE_AA)
        center = self._pixel_xy(gate.get("center"))
        if center is not None:
            cv2.drawMarker(
                overlay,
                center,
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=marker_size,
                thickness=thickness,
                line_type=cv2.LINE_AA,
            )

    @staticmethod
    def summarize_gate_rejection(aux):
        summary = aux.get("gate_rejection_summary") if isinstance(aux, dict) else None
        if not isinstance(summary, dict):
            return "reject=unknown"

        parts = [
            f"contours={summary.get('contours', 0)}",
            f"geometry_candidates={summary.get('geometry_candidates', 0)}",
        ]

        geometry = summary.get("geometry_rejections")
        if isinstance(geometry, dict):
            geometry_nonzero = [
                f"{key}:{value}" for key, value in geometry.items() if int(value or 0) > 0
            ]
            if geometry_nonzero:
                parts.append("geometry_rejects=" + ",".join(geometry_nonzero))

        depth = summary.get("depth_rejections")
        if isinstance(depth, dict):
            depth_nonzero = [
                f"{key}:{value}" for key, value in depth.items() if int(value or 0) > 0
            ]
            if depth_nonzero:
                parts.append("depth_rejects=" + ",".join(depth_nonzero))

        rejected = summary.get("depth_rejected_candidates")
        if isinstance(rejected, list) and rejected:
            candidate_bits = []
            for candidate in rejected[:3]:
                if not isinstance(candidate, dict):
                    continue
                reason = candidate.get("reason", "unknown")
                depth_m = candidate.get("depth_m")
                previous = candidate.get("previous_depth_m")
                diff = candidate.get("depth_difference_m")
                depth_text = "--" if depth_m is None else f"{float(depth_m):.2f}m"
                bit = f"{reason}@{depth_text}"
                if previous is not None:
                    bit += f",prev={float(previous):.2f}m"
                if diff is not None:
                    bit += f",diff={float(diff):.2f}m"
                candidate_bits.append(bit)
            if candidate_bits:
                parts.append("candidates=" + ";".join(candidate_bits))

        return " ".join(parts)

    def build_rgb_overlay(self, image_rgb):
        overlay = image_rgb.copy()
        aux = self.aux or {}
        rejected_geometry = aux.get("geometry_rejected_candidates") or []
        if isinstance(rejected_geometry, dict):
            rejected_geometry = [rejected_geometry]
        for candidate in rejected_geometry:
            self.draw_gate_candidate(
                overlay,
                candidate,
                color=(239, 68, 68),
                thickness=1,
                marker_size=6,
            )

        all_candidates = aux.get("corner_gate_candidates_all") or aux.get("corner_gate_candidates") or []
        if isinstance(all_candidates, dict):
            all_candidates = [all_candidates]
        for candidate in all_candidates:
            self.draw_gate_candidate(
                overlay,
                candidate,
                color=(56, 189, 248),
                thickness=1,
                marker_size=7,
            )

        backup = aux.get("segmentation_backup_rect")
        if isinstance(backup, dict):
            self.draw_gate_candidate(
                overlay,
                backup,
                color=(168, 85, 247),
                thickness=2,
                marker_size=10,
            )

        primary = aux.get("segmentation_primary_rect") or aux.get("segmentation_rect")
        if isinstance(primary, dict):
            is_cached = bool(aux.get("gate_detection_target_cache_used"))
            color = (250, 204, 21) if is_cached else (34, 197, 85)
            self.draw_gate_candidate(
                overlay,
                primary,
                color=color,
                thickness=2,
                marker_size=11,
            )
        depth_m = aux.get("gate_depth_m")
        if isinstance(depth_m, (int, float)) and math.isfinite(float(depth_m)):
            text = f"depth {float(depth_m):.2f}m"
        else:
            text = "depth --"
        cv2.putText(
            overlay,
            text,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if aux.get("gate_detection_target_cache_used"):
            reason = str(aux.get("gate_detection_target_cache_reason", "cached"))
            cv2.putText(
                overlay,
                f"cached {reason}",
                (8, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (250, 204, 21),
                1,
                cv2.LINE_AA,
            )
        return overlay

    def show_or_save_rgb_overlay(self, image_rgb, frame_id=None, aux=None):
        if not self.args.viz_rgb and self.save_rgb_overlay_dir is None:
            return
        previous_aux = self.aux
        if aux is not None:
            self.aux = aux
        overlay_rgb = self.build_rgb_overlay(image_rgb)
        self.aux = previous_aux
        self.save_rgb_overlay(overlay_rgb, frame_id=frame_id)
        if self.args.viz_rgb:
            cv2.imshow("rgb", cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def fit_rgb_for_segmentation_gate(self, rgb):
        return self.fit_rgb_to_size(
            rgb,
            self.args.seg_gate_input_width,
            self.args.seg_gate_input_height,
            self.args.seg_gate_crop_width,
            self.args.seg_gate_crop_height,
        )

    def estimate_segmentation_gate_target_resized(self, rgb, depth):
        gate_rgb, transform = self.fit_rgb_for_segmentation_gate(rgb)
        gate_h, gate_w = gate_rgb.shape[:2]

        previous_intrinsics = getattr(self.gate_detector, "fixed_camera_intrinsics", None)
        previous_camera_intrinsics = getattr(self.gate_detector, "camera_intrinsics", None)
        self.gate_detector.fixed_camera_intrinsics = {
            "fx": self.base_camera_intrinsics["fx"] * transform.get("resize_scale_x", 1.0),
            "fy": self.base_camera_intrinsics["fy"] * transform.get("resize_scale_y", 1.0),
            "cx": (
                self.base_camera_intrinsics["cx"]
                - transform["crop_left"]
                + transform["pad_left"]
            )
            * transform.get("resize_scale_x", 1.0),
            "cy": (
                self.base_camera_intrinsics["cy"]
                - transform["crop_top"]
                + transform["pad_top"]
            )
            * transform.get("resize_scale_y", 1.0),
        }
        self.gate_detector.camera_intrinsics = None
        try:
            target_rel_drone, aux = self.gate_detector.estimate_target_point_airsim(gate_rgb, depth)
        finally:
            self.gate_detector.fixed_camera_intrinsics = previous_intrinsics
            self.gate_detector.camera_intrinsics = previous_camera_intrinsics

        aux = aux or {}
        display_aux = self.map_gate_aux_to_original_pixels(aux, transform)
        display_aux["gate_input_size"] = (int(gate_w), int(gate_h))
        display_aux["gate_fit_transform"] = transform
        display_aux["gate_model_aux"] = aux
        return target_rel_drone, display_aux

    def gate_callback(self):
        with self._sensor_cond:
            while self.is_gate_thread_active:
                if self.depth_buffer:
                    item = self.depth_buffer[-1]
                    if item["depth_id"] != self.gate_used_depth_id:
                        depth_item = {
                            "depth_id": item["depth_id"],
                            "frame_id": item["frame_id"],
                            "image_bgr": item["image_bgr"].copy(),
                            "depth": item["depth"].copy(),
                        }
                        break
                self._sensor_cond.wait(timeout=0.05)
            else:
                return

        raw_rgb = cv2.cvtColor(depth_item["image_bgr"], cv2.COLOR_BGR2RGB)
        target_rel_drone, display_aux = self.estimate_segmentation_gate_target_resized(
            raw_rgb,
            depth_item["depth"],
        )
        gate_model_aux = display_aux.get("gate_model_aux", display_aux) or {}
        backup_target_rel_drone = display_aux.get(
            "segmentation_backup_target_rel_drone",
            gate_model_aux.get("segmentation_backup_target_rel_drone")
            if isinstance(gate_model_aux, dict)
            else None,
        )

        if target_rel_drone is None:
            target_v = None
        else:
            target_v = airsim_to_normal_vector(target_rel_drone)
        gate_center = gate_model_aux.get("gate_center_px") if isinstance(gate_model_aux, dict) else None
        cache_used = bool(gate_model_aux.get("gate_detection_target_cache_used")) if isinstance(gate_model_aux, dict) else False
        cache_rank = gate_model_aux.get("gate_detection_target_cache_rank") if isinstance(gate_model_aux, dict) else None
        cache_reason = gate_model_aux.get("gate_detection_target_cache_reason") if isinstance(gate_model_aux, dict) else None
        state_snapshot = self.get_state()
        gate_world_estimate = None
        backup_gate_world_estimate = None
        if target_rel_drone is not None:
            if cache_used and self.last_gate_world_estimate is not None:
                gate_world_estimate = np.asarray(self.last_gate_world_estimate, dtype=np.float32).copy()
            else:
                gate_world_estimate = self.calculate_gate_world_estimate(target_rel_drone, state_snapshot)
                if gate_world_estimate is not None and not cache_used:
                    self.last_gate_world_estimate = gate_world_estimate.copy()
        if backup_target_rel_drone is not None:
            if cache_used and self.last_backup_gate_world_estimate is not None:
                backup_gate_world_estimate = np.asarray(
                    self.last_backup_gate_world_estimate,
                    dtype=np.float32,
                ).copy()
            else:
                backup_gate_world_estimate = self.calculate_gate_world_estimate(
                    backup_target_rel_drone,
                    state_snapshot,
                )
                if backup_gate_world_estimate is not None and not cache_used:
                    self.last_backup_gate_world_estimate = backup_gate_world_estimate.copy()
        elif self.last_backup_gate_world_estimate is not None:
            backup_gate_world_estimate = np.asarray(
                self.last_backup_gate_world_estimate,
                dtype=np.float32,
            ).copy()

        with self._sensor_cond:
            self.target_v = target_v
            self.target_info = {
                "source": "gate" if target_rel_drone is not None else "none",
                "depth_id": depth_item["depth_id"],
                "frame_id": depth_item["frame_id"],
                "pose_source": None if state_snapshot is None else state_snapshot.get("source"),
                "pose_frame_id": None if state_snapshot is None else state_snapshot.get("frame_id"),
                "pose_child_frame_id": None if state_snapshot is None else state_snapshot.get("child_frame_id"),
                "pose_time_boot_us": None if state_snapshot is None else state_snapshot.get("time_boot_us"),
                "pose_time_boot_ms": None if state_snapshot is None else state_snapshot.get("time_boot_ms"),
                "cache_used": cache_used,
                "cache_rank": cache_rank,
                "cache_reason": cache_reason,
                "gate_world_estimate": (
                    None
                    if gate_world_estimate is None
                    else np.asarray(gate_world_estimate, dtype=np.float32).copy()
                ),
                "backup_gate_world_estimate": (
                    None
                    if backup_gate_world_estimate is None
                    else np.asarray(backup_gate_world_estimate, dtype=np.float32).copy()
                ),
                "target_rel_drone": (
                    None
                    if target_rel_drone is None
                    else np.asarray(target_rel_drone, dtype=np.float32).copy()
                ),
                "backup_target_rel_drone": (
                    None
                    if backup_target_rel_drone is None
                    else np.asarray(backup_target_rel_drone, dtype=np.float32).copy()
                ),
            }
            self.aux = display_aux
            self.gate_used_depth_id = depth_item["depth_id"]
            self._sensor_cond.notify_all()

        if self.args.debug_print:
            center_disp = None
            if gate_center is not None:
                gate_center = np.asarray(gate_center, dtype=np.float32).reshape(-1)
                if gate_center.size >= 2:
                    center_disp = (round(float(gate_center[0]), 3), round(float(gate_center[1]), 3))
            print(
                "[gate]",
                "center=", center_disp,
                "target_rel_drone=", np.round(target_rel_drone, 3) if target_rel_drone is not None else None,
                "gate_world_est=", None if gate_world_estimate is None else np.round(gate_world_estimate, 3),
                "backup_target_rel_drone=", np.round(backup_target_rel_drone, 3) if backup_target_rel_drone is not None else None,
                "backup_gate_world_est=", None if backup_gate_world_estimate is None else np.round(backup_gate_world_estimate, 3),
                "pose_src=", None if state_snapshot is None else state_snapshot.get("source"),
                "pose_frame_id=", None if state_snapshot is None else state_snapshot.get("frame_id"),
                "pose_child_frame_id=", None if state_snapshot is None else state_snapshot.get("child_frame_id"),
                "pose_time_boot_us=", None if state_snapshot is None else state_snapshot.get("time_boot_us"),
                "target_v=", None if target_v is None else np.round(target_v, 3),
                "cache_used=", cache_used,
                "cache_rank=", cache_rank,
                "cache_reason=", cache_reason,
                flush=True,
            )
            rejection_summary = gate_model_aux.get("gate_rejection_summary")
            if rejection_summary:
                print("[gate_reject]", rejection_summary, flush=True)

        if self.seg_debug_print and gate_model_aux.get("gate_detection_target_cache_used"):
            print(
                "[gate_cache]",
                "reason=", gate_model_aux.get("gate_detection_target_cache_reason", "unknown"),
                "depth_id=", depth_item["depth_id"],
                "frame=", depth_item["frame_id"],
                "previous_depth=", gate_model_aux.get("gate_depth_m"),
                self.summarize_gate_rejection(gate_model_aux),
                flush=True,
            )

        self.show_or_save_rgb_overlay(
            raw_rgb,
            frame_id=depth_item["frame_id"],
            aux=display_aux,
        )
        self.save_segmentation_mask(
            display_aux.get("segmentation_mask"),
            aux=display_aux,
            frame_id=depth_item["frame_id"],
        )

    def control_callback(self):
        with self._sensor_cond:
            while self.is_control_thread_active:
                if self.depth_buffer:
                    item = self.depth_buffer[-1]
                    if item["depth_id"] != self.control_last_depth_id:
                        depth_item = {
                            "depth_id": item["depth_id"],
                            "frame_id": item["frame_id"],
                            "depth": item["depth"].copy(),
                        }
                        target_v_snapshot = (
                            None if self.target_v is None else np.array(self.target_v, copy=True)
                        )
                        target_info_snapshot = dict(self.target_info)
                        for key in (
                            "target_rel_drone",
                            "gate_world_estimate",
                            "backup_gate_world_estimate",
                        ):
                            if target_info_snapshot.get(key) is not None:
                                target_info_snapshot[key] = np.array(
                                    target_info_snapshot[key],
                                    copy=True,
                                )
                        break
                self._sensor_cond.wait(timeout=0.05)
            else:
                return

        state = self.get_state()
        attitude = self.data.get("attitude")
        if state is None or attitude is None:
            reason = "missing_state" if state is None else "missing_attitude"
            self.debug_idle(reason, frame={"frame_id": depth_item["frame_id"]}, state=state)
            return

        env_rot_ned = quaternion_to_rotation_matrix(state["orientation"])
        env_rot = sim_to_normal_rotation(env_rot_ned)
        linear_velocity_body_frd = state["linear_velocity"]
        linear_velocity_ned = env_rot_ned @ linear_velocity_body_frd
        linear_velocity = sim_to_normal(linear_velocity_ned)

        forward = env_rot[:, 0].copy()
        forward[2] = 0.0
        if np.linalg.norm(forward) < 1e-6:
            forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        forward = normalize(forward)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        left = normalize(np.cross(up, forward))
        yaw_only_rot = np.stack([forward, left, up], axis=-1)

        raw_target_v = target_v_snapshot
        target_source = target_info_snapshot.get("source", "unknown")
        gate_cache_used = bool(target_info_snapshot.get("cache_used", False))
        gate_world_estimate = target_info_snapshot.get("gate_world_estimate")
        if gate_world_estimate is not None:
            gate_world_estimate = np.asarray(gate_world_estimate, dtype=np.float32).copy()
        backup_gate_world_estimate = target_info_snapshot.get("backup_gate_world_estimate")
        if backup_gate_world_estimate is not None:
            backup_gate_world_estimate = np.asarray(
                backup_gate_world_estimate,
                dtype=np.float32,
            ).copy()
        cached_gate_distance = None
        cached_gate_switched_to_backup = False

        if raw_target_v is None or self.args.always_world_fallback:
            sim_world_target_delta = self.world_target_v - np.asarray(state["position"], dtype=np.float32)
            env_target_v = sim_to_normal(sim_world_target_delta)
            raw_target_v = env_target_v.copy()
            target_source = "world_fallback_forced" if self.args.always_world_fallback else "world_fallback"
            local_target_v = env_target_v @ yaw_only_rot
            target_v_from_local = lambda vec: vec @ yaw_only_rot.T
        elif (
            target_source == "gate"
            and gate_cache_used
            and gate_world_estimate is not None
        ):
            current_position = np.asarray(state["position"], dtype=np.float32)
            cached_gate_distance = float(np.linalg.norm(gate_world_estimate - current_position))
            switch_distance = float(self.args.seg_cached_gate_switch_distance_m)
            if (
                switch_distance > 0.0
                and cached_gate_distance < switch_distance
                and backup_gate_world_estimate is not None
            ):
                gate_world_estimate = backup_gate_world_estimate.copy()
                cached_gate_switched_to_backup = True
            sim_gate_delta = gate_world_estimate - current_position
            env_target_v = sim_to_normal(sim_gate_delta)
            raw_target_v = env_target_v.copy()
            target_source = (
                "seg_gate_cached_backup_world"
                if cached_gate_switched_to_backup
                else "gate_cached_world"
            )
            local_target_v = env_target_v @ yaw_only_rot
            target_v_from_local = lambda vec: vec @ yaw_only_rot.T
        else:
            local_target_v = raw_target_v.copy()
            target_v_from_local = lambda vec: vec @ yaw_only_rot.T

        target_v_norm = np.linalg.norm(local_target_v)
        if target_v_norm > 1e-6:
            if self.args.target_type == "max":
                local_target_v = (local_target_v / target_v_norm) * self.args.target_speed
            elif self.args.target_type == "min":
                local_target_v = (
                    local_target_v / target_v_norm * min(target_v_norm, self.args.target_speed)
                )
        else:
            local_target_v = np.array([self.args.target_speed, 0.0, 0.0], dtype=np.float32)

        target_v = target_v_from_local(local_target_v)
        local_velocity = linear_velocity @ yaw_only_rot
        state_parts = [local_target_v, env_rot[:, 2], np.array([self.args.margin], dtype=np.float32)]
        if not self.args.no_odom:
            state_parts.insert(0, local_velocity)
        state_tensor = torch.as_tensor(np.concatenate(state_parts), dtype=torch.float32)[None]

        if self.args.control_fake_infinite_depth:
            control_depth = np.full_like(depth_item["depth"], np.inf, dtype=np.float32)
            depth_input_label = "fake_infinite"
        else:
            control_depth = depth_item["depth"]
            depth_input_label = "actual"

        depth_tensor = self.preprocess_depth(control_depth)
        act, self.hidden = self.model.predict_action(depth_tensor, state_tensor)
        act = yaw_only_rot @ act.reshape(3, -1)
        a_pred = act[:, 0] - act[:, 1]
        roll, pitch, yaw, throttle, thrust = self.acceleration_to_attitude_command(
            a_pred, local_velocity, target_v, env_rot
        )
        target_rpy = (roll, pitch, yaw)
        current_rpy = (
            float(attitude["roll"]),
            float(attitude["pitch"]),
            float(attitude["yaw"]),
        )
        error_rpy, command_delta_rpy = self.build_attitude_command(target_rpy, current_rpy)
        self.send_attitude_command(command_delta_rpy, throttle)

        if self.args.debug_print:
            print(
                "[control_input]",
                "depth_input=", depth_input_label,
                "drone_position=", np.round(state["position"], 3),
                "target_src=", target_source,
                "gate_cache_used=", gate_cache_used,
                "seg_cached_gate_distance=", None if cached_gate_distance is None else round(cached_gate_distance, 3),
                "seg_cached_gate_switch_threshold=", round(float(self.args.seg_cached_gate_switch_distance_m), 3),
                "seg_cached_gate_switched_to_backup=", cached_gate_switched_to_backup,
                "raw_target_v=", np.round(raw_target_v, 3),
                "local_target_v=", np.round(local_target_v, 3),
                "gate_world_est=", None if gate_world_estimate is None else np.round(gate_world_estimate, 3),
                "backup_gate_world_est=", None if backup_gate_world_estimate is None else np.round(backup_gate_world_estimate, 3),
                "linear_velocity_ned=", np.round(linear_velocity_ned, 3),
                "linear_velocity=", np.round(linear_velocity, 3),
                "a_pred=", np.round(a_pred, 3),
                "local_velocity=", np.round(local_velocity, 3),
                "env_rot", np.round(env_rot, 3),
                "env_rot_ned", np.round(env_rot_ned, 3),
                flush=True,
            )

            print(
                "[control_output]",
                "attitude_rpy=", np.round(current_rpy, 3),
                "target_rpy=", np.round(target_rpy, 3),
                "error_rpy=", np.round(error_rpy, 3),
                "command_delta_rpy=", np.round(command_delta_rpy, 3),
                "thrust=", round(thrust, 3),
                "throttle=", round(throttle, 3),
                flush=True,
            )
        self.debug_counter += 1
        self.control_last_depth_id = depth_item["depth_id"]

    def save_segmentation_mask(self, mask, aux=None, frame_id=None):
        if self.save_segmentation_dir is None or mask is None:
            return
        save_every = max(1, int(getattr(self.args, "save_segmentation_every", 1)))
        if self.segmentation_save_counter % save_every == 0:
            if frame_id is None:
                name = f"segmentation_{self.segmentation_save_counter:06d}.png"
            else:
                name = f"segmentation_frame_{int(frame_id):06d}.png"
            out_path = self.save_segmentation_dir / name
            mask_u8 = np.asarray(mask, dtype=np.uint8)
            if mask_u8.ndim == 2:
                overlay = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2RGB)
            else:
                overlay = mask_u8.copy()

            previous_aux = self.aux
            if aux is not None:
                self.aux = aux
            overlay = self.build_rgb_overlay(overlay)
            self.aux = previous_aux
            cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        self.segmentation_save_counter += 1

    def save_rgb_overlay(self, overlay_rgb, frame_id=None):
        if self.save_rgb_overlay_dir is None:
            return
        save_every = max(1, int(self.args.save_rgb_overlay_every))
        if self.rgb_overlay_save_counter % save_every == 0:
            if frame_id is None:
                name = f"rgb_overlay_{self.rgb_overlay_save_counter:06d}.png"
            else:
                name = f"rgb_overlay_frame_{int(frame_id):06d}.png"
            out_path = self.save_rgb_overlay_dir / name
            cv2.imwrite(str(out_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        self.rgb_overlay_save_counter += 1

    def estimate_gate_target_resized(self, rgb_or_bgr, depth, transform=None):
        if transform is None:
            gate_rgb, transform = self.fit_rgb_for_gate(rgb_or_bgr)
        else:
            # The base gate thread passes gate_bgr. Convert it back so segmentation
            # consistently operates on RGB.
            gate_rgb = cv2.cvtColor(rgb_or_bgr, cv2.COLOR_BGR2RGB)
        gate_h, gate_w = gate_rgb.shape[:2]

        previous_intrinsics = getattr(self.gate_detector, "fixed_camera_intrinsics", None)
        previous_camera_intrinsics = getattr(self.gate_detector, "camera_intrinsics", None)
        self.gate_detector.fixed_camera_intrinsics = {
            "fx": self.base_camera_intrinsics["fx"] * transform.get("resize_scale_x", 1.0),
            "fy": self.base_camera_intrinsics["fy"] * transform.get("resize_scale_y", 1.0),
            "cx": (
                self.base_camera_intrinsics["cx"]
                - transform["crop_left"]
                + transform["pad_left"]
            )
            * transform.get("resize_scale_x", 1.0),
            "cy": (
                self.base_camera_intrinsics["cy"]
                - transform["crop_top"]
                + transform["pad_top"]
            )
            * transform.get("resize_scale_y", 1.0),
        }
        self.gate_detector.camera_intrinsics = None
        try:
            target_v_airsim, aux = self.gate_detector.estimate_target_point_airsim(gate_rgb, depth)
        finally:
            self.gate_detector.fixed_camera_intrinsics = previous_intrinsics
            self.gate_detector.camera_intrinsics = previous_camera_intrinsics

        aux = aux or {}
        display_aux = self.map_gate_aux_to_original_pixels(aux, transform)
        display_aux["gate_input_size"] = (int(gate_w), int(gate_h))
        display_aux["gate_fit_transform"] = transform
        display_aux["gate_model_aux"] = aux
        return target_v_airsim, display_aux


def main():
    args = build_args()
    args.seg_sat_min = 30
    args.seg_val_min = 30
    args.seg_hue_low_max = 10
    args.seg_hue_high_min = 160
    args.seg_open_kernel = 3
    args.seg_close_kernel = 5
    args.seg_min_area = 0.0
    args.seg_square_ratio_tol = 0.1
    args.seg_min_edge_coverage = 0
    args.seg_edge_samples = 30
    args.seg_edge_thickness = 2
    args.seg_max_gate_depth_m = 50.0
    args.seg_free_select_depth_m = 5.0
    args.seg_depth_consistency_tol_m = 3
    print(
        "Segmentation gate input:",
        f"{args.seg_gate_input_width}x{args.seg_gate_input_height}",
        "from crop",
        f"{args.seg_gate_crop_width}x{args.seg_gate_crop_height}",
        flush=True,
    )

    print("Startup complete. Opening MAVLink connection...", flush=True)
    shared_data = {}
    system_boot_ms = int(time.time() * 1000)

    sim_conn = mavutil.mavlink_connection(f"udpin:{args.server_ip}:{args.server_udp_port}")
    mavlink_rx = MAVLinkRX.create_mavlink_rx(sim_conn, shared_data)
    print("Waiting for heartbeat...", flush=True)
    if not mavlink_rx.wait_heartbeat(timeout=10.0):
        raise TimeoutError("Timed out waiting for MAVLink heartbeat.")
    print(f"Connected to system: {sim_conn.target_system}", flush=True)

    ts_loop = TimeSync.create_timesync(sim_conn, shared_data)
    vision_rx = VisionRX(shared_data)
    print("Loading segmentation gate detector and control model...", flush=True)
    racer = SegmentationDepthGateRacer(sim_conn, shared_data, system_boot_ms, args)

    print("Using segmentation geometry gate detector.", flush=True)
    print("Arming drone...", flush=True)
    racer.arm()
    print("Starting MAVLink threaded frame/gate/control racer...", flush=True)
    racer.start_threads()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        racer.stop_threads()
        stop_and_join(ts_loop)
        stop_and_join(mavlink_rx)
        stop_and_join(vision_rx)
        if args.viz_rgb:
            cv2.destroyWindow("rgb")
        print("Client exited!", flush=True)


if __name__ == "__main__":
    main()
