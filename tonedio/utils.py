from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path

import json
import math
import numpy as np
import os
import site
import sys
import time
import torch
import torch.nn as nn
import ctypes

try:
    import airsimdroneracinglab as airsim
except ImportError:  # MAVLink simulator does not provide the AirSim Python API.
    airsim = None


def load_trusted_torch_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)





try:
    from .train_corner_affinity_detection import (
        CORNER_NAMES,
        EDGE_TYPES,
        assemble_gates_from_edges,
        extract_corner_candidates,
        score_and_match_edges,
    )
except ImportError:
    from train_corner_affinity_detection import (
        CORNER_NAMES,
        EDGE_TYPES,
        assemble_gates_from_edges,
        extract_corner_candidates,
        score_and_match_edges,
    )

class GateDetector:
    def __init__(self, 
                 checkpoint_path=None, 
                 device="cpu",
                 gate_switch_depth_m=20,
                 gate_depth_switch_tol_m=1,
                 gate_max_depth_m=25,
                 profile_gate=False,
                 corner_conf_threshold=0.25,
                 corner_topk=50,
                 corner_nms_radius=5,
                 edge_min_score=0.05,
                 integral_samples=15,
                 debug_print=False,
                 debug_print_every=1,
                 drone_name = "drone_1",
                 camera_fov_degrees=90.0,
                 camera_fx=None,
                 camera_fy=None,
                 camera_cx=None,
                 camera_cy=None,
                 camera_pose=None,
                 load_airsim_camera_settings=True):
        
       

        try:
            from .corner_unet import CornerUNet
        except ImportError:
            from corner_unet import CornerUNet

        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Corner affinity checkpoint does not exist: {checkpoint_path}")

        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.model = CornerUNet(out_channels=12).to(self.device)

        checkpoint = load_trusted_torch_checkpoint(str(self.checkpoint_path), map_location=self.device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

      
        self.last_corner_target_airsim = None
        self.last_selected_gate_center_px = None
        self.last_backup_gate_center_px = None
        self.last_selected_gate_points_px = None
        self.last_backup_gate_points_px = None
        self.last_selected_gate_depth_m = None
        self.last_selected_gate_depth_source = None
        self.last_backup_gate_depth_m = None
        self.last_backup_gate_depth_source = None
        self.last_corner_backup_target_airsim = None
        self.last_corner_candidate_targets_airsim = []
        self.last_corner_candidate_timestamp = 0.0
        self.last_viz_center_px = None
        self.gate_max_depth_m = float(gate_max_depth_m)
        self.profile_gate = bool(profile_gate)
        self._gate_profile_counter = 0
        self.gate_switch_depth_m = float(gate_switch_depth_m)
        self.gate_depth_switch_tol_m = float(gate_depth_switch_tol_m)
        self.corner_conf_threshold = float(corner_conf_threshold)
        self.corner_topk = int(corner_topk)
        self.corner_nms_radius = int(corner_nms_radius)
        self.edge_min_score = float(edge_min_score)
        self.integral_samples = int(integral_samples)
        self.debug_print = bool(debug_print)
        self.debug_print_every = int(debug_print_every)
        self._gate_postproc_debug_counter = 0
        self._gate_select_debug_counter = 0
        self._gate_target_debug_counter= 0
        self.camera_intrinsics = None
        self.airsim_client_images = None
        if airsim is not None:
            self.airsim_client_images = airsim.MultirotorClient()
            self.airsim_client_images.confirmConnection()
        self.drone_name = drone_name
        self.camera_fov_degrees = float(camera_fov_degrees)
        self.fixed_camera_intrinsics = None
        if all(value is not None for value in (camera_fx, camera_fy, camera_cx, camera_cy)):
            self.fixed_camera_intrinsics = {
                "fx": float(camera_fx),
                "fy": float(camera_fy),
                "cx": float(camera_cx),
                "cy": float(camera_cy),
            }
        self.camera_pose = {
            "X": 0.0,
            "Y": 0.0,
            "Z": 0.0,
            "Roll": 0.0,
            "Pitch": 0.0,
            "Yaw": 0.0,
        }
        if camera_pose:
            self.camera_pose.update(camera_pose)
        self.load_airsim_camera_settings = bool(load_airsim_camera_settings)
     
    def get_camera_intrinsics(self, width, height):
        if (
            self.camera_intrinsics is not None
            and self.camera_intrinsics["width"] == width
            and self.camera_intrinsics["height"] == height
        ):
            return self.camera_intrinsics

        fov_degrees = self.camera_fov_degrees
        if self.fixed_camera_intrinsics is not None:
            fx = self.fixed_camera_intrinsics["fx"]
            fy = self.fixed_camera_intrinsics["fy"]
            cx = self.fixed_camera_intrinsics["cx"]
            cy = self.fixed_camera_intrinsics["cy"]
        elif self.airsim_client_images is not None:
            camera_info = self.airsim_client_images.simGetCameraInfo(
                "fpv_cam", vehicle_name=self.drone_name
            )
            fov_degrees = float(camera_info.fov)
            if not math.isfinite(fov_degrees) or fov_degrees <= 0.0:
                fov_degrees = 90.0
            fx, fy, cx, cy = self.compute_pinhole_intrinsics(width, height, fov_degrees)
        else:
            if not math.isfinite(fov_degrees) or fov_degrees <= 0.0:
                fov_degrees = 90.0
            fx, fy, cx, cy = self.compute_pinhole_intrinsics(width, height, fov_degrees)
        self.camera_intrinsics = {
            "width": width,
            "height": height,
            "fov_degrees": fov_degrees,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        }
        return self.camera_intrinsics
    
    def euler_to_rotation_matrix(self, roll, pitch, yaw, degrees=True):
        if degrees:
            roll = math.radians(roll)
            pitch = math.radians(pitch)
            yaw = math.radians(yaw)

        cr = math.cos(roll)
        sr = math.sin(roll)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)

        R_x = np.array([
            [1, 0, 0],
            [0, cr, -sr],
            [0, sr, cr],
        ], dtype=np.float32)

        R_y = np.array([
            [cp, 0, sp],
            [0, 1, 0],
            [-sp, 0, cp],
        ], dtype=np.float32)

        R_z = np.array([
            [cy, -sy, 0],
            [sy, cy, 0],
            [0, 0, 1],
        ], dtype=np.float32)

        return R_z @ R_y @ R_x

    def compute_pinhole_intrinsics(self, width, height, fov_degrees):
        fov_radians = math.radians(float(fov_degrees))
        fx = 0.5 * float(width) / math.tan(0.5 * fov_radians)
        fy = fx
        cx = 0.5 * float(width)
        cy = 0.5 * float(height)
        return fx, fy, cx, cy

    
    def _points_to_center_and_size(self, points: dict[str, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
        tl = np.array(points["TL"], dtype=np.float32)
        tr = np.array(points["TR"], dtype=np.float32)
        br = np.array(points["BR"], dtype=np.float32)
        bl = np.array(points["BL"], dtype=np.float32)
        center = (tl + tr + br + bl) / 4.0
        width_px = float(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        height_px = float(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
        return center.astype(np.float32), np.array([width_px, height_px], dtype=np.float32)


    def _preprocess_rgb(self, rgb_image: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        if rgb_image is None:
            raise ValueError("rgb_image cannot be None")
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb_image.shape}")
        height, width = int(rgb_image.shape[0]), int(rgb_image.shape[1])

        image = rgb_image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))[None, ...]).contiguous().to(self.device)

        # CornerUNet downsamples 4x (stride 16). Pad to avoid shape mismatches.
        stride = 16
        pad_h = (stride - (height % stride)) % stride
        pad_w = (stride - (width % stride)) % stride
        if pad_h or pad_w:
            tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        return tensor, (height, width)

    @torch.no_grad()
    def predict_maps(self, rgb_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        input_tensor, unpad_hw = self._preprocess_rgb(rgb_image)
        logits = self.model(input_tensor)
        corner_maps = torch.sigmoid(logits[:, :4])[0].detach().float().cpu().numpy().astype(np.float32)
        paf_maps = torch.tanh(logits[:, 4:])[0].detach().float().cpu().numpy().astype(np.float32)
        out_h, out_w = unpad_hw
        return corner_maps[:, :out_h, :out_w], paf_maps[:, :out_h, :out_w]
    
    def _extract_gate_candidates(
        self,
        corner_maps: np.ndarray,
        paf_maps: np.ndarray,
        image_width: int,
        image_height: int,
        max_gates: int = 5,
    ) -> list[dict]:
        if corner_maps.ndim != 3 or corner_maps.shape[0] != 4:
            return []
        if paf_maps.ndim != 3 or paf_maps.shape[0] != 8:
            return []

        candidates = extract_corner_candidates(
            corner_maps,
            threshold=self.corner_conf_threshold,
            topk=self.corner_topk,
            nms_radius=self.corner_nms_radius,
        )
        if any(len(candidates[name]) == 0 for name in CORNER_NAMES):
            return []

        edge_matches = score_and_match_edges(
            candidates,
            paf_maps,
            edge_min_score=self.edge_min_score,
            integral_samples=self.integral_samples,
        )
        raw_gates = assemble_gates_from_edges(edge_matches)
        if not raw_gates:
            return []

        if self.debug_print:
            self._gate_postproc_debug_counter += 1
            if (self._gate_postproc_debug_counter - 1) % self.debug_print_every == 0:
                cand_counts = {k: len(v) for k, v in candidates.items()}
                match_counts = {f"{a}_{b}": len(edge_matches.get((a, b), [])) for a, b in EDGE_TYPES}
                print("[gate]", "num", len(raw_gates))

        gate_candidates: list[dict] = []
        for gate in raw_gates:
            points = gate.get("points", {})
            if not all(k in points for k in CORNER_NAMES):
                continue

            center, size = self._points_to_center_and_size(points)

            scores = gate.get("scores", {})
            heatmap_score = float(np.mean([float(scores.get(k, 0.0)) for k in CORNER_NAMES]))
            gate_score = float(gate.get("gate_score", heatmap_score))

            gate_candidates.append(
                {
                    "points": {k: np.array(points[k], dtype=np.float32) for k in CORNER_NAMES},
                    "scores": {k: float(scores.get(k, 0.0)) for k in CORNER_NAMES},
                    "edge_scores": gate.get("edge_scores", {}),
                    "center": center,
                    "size": size,
                    "confidence": heatmap_score,
                    "gate_score": gate_score,
                }
            )

        gate_candidates.sort(key=lambda g: float(g.get("gate_score", 0.0)), reverse=True)

        return gate_candidates[: int(max_gates)]

    
    def _sample_depth_at_pixel(self, depth: np.ndarray | None, pixel: np.ndarray, source_size: tuple[int, int]) -> float:
        if depth is None or pixel is None:
            return np.inf
        if depth.ndim != 2:
            return np.inf

        src_h, src_w = source_size
        depth_h, depth_w = depth.shape[:2]
        # Map from RGB pixel coordinates into the depth-map pixel coordinates by
        # reversing the crop/pad performed by the depth estimator input fitting.
        x_src = float(pixel[0])
        y_src = float(pixel[1])

        crop_left = (int(src_w) - int(depth_w)) // 2 if src_w > depth_w else 0
        crop_top = (int(src_h) - int(depth_h)) // 2 if src_h > depth_h else 0
        cropped_w = int(depth_w) if src_w > depth_w else int(src_w)
        cropped_h = int(depth_h) if src_h > depth_h else int(src_h)

        pad_left = (int(depth_w) - int(cropped_w)) // 2 if cropped_w < depth_w else 0
        pad_top = (int(depth_h) - int(cropped_h)) // 2 if cropped_h < depth_h else 0

        x = (x_src - float(crop_left)) if src_w > depth_w else (x_src + float(pad_left))
        y = (y_src - float(crop_top)) if src_h > depth_h else (y_src + float(pad_top))
        xi = int(round(x))
        yi = int(round(y))
        if xi < 0 or yi < 0 or xi >= depth_w or yi >= depth_h:
            return np.inf

        # Robust sampling: use a small patch around the projected pixel and take
        # the minimum of valid (finite, >0) values. This biases towards nearer
        # surfaces and is more stable than a raw mean when the patch overlaps
        # edges/occluders.
        patch_radius = int(getattr(self, "depth_patch_radius", 2))
        if patch_radius < 0:
            patch_radius = 0

        x0 = max(0, xi - patch_radius)
        x1 = min(depth_w, xi + patch_radius + 1)
        y0 = max(0, yi - patch_radius)
        y1 = min(depth_h, yi + patch_radius + 1)
        patch = depth[y0:y1, x0:x1]
        patch = patch[np.isfinite(patch) & (patch > 0.0)]
        if patch.size == 0:
            return np.inf
        return float(np.min(patch))

    #calculate target_V in airsim coordinates
    def _gate_target_to_airsim(self, candidate, intr, depth, rgb_width, rgb_height): 
        camera = self.camera_pose
        settings_path = Path.home() / "Documents" / "AirSim" / "settings.json"
        if self.load_airsim_camera_settings and airsim is not None and settings_path.exists():
            with settings_path.open("r") as f:
                settings = json.load(f)
            camera = settings["Vehicles"].get(self.drone_name, {}).get("Cameras", {}).get("fpv_cam", camera)
        
        if candidate is None:
            return np.inf, "unavailable", {"reason": "candidate_none"}, None, None

        center_u, center_v = float(candidate["center"][0]), float(candidate["center"][1])
        rect_w_px = float(candidate["size"][0])
        rect_h_px = float(candidate["size"][1])

        source_size = (int(rgb_height), int(rgb_width))

        # Use a robust aggregate of the gate corner depths (more stable than a single
        # center sample). We no longer reject candidates based on "corner depth
        # inconsistency"; instead we proceed whenever at least one valid corner
        # depth is available.
        depth_dbg: dict = {
            "reason": "unavailable",
            "center_px": (float(center_u), float(center_v)),
            "corner_depths_m": {},
            "max_corner_pair_diff_m": None,
        }
        corner_depths_by_name: dict[str, float] = {}
        points = candidate.get("points", {})
        for name in CORNER_NAMES:
            corner_px = points.get(name)
            if corner_px is None:
                continue
            depth_corner = self._sample_depth_at_pixel(depth, corner_px, source_size)
            depth_dbg["corner_depths_m"][str(name)] = (
                None if (not np.isfinite(depth_corner)) else float(depth_corner)
            )
            if np.isfinite(depth_corner) and depth_corner > 1e-6:
                corner_depths_by_name[str(name)] = float(depth_corner)

        depth_m = np.inf
        depth_source = "unavailable"

        if corner_depths_by_name:
            ordered = [corner_depths_by_name[name] for name in CORNER_NAMES if name in corner_depths_by_name]
            corner_depths = np.array(ordered, dtype=np.float32)
            if corner_depths.size >= 2:
                pair_diffs = []
                for i in range(int(corner_depths.shape[0])):
                    for j in range(i + 1, int(corner_depths.shape[0])):
                        pair_diffs.append(float(abs(float(corner_depths[i]) - float(corner_depths[j]))))
                depth_dbg["max_corner_pair_diff_m"] = float(max(pair_diffs)) if pair_diffs else None
            depth_m = float(np.mean(corner_depths))
            depth_source = (
                "depth_map_corners_mean"
                if len(corner_depths_by_name) == len(CORNER_NAMES)
                else "depth_map_corners_mean_partial"
            )
        else:
            depth_m = np.inf
            depth_source = "depth_map_corners_missing"

        if not np.isfinite(depth_m) or depth_m <= 1e-6:
            depth_dbg["reason"] = depth_source
            return np.inf, depth_source, depth_dbg, None, None

        x_off = (center_u - intr["cx"]) * depth_m / intr["fx"]
        y_off = (center_v - intr["cy"]) * depth_m / intr["fy"]
        target_rel_camera = np.array([depth_m, x_off, y_off], dtype=np.float32) #target point in camera frame 

        p_camera_drone = np.array(
            [
                camera["X"], 
                camera["Y"],
                camera["Z"]
            ],
            dtype=np.float32,
        )
   
    
        rot_camera_drone  =  self.euler_to_rotation_matrix(camera["Roll"], camera["Pitch"], camera["Yaw"], degrees=True)

        target_rel_drone = rot_camera_drone @ target_rel_camera
        target_rel_drone = p_camera_drone + target_rel_drone #target position in drone frame
        if self.debug_print:
            self._gate_target_debug_counter += 1
            if (self._gate_target_debug_counter - 1) % self.debug_print_every == 0:
                print(
                    "[gate_project]",
                    "center=",
                    (round(center_u, 3), round(center_v, 3)),
                    "intr=",
                    {
                        "fx": round(float(intr["fx"]), 3),
                        "fy": round(float(intr["fy"]), 3),
                        "cx": round(float(intr["cx"]), 3),
                        "cy": round(float(intr["cy"]), 3),
                    },
                    "corner_depths=",
                    {
                        str(k): (
                            None
                            if v is None or (isinstance(v, float) and not np.isfinite(v))
                            else round(float(v), 3)
                        )
                        for k, v in depth_dbg.get("corner_depths_m", {}).items()
                    },
                    "depth_m=",
                    round(float(depth_m), 3),
                    "offset=",
                    (round(float(x_off), 3), round(float(y_off), 3)),
                    "target_rel_camera=",
                    np.round(target_rel_camera, 3),
                    "camera_pose=",
                    {
                        "X": round(float(camera["X"]), 3),
                        "Y": round(float(camera["Y"]), 3),
                        "Z": round(float(camera["Z"]), 3),
                        "Roll": round(float(camera["Roll"]), 3),
                        "Pitch": round(float(camera["Pitch"]), 3),
                        "Yaw": round(float(camera["Yaw"]), 3),
                    },
                    "target_rel_drone=",
                    np.round(target_rel_drone, 3),
                    flush=True,
                )
                print( 
                    "[gates]",
                    "center=",
                    (round(center_u, 3), round(center_v, 3)),
                    "target_rel_drone=",
                    np.array2string(target_rel_drone, precision=6, suppress_small=False),
                )
        depth_dbg["reason"] = "ok"
        return (
            depth_m,
            depth_source,
            depth_dbg, #depth rejection debug
            target_rel_drone.astype(np.float32),
            target_rel_camera.astype(np.float32),
        )


    def estimate_target_point_airsim(self, rgb, depth):
        t0_total = time.perf_counter() #high resolution timer. records the current time right before the function starts

        if rgb is None:
            if self.last_corner_target_airsim is not None:
                return self.last_corner_target_airsim, {
                    "gate_depth_source": "cached_primary",
                    "gate_detection_target_cache_used": True,
                    "gate_detection_target_cache_rank": 1,
                    "gate_center_px": self.last_selected_gate_center_px,
                    "gate_depth_m": self.last_selected_gate_depth_m,
                    "gate_detection_target_rel_drone" : self.last_corner_target_airsim
                }

            if self.last_corner_backup_target_airsim is not None:
                return self.last_corner_backup_target_airsim, {
                    "gate_depth_source": "cached_secondary",
                    "gate_detection_target_cache_used": True,
                    "gate_detection_target_cache_rank": 2,
                    "gate_center_px": self.last_backup_gate_center_px,
                    "gate_depth_m": self.last_backup_gate_depth_m,
                    "gate_detection_target_rel_drone" : self.last_corner_backup_target_airsim,
                }
            return None, {}

        t0_predict = time.perf_counter()
        corner_maps, paf_maps = self.predict_maps(rgb)
        t_ms_predict = (time.perf_counter() - t0_predict) * 1000.0 #time it took gate detector to run 
        t0_post = time.perf_counter()
        rgb_height, rgb_width = rgb.shape[:2]
        gate_candidates = self._extract_gate_candidates(
            corner_maps,
            paf_maps,
            int(rgb_width), 
            int(rgb_height),
        )
        t_ms_post = (time.perf_counter() - t0_post) * 1000.0 # time it took for extracting gate candidates 
        self.last_corner_candidate_targets_airsim = [] #storage for drone relative target position for all candidates
        
        if not gate_candidates: #if there are no gate candidates
            # If the detector loses the gate temporarily, keep flying toward the last target
            # instead of switching to the forward-fallback target_v.
            if self.last_corner_target_airsim is not None:
                return self.last_corner_target_airsim, {
                    "gate_depth_source": "cached_primary",
                    "gate_detection_target_cache_used": True,
                    "gate_detection_target_cache_rank": 1,
                    "gate_center_px": self.last_selected_gate_center_px,
                    "gate_depth_m": self.last_selected_gate_depth_m,
                    "gate_detection_target_rel_drone" : self.last_corner_target_airsim,

            
                }
            if self.last_corner_backup_target_airsim is not None:
                return self.last_corner_backup_target_airsim, {
                    "gate_depth_source": "cached_secondary",
                    "gate_detection_target_cache_used": True,
                    "gate_detection_target_cache_rank": 2,
                    "gate_center_px": self.last_backup_gate_center_px,
                    "gate_depth_m": self.last_backup_gate_depth_m,
                    "gate_detection_target_rel_drone" : self.last_corner_backup_target_airsim,
                
                }
            return None, {}

        intr = self.get_camera_intrinsics(
            int(rgb_width),
            int(rgb_height),
        )

        t0_targets = time.perf_counter()
        candidate_targets_all = []
        rejected_targets = []
        for gate_candidate in gate_candidates:
            (
                target_depth_m,
                target_depth_source,
                target_dbg, 
                target_rel_drone,
                target_rel_camera,
            ) = self._gate_target_to_airsim(
                gate_candidate,
                intr, depth , rgb_width, rgb_height
                  )
            if target_rel_drone is None:
                rejected_targets.append(
                    {
                        "candidate": gate_candidate,
                        "depth_source": target_depth_source, #how depth was computed
                        "dbg": target_dbg, #regection debug 
                    }
                )
                continue
            candidate_targets_all.append( #if there exists a target_rel_drone
                {
                    "candidate": gate_candidate,
                    "target_rel_drone": target_rel_drone,
                    "depth_m": target_depth_m,
                    "depth_source": target_depth_source,
                    "target_rel_camera": target_rel_camera,
                }
            )

        t_ms_targets = (time.perf_counter() - t0_targets) * 1000.0 #time for calculating target_rel_drone
        t_ms_total = (time.perf_counter() - t0_total) * 1000.0 #time for total gate detecting + target calculation
        if self.profile_gate and self.debug_print:
            self._gate_profile_counter += 1
            if (self._gate_profile_counter - 1) % self.debug_print_every == 0:
                fps = (1000.0 / t_ms_total) if t_ms_total > 1e-6 else float("inf")
                print(
                    "[gate_profile]",
                    "predict_ms=",
                    round(t_ms_predict, 3),
                    "post_ms=",
                    round(t_ms_post, 3),
                    "targets_ms=",
                    round(t_ms_targets, 3),
                    "total_ms=",
                    round(t_ms_total, 3),
                    "fps=",
                    round(fps, 2),
                    "candidates=",
                    len(gate_candidates),
                )

        max_depth_m = float(self.gate_max_depth_m)
        candidate_targets = [] #candidates that passed 각종 조건 
        for item in candidate_targets_all:
            depth_m = float(item.get("depth_m", np.inf))
            depth_ok = np.isfinite(depth_m) and depth_m > 1e-6 #depth is finite and bigger than 0 
            if depth_ok and depth_m <= max_depth_m:  # depth is smaller than max depth threshold
                candidate_targets.append(item)
            else:
                rejected_targets.append(
                    {
                        "candidate": item.get("candidate"),
                        "depth_source": item.get("depth_source"),
                        "dbg": {
                            "reason": "depth_too_far",
                            "depth_m": None if not np.isfinite(depth_m) else depth_m,
                            "max_depth_m": max_depth_m,
                        },
                    }
                )

        self.last_corner_candidate_targets_airsim = [item["target_rel_drone"] for item in candidate_targets]
        if not candidate_targets:
            if self.debug_print:
                rejected_preview = []
                for item in rejected_targets[:3]:
                    cand = item.get("candidate", {}) or {}
                    center = cand.get("center")
                    center_disp = None
                    if center is not None:
                        center_disp = (round(float(center[0]), 1), round(float(center[1]), 1))
                    dbg = item.get("dbg", {}) or {}
                    corner_depths = dbg.get("corner_depths_m", {}) or {}
                    rejected_preview.append(
                        {
                            "reason": str(dbg.get("reason")),
                            "max_pair_diff": dbg.get("max_corner_pair_diff_m"),
                            "max_depth_m": dbg.get("max_depth_m"),
                            "depth_m": dbg.get("depth_m"),
                            "center": center_disp,
                            "corners": {
                                k: (
                                    None
                                    if v is None or (isinstance(v, float) and (not np.isfinite(v)))
                                    else round(float(v), 3)
                                )
                                for k, v in corner_depths.items()
                            },
                        }
                    )
                if rejected_preview and ((self._gate_postproc_debug_counter - 1) % self.debug_print_every == 0):
                    print("[gate_depth_reject]", "rejected=", len(rejected_targets), "preview=", rejected_preview)
            if self.last_corner_target_airsim is not None:
                return self.last_corner_target_airsim, {
                    "gate_depth_source": "cached_primary",
                    "gate_detection_target_cache_used": True,
                    "gate_detection_target_cache_rank": 1,
                    "gate_center_px": self.last_selected_gate_center_px,
                    "gate_depth_m": self.last_selected_gate_depth_m,
                    "gate_detection_target_rel_drone" : self.last_corner_target_airsim,
                }
            if self.last_corner_backup_target_airsim is not None:
                return self.last_corner_backup_target_airsim, {
                    "gate_depth_source": "cached_secondary",
                    "gate_detection_target_cache_used": True,
                    "gate_detection_target_cache_rank": 2,
                    "gate_center_px": self.last_backup_gate_center_px,
                    "gate_depth_m": self.last_backup_gate_depth_m,
                    "gate_detection_target_rel_drone" : self.last_corner_backup_target_airsim,
                }
            return None, {}
        
        def _closest_gate_key(item: dict) -> tuple[int, float]:
            depth_m = float(item.get("depth_m", np.inf))
            depth_ok = np.isfinite(depth_m) and depth_m > 1e-6
            tier = 0 if depth_ok else 1
            return (tier, depth_m if depth_ok else np.inf)

        # Select the closest gate by estimated depth (smaller = closer), but with
        # a depth-switch constraint:
        # - If the previously selected gate is close (< gate_switch_depth_m),
        #   allow switching to anything (even if depth increases).
        # - Otherwise (>= gate_switch_depth_m), only allow switching to candidates
        #   whose depth is within +/- gate_depth_switch_tol_m of the previous depth.
        #   If none exist, keep the previous target (target_v stays the same).
        gate_switch_depth_m = float(getattr(self, "gate_switch_depth_m", 3.0))
        gate_depth_switch_tol_m = float(getattr(self, "gate_depth_switch_tol_m", 1.0))
        if not np.isfinite(gate_depth_switch_tol_m) or gate_depth_switch_tol_m < 0.0:
            gate_depth_switch_tol_m = 0.0
        prev_depth = self.last_selected_gate_depth_m
        prev_depth_ok = isinstance(prev_depth, (float, int)) and np.isfinite(float(prev_depth)) and float(prev_depth) > 1e-6

        sorted_targets = sorted(candidate_targets, key=_closest_gate_key)
        indexed_targets = list(enumerate(sorted_targets))

        allow_any_switch = (not prev_depth_ok) or (float(prev_depth) < gate_switch_depth_m)
        constrained_indexed = indexed_targets
        selection_mode = "unconstrained"
        if (not allow_any_switch) and prev_depth_ok:
            prev_d = float(prev_depth)
            smaller: list[tuple[int, dict]] = []
            larger_within_tol: list[tuple[int, dict]] = []
            for idx, item in indexed_targets:
                d = float(item.get("depth_m", np.inf))
                if not (np.isfinite(d) and d > 1e-6):
                    continue
                if d < prev_d:
                    smaller.append((idx, item))
                elif (d - prev_d) <= gate_depth_switch_tol_m:
                    larger_within_tol.append((idx, item))

            if smaller:
                constrained_indexed = smaller
                selection_mode = "prefer_smaller_depth"
            elif larger_within_tol:
                constrained_indexed = larger_within_tol
                selection_mode = "larger_within_tol"
            else:
                if self.last_corner_target_airsim is not None:
                    return self.last_corner_target_airsim, {
                        "gate_depth_source": "cached_primary_no_allowed_depth",
                        "gate_detection_target_cache_used": True,
                        "gate_detection_target_cache_rank": 1,
                        "gate_center_px": self.last_selected_gate_center_px,
                        "gate_depth_m": self.last_selected_gate_depth_m,
                        "segmentation_gate_lock_depth_m": gate_switch_depth_m,
                        "segmentation_prev_depth_m": prev_d,
                        "segmentation_gate_depth_switch_tol_m": gate_depth_switch_tol_m,
                        "segmentation_gate_selection_mode": "locked_keep_cached",
                        "gate_detection_target_rel_drone" : self.last_corner_target_airsim,
                    }

        selected_idx, selected = constrained_indexed[0]
        selected_rank = int(selected_idx) + 1

        backup_idx, backup = selected_idx, selected
        backup_rank = selected_rank
        if len(constrained_indexed) > 1:
            backup_idx, backup = constrained_indexed[1]
            backup_rank = int(backup_idx) + 1

        candidate = selected["candidate"]
        target_rel_drone = selected["target_rel_drone"]
        depth_m = float(selected["depth_m"])
        depth_source = str(selected["depth_source"])

        backup_candidate = backup["candidate"]
        backup_target_rel_drone = backup["target_rel_drone"]
        backup_depth_m = float(backup["depth_m"])
        backup_depth_source = str(backup["depth_source"])

        if self.debug_print:
            self._gate_select_debug_counter += 1
            if (self._gate_select_debug_counter - 1) % self.debug_print_every == 0:
                summary = []
                for idx, item in enumerate(sorted_targets[:5], start=1):
                    cand = item.get("candidate", {})
                    center = cand.get("center")
                    center_disp = None
                    if center is not None:
                        center_disp = (round(float(center[0]), 1), round(float(center[1]), 1))
                    depth_val = float(item.get("depth_m", np.inf))
                    depth_disp = None if (not np.isfinite(depth_val) or depth_val <= 1e-6) else round(depth_val, 3)
                    summary.append(
                        {
                            "rank": idx,
                            "depth": depth_disp,
                            "src": str(item.get("depth_source", "unavailable")),
                            "center": center_disp,
                        }
                    )
                print(
                    "[gate_select]",
                    "selected_rank=",
                    selected_rank,
                    "selected_depth=",
                    round(depth_m, 3) if np.isfinite(depth_m) else depth_m,
                    "selected_src=",
                    depth_source,
                )

        self.last_corner_target_airsim = target_rel_drone
        self.last_corner_backup_target_airsim = backup_target_rel_drone
        self.last_corner_candidate_timestamp = time.time()
        self.last_selected_gate_center_px = np.asarray(candidate["center"], dtype=np.float32).copy()
        self.last_backup_gate_center_px = np.asarray(backup_candidate["center"], dtype=np.float32).copy()
        self.last_selected_gate_points_px = {
            str(k): np.asarray(v, dtype=np.float32).copy() for k, v in (candidate.get("points", {}) or {}).items()
        }
        self.last_backup_gate_points_px = {
            str(k): np.asarray(v, dtype=np.float32).copy() for k, v in (backup_candidate.get("points", {}) or {}).items()
        }
        self.last_selected_gate_depth_m = depth_m
        self.last_selected_gate_depth_source = depth_source
        self.last_backup_gate_depth_m = backup_depth_m
        self.last_backup_gate_depth_source = backup_depth_source

        aux = {
            "segmentation_mask": None,
            "segmentation_rect": candidate,
            "segmentation_primary_rect": candidate,
            "segmentation_backup_rect": backup_candidate,
            "gate_center_px": candidate["center"],
            "segmentation_rect_size_px": candidate["size"],
            "gate_depth_m": depth_m,
            "gate_depth_source": depth_source,
            "segmentation_primary_depth_m": depth_m,
            "segmentation_primary_rank": selected_rank,
            "segmentation_selected_rank": selected_rank,
            "segmentation_backup_depth_m": backup_depth_m,
            "segmentation_backup_depth_source": backup_depth_source,
            "segmentation_backup_rank": backup_rank,
            "segmentation_blob_count": len(candidate_targets),
            "segmentation_blob_selection": "corner_affinity",
            "segmentation_blob_depth_m": depth_m,
            "segmentation_promoted": False,
            "segmentation_promote_depth_threshold": self.corner_conf_threshold,
            "segmentation_blob_backup_depth_m": backup_depth_m,
            "segmentation_gate_switch_depth_m": float(getattr(self, "gate_switch_depth_m", 3.0)),
            "segmentation_gate_depth_switch_tol_m": float(getattr(self, "gate_depth_switch_tol_m", 1.0)),
            "segmentation_gate_selection_mode": selection_mode if "selection_mode" in locals() else "unconstrained",
            "segmentation_target_airsim": target_rel_drone,
            "segmentation_backup_target_airsim": backup_target_rel_drone,
            "gate_detection_target_rel_drone": self.last_corner_target_airsim,
            "segmentation_backup_target_vec_airsim_world": backup.get("target_rel_drone"),
            "segmentation_target_rel_camera": selected.get("target_rel_camera"),
            "segmentation_backup_target_rel_camera": backup.get("target_rel_camera"),
            "camera_intrinsics": intr,
            "gate_corner_points_px": candidate["points"],
            "gate_corner_scores": candidate["scores"],
            "gate_center_px": candidate["center"],
            "gate_confidence": candidate["confidence"],
            "corner_gate_candidates": gate_candidates,
            "corner_gate_count": len(gate_candidates),
            "corner_gate_targets_airsim": self.last_corner_candidate_targets_airsim,
            "corner_gate_target_records": candidate_targets,
        }
        return target_rel_drone, aux





SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DEPTH_ONNX_PATH = (
    REPO_ROOT
    / "depth_estimation"
    / "results"
    / "run_224224_moreepoch"
    / "export"
    / "dn_model_latest.onnx"
)
DEFAULT_DEPTH_CHECKPOINT = DEFAULT_DEPTH_ONNX_PATH
DEFAULT_DEPTH_INPUT_WIDTH = 224
DEFAULT_DEPTH_INPUT_HEIGHT = 224
DEFAULT_DEPTH_INPUT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DEFAULT_DEPTH_INPUT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    ort = None
    ONNXRUNTIME_IMPORT_ERROR = exc
else:
    ONNXRUNTIME_IMPORT_ERROR = None

try:
    from .diffsim_racer import DEFAULT_MODEL_PATH, DiffSimRacer
except ImportError:
    try:
        from diffsim_racer import DEFAULT_MODEL_PATH, DiffSimRacer
    except ImportError:
        DEFAULT_MODEL_PATH = None
        DiffSimRacer = None
except Exception:
    DEFAULT_MODEL_PATH = None
    DiffSimRacer = None


DEFAULT_DEPTH_DEVICE = "auto"


class DepthAnythingOnnxEstimator:
    def __init__(
        self,
        onnx_path=DEFAULT_DEPTH_ONNX_PATH,
        input_width=DEFAULT_DEPTH_INPUT_WIDTH,
        input_height=DEFAULT_DEPTH_INPUT_HEIGHT,
        input_mean=DEFAULT_DEPTH_INPUT_MEAN,
        input_std=DEFAULT_DEPTH_INPUT_STD,
        device=DEFAULT_DEPTH_DEVICE,
    ):
        onnx_path = Path(onnx_path).expanduser().resolve()
        if not onnx_path.exists():
            raise FileNotFoundError(f"Depth ONNX model does not exist: {onnx_path}")
        if ort is None and cv2 is None:
            raise ImportError(
                "onnxruntime or opencv-python is required for ONNX depth inference."
            ) from ONNXRUNTIME_IMPORT_ERROR

        self.onnx_path = onnx_path
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.input_mean = np.asarray(input_mean, dtype=np.float32).reshape(1, 1, 3)
        self.input_std = np.asarray(input_std, dtype=np.float32).reshape(1, 1, 3)
        self.device = str(device).lower()
        self.runtime = None
        self.session = None
        self.net = None
        
        def _ensure_cuda_shared_libs_visible() -> None:
            lib_dirs: list[str] = []
            candidates = []
            try:
                candidates.extend(site.getsitepackages() or [])
            except Exception:
                pass
            try:
                user_sp = site.getusersitepackages()
                if user_sp:
                    candidates.append(user_sp)
            except Exception:
                pass

            for sp in candidates:
                if not sp:
                    continue
                nvidia_root = Path(sp) / "nvidia"
                if not nvidia_root.exists():
                    continue
                for lib_dir in nvidia_root.glob("*/lib"):
                    if lib_dir.is_dir():
                        lib_dirs.append(str(lib_dir))

            if not lib_dirs:
                return

            existing = os.environ.get("LD_LIBRARY_PATH", "")
            parts = [p for p in existing.split(":") if p]
            for lib_dir in reversed(lib_dirs):
                if lib_dir not in parts:
                    parts.insert(0, lib_dir)
            os.environ["LD_LIBRARY_PATH"] = ":".join(parts)

            # Best-effort preload of common CUDA deps so subsequent dlopen succeeds.
            for soname in (
                "libcublasLt.so.12",
                "libcublas.so.12",
                "libcufft.so.11",
                "libcurand.so.10",
                "libcusolver.so.11",
                "libcusparse.so.12",
                "libcudnn.so.9",
                "libcudart.so.12",
            ):
                for lib_dir in lib_dirs:
                    candidate = Path(lib_dir) / soname
                    if candidate.exists():
                        try:
                            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                        except OSError:
                            pass
                        break

        if ort is not None:
            available_providers = ort.get_available_providers()
            providers = ["CPUExecutionProvider"]
            if self.device in {"auto", "cuda", "gpu"} and "CUDAExecutionProvider" in available_providers:
                _ensure_cuda_shared_libs_visible()
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            try:
                self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
            except Exception:
                # Fallback: keep the process running even if CUDA deps are missing.
                providers = ["CPUExecutionProvider"]
                self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.runtime = "onnxruntime"
            self.providers = providers
        else:
            self.net = cv2.dnn.readNetFromONNX(str(self.onnx_path))
            self.runtime = "opencv_dnn"
            self.providers = ["opencv_dnn_cpu"]

    def _fit_image_to_size(self, image: np.ndarray, target_width: int, target_height: int) -> tuple[np.ndarray, str]:
        """Center-crop or edge-pad an image to the requested size."""
        src_height, src_width = image.shape[:2]
        out = image
        actions: list[str] = []

        if src_width > target_width:
            left = (src_width - target_width) // 2
            out = out[:, left : left + target_width]
            actions.append(f"crop_w({src_width}->{target_width})")
        if src_height > target_height:
            top = (src_height - target_height) // 2
            out = out[top : top + target_height, :]
            actions.append(f"crop_h({src_height}->{target_height})")

        pad_height = max(0, target_height - out.shape[0])
        pad_width = max(0, target_width - out.shape[1])
        if pad_height > 0 or pad_width > 0:
            pre_pad_h, pre_pad_w = out.shape[:2]
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            out = np.pad(
                out,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="edge",
            )
            pad_parts = []
            if pad_width > 0:
                pad_parts.append(f"pad_w({pre_pad_w}->{target_width})")
            if pad_height > 0:
                pad_parts.append(f"pad_h({pre_pad_h}->{target_height})")
            actions.extend(pad_parts)

        if out.shape[0] != target_height or out.shape[1] != target_width:
            raise ValueError(
                f"Failed to fit image to {(target_height, target_width)}; got {out.shape[:2]}"
            )

        if not actions:
            actions.append("ok")
        return out, "+".join(actions)
    
    
    
    
    def _preprocess_rgb(self, rgb_image):
        if rgb_image is None:
            raise ValueError("rgb_image cannot be None")
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError(
                f"Expected an RGB image with shape (H, W, 3), got {rgb_image.shape}"
            )

        rgb = np.asarray(rgb_image)
        fitted_rgb, action = self._fit_image_to_size(rgb, self.input_width, self.input_height)
        rgb = fitted_rgb.astype(np.float32, copy=False) / 255.0
        rgb = (rgb - self.input_mean) / self.input_std
        rgb = np.transpose(rgb, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(rgb, dtype=np.float32)

    def predict_depth(self, rgb_image):
        input_tensor = self._preprocess_rgb(rgb_image)
        if self.runtime == "onnxruntime":
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            depth = outputs[0]
        elif self.runtime == "opencv_dnn":
            self.net.setInput(input_tensor)
            depth = self.net.forward()
        else:  # pragma: no cover
            raise RuntimeError(f"Unsupported ONNX runtime backend: {self.runtime}")

        depth = np.asarray(depth, dtype=np.float32)
        depth = np.squeeze(depth)
        if depth.ndim != 2:
            raise ValueError(
                f"Expected depth output to be 2D after squeeze, got shape {depth.shape}"
            )
        return depth
    


class DiffPhysModel:
    def __init__(self, control_model_path, dim_obs=10, dim_action=6, device="cpu"):
        self.device = torch.device(device)
        self.model = Model(dim_obs=dim_obs, dim_action=dim_action).to(self.device)
        checkpoint = load_trusted_torch_checkpoint(control_model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.hidden = None

    @torch.no_grad()
    def predict_action(self, depth_tensor, state_tensor):
        depth_tensor = depth_tensor.to(self.device)
        state_tensor = state_tensor.to(self.device)
        action, self.hidden = self.model(depth_tensor, state_tensor, self.hidden)
        action = action.reshape(-1).detach().cpu().numpy()
        if action.shape[0] < 6:
            raise ValueError(
                "Model output must contain 6 values so it can be reshaped to (3, 2)."
            )
        return action, self.hidden
    


class Model(nn.Module):
        def __init__(self, dim_obs=10, dim_action=6) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(1, 32, 2, 2, bias=False),
                nn.LeakyReLU(0.05),
                nn.Conv2d(32, 64, 3, bias=False),
                nn.LeakyReLU(0.05),
                nn.Conv2d(64, 128, 3, bias=False),
                nn.LeakyReLU(0.05),
                nn.Flatten(),
                nn.Linear(128 * 2 * 4, 192, bias=False),
            )
            self.dim_obs = dim_obs
            self.observation_fc = nn.Linear(dim_obs, 192)
            self.gru = nn.GRUCell(192, 192)
            self.action_fc = nn.Linear(192, dim_action, bias=False)
            self.activation = nn.LeakyReLU(0.05)

        def forward(self, x: torch.Tensor, v, hx=None):
            img_feat = self.stem(x)
            x = self.activation(img_feat + self.observation_fc(v))
            hx = self.gru(x, hx)
            action = self.action_fc(self.activation(hx))
            return action, hx
        
AIRSIM_TO_FLIGHTMARE = np.diag([1.0, -1.0, -1.0]).astype(np.float32)



def airsim_to_normal_rotation(rot):
    rot = np.asarray(rot, dtype=np.float32)
    return AIRSIM_TO_FLIGHTMARE @ rot @ AIRSIM_TO_FLIGHTMARE
    
def airsim_to_normal_vector(vec):
    return AIRSIM_TO_FLIGHTMARE @ np.asarray(vec, dtype=np.float32)
    

def normalize(vec, eps=1e-6):
    norm = np.linalg.norm(vec)
    if norm < eps:
        return vec.copy()
    return vec / norm
    
def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z #sqaured norm 
    if n < 1e-12: #if norm is too small, it means the quaternion may not represent a valid rotation. 
        return np.eye(3, dtype=np.float32) #return identity matrix 
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )
