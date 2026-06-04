from __future__ import annotations

from argparse import ArgumentParser
from collections import deque
from pathlib import Path
import math
import sys
import threading
import time

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for import_path in (REPO_ROOT, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pymavlink import mavutil

from tonedio.mavlink_rx import MAVLinkRX
from example.timesync import TimeSync
from tonedio.utils import (
    DepthAnythingOnnxEstimator,
    DiffPhysModel,
    GateDetector,
    airsim_to_normal_vector,
    normalize,
    quaternion_to_rotation_matrix,
)
from tonedio.vision_rx import VisionRX


DEFAULT_MODELS_DIR = SCRIPT_DIR / "models"
DEFAULT_GATE_DETECTION_PATH = DEFAULT_MODELS_DIR / "gate_detection_112112.pt"
DEFAULT_DEPTH_ONNX_PATH = DEFAULT_MODELS_DIR / "dn_model_latest.onnx"
DEFAULT_CONTROL_MODEL_PATH = DEFAULT_MODELS_DIR / "controlmodel.pth"

SIM_TO_NORMAL = np.diag([1.0, 1.0, -1.0]).astype(np.float32)
SIM_TO_NORMAL_ROT = np.diag([1.0, 1.0, -1.0]).astype(np.float32)
INITIAL_ALIGNED_ROLL = 0.0
INITIAL_ALIGNED_YAW = -math.pi
GATE_EDGE_TYPES = (("TL", "TR"), ("TR", "BR"), ("BR", "BL"), ("BL", "TL"))


def sim_to_normal(vector):
    return SIM_TO_NORMAL @ np.asarray(vector, dtype=np.float32)


def sim_to_normal_rotation(rot):
    rot = np.asarray(rot, dtype=np.float32)
    return SIM_TO_NORMAL_ROT @ rot @ SIM_TO_NORMAL_ROT


def stop_and_join(component, timeout=0.5):
    if component is None:
        return
    try:
        thread = component.get_thread_for_join()
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
    except KeyboardInterrupt:
        pass


def build_args():
    parser = ArgumentParser(description="Run the depth/gate policy against the AI-GP MAVLink simulator.")
    parser.add_argument("--server_ip", type=str, default="127.0.0.1")
    parser.add_argument("--server_udp_port", type=int, default=14550)
    parser.add_argument("--control_hz", type=float, default=20.0)
    parser.add_argument("--depth_sleep", type=float, default=0.01)
    parser.add_argument("--gate_sleep", type=float, default=0.01)
    parser.add_argument("--control_sleep", type=float, default=0.01)
    parser.add_argument("--target_speed", type=float, default=1.0)
    parser.add_argument("--target_type", type=str, choices=["max", "raw", "min"], default="min")
    parser.add_argument("--world_target_x", type=float, default=-100.0)
    parser.add_argument("--world_target_y", type=float, default=0.0)
    parser.add_argument("--world_target_z", type=float, default=0.0)
    parser.add_argument("--hover_throttle", type=float, default=0.27)
    parser.add_argument("--attitude_p_gain", type=float, default=1)
    parser.add_argument("--max_delta_roll", type=float, default=1)
    parser.add_argument("--max_delta_pitch", type=float, default=1)
    parser.add_argument("--max_delta_yaw", type=float, default=1)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--no_odom", action="store_true", default=False)
    parser.add_argument("--debug_print", action="store_true", default=False)
    parser.add_argument("--debug_print_every", type=int, default=10)
    parser.add_argument("--debug_interval_sec", type=float, default=1.0)
    parser.add_argument("--debug_every_command", action="store_true", default=False)
    parser.add_argument("--viz_rgb", action="store_true", default=False)
    parser.add_argument("--save_rgb_overlay_dir", type=str, default="")
    parser.add_argument("--save_rgb_overlay_every", type=int, default=10)
    parser.add_argument("--save_depth_dir", type=str, default="")
    parser.add_argument("--save_depth_every", type=int, default=10)

    parser.add_argument("--control_model_path", type=str, default=str(DEFAULT_CONTROL_MODEL_PATH))
    parser.add_argument("--gate_model_path", type=str, default=str(DEFAULT_GATE_DETECTION_PATH))
    parser.add_argument("--depth_onnx_path", type=str, default=str(DEFAULT_DEPTH_ONNX_PATH))
    parser.add_argument("--depth_input_width", type=int, default=112)
    parser.add_argument("--depth_input_height", type=int, default=112)
    parser.add_argument("--gate_input_width", type=int, default=112)
    parser.add_argument("--gate_input_height", type=int, default=112)
    parser.add_argument("--gate_crop_size", type=int, default=336)
    parser.add_argument("--depth_device", type=str, default="auto")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dim_obs", type=int, default=10)
    parser.add_argument("--dim_action", type=int, default=6)

    parser.add_argument("--corner_conf_threshold", type=float, default=0.5)
    parser.add_argument("--corner_topk", type=int, default=50)
    parser.add_argument("--corner_nms_radius", type=int, default=5)
    parser.add_argument("--edge_min_score", type=float, default=0.1)
    parser.add_argument("--integral_samples", type=int, default=15)
    parser.add_argument("--gate_switch_depth_m", type=float, default=100.0)
    parser.add_argument("--gate_depth_switch_tol_m", type=float, default=100.0)
    parser.add_argument("--gate_max_depth_m", type=float, default=1000.0)
    parser.add_argument("--camera_fov_degrees", type=float, default=90.0)
    parser.add_argument("--camera_fx", type=float, default=320.0)
    parser.add_argument("--camera_fy", type=float, default=320.0)
    parser.add_argument("--camera_cx", type=float, default=320.0)
    parser.add_argument("--camera_cy", type=float, default=180.0)
    return parser.parse_args()


def euler_to_quaternion(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def normalize_quaternion(q):
    norm = math.sqrt(sum(float(value) * float(value) for value in q))
    if norm < 1e-9:
        return [1.0, 0.0, 0.0, 0.0]
    return [float(value) / norm for value in q]


def wrap_pi(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def subtract_rpy(target_rpy, current_rpy):
    return (
        wrap_pi(target_rpy[0] - current_rpy[0]),
        wrap_pi(target_rpy[1] - current_rpy[1]),
        wrap_pi(target_rpy[2] - current_rpy[2]),
    )


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, float(value)))


def attitude_error_to_body_delta_command(error_rpy, gain, max_delta):
    return (
        clamp(-error_rpy[0] * gain, -max_delta[0], max_delta[0]),
        clamp(-error_rpy[1] * gain, -max_delta[1], max_delta[1]),
        clamp(error_rpy[2] * gain, -max_delta[2], max_delta[2]),
    )


class MavlinkDepthGateRacer:
    def __init__(self, mavlink_conn, shared_data, system_boot_ms, args):
        self.mavlink_conn = mavlink_conn
        self.data = shared_data
        self.system_boot_ms = system_boot_ms
        self.args = args
        self.gravity = 9.81
        self.hidden = None
        self.last_frame_id = None
        self.target_v = None
        self.target_info = {}
        self.world_target_v = np.array(
            [args.world_target_x, args.world_target_y, args.world_target_z],
            dtype=np.float32,
        )
        self.aux = {}
        self.debug_counter = 0
        self.last_idle_debug_time = 0.0
        self.last_control_debug_time = 0.0
        self.rgb_overlay_save_counter = 0
        self.depth_save_counter = 0
        self.save_rgb_overlay_dir = None
        if args.save_rgb_overlay_dir:
            self.save_rgb_overlay_dir = Path(args.save_rgb_overlay_dir)
            self.save_rgb_overlay_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"Saving RGB gate overlays to: {self.save_rgb_overlay_dir.resolve()}",
                flush=True,
            )
        self.save_depth_dir = None
        if args.save_depth_dir:
            self.save_depth_dir = Path(args.save_depth_dir)
            self.save_depth_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving depth images to: {self.save_depth_dir.resolve()}", flush=True)
        self.base_camera_intrinsics = {
            "fx": float(args.camera_fx),
            "fy": float(args.camera_fy),
            "cx": float(args.camera_cx),
            "cy": float(args.camera_cy),
        }
        self.last_depth_frame_id = None
        self.depth_buffer = deque(maxlen=10)
        self.current_depth_id = 0
        self.gate_used_depth_id = -1
        self.control_last_depth_id = -1
        self._sensor_cond = threading.Condition()
        self.is_depth_thread_active = False
        self.is_gate_thread_active = False
        self.is_control_thread_active = False
        self.depth_thread = threading.Thread(
            target=self.repeat_timer_depth_callback,
            args=(self.depth_callback, float(args.depth_sleep)),
            daemon=True,
        )
        self.gate_thread = threading.Thread(
            target=self.repeat_timer_gate_callback,
            args=(self.gate_callback, float(args.gate_sleep)),
            daemon=True,
        )
        self.control_thread = threading.Thread(
            target=self.repeat_timer_control_callback,
            args=(self.control_callback, float(args.control_sleep)),
            daemon=True,
        )

        self.depth_estimator = DepthAnythingOnnxEstimator(
            onnx_path=args.depth_onnx_path,
            input_width=args.depth_input_width,
            input_height=args.depth_input_height,
            device=args.depth_device,
        )
        self.gate_detector = GateDetector(
            checkpoint_path=args.gate_model_path,
            device=args.device,
            gate_switch_depth_m=args.gate_switch_depth_m,
            gate_depth_switch_tol_m=args.gate_depth_switch_tol_m,
            gate_max_depth_m=args.gate_max_depth_m,
            corner_conf_threshold=args.corner_conf_threshold,
            corner_topk=args.corner_topk,
            corner_nms_radius=args.corner_nms_radius,
            edge_min_score=args.edge_min_score,
            integral_samples=args.integral_samples,
            debug_print=False,
            debug_print_every=max(1, args.debug_print_every),
            camera_fov_degrees=args.camera_fov_degrees,
            camera_fx=args.camera_fx,
            camera_fy=args.camera_fy,
            camera_cx=args.camera_cx,
            camera_cy=args.camera_cy,
            camera_pose={
                "X": 0.0,
                "Y": 0.0,
                "Z": 0.0,
                "Roll": 0.0,
                "Pitch": 20.0,
                "Yaw": 0.0,
            },
            load_airsim_camera_settings=False,
        )
        self.model = DiffPhysModel(
            args.control_model_path,
            dim_obs=args.dim_obs,
            dim_action=args.dim_action,
            device=args.device,
        )

    def arm(self):
        self.mavlink_conn.mav.command_long_send(
            self.mavlink_conn.target_system,
            self.mavlink_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def get_state(self):
        odom = self.data.get("odometry")
        if odom is not None:
            return {
                "source": "odometry",
                "position": np.asarray(odom["position"], dtype=np.float32),
                "orientation": np.asarray(odom["orientation"], dtype=np.float32),
                "linear_velocity": np.asarray(odom["linear_velocity"], dtype=np.float32),
            }

        local_position = self.data.get("local_position_ned")
        attitude = self.data.get("attitude")
        if local_position is None or attitude is None:
            return None

        roll = float(attitude["roll"])
        pitch = float(attitude["pitch"])
        yaw = float(attitude["yaw"])
        return {
            "source": "local_position_ned",
            "position": np.asarray(local_position["position"], dtype=np.float32),
            "orientation": np.asarray(euler_to_quaternion(roll, pitch, yaw), dtype=np.float32),
            "linear_velocity": np.asarray(local_position["linear_velocity"], dtype=np.float32),
        }

    def get_closest_track_gate(self, drone_position):
        gates = self.data.get("track_gates") or []
        if not gates:
            return None

        drone_position = np.asarray(drone_position, dtype=np.float32).reshape(-1)
        if drone_position.size < 3:
            return None

        closest = None
        closest_distance = None
        for gate in gates:
            if not isinstance(gate, dict) or gate.get("position_ned") is None:
                continue
            gate_position = np.asarray(gate["position_ned"], dtype=np.float32).reshape(-1)
            if gate_position.size < 3:
                continue
            distance = float(np.linalg.norm(gate_position[:3] - drone_position[:3]))
            if closest_distance is None or distance < closest_distance:
                closest_distance = distance
                closest = {
                    "gate_id": gate.get("gate_id"),
                    "position_ned": gate_position[:3],
                    "distance": distance,
                }
        return closest

    def preprocess_depth(self, depth):
        depth = depth.copy()
        depth[~np.isfinite(depth)] = 24.0
        depth[depth <= 0.0] = 24.0
        depth = 3.0 / np.clip(depth, 0.3, 24.0) - 0.6
        h, w = depth.shape
        crop_ratio = 0.82
        crop_h = max(1, int(round(h * crop_ratio)))
        crop_w = max(1, int(round(w * crop_ratio)))
        start_h = max(0, (h - crop_h) // 2)
        start_w = max(0, (w - crop_w) // 2)
        depth = depth[start_h : start_h + crop_h, start_w : start_w + crop_w]
        depth_tensor = torch.as_tensor(depth, dtype=torch.float32)[None, None]
        depth_tensor = F.interpolate(depth_tensor, (48, 64), mode="area")
        return F.max_pool2d(depth_tensor, (4, 4))

    def infinite_depth_like(self, depth):
        return np.full_like(np.asarray(depth, dtype=np.float32), np.inf, dtype=np.float32)

    def build_attitude_command(self, target_rpy, current_rpy):
        error_rpy = subtract_rpy(target_rpy, current_rpy)
        command_delta_rpy = attitude_error_to_body_delta_command(
            error_rpy,
            self.args.attitude_p_gain,
            (self.args.max_delta_roll, self.args.max_delta_pitch, self.args.max_delta_yaw),
        )
        return error_rpy, command_delta_rpy

    def send_attitude_command(self, command_delta_rpy, throttle):
        now_ms = int(time.time() * 1000)
        type_mask = (
            mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE
            | mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE
            | mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE
        )
        self.mavlink_conn.mav.set_attitude_target_send(
            now_ms - self.system_boot_ms,
            self.mavlink_conn.target_system,
            self.mavlink_conn.target_component,
            type_mask,
            normalize_quaternion(euler_to_quaternion(*command_delta_rpy)),
            0.0,
            0.0,
            0.0,
            float(np.clip(throttle, 0.0, 1.0)),
        )

    def acceleration_to_attitude_command(self, a_pred, input_velocity, input_target_v, env_rot):
        a_setpoint = a_pred.astype(np.float32)
        a_setpoint[2] += self.gravity
        thrust = float(np.linalg.norm(a_setpoint))
        if thrust < 1e-6:
            current_forward = env_rot[:, 0]
            yaw = math.atan2(float(current_forward[1]), float(current_forward[0]))
            return 0.0, 0.0, yaw, float(self.args.hover_throttle), thrust

        up_vec = a_setpoint / thrust
        throttle = thrust + float(input_velocity[2] * abs(input_velocity[2]) * 0.01)
        forward_vec = env_rot[:, 0] * 1.0 + input_target_v
        if abs(up_vec[2]) > 1e-6:
            forward_vec[2] = -(
                forward_vec[0] * up_vec[0] + forward_vec[1] * up_vec[1]
            ) / up_vec[2]
        else:
            forward_vec[2] = 0.0
        forward_vec = normalize(forward_vec)
        left_vec = normalize(np.cross(up_vec, forward_vec))

        roll = math.atan2(float(left_vec[2]), float(up_vec[2]))
        pitch = math.asin(float(np.clip(-forward_vec[2], -1.0, 1.0)))
        yaw = math.atan2(float(forward_vec[1]), float(forward_vec[0]))
        throttle = float(np.clip(throttle / 9.8 * self.args.hover_throttle, 0.0, 1.0))
        return float(roll), float(pitch), float(yaw), throttle, thrust

    def update_gate_target(self, rgb, depth):
        target_v_airsim, aux = self.estimate_gate_target_resized(rgb, depth)
        self.aux = aux or {}
        target_rel_drone = self.aux.get("gate_detection_target_rel_drone", target_v_airsim)
        if target_rel_drone is None:
            self.target_v = None
        else:
            self.target_v = airsim_to_normal_vector(target_rel_drone)
       

    def fit_rgb_for_gate(self, rgb):
        gate_w = max(1, int(self.args.gate_input_width))
        gate_h = max(1, int(self.args.gate_input_height))
        crop_size = max(1, int(self.args.gate_crop_size))
        original_h, original_w = rgb.shape[:2]
        fitted = rgb

        crop_left = 0
        crop_top = 0
        crop_w = min(crop_size, original_w)
        crop_h = min(crop_size, original_h)
        if original_w > crop_w:
            crop_left = (original_w - crop_w) // 2
            fitted = fitted[:, crop_left : crop_left + crop_w]
        if original_h > crop_h:
            crop_top = (original_h - crop_h) // 2
            fitted = fitted[crop_top : crop_top + crop_h, :]

        cropped_h, cropped_w = fitted.shape[:2]
        pad_left = 0
        pad_top = 0
        pad_right = 0
        pad_bottom = 0
        if cropped_w < crop_size or cropped_h < crop_size:
            pad_w = max(0, crop_size - cropped_w)
            pad_h = max(0, crop_size - cropped_h)
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            fitted = np.pad(
                fitted,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="edge",
            )
        crop_box_h, crop_box_w = fitted.shape[:2]
        scale_x = float(gate_w) / float(crop_box_w)
        scale_y = float(gate_h) / float(crop_box_h)
        if crop_box_w != gate_w or crop_box_h != gate_h:
            fitted = cv2.resize(fitted, (gate_w, gate_h), interpolation=cv2.INTER_LINEAR)

        transform = {
            "crop_left": float(crop_left),
            "crop_top": float(crop_top),
            "pad_left": float(pad_left),
            "pad_top": float(pad_top),
            "pad_right": float(pad_right),
            "pad_bottom": float(pad_bottom),
            "resize_scale_x": float(scale_x),
            "resize_scale_y": float(scale_y),
            "crop_box_w": int(crop_box_w),
            "crop_box_h": int(crop_box_h),
            "crop_size": int(crop_size),
            "gate_w": int(gate_w),
            "gate_h": int(gate_h),
            "original_w": int(original_w),
            "original_h": int(original_h),
        }
        return fitted.copy(), transform

    def resize_rgb_for_gate(self, rgb):
        fitted, _ = self.fit_rgb_for_gate(rgb)
        return fitted

    @staticmethod
    def _scale_point(point, scale_x, scale_y):
        if point is None:
            return None
        arr = np.asarray(point, dtype=np.float32).copy()
        if arr.shape[0] >= 2:
            arr[0] *= float(scale_x)
            arr[1] *= float(scale_y)
        return arr

    def _scale_candidate_pixels(self, candidate, scale_x, scale_y):
        if not isinstance(candidate, dict):
            return candidate
        scaled = dict(candidate)
        if "center" in scaled:
            scaled["center"] = self._scale_point(scaled.get("center"), scale_x, scale_y)
        if "size" in scaled and scaled.get("size") is not None:
            size = np.asarray(scaled["size"], dtype=np.float32).copy()
            if size.shape[0] >= 2:
                size[0] *= float(scale_x)
                size[1] *= float(scale_y)
            scaled["size"] = size
        points = scaled.get("points")
        if isinstance(points, dict):
            scaled["points"] = {
                key: self._scale_point(value, scale_x, scale_y) for key, value in points.items()
            }
        return scaled

    def _map_gate_point_to_original(self, point, transform):
        if point is None:
            return None
        arr = np.asarray(point, dtype=np.float32).copy()
        if arr.shape[0] >= 2:
            scale_x = float(transform.get("resize_scale_x", 1.0))
            scale_y = float(transform.get("resize_scale_y", 1.0))
            if abs(scale_x) > 1e-9:
                arr[0] /= scale_x
            if abs(scale_y) > 1e-9:
                arr[1] /= scale_y
            arr[0] = arr[0] - float(transform["pad_left"]) + float(transform["crop_left"])
            arr[1] = arr[1] - float(transform["pad_top"]) + float(transform["crop_top"])
        return arr

    def _map_gate_candidate_to_original(self, candidate, transform):
        if not isinstance(candidate, dict):
            return candidate
        mapped = dict(candidate)
        if "center" in mapped:
            mapped["center"] = self._map_gate_point_to_original(mapped.get("center"), transform)
        points = mapped.get("points")
        if isinstance(points, dict):
            mapped["points"] = {
                key: self._map_gate_point_to_original(value, transform)
                for key, value in points.items()
            }
        return mapped

    def map_gate_aux_to_original_pixels(self, aux, transform):
        if not isinstance(aux, dict):
            return aux
        mapped = dict(aux)
        for key in (
            "segmentation_rect",
            "segmentation_primary_rect",
            "segmentation_backup_rect",
        ):
            if isinstance(mapped.get(key), dict):
                mapped[key] = self._map_gate_candidate_to_original(mapped[key], transform)
        if isinstance(mapped.get("corner_gate_candidates"), list):
            mapped["corner_gate_candidates"] = [
                self._map_gate_candidate_to_original(candidate, transform)
                for candidate in mapped["corner_gate_candidates"]
            ]
        if "gate_center_px" in mapped:
            mapped["gate_center_px"] = self._map_gate_point_to_original(
                mapped.get("gate_center_px"), transform
            )
        if isinstance(mapped.get("gate_corner_points_px"), dict):
            mapped["gate_corner_points_px"] = {
                key: self._map_gate_point_to_original(value, transform)
                for key, value in mapped["gate_corner_points_px"].items()
            }
        return mapped

    def scale_gate_aux_pixels(self, aux, scale_x, scale_y):
        if not isinstance(aux, dict):
            return aux
        scaled = dict(aux)
        for key in (
            "segmentation_rect",
            "segmentation_primary_rect",
            "segmentation_backup_rect",
        ):
            if isinstance(scaled.get(key), dict):
                scaled[key] = self._scale_candidate_pixels(scaled[key], scale_x, scale_y)
        if isinstance(scaled.get("corner_gate_candidates"), list):
            scaled["corner_gate_candidates"] = [
                self._scale_candidate_pixels(candidate, scale_x, scale_y)
                for candidate in scaled["corner_gate_candidates"]
            ]
        if "gate_center_px" in scaled:
            scaled["gate_center_px"] = self._scale_point(scaled.get("gate_center_px"), scale_x, scale_y)
        if isinstance(scaled.get("gate_corner_points_px"), dict):
            scaled["gate_corner_points_px"] = {
                key: self._scale_point(value, scale_x, scale_y)
                for key, value in scaled["gate_corner_points_px"].items()
            }
        return scaled

    def estimate_gate_target_resized(self, rgb, depth, transform=None):
        if transform is None:
            gate_rgb, transform = self.fit_rgb_for_gate(rgb)
        else:
            gate_rgb = rgb
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

    @staticmethod
    def _pixel_xy(point):
        if point is None:
            return None
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError, IndexError):
            return None
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        return int(round(x)), int(round(y))

    def draw_gate_candidate(self, image_bgr, candidate, color, label=None, thickness=2):
        if not isinstance(candidate, dict):
            return
        points = candidate.get("points") or {}
        for a, b in GATE_EDGE_TYPES:
            p0 = self._pixel_xy(points.get(a))
            p1 = self._pixel_xy(points.get(b))
            if p0 is not None and p1 is not None:
                cv2.line(image_bgr, p0, p1, color, thickness, lineType=cv2.LINE_AA)

        for name, point in points.items():
            xy = self._pixel_xy(point)
            if xy is None:
                continue
            cv2.circle(image_bgr, xy, 2, color, -1, lineType=cv2.LINE_AA)
            cv2.putText(
                image_bgr,
                str(name),
                (xy[0] + 3, xy[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        center = self._pixel_xy(candidate.get("center"))
        if center is not None:
            cv2.drawMarker(
                image_bgr,
                center,
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=9,
                thickness=1,
                line_type=cv2.LINE_AA,
            )
            if label:
                cv2.putText(
                    image_bgr,
                    label,
                    (center[0] + 5, center[1] + 11),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    def build_rgb_overlay(self, image_bgr):
        overlay = image_bgr.copy()
        aux = self.aux or {}

        candidates = aux.get("corner_gate_candidates")
        if isinstance(candidates, list):
            palette = [(0, 170, 255), (255, 120, 0), (180, 180, 180)]
            for idx, candidate in enumerate(candidates[:3]):
                self.draw_gate_candidate(
                    overlay,
                    candidate,
                    palette[idx % len(palette)],
                    label=f"cand {idx + 1}",
                    thickness=1,
                )

        backup = aux.get("segmentation_backup_rect")
        if isinstance(backup, dict):
            self.draw_gate_candidate(overlay, backup, (255, 0, 255), label="backup", thickness=1)

        primary = aux.get("segmentation_primary_rect") or aux.get("segmentation_rect")
        if isinstance(primary, dict):
            depth_m = aux.get("gate_depth_m")
            confidence = aux.get("gate_confidence")
            label_parts = ["selected"]
            if isinstance(depth_m, (int, float)) and math.isfinite(float(depth_m)):
                label_parts.append(f"{float(depth_m):.1f}m")
            if isinstance(confidence, (int, float)) and math.isfinite(float(confidence)):
                label_parts.append(f"conf {float(confidence):.2f}")
            self.draw_gate_candidate(
                overlay,
                primary,
                (0, 255, 0),
                label=" ".join(label_parts),
                thickness=1,
            )
        else:
            center = self._pixel_xy(aux.get("gate_center_px"))
            if center is not None:
                cv2.drawMarker(
                    overlay,
                    center,
                    (0, 255, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=9,
                    thickness=1,
                    line_type=cv2.LINE_AA,
                )

        status = "gate"
        if aux.get("gate_detection_target_cache_used"):
            status = f"gate cached rank {aux.get('gate_detection_target_cache_rank', '?')}"
        elif not aux:
            status = "gate not detected"
        cv2.putText(
            overlay,
            status,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0) if aux else (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        return overlay

    def save_rgb_overlay(self, overlay_bgr, frame_id=None):
        if self.save_rgb_overlay_dir is None:
            return
        save_every = max(1, int(self.args.save_rgb_overlay_every))
        if self.rgb_overlay_save_counter % save_every == 0:
            if frame_id is None:
                name = f"rgb_overlay_{self.rgb_overlay_save_counter:06d}.png"
            else:
                name = f"rgb_overlay_frame_{int(frame_id):06d}.png"
            out_path = self.save_rgb_overlay_dir / name
            cv2.imwrite(str(out_path), overlay_bgr)
        self.rgb_overlay_save_counter += 1

    def depth_to_display_image(self, depth):
        depth_display = np.asarray(depth, dtype=np.float32).copy()
        depth_display[~np.isfinite(depth_display)] = np.nan
        valid_depth = depth_display[np.isfinite(depth_display) & (depth_display > 0.0)]
        if valid_depth.size == 0:
            return np.zeros(depth_display.shape, dtype=np.uint8)

        near = float(np.percentile(valid_depth, 2.0))
        far = float(np.percentile(valid_depth, 98.0))
        if far <= near:
            far = near + 1.0

        depth_display = np.clip(depth_display, near, far)
        depth_display = (depth_display - near) / (far - near)
        depth_display[~np.isfinite(depth_display)] = 0.0
        return (depth_display * 255.0).astype(np.uint8)

    def save_depth_image(self, depth, frame_id=None):
        if self.save_depth_dir is None:
            return
        save_every = max(1, int(self.args.save_depth_every))
        if self.depth_save_counter % save_every == 0:
            if frame_id is None:
                name = f"depth_{self.depth_save_counter:06d}.png"
            else:
                name = f"depth_frame_{int(frame_id):06d}.png"
            out_path = self.save_depth_dir / name
            cv2.imwrite(str(out_path), self.depth_to_display_image(depth))
        self.depth_save_counter += 1

    def show_or_save_rgb_overlay(self, image_bgr, frame_id=None, aux=None):
        if not self.args.viz_rgb and self.save_rgb_overlay_dir is None:
            return
        previous_aux = self.aux
        if aux is not None:
            self.aux = aux
        overlay = self.build_rgb_overlay(image_bgr)
        self.aux = previous_aux
        self.save_rgb_overlay(overlay, frame_id=frame_id)
        if self.args.viz_rgb:
            cv2.imshow("rgb", overlay)
            cv2.waitKey(1)

    def depth_callback(self):
        frame = self.data.get("latest_frame")
        if frame is None:
            self.debug_idle("missing_frame")
            return

        frame_id = frame.get("frame_id")
        if frame_id == self.last_depth_frame_id:
            return
        self.last_depth_frame_id = frame_id

        image_bgr = frame["image_bgr"]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        depth_rgb, gate_transform = self.fit_rgb_for_gate(rgb)
        gate_bgr, _ = self.fit_rgb_for_gate(image_bgr)
        depth = self.depth_estimator.predict_depth(depth_rgb)
        self.save_depth_image(depth, frame_id=frame_id)

        with self._sensor_cond:
            self.current_depth_id += 1
            depth_id = self.current_depth_id
            self.depth_buffer.append(
                {
                    "depth_id": depth_id,
                    "frame_id": frame_id,
                    "image_bgr": image_bgr.copy(),
                    "rgb": depth_rgb.copy(),
                    "gate_bgr": gate_bgr.copy(),
                    "gate_transform": gate_transform,
                    "depth": depth.copy(),
                }
            )
            self._sensor_cond.notify_all()
        if self.args.debug_print:
            print(
                "[depth]",
                "depth_id=", depth_id,
                "frame=", frame_id,
                flush=True,
            )

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
                            "rgb": item["rgb"].copy(),
                            "gate_bgr": item["gate_bgr"].copy(),
                            "gate_transform": dict(item["gate_transform"]),
                            "depth": item["depth"].copy(),
                        }
                        break
                self._sensor_cond.wait(timeout=0.05)
            else:
                return

        target_v_airsim, aux = self.estimate_gate_target_resized(
            depth_item["gate_bgr"],
            depth_item["depth"],
            transform=depth_item["gate_transform"],
        )
        aux = aux or {}
        target_rel_drone = aux.get("gate_detection_target_rel_drone", target_v_airsim)
        gate_model_aux = aux.get("gate_model_aux", aux)
        if target_rel_drone is None:
            target_v = None
        else:
            target_v = airsim_to_normal_vector(target_rel_drone)
        gate_center = gate_model_aux.get("gate_center_px") if isinstance(gate_model_aux, dict) else None

        with self._sensor_cond:
            self.target_v = target_v
            self.target_info = {
                "source": "gate" if target_rel_drone is not None else "none",
                "depth_id": depth_item["depth_id"],
                "frame_id": depth_item["frame_id"],
                "target_rel_drone": (
                    None
                    if target_rel_drone is None
                    else np.asarray(target_rel_drone, dtype=np.float32).copy()
                ),
            }
            self.aux = aux
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
                "target_v=", None if target_v is None else np.round(target_v, 3),
                flush=True,
            )

        self.show_or_save_rgb_overlay(
            depth_item["rgb"],
            frame_id=depth_item["frame_id"],
            aux=aux.get("gate_model_aux", aux),
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
                        if target_info_snapshot.get("target_rel_drone") is not None:
                            target_info_snapshot["target_rel_drone"] = np.array(
                                target_info_snapshot["target_rel_drone"],
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
        #env_rot = env_rot_ned.copy()
        linear_velocity = sim_to_normal(state["linear_velocity"])

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
        if raw_target_v is None:
            sim_world_target_delta = self.world_target_v - np.asarray(state["position"], dtype=np.float32)
            env_target_v = sim_to_normal(sim_world_target_delta)
            env_target_v[1] *= -1.0
            raw_target_v = env_target_v.copy()
            target_source = "world_fallback"
            target_info_snapshot = {
                "source": target_source,
                "depth_id": None,
                "frame_id": None,
                "target_rel_drone": None,
            }
            local_target_v = env_target_v @ yaw_only_rot
            target_v_from_local = lambda vec: vec @ yaw_only_rot.T
        else:
            env_target_v = env_rot @ raw_target_v
            local_target_v = env_target_v @ yaw_only_rot
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

        depth_tensor = self.preprocess_depth(self.infinite_depth_like(depth_item["depth"]))
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
            closest_gate = self.get_closest_track_gate(state["position"])
            print(
                "[debug]",
                "depth_input=", "infinite",
                "drone_position=", np.round(state["position"], 3),
                "closest_gate_id=", None if closest_gate is None else closest_gate["gate_id"],
                "closest_gate_position=", None if closest_gate is None else np.round(closest_gate["position_ned"], 3),
                "closest_gate_distance=", None if closest_gate is None else round(closest_gate["distance"], 3),
                "target_src=", target_source,
                "attitude_rpy=", np.round(current_rpy, 3),
                "raw_target_v=", np.round(raw_target_v, 3),
                "local_target_v=", np.round(local_target_v, 3),
                "a_pred=", np.round(a_pred, 3),
                "local_velocity=", np.round(local_velocity, 3),
                "thrust=", round(thrust, 3),
                "throttle=", round(throttle, 3),
                flush=True,
            )
        self.debug_counter += 1
        self.control_last_depth_id = depth_item["depth_id"]

    def repeat_timer_depth_callback(self, task, period):
        period = max(0.0, float(period))
        while self.is_depth_thread_active:
            try:
                task()
            except Exception as exc:
                print("[depth_thread] callback failed:", exc, flush=True)
            if period > 0.0:
                time.sleep(period)

    def repeat_timer_gate_callback(self, task, period):
        period = max(0.0, float(period))
        while self.is_gate_thread_active:
            try:
                task()
            except Exception as exc:
                print("[gate_thread] callback failed:", exc, flush=True)
            if period > 0.0:
                time.sleep(period)

    def repeat_timer_control_callback(self, task, period):
        period = max(0.0, float(period))
        while self.is_control_thread_active:
            try:
                task()
            except Exception as exc:
                print("[control_thread] callback failed:", exc, flush=True)
            if period > 0.0:
                time.sleep(period)

    def start_threads(self):
        if not self.is_depth_thread_active:
            self.is_depth_thread_active = True
            self.depth_thread.start()

        if not self.is_gate_thread_active:
            self.is_gate_thread_active = True
            self.gate_thread.start()

        if not self.is_control_thread_active:
            self.is_control_thread_active = True
            self.control_thread.start()

    def stop_threads(self):
        with self._sensor_cond:
            self.is_control_thread_active = False
            self.is_gate_thread_active = False
            self.is_depth_thread_active = False
            self._sensor_cond.notify_all()

        for thread in (self.control_thread, self.gate_thread, self.depth_thread):
            if thread.is_alive():
                thread.join(timeout=1.0)

    def debug_idle(self, reason, frame=None, state=None):
        if not self.args.debug_print:
            return
        now = time.time()
        if now - self.last_idle_debug_time < 1.0:
            return
        self.last_idle_debug_time = now
        print(
            "[debug_idle]",
            "reason=", reason,
            "has_frame=", frame is not None,
            "has_state=", state is not None,
            "has_attitude=", self.data.get("attitude") is not None,
            flush=True,
        )

    def update(self):
        frame = self.data.get("latest_frame")
        state = self.get_state()
        if frame is None or state is None:
            reason = "missing_frame" if frame is None else "missing_state"
            self.debug_idle(reason, frame=frame, state=state)
            time.sleep(1.0 / max(self.args.control_hz, 1.0))
            return

        attitude = self.data.get("attitude")
        if attitude is None:
            self.debug_idle("missing_attitude", frame=frame, state=state)
            time.sleep(1.0 / max(self.args.control_hz, 1.0))
            return

        frame_id = frame["frame_id"]
        if frame_id == self.last_frame_id:
            self.debug_idle("same_frame", frame=frame, state=state)
            time.sleep(1.0 / max(self.args.control_hz, 1.0))
            return
        self.last_frame_id = frame_id

        image_bgr = frame["image_bgr"]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        depth_rgb, gate_transform = self.fit_rgb_for_gate(rgb)
        gate_bgr, _ = self.fit_rgb_for_gate(image_bgr)
        depth = self.depth_estimator.predict_depth(depth_rgb)
        self.save_depth_image(depth, frame_id=frame_id)
        if self.args.debug_print:
            print("[depth]", "frame=", frame_id, flush=True)
        target_v_airsim, aux = self.estimate_gate_target_resized(
            gate_bgr,
            depth,
            transform=gate_transform,
        )
        self.aux = aux or {}
        target_rel_drone = self.aux.get("gate_detection_target_rel_drone", target_v_airsim)
        if target_rel_drone is None:
            self.target_v = None
            target_source = "none"
        else:
            self.target_v = airsim_to_normal_vector(target_rel_drone)
            target_source = "gate"
        if self.args.debug_print:
            gate_model_aux = self.aux.get("gate_model_aux", self.aux)
            gate_center = gate_model_aux.get("gate_center_px") if isinstance(gate_model_aux, dict) else None
            center_disp = None
            if gate_center is not None:
                gate_center = np.asarray(gate_center, dtype=np.float32).reshape(-1)
                if gate_center.size >= 2:
                    center_disp = (round(float(gate_center[0]), 3), round(float(gate_center[1]), 3))
            print(
                "[gate]",
                "center=", center_disp,
                "target_v=", None if self.target_v is None else np.round(self.target_v, 3),
                flush=True,
            )

        env_rot_ned = quaternion_to_rotation_matrix(state["orientation"])
        env_rot = sim_to_normal_rotation(env_rot_ned)
        linear_velocity = sim_to_normal(state["linear_velocity"])

        forward = env_rot[:, 0].copy()
        forward[2] = 0.0
        if np.linalg.norm(forward) < 1e-6:
            forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        forward = normalize(forward)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        left = normalize(np.cross(up, forward))
        yaw_only_rot = np.stack([forward, left, up], axis=-1)

        raw_target_v = None if self.target_v is None else np.array(self.target_v, copy=True)
        if raw_target_v is None:
            sim_world_target_delta = self.world_target_v - np.asarray(state["position"], dtype=np.float32)
            env_target_v = sim_to_normal(sim_world_target_delta)
            env_target_v[1] *= -1.0
            raw_target_v = env_target_v.copy()
            target_source = "world_fallback"
            local_target_v = env_target_v @ yaw_only_rot
            target_v_from_local = lambda vec: vec @ yaw_only_rot.T
        else:
            env_target_v = env_rot @ raw_target_v
            local_target_v = env_target_v @ yaw_only_rot
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

        depth_tensor = self.preprocess_depth(self.infinite_depth_like(depth))
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
            closest_gate = self.get_closest_track_gate(state["position"])
            print(
                "[debug]",
                "depth_input=", "infinite",
                "drone_position=", np.round(state["position"], 3),
                "closest_gate_id=", None if closest_gate is None else closest_gate["gate_id"],
                "closest_gate_position=", None if closest_gate is None else np.round(closest_gate["position_ned"], 3),
                "closest_gate_distance=", None if closest_gate is None else round(closest_gate["distance"], 3),
                "target_src=", target_source,
                "attitude_rpy=", np.round(current_rpy, 3),
                "raw_target_v=", np.round(raw_target_v, 3),
                "local_target_v=", np.round(local_target_v, 3),
                "a_pred=", np.round(a_pred, 3),
                "current_velocity=", np.round(local_velocity, 3),
                "thrust=", round(thrust, 3),
                "throttle=", round(throttle, 3),
                flush=True,
            )
        self.debug_counter += 1

        self.show_or_save_rgb_overlay(
            depth_rgb,
            frame_id=frame_id,
            aux=self.aux.get("gate_model_aux", self.aux),
        )


def main():
    args = build_args()
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
    print("Loading depth, gate, and control models...", flush=True)
    racer = MavlinkDepthGateRacer(sim_conn, shared_data, system_boot_ms, args)

    print("Arming drone...", flush=True)
    racer.arm()
    print("Starting MAVLink threaded depth/gate/control racer...", flush=True)
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
