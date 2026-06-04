from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
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
import torch.nn.functional as F
from pymavlink import mavutil

from tonedio.mavlink_rx import MAVLinkRX
from timesync import TimeSync
from tonedio.utils import (
    DepthAnythingOnnxEstimator,
    DiffPhysModel,
    normalize,
    quaternion_to_rotation_matrix,
)
from tonedio.vision_rx import VisionRX


DEFAULT_MODELS_DIR = SCRIPT_DIR / "models"
DEFAULT_DEPTH_ONNX_PATH = DEFAULT_MODELS_DIR / "dn_model_latest.onnx"
DEFAULT_CONTROL_MODEL_PATH = DEFAULT_MODELS_DIR / "controlmodel.pth"

SIM_TO_NORMAL = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def sim_to_normal(vector):
    return SIM_TO_NORMAL @ np.asarray(vector, dtype=np.float32)


def sim_to_normal_rotation(rot):
    rot = np.asarray(rot, dtype=np.float32)
    return SIM_TO_NORMAL @ rot @ SIM_TO_NORMAL


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
    parser = ArgumentParser(description="Run the depth policy with a fixed forward target vector.")
    parser.add_argument("--server_ip", type=str, default="127.0.0.1")
    parser.add_argument("--server_udp_port", type=int, default=14550)
    parser.add_argument("--control_hz", type=float, default=20.0)
    parser.add_argument("--target_speed", type=float, default=7.0)
    parser.add_argument("--world_target_x", type=float, default=-100.0)
    parser.add_argument("--world_target_y", type=float, default=0.0)
    parser.add_argument("--world_target_z", type=float, default=0.0)
    parser.add_argument("--hover_throttle", type=float, default=0.6)
    parser.add_argument("--attitude_p_gain", type=float, default=1)
    parser.add_argument("--max_delta_roll", type=float, default=1)
    parser.add_argument("--max_delta_pitch", type=float, default=1)
    parser.add_argument("--max_delta_yaw", type=float, default=1)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--invert_forward_x", action="store_true", default=False)
    parser.add_argument("--no_odom", action="store_true", default=False)
    parser.add_argument("--debug_print", action="store_true", default=False)
    parser.add_argument("--debug_print_every", type=int, default=10)
    parser.add_argument("--debug_interval_sec", type=float, default=1.0)
    parser.add_argument("--debug_every_command", action="store_true", default=False)
    parser.add_argument("--viz_rgb", action="store_true", default=False)
    parser.add_argument("--viz_depth", action="store_true", default=False)
    parser.add_argument("--viz_depth_scale", type=int, default=4)
    parser.add_argument("--save_image_dir", type=str, default="")
    parser.add_argument("--save_image_every", type=int, default=10)
    parser.add_argument("--save_depth_dir", type=str, default="")
    parser.add_argument("--save_depth_every", type=int, default=10)

    parser.add_argument("--control_model_path", type=str, default=str(DEFAULT_CONTROL_MODEL_PATH))
    parser.add_argument("--depth_onnx_path", type=str, default=str(DEFAULT_DEPTH_ONNX_PATH))
    parser.add_argument("--depth_input_width", type=int, default=112)
    parser.add_argument("--depth_input_height", type=int, default=112)
    parser.add_argument("--depth_device", type=str, default="auto")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dim_obs", type=int, default=10)
    parser.add_argument("--dim_action", type=int, default=6)
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


class MavlinkDepthForwardRacer:
    def __init__(self, mavlink_conn, shared_data, system_boot_ms, args):
        self.mavlink_conn = mavlink_conn
        self.data = shared_data
        self.system_boot_ms = system_boot_ms
        self.args = args
        self.gravity = 9.81
        self.hidden = None
        self.last_frame_id = None
        self.debug_counter = 0
        self.last_idle_debug_time = 0.0
        self.last_control_debug_time = 0.0
        self.last_depth_debug_time = 0.0
        self.image_save_counter = 0
        self.depth_save_counter = 0
        self.world_target_v = np.array(
            [args.world_target_x, args.world_target_y, args.world_target_z],
            dtype=np.float32,
        )

        self.depth_estimator = DepthAnythingOnnxEstimator(
            onnx_path=args.depth_onnx_path,
            input_width=args.depth_input_width,
            input_height=args.depth_input_height,
            device=args.depth_device,
        )
        self.model = DiffPhysModel(
            args.control_model_path,
            dim_obs=args.dim_obs,
            dim_action=args.dim_action,
            device=args.device,
        )
        if args.viz_rgb:
            cv2.namedWindow("rgb", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("rgb", 640, 360)
            cv2.moveWindow("rgb", 40, 40)
        if args.viz_depth:
            cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("depth", 448, 448)
            cv2.moveWindow("depth", 720, 40)
        self.save_depth_dir = None
        self.save_image_dir = None
        if args.save_image_dir:
            self.save_image_dir = Path(args.save_image_dir)
            self.save_image_dir.mkdir(parents=True, exist_ok=True)
        if args.save_depth_dir:
            self.save_depth_dir = Path(args.save_depth_dir)
            self.save_depth_dir.mkdir(parents=True, exist_ok=True)

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

    def show_depth_map(self, depth):
        depth_display = depth.copy()
        depth_display[~np.isfinite(depth_display)] = np.nan
        valid_depth = depth_display[np.isfinite(depth_display) & (depth_display > 0.0)]
        if valid_depth.size == 0:
            now = time.time()
            if self.args.debug_print and now - self.last_depth_debug_time >= 1.0:
                self.last_depth_debug_time = now
                print("[debug_depth] no valid depth values to display", flush=True)
            return

        near = float(np.percentile(valid_depth, 2.0))
        far = float(np.percentile(valid_depth, 98.0))
        if far <= near:
            far = near + 1.0

        depth_display = np.clip(depth_display, near, far)
        depth_display = (depth_display - near) / (far - near)
        depth_display[~np.isfinite(depth_display)] = 0.0
        depth_display = (depth_display * 255.0).astype(np.uint8)
        scale = max(1, int(self.args.viz_depth_scale))
        if scale != 1:
            depth_display = cv2.resize(
                depth_display,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_NEAREST,
            )
        if self.save_depth_dir is not None:
            save_every = max(1, int(self.args.save_depth_every))
            if self.depth_save_counter % save_every == 0:
                out_path = self.save_depth_dir / f"depth_{self.depth_save_counter:06d}.png"
                cv2.imwrite(str(out_path), depth_display)
            self.depth_save_counter += 1
        cv2.imshow("depth", depth_display)
        cv2.waitKey(1)

    def save_image(self, image_bgr):
        if self.save_image_dir is None:
            return
        save_every = max(1, int(self.args.save_image_every))
        if self.image_save_counter % save_every == 0:
            out_path = self.save_image_dir / f"image_{self.image_save_counter:06d}.png"
            cv2.imwrite(str(out_path), image_bgr)
        self.image_save_counter += 1

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
        forward_vec = env_rot[:, 0] * 0.2 + input_target_v
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

    def debug_idle(self, reason, frame=None, state=None):
        if not self.args.debug_print:
            return
        now = time.time()
        if not self.args.debug_every_command and now - self.last_idle_debug_time < 1.0:
            return
        self.last_idle_debug_time = now
        frame_id = None if frame is None else frame.get("frame_id")
        print(
            "[debug_idle]",
            "reason=", reason,
            "frame=", frame_id,
            "last_frame=", self.last_frame_id,
            "has_state=", state is not None,
            "mav_msgs=", sorted(list(self.data.get("mavlink_message_types", []))),
            flush=True,
        )

    def update(self):
        frame = self.data.get("latest_frame")
        state = self.get_state()
        attitude = self.data.get("attitude")
        if frame is None or state is None:
            reason = "missing_frame" if frame is None else "missing_state"
            self.debug_idle(reason, frame=frame, state=state)
            time.sleep(1.0 / max(self.args.control_hz, 1.0))
            return
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
        self.save_image(image_bgr)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        depth = self.depth_estimator.predict_depth(rgb)

        env_rot_ned = quaternion_to_rotation_matrix(state["orientation"])
        env_rot = sim_to_normal_rotation(env_rot_ned)
        sim_position = np.asarray(state["position"], dtype=np.float32)
        linear_velocity = sim_to_normal(state["linear_velocity"])

        forward = env_rot[:, 0].copy()
        if self.args.invert_forward_x:
            forward = -forward
        forward[2] = 0.0
        if np.linalg.norm(forward) < 1e-6:
            forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        forward = normalize(forward)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        left = normalize(np.cross(up, forward))
        yaw_only_rot = np.stack([forward, left, up], axis=-1)
        current_yaw = math.atan2(float(forward[1]), float(forward[0]))
        attitude_yaw = None if attitude is None else float(attitude["yaw"])
        yaw_delta = None
        if attitude_yaw is not None:
            yaw_delta = math.atan2(
                math.sin(current_yaw - attitude_yaw),
                math.cos(current_yaw - attitude_yaw),
            )

        raw_world_target_v = self.world_target_v.copy()
        sim_world_target_delta = raw_world_target_v - sim_position
        world_target_delta = sim_to_normal(sim_world_target_delta)
        world_target_delta[1] *= -1.0
        target_v_norm = np.linalg.norm(world_target_delta)
        if target_v_norm > 1e-6:
            world_target_delta = (
                world_target_delta / target_v_norm * min(target_v_norm, self.args.target_speed)
            )
        else:
            world_target_delta = np.array([self.args.target_speed, 0.0, 0.0], dtype=np.float32)
        local_target_v = world_target_delta @ yaw_only_rot
        target_v = world_target_delta
        local_velocity = linear_velocity @ yaw_only_rot
        state_parts = [local_target_v, env_rot[:, 2], np.array([self.args.margin], dtype=np.float32)]
        if not self.args.no_odom:
            state_parts.insert(0, local_velocity)
        state_tensor = torch.as_tensor(np.concatenate(state_parts), dtype=torch.float32)[None]

        depth_tensor = self.preprocess_depth(depth)
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

        debug_print_every = max(1, int(self.args.debug_print_every))
        now = time.time()
        debug_due = (
            self.args.debug_every_command
            or self.debug_counter % debug_print_every == 0
            or now - self.last_control_debug_time >= max(0.0, float(self.args.debug_interval_sec))
        )
        if self.args.debug_print and debug_due:
            self.last_control_debug_time = now
            attitude_rates = (
                float(attitude.get("rollspeed", 0.0)),
                float(attitude.get("pitchspeed", 0.0)),
                float(attitude.get("yawspeed", 0.0)),
            )
            print(
                "[debug]",
                "frame=", frame_id,
                "state_src=", state.get("source", "unknown"),
                "raw_position=", np.round(sim_position, 3),
                "raw_vel=", np.round(state["linear_velocity"], 3),
                "attitude_rpy=", np.round(current_rpy, 3),
                "attitude_rates=", np.round(attitude_rates, 3),
                "raw_forward=", np.round(env_rot[:, 0], 3),
                "forward=", np.round(forward, 3),
                "invert_forward_x=", self.args.invert_forward_x,
                "target_v=", np.round(local_target_v, 3),
                "raw_world_target_v=", np.round(raw_world_target_v, 3),
                "sim_world_target_delta=", np.round(sim_world_target_delta, 3),
                "world_target_delta=", np.round(world_target_delta, 3),
                "attitude_target_v=", np.round(target_v, 3),
                "velocity=", np.round(local_velocity, 3),
                "a_pred=", np.round(a_pred, 3),
                "current_yaw=", round(current_yaw, 3),
                "attitude_yaw=", None if attitude_yaw is None else round(attitude_yaw, 3),
                "yaw_delta=", None if yaw_delta is None else round(yaw_delta, 3),
                "target_rpy=", np.round(target_rpy, 3),
                "current_rpy=", np.round(current_rpy, 3),
                "error_rpy=", np.round(error_rpy, 3),
                "command_delta_rpy=", np.round(command_delta_rpy, 3),
                "target_rpy_deg=", np.round(np.degrees(target_rpy), 1),
                "command_delta_rpy_deg=", np.round(np.degrees(command_delta_rpy), 1),
                "throttle=", round(throttle, 3),
                "thrust=", round(thrust, 3),
                flush=True,
            )
        self.debug_counter += 1

        if self.args.viz_rgb:
            cv2.imshow("rgb", image_bgr)
            cv2.waitKey(1)
        if self.args.viz_depth:
            self.show_depth_map(depth)


def main():
    args = build_args()
    shared_data = {}
    system_boot_ms = int(time.time() * 1000)

    sim_conn = mavutil.mavlink_connection(f"udpin:{args.server_ip}:{args.server_udp_port}")
    print("Waiting for heartbeat...", flush=True)
    sim_conn.wait_heartbeat()
    print(f"Connected to system: {sim_conn.target_system}", flush=True)

    mavlink_rx = MAVLinkRX.create_mavlink_rx(sim_conn, shared_data)
    ts_loop = TimeSync.create_timesync(sim_conn, shared_data)
    vision_rx = VisionRX(shared_data)
    racer = MavlinkDepthForwardRacer(sim_conn, shared_data, system_boot_ms, args)

    print("Arming drone...", flush=True)
    racer.arm()
    print("Starting MAVLink depth/forward racer...", flush=True)
    try:
        while True:
            racer.update()
    except KeyboardInterrupt:
        pass
    finally:
        stop_and_join(ts_loop)
        stop_and_join(mavlink_rx)
        stop_and_join(vision_rx)
        if args.viz_rgb:
            cv2.destroyWindow("rgb")
        if args.viz_depth:
            cv2.destroyWindow("depth")
        print("Client exited!", flush=True)


if __name__ == "__main__":
    main()
