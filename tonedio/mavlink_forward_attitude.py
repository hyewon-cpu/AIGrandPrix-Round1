from __future__ import annotations

from argparse import ArgumentParser
import math
import time

from pymavlink import mavutil


def build_args():
    parser = ArgumentParser(description="Command the AI-GP drone forward with SET_ATTITUDE_TARGET.")
    parser.add_argument("--server_ip", type=str, default="127.0.0.1")
    parser.add_argument("--server_udp_port", type=int, default=14550)
    parser.add_argument("--roll", type=float, default=0.0, help="Commanded roll in radians.")
    parser.add_argument("--pitch", type=float, default=0.12, help="Commanded pitch in radians. Flip sign if it goes backward.")
    parser.add_argument("--yaw", type=float, default=0.0, help="Commanded yaw in radians.")
    parser.add_argument("--throttle", type=float, default=0.5)
    parser.add_argument(
        "--absolute_attitude",
        action="store_true",
        default=False,
        help="Send the target quaternion directly instead of current-to-target delta quaternion.",
    )
    parser.add_argument("--delta_gain", type=float, default=0.25, help="Gain applied to delta RPY before quaternion conversion.")
    parser.add_argument("--max_delta_roll", type=float, default=0.08, help="Max roll correction per command in radians.")
    parser.add_argument("--max_delta_pitch", type=float, default=0.08, help="Max pitch correction per command in radians.")
    parser.add_argument("--max_delta_yaw", type=float, default=0.08, help="Max yaw correction per command in radians.")
    parser.add_argument("--command_hz", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run. 0 means run until Ctrl+C.")
    parser.add_argument("--no_arm", action="store_true", default=False)
    parser.add_argument("--debug_print", action="store_true", default=False)
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


def quaternion_conjugate(q):
    return [q[0], -q[1], -q[2], -q[3]]


def quaternion_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


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


def limit_delta_rpy(delta_rpy, gain, max_delta):
    return (
        clamp(-delta_rpy[0] * gain, -max_delta[0], max_delta[0]),
        clamp(-delta_rpy[1] * gain, -max_delta[1], max_delta[1]),
        clamp(delta_rpy[2] * gain, -max_delta[2], max_delta[2]),
    )


def attitude_delta_quaternion(delta_rpy):
    return normalize_quaternion(euler_to_quaternion(*delta_rpy))


def arm(conn):
    conn.mav.command_long_send(
        conn.target_system,
        conn.target_component,
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


def send_attitude_quaternion(conn, system_boot_ms, q, throttle):
    now_ms = int(time.time() * 1000)
    type_mask = (
        mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE
        | mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE
        | mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE
    )
    conn.mav.set_attitude_target_send(
        now_ms - system_boot_ms,
        conn.target_system,
        conn.target_component,
        type_mask,
        normalize_quaternion(q),
        0.0,
        0.0,
        0.0,
        float(max(0.0, min(1.0, throttle))),
    )


def send_attitude(conn, system_boot_ms, roll, pitch, yaw, throttle):
    send_attitude_quaternion(conn, system_boot_ms, euler_to_quaternion(roll, pitch, yaw), throttle)


def update_telemetry(conn, telemetry):
    while True:
        msg = conn.recv_match(blocking=False)
        if msg is None:
            return telemetry
        msg_type = msg.get_type()
        if msg_type == "BAD_DATA":
            continue
        telemetry.setdefault("message_counts", {})
        telemetry["message_counts"][msg_type] = telemetry["message_counts"].get(msg_type, 0) + 1
        telemetry["last_msg_type"] = msg_type
        if msg_type == "ATTITUDE":
            telemetry["attitude"] = {
                "rpy": (float(msg.roll), float(msg.pitch), float(msg.yaw)),
                "rates": (float(msg.rollspeed), float(msg.pitchspeed), float(msg.yawspeed)),
            }
        elif msg_type == "LOCAL_POSITION_NED":
            telemetry["local_position_ned"] = {
                "position": (float(msg.x), float(msg.y), float(msg.z)),
                "linear_velocity": (float(msg.vx), float(msg.vy), float(msg.vz)),
            }
        elif msg_type == "ODOMETRY":
            telemetry["odometry"] = {
                "position": (float(msg.x), float(msg.y), float(msg.z)),
                "linear_velocity": (float(msg.vx), float(msg.vy), float(msg.vz)),
            }
        elif msg_type == "HIGHRES_IMU":
            telemetry["highres_imu"] = {
                "acc": (float(msg.xacc), float(msg.yacc), float(msg.zacc)),
                "gyro": (float(msg.xgyro), float(msg.ygyro), float(msg.zgyro)),
            }
        elif msg_type == "COLLISION":
            telemetry["collision"] = msg.to_dict()


def rounded_tuple(values):
    if values is None:
        return None
    return tuple(round(float(value), 3) for value in values)


def main():
    args = build_args()
    conn = mavutil.mavlink_connection(f"udpin:{args.server_ip}:{args.server_udp_port}")
    system_boot_ms = int(time.time() * 1000)

    print("Waiting for heartbeat...", flush=True)
    conn.wait_heartbeat()
    print(f"Connected to system: {conn.target_system}", flush=True)

    if not args.no_arm:
        print("Arming drone...", flush=True)
        arm(conn)

    print(
        f"Sending attitude command: roll={args.roll:.3f}, pitch={args.pitch:.3f}, throttle={args.throttle:.3f}",
        flush=True,
    )
    period = 1.0 / max(args.command_hz, 1.0)
    start_time = time.time()
    telemetry = {}
    command_count = 0
    command_yaw = args.yaw

    try:
        while args.duration <= 0.0 or time.time() - start_time < args.duration:
            telemetry = update_telemetry(conn, telemetry)
            attitude = telemetry.get("attitude", {})
            current_rpy = attitude.get("rpy")
            target_rpy = (args.roll, args.pitch, command_yaw)
            target_q = normalize_quaternion(euler_to_quaternion(*target_rpy))
            if args.absolute_attitude or current_rpy is None:
                out_q = target_q
                attitude_mode = "absolute"
                delta_rpy = None
                raw_delta_rpy = None
            else:
                raw_delta_rpy = subtract_rpy(target_rpy, current_rpy)
                delta_rpy = limit_delta_rpy(
                    raw_delta_rpy,
                    args.delta_gain,
                    (args.max_delta_roll, args.max_delta_pitch, args.max_delta_yaw),
                )
                out_q = attitude_delta_quaternion(delta_rpy)
                attitude_mode = "delta"

            send_attitude_quaternion(conn, system_boot_ms, out_q, args.throttle)

            if args.debug_print:
                local_position = telemetry.get("local_position_ned", {})
                odometry = telemetry.get("odometry", {})
                imu = telemetry.get("highres_imu", {})
                print(
                    "[debug]",
                    "cmd=", command_count,
                    "mode=", attitude_mode,
                    "target_rpy=", rounded_tuple(target_rpy),
                    "out_rpy_deg=", rounded_tuple(
                        (math.degrees(target_rpy[0]), math.degrees(target_rpy[1]), math.degrees(target_rpy[2]))
                    ),
                    "raw_delta_rpy=", rounded_tuple(raw_delta_rpy),
                    "raw_delta_rpy_deg=", None
                    if raw_delta_rpy is None
                    else rounded_tuple(tuple(math.degrees(value) for value in raw_delta_rpy)),
                    "delta_rpy=", rounded_tuple(delta_rpy),
                    "delta_rpy_deg=", None
                    if delta_rpy is None
                    else rounded_tuple(tuple(math.degrees(value) for value in delta_rpy)),
                    "out_q=", rounded_tuple(out_q),
                    "out_throttle=", round(args.throttle, 3),
                    "last_in=", telemetry.get("last_msg_type"),
                    "local_pos=", rounded_tuple(local_position.get("position")),
                    "local_vel=", rounded_tuple(local_position.get("linear_velocity")),
                    "odom_pos=", rounded_tuple(odometry.get("position")),
                    "odom_vel=", rounded_tuple(odometry.get("linear_velocity")),
                    "att_rpy=", rounded_tuple(attitude.get("rpy")),
                    "imu_acc=", rounded_tuple(imu.get("acc")),
                    "collision=", telemetry.get("collision"),
                    flush=True,
                )
            command_count += 1
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        send_attitude(conn, system_boot_ms, 0.0, 0.0, command_yaw, 0.0)
        print("Client exited!", flush=True)


if __name__ == "__main__":
    main()
