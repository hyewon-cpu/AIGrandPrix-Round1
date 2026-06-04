from __future__ import annotations

from argparse import ArgumentParser
import time

from pymavlink import mavutil


CLIENT_TO_SIM_MESSAGES = {
    "COMMAND_LONG": "Used here only for arm/disarm commands when requested by a script.",
    "SET_POSITION_TARGET_LOCAL_NED": "Position/velocity/acceleration/yaw/yaw-rate setpoints via type_mask.",
    "SET_ATTITUDE_TARGET": "Attitude quaternion/body-rate/throttle setpoints via type_mask.",
}


def build_args():
    parser = ArgumentParser(description="Monitor MAVLink telemetry published by the AI-GP simulator.")
    parser.add_argument("--server_ip", type=str, default="127.0.0.1")
    parser.add_argument("--server_udp_port", type=int, default=14550)
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run. 0 means run until Ctrl+C.")
    parser.add_argument("--print_every", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true", default=False, help="Print every received message.")
    parser.add_argument("--include_bad_data", action="store_true", default=False)
    return parser.parse_args()


def compact_value(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, list):
        return [compact_value(item) for item in value]
    return value


def compact_message(msg):
    data = msg.to_dict()
    data.pop("mavpackettype", None)
    return {key: compact_value(value) for key, value in data.items()}


def print_client_surface():
    print("Client -> Simulator messages this project can publish:", flush=True)
    for name, description in CLIENT_TO_SIM_MESSAGES.items():
        print(f"  {name}: {description}", flush=True)
    print("", flush=True)


def print_summary(message_counts, latest_messages):
    print("[summary]", flush=True)
    for msg_type in sorted(message_counts):
        fields = latest_messages.get(msg_type, {})
        print(
            f"  {msg_type}: count={message_counts[msg_type]} fields={list(fields.keys())}",
            flush=True,
        )


def main():
    args = build_args()
    conn = mavutil.mavlink_connection(f"udpin:{args.server_ip}:{args.server_udp_port}")

    print_client_surface()
    print("Waiting for heartbeat...", flush=True)
    conn.wait_heartbeat()
    print(f"Connected to system: {conn.target_system}", flush=True)
    print("Simulator -> Client telemetry live monitor started.", flush=True)

    start_time = time.time()
    last_summary_time = 0.0
    message_counts = {}
    latest_messages = {}

    try:
        while args.duration <= 0.0 or time.time() - start_time < args.duration:
            msg = conn.recv_match(blocking=False)
            if msg is None:
                now = time.time()
                if now - last_summary_time >= max(0.1, args.print_every):
                    last_summary_time = now
                    print_summary(message_counts, latest_messages)
                time.sleep(0.001)
                continue

            msg_type = msg.get_type()
            if msg_type == "BAD_DATA" and not args.include_bad_data:
                continue

            message_counts[msg_type] = message_counts.get(msg_type, 0) + 1
            latest_messages[msg_type] = compact_message(msg)

            if args.verbose:
                print(
                    "[recv]",
                    "type=", msg_type,
                    "count=", message_counts[msg_type],
                    "fields=", latest_messages[msg_type],
                    flush=True,
                )
    except KeyboardInterrupt:
        pass
    finally:
        print_summary(message_counts, latest_messages)
        print("Monitor exited!", flush=True)


if __name__ == "__main__":
    main()
