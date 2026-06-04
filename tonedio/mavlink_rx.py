import struct
import time
import threading

from pymavlink import mavutil

ENCAPSULATED_RACE_STATUS_MSG_ID = 1

class MAVLinkRX:

    def __init__(self, mavlink_connection, data):
        self.mavlink_conn = mavlink_connection
        self.data = data
        self.thread = None
        self.is_running = False
        self.heartbeat_event = threading.Event()

    @classmethod
    def create_mavlink_rx(cls, mavlink_connection, data):
        rx = cls(mavlink_connection, data)
        rx.thread = threading.Thread(
            target=rx.mavlink_receive_loop,
            daemon = False
        )
        rx.is_running = True
        rx.thread.start()
        return rx

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def wait_heartbeat(self, timeout=None):
        return self.heartbeat_event.wait(timeout=timeout)

    def mavlink_receive_loop(self):
        """
        Continuously receive MAVLink messages without blocking.
        """
        while self.is_running:

            try:
                msg = self.mavlink_conn.recv_match(blocking=False)
            except ConnectionResetError:
                print('WARNING: ConnectionResetError was thrown. No longer listening to MAVLink port.')
                return

            if msg is None:
                time.sleep(0.001)
                continue

            msg_type = msg.get_type()

            if msg_type == "BAD_DATA":
                continue

            # --------------------------------------------------------------------------------------
            # HEARTBEAT
            # --------------------------------------------------------------------------------------
            if msg_type == "HEARTBEAT":
                self.on_heartbeat(msg)

            # --------------------------------------------------------------------------------------
            # TIMESYNC
            # --------------------------------------------------------------------------------------
            elif msg_type == "TIMESYNC":
                self.on_timesync(msg)

            # --------------------------------------------------------------------------------------
            # ATTITUDE
            # --------------------------------------------------------------------------------------
            elif msg_type == "ATTITUDE":
                self.on_attitude(msg)

            # --------------------------------------------------------------------------------------
            # LOCAL_POSITION_NED
            # --------------------------------------------------------------------------------------
            elif msg_type == "LOCAL_POSITION_NED":
                self.on_local_position_ned(msg)

            # --------------------------------------------------------------------------------------
            # ODOMETRY
            # --------------------------------------------------------------------------------------
            elif msg_type == "ODOMETRY":
                self.on_odometry(msg)

            # --------------------------------------------------------------------------------------
            # HIGHRES_IMU
            # --------------------------------------------------------------------------------------
            elif msg_type == "HIGHRES_IMU":
                self.on_highres_imu(msg)

            # --------------------------------------------------------------------------------------
            # ENCAPSULATED_DATA
            # --------------------------------------------------------------------------------------
            elif msg_type == "ENCAPSULATED_DATA":
                self.on_encapsulated_data(msg)

            # --------------------------------------------------------------------------------------
            # ACTUATOR_OUTPUT_STATUS
            # --------------------------------------------------------------------------------------
            elif msg_type == "ACTUATOR_OUTPUT_STATUS":
                self.on_actuator_output_status(msg)

            # --------------------------------------------------------------------------------------
            # COLLISION
            # --------------------------------------------------------------------------------------
            elif msg_type == "COLLISION":
                self.on_collision(msg)

    def on_heartbeat(self, msg):
        armed = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        self.data["heartbeat"] = {
            "armed": bool(armed),
            "base_mode": msg.base_mode,
            "custom_mode": msg.custom_mode,
            "type": msg.type,
            "autopilot": msg.autopilot,
            "timestamp": time.time(),
        }
        self.heartbeat_event.set()

    def on_timesync(self, msg):
        request_time = msg.ts1
        response_time = msg.tc1

    def on_attitude(self, msg):
        roll = msg.roll
        pitch = msg.pitch
        yaw = msg.yaw
        roll_speed = msg.rollspeed
        pitch_speed = msg.pitchspeed
        yaw_speed = msg.yawspeed
        time_boot_ms = msg.time_boot_ms
        self.data["attitude"] = {
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "rollspeed": roll_speed,
            "pitchspeed": pitch_speed,
            "yawspeed": yaw_speed,
            "time_boot_ms": time_boot_ms,
        }

    def on_local_position_ned(self, msg):
        pos_x = msg.x
        pos_y = msg.y
        pos_z = msg.z
        vel_x = msg.vx
        vel_y = msg.vy
        vel_z = msg.vz
        time_boot_ms = msg.time_boot_ms
        self.data["local_position_ned"] = {
            "position": (pos_x, pos_y, pos_z),
            "linear_velocity": (vel_x, vel_y, vel_z),
            "time_boot_ms": time_boot_ms,
        }

    def on_odometry(self, msg):
        pos_x, pos_y, pos_z = msg.x, msg.y, msg.z
        qx, qy, qz, qw = msg.q[1], msg.q[2], msg.q[3], msg.q[0]
        vel_x, vel_y, vel_z = msg.vx, msg.vy, msg.vz
        roll_speed = msg.rollspeed
        pitch_speed = msg.pitchspeed
        yaw_speed = msg.yawspeed
        time_boot_us = msg.time_usec
        reset_count = msg.reset_counter
        self.data["odometry"] = {
            "position": (pos_x, pos_y, pos_z),
            "orientation": (qw, qx, qy, qz),
            "linear_velocity": (vel_x, vel_y, vel_z),
            "angular_velocity": (roll_speed, pitch_speed, yaw_speed),
            "time_boot_us": time_boot_us,
            "reset_count": reset_count,
        }

    def on_highres_imu(self, msg):
        acceleration_x, acceleration_y, acceleration_z = msg.xacc, msg.yacc, msg.zacc
        gyro_x, gyro_y, gyro_z = msg.xgyro, msg.ygyro, msg.zgyro
        time_boot_us = msg.time_usec

    def on_encapsulated_data(self, msg):
        if msg:
            payload_size = int(getattr(msg, "size", len(msg.data)))
            raw_payload = bytes(msg.data[:payload_size])
            if not raw_payload:
                return
            data_type = raw_payload[0]

            if int(data_type) == ENCAPSULATED_RACE_STATUS_MSG_ID:
                self.on_race_status(msg)

    def on_race_status(self, msg):
        payload_size = int(getattr(msg, "size", len(msg.data)))
        raw_payload = bytes(msg.data[:payload_size])
        # data_type - ID of this message
        # sim_boot_time_ms - elapsed ms on server since sim boot
        # race_start_boot_time_ms - elapsed ms on server since sim boot when race started. None or < 0 if race has not started
        # race_finish_time_ns - elapsed ns on server since sim boot when race finished. None or < 0 if race is ongoing
        # active_gate_index - current index of target race gate
        # last_gate_race_time - race time in seconds when last gate was passed
        data_type, sim_boot_time_ms, race_start_boot_time_ms, race_finish_time_ns, active_gate_index, last_gate_race_time = struct.unpack_from(
            "<BQqqIq", raw_payload)
        self.data["race_status"] = {
            "sim_boot_time_ms": sim_boot_time_ms,
            "race_start_boot_time_ms": race_start_boot_time_ms,
            "race_finish_time_ns": race_finish_time_ns,
            "active_gate_index": active_gate_index,
            "last_gate_race_time": last_gate_race_time,
            "timestamp": time.time(),
        }

    def on_actuator_output_status(self, msg):
        time_boot_us = msg.time_usec
        motor_front_left = msg.actuator[0]
        motor_front_right = msg.actuator[1]
        motor_back_left = msg.actuator[2]
        motor_back_right = msg.actuator[3]

    def on_collision(self, msg):
        # Collision IDs
        # 1001 - Gate
        # 1002 - Environment
        collision_id = msg.id

        threat_level = msg.threat_level # 1-2 with 2 being higher impact collision
        impact = msg.horizontal_minimum_delta # this is not a delta - it is the impulse magnitude in kg m/s
