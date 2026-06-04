import struct
import time
import threading

from pymavlink import mavutil

ENCAPSULATED_RACE_STATUS_MSG_ID = 1
ENCAPSULATED_TRACK_INFO_MSG_ID = 2

class MAVLinkRX:

    def __init__(self, mavlink_connection, data):
        self.mavlink_conn = mavlink_connection
        self.data = data
        self.thread = None
        self.is_running = False
        self.heartbeat_event = threading.Event()

        self.track_chunks = {}
        self.expected_num_track_chunks = {}

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

            # --------------------------------------------------------------------------------------
            # DATA_TRANSMISSION_HANDSHAKE - Repurposed and used for upcoming 'Track Data' packets
            # --------------------------------------------------------------------------------------
            elif msg.get_type() == "DATA_TRANSMISSION_HANDSHAKE":
                track_data_transfer_id = msg.width
                self.track_chunks[track_data_transfer_id] = {}
                self.expected_num_track_chunks[track_data_transfer_id] = msg.packets

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
            elif int(data_type) == ENCAPSULATED_TRACK_INFO_MSG_ID:
                self.on_track_data_packet(msg)

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

    def on_track_data_packet(self, msg):
        payload_size = int(getattr(msg, "size", len(msg.data)))
        raw_payload = bytes(msg.data[:payload_size])
        if len(raw_payload) < 3:
            return

        # header:
        #   data_type - ID of this message
        #   transfer_id - ID of the group of packets this chunk belongs to
        data_type, transfer_id = struct.unpack_from("<BH", raw_payload)
        if transfer_id not in self.expected_num_track_chunks:
            return

        self.track_chunks[transfer_id][msg.seqnr] = raw_payload[3:]
        if len(self.track_chunks[transfer_id]) == self.expected_num_track_chunks[transfer_id]:
            full_payload = bytes()
            for i in range(len(self.track_chunks[transfer_id])):
                full_payload += self.track_chunks[transfer_id][i]
            del self.track_chunks[transfer_id]
            del self.expected_num_track_chunks[transfer_id]
            self.on_track_data(full_payload)

    def on_track_data(self, payload):
        # header:
        #   num_gates - track gate count
        if len(payload) < 2:
            return

        num_gates, = struct.unpack_from("<H", payload)
        payload = payload[2:]
        gates = []
        gate_payload_size = struct.calcsize("<Hfffffffff")
        for i in range(num_gates):
            if len(payload) < gate_payload_size:
                break

            # Gate Info
            #   gate_id - range is 0 - num_gates
            #   position_ned_x, position_ned_y, position_ned_z - Position of gate in NED coordinates
            #   orientation_ned_w, orientation_ned_x, orientation_ned_y, orientation_ned_z - Orientation of gate in NED coordinates
            #   width - gate width in metres
            #   height - gate height in metres
            gate_id, position_ned_x, position_ned_y, position_ned_z, orientation_ned_w, orientation_ned_x, orientation_ned_y, orientation_ned_z, width, height = struct.unpack_from(
                "<Hfffffffff", payload)
            payload = payload[gate_payload_size:]
            gates.append({
                "gate_id": gate_id,
                "position_ned": (position_ned_x, position_ned_y, position_ned_z),
                "orientation_ned": (
                    orientation_ned_w,
                    orientation_ned_x,
                    orientation_ned_y,
                    orientation_ned_z,
                ),
                "width": width,
                "height": height,
            })

        self.data["track_gates"] = gates
        self.data["track_data"] = {
            "num_gates": len(gates),
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
