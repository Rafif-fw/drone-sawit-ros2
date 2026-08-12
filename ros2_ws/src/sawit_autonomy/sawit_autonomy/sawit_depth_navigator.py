#!/usr/bin/env python3

import math
import time
from enum import Enum

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)


class MissionState(Enum):
    INIT = 0
    TAKEOFF = 1
    SCAN = 2
    APPROACH_DEPTH_TARGET = 3
    MARK_SCANNED = 4
    ROTATE_SEARCH = 5
    BACKTRACK = 6
    FORWARD_SEARCH = 7
    FINISHED = 8


class SawitDepthNavigator(Node):
    def __init__(self):
        super().__init__("sawit_depth_distance_navigator")

        # =========================================================
        # BASIC MISSION PARAMETER
        # =========================================================

        # PX4 NED: z negatif = naik
        self.takeoff_alt = -2.5
        self.flight_alt = -2.5

        # Untuk uji coba awal bisa ganti 3.
        # Kalau sudah stabil, pakai 16.
        self.max_scanned_trees = 16

        # Topic depth camera kamu sudah 32FC1
        self.camera_topic = "/camera"

        # =========================================================
        # DEPTH DETECTION PARAMETER
        # =========================================================

        # Depth 32FC1 = meter.
        self.depth_min_m = 1.5
        self.depth_max_m = 20.0

        # Kalau median depth target <= ini, pohon dianggap sudah dikunjungi.
        # Dinaikkan dari 3.0 ke 4.5 agar percobaan depth tidak terlalu sulit.
        self.scan_depth_m = 4.5

        # Kalau min depth terlalu dekat, langsung scan supaya tidak nabrak.
        self.danger_depth_m = 3.0

        # ROI dibuat tengah-atas agar lantai bawah tidak ikut terdeteksi.
        # Dari log sebelumnya target palsu ada di cy sekitar 411.
        self.roi_x_min_ratio = 0.20
        self.roi_x_max_ratio = 0.80
        self.roi_y_min_ratio = 0.05
        self.roi_y_max_ratio = 0.55

        # Ambil bagian depth terdekat saja.
        self.closest_percentile = 15.0

        # Minimal pixel objek.
        self.min_target_pixels = 80

        # Area terlalu besar kemungkinan ground/background.
        self.max_target_area_ratio = 0.18

        # Reject jika centroid target terlalu bawah frame.
        # Walau ROI sudah dipotong, ini safety tambahan.
        self.max_target_cy_ratio = 0.65

        # =========================================================
        # NAVIGATION PARAMETER
        # =========================================================

        # Saat target di tengah dan belum dekat, drone maju pelan.
        self.forward_step = 0.35

        # Kalau tidak ada target, maju eksplorasi.
        self.search_forward_step = 2.0

        # Jangan langsung dianggap reached.
        self.forward_search_reached_radius = 0.25

        # Setelah scan, mundur sedikit.
        self.backtrack_step = 0.70

        # Target dianggap tengah kalau error_x kurang dari ini.
        self.center_tolerance_px = 65

        # Koreksi yaw maksimum.
        self.max_yaw_correction_deg = 15.0

        # Rotasi pencarian.
        self.rotate_angle = math.radians(180.0)
        self.rotate_duration = 3.5

        # Kalau target hilang saat approach.
        self.lost_target_limit = 8

        # Kalau scan gagal beberapa kali, maju eksplorasi.
        self.scan_fail_limit = 2

        # Minimal step sebelum boleh scan berdasarkan depth.
        self.min_approach_steps_before_scan = 4

        # Penting:
        # Jangan scan paksa berdasarkan jumlah step.
        # Sebelumnya ini bikin target lantai tetap dianggap scanned.
        self.max_approach_steps = 999

        # =========================================================
        # STATE
        # =========================================================

        self.state = MissionState.INIT

        self.local_pos = np.array([0.0, 0.0, 0.0], dtype=float)
        self.yaw = 0.0
        self.vehicle_status = None

        self.offboard_counter = 0

        self.current_target = None
        self.previous_point = None

        self.latest_target = None
        self.last_image_time = 0.0

        self.scanned_count = 0
        self.visited_points = []

        self.lost_target_counter = 0
        self.scan_fail_counter = 0
        self.approach_step_counter = 0

        self.rotate_start_time = None
        self.target_yaw = 0.0

        self.land_sent = False

        # =========================================================
        # QOS
        # =========================================================

        self.px4_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # =========================================================
        # SUBSCRIBER
        # =========================================================

        self.sub_image = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10,
        )

        self.sub_local_pos = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self.local_position_callback,
            self.px4_qos,
        )

        self.sub_status = self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status_v4",
            self.vehicle_status_callback,
            self.px4_qos,
        )

        # =========================================================
        # PUBLISHER PX4
        # =========================================================

        self.pub_offboard = self.create_publisher(
            OffboardControlMode,
            "/fmu/in/offboard_control_mode",
            10,
        )

        self.pub_setpoint = self.create_publisher(
            TrajectorySetpoint,
            "/fmu/in/trajectory_setpoint",
            10,
        )

        self.pub_command = self.create_publisher(
            VehicleCommand,
            "/fmu/in/vehicle_command",
            10,
        )

        # Marker RViz
        self.pub_markers = self.create_publisher(
            MarkerArray,
            "/sawit/depth_distance_markers",
            10,
        )

        self.timer = self.create_timer(0.1, self.main_loop)

        self.get_logger().info("Sawit Depth Distance Navigator started")
        self.get_logger().info(f"Camera topic: {self.camera_topic}")
        self.get_logger().info("Input depth: 32FC1 distance image")
        self.get_logger().info("Logic: detect depth target in upper ROI, align yaw, move forward, scan by distance")
        self.get_logger().info(
            f"depth_min={self.depth_min_m}, depth_max={self.depth_max_m}, "
            f"scan_depth={self.scan_depth_m}, danger_depth={self.danger_depth_m}"
        )
        self.get_logger().info(
            f"ROI x={self.roi_x_min_ratio}-{self.roi_x_max_ratio}, "
            f"y={self.roi_y_min_ratio}-{self.roi_y_max_ratio}"
        )
        self.get_logger().info(
            f"forward_search_step={self.search_forward_step}, "
            f"forward_search_reached_radius={self.forward_search_reached_radius}"
        )
        self.get_logger().info("Step fallback scan disabled. Target must be close by depth.")

    # =============================================================
    # CALLBACK
    # =============================================================

    def local_position_callback(self, msg: VehicleLocalPosition):
        self.local_pos = np.array([msg.x, msg.y, msg.z], dtype=float)

        if math.isfinite(msg.heading):
            self.yaw = float(msg.heading)

    def vehicle_status_callback(self, msg: VehicleStatus):
        self.vehicle_status = msg

    def image_callback(self, msg: Image):
        depth = self.image_to_depth_meters(msg)

        if depth is None:
            self.latest_target = None
            return

        self.latest_target = self.detect_depth_target(depth)
        self.last_image_time = time.time()

    # =============================================================
    # DEPTH IMAGE PROCESSING
    # =============================================================

    def image_to_depth_meters(self, msg: Image):
        h = msg.height
        w = msg.width
        enc = msg.encoding.lower()

        try:
            if enc == "32fc1":
                arr = np.frombuffer(msg.data, dtype=np.float32)
                arr = arr.reshape((h, w))

                depth = np.nan_to_num(
                    arr,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                return depth.astype(np.float32)

            if enc in ["16uc1", "mono16"]:
                arr = np.frombuffer(msg.data, dtype=np.uint16)
                arr = arr.reshape((h, w)).astype(np.float32)

                # Biasanya 16UC1 dalam milimeter.
                depth = arr / 1000.0

                return depth.astype(np.float32)

            self.get_logger().warn(
                f"Unsupported depth encoding: {msg.encoding}. "
                f"Expected 32FC1 or 16UC1."
            )
            return None

        except Exception as e:
            self.get_logger().error(f"image_to_depth_meters error: {e}")
            return None

    def detect_depth_target(self, depth):
        h, w = depth.shape

        x1 = int(w * self.roi_x_min_ratio)
        x2 = int(w * self.roi_x_max_ratio)
        y1 = int(h * self.roi_y_min_ratio)
        y2 = int(h * self.roi_y_max_ratio)

        roi = depth[y1:y2, x1:x2]

        if roi.size == 0:
            return None

        valid_mask = (
            np.isfinite(roi)
            & (roi > self.depth_min_m)
            & (roi < self.depth_max_m)
        )

        ys, xs = np.where(valid_mask)

        if len(xs) < self.min_target_pixels:
            return None

        valid_depth = roi[ys, xs]

        # Ambil depth yang paling dekat dari ROI.
        cutoff = np.percentile(valid_depth, self.closest_percentile)
        close_mask = valid_depth <= cutoff

        xs_close = xs[close_mask]
        ys_close = ys[close_mask]
        depth_close = valid_depth[close_mask]

        if len(xs_close) < self.min_target_pixels:
            return None

        area_ratio = len(xs_close) / float(w * h)

        if area_ratio > self.max_target_area_ratio:
            return None

        cx = float(np.mean(xs_close)) + x1
        cy = float(np.mean(ys_close)) + y1

        # Safety: reject target kalau centroid terlalu bawah.
        if cy > h * self.max_target_cy_ratio:
            return None

        mean_depth = float(np.mean(depth_close))
        median_depth = float(np.median(depth_close))
        min_depth = float(np.min(depth_close))

        error_x = cx - (w / 2.0)

        return {
            "type": "depth_distance",
            "cx": cx,
            "cy": cy,
            "error_x": error_x,
            "area_ratio": area_ratio,
            "mean_depth": mean_depth,
            "median_depth": median_depth,
            "min_depth": min_depth,
            "width": w,
            "height": h,
            "pixels": int(len(xs_close)),
            "cutoff_depth": float(cutoff),
        }

    # =============================================================
    # PX4 HELPERS
    # =============================================================

    def timestamp(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    def publish_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.timestamp = self.timestamp()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False

        self.pub_offboard.publish(msg)

    def publish_position_setpoint(self, target, yaw=None):
        msg = TrajectorySetpoint()
        msg.timestamp = self.timestamp()

        msg.position = [
            float(target[0]),
            float(target[1]),
            float(target[2]),
        ]

        msg.velocity = [float("nan"), float("nan"), float("nan")]
        msg.acceleration = [float("nan"), float("nan"), float("nan")]
        msg.jerk = [float("nan"), float("nan"), float("nan")]

        if yaw is None:
            msg.yaw = float(self.yaw)
        else:
            msg.yaw = float(yaw)

        msg.yawspeed = float("nan")

        self.pub_setpoint.publish(msg)

    def send_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.timestamp = self.timestamp()

        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.param3 = 0.0
        msg.param4 = 0.0
        msg.param5 = 0.0
        msg.param6 = 0.0
        msg.param7 = 0.0

        msg.command = command

        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True

        self.pub_command.publish(msg)

    def set_offboard_mode(self):
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            1.0,
            6.0,
        )

    def arm(self):
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            1.0,
        )

    def land(self):
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_NAV_LAND,
            0.0,
            0.0,
        )

    # =============================================================
    # NAV HELPERS
    # =============================================================

    def make_forward_target(self, step):
        x = self.local_pos[0] + step * math.cos(self.yaw)
        y = self.local_pos[1] + step * math.sin(self.yaw)

        return np.array([x, y, self.flight_alt], dtype=float)

    def make_back_target(self, step):
        x = self.local_pos[0] - step * math.cos(self.yaw)
        y = self.local_pos[1] - step * math.sin(self.yaw)

        return np.array([x, y, self.flight_alt], dtype=float)

    def distance_to(self, target):
        if target is None:
            return 9999.0

        return float(np.linalg.norm(self.local_pos - target))

    def is_target_centered(self, target):
        if target is None:
            return False

        return abs(target["error_x"]) <= self.center_tolerance_px

    def is_target_close_enough(self, target):
        if target is None:
            return False

        return target["median_depth"] <= self.scan_depth_m

    def is_target_danger_close(self, target):
        if target is None:
            return False

        return target["min_depth"] <= self.danger_depth_m

    # =============================================================
    # RVIZ MARKER
    # =============================================================

    def publish_markers(self):
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()

        for i, p in enumerate(self.visited_points):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = now
            marker.ns = "depth_visited_points"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = float(p[0])
            marker.pose.position.y = float(p[1])
            marker.pose.position.z = 0.5
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.8
            marker.scale.y = 0.8
            marker.scale.z = 0.8

            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            arr.markers.append(marker)

        self.pub_markers.publish(arr)

    # =============================================================
    # MAIN LOOP
    # =============================================================

    def main_loop(self):
        self.publish_offboard_heartbeat()
        self.publish_markers()

        if self.state == MissionState.INIT:
            self.handle_init()

        elif self.state == MissionState.TAKEOFF:
            self.handle_takeoff()

        elif self.state == MissionState.SCAN:
            self.handle_scan()

        elif self.state == MissionState.APPROACH_DEPTH_TARGET:
            self.handle_approach_depth_target()

        elif self.state == MissionState.MARK_SCANNED:
            self.handle_mark_scanned()

        elif self.state == MissionState.ROTATE_SEARCH:
            self.handle_rotate_search()

        elif self.state == MissionState.BACKTRACK:
            self.handle_backtrack()

        elif self.state == MissionState.FORWARD_SEARCH:
            self.handle_forward_search()

        elif self.state == MissionState.FINISHED:
            self.handle_finished()

    def handle_init(self):
        target = np.array(
            [
                self.local_pos[0],
                self.local_pos[1],
                self.takeoff_alt,
            ],
            dtype=float,
        )

        self.publish_position_setpoint(target)

        self.offboard_counter += 1

        if self.offboard_counter == 20:
            self.set_offboard_mode()
            self.arm()
            self.get_logger().info("OFFBOARD + ARM command sent")

        if self.offboard_counter > 35:
            self.current_target = target
            self.state = MissionState.TAKEOFF
            self.get_logger().info("STATE -> TAKEOFF")

    def handle_takeoff(self):
        if self.current_target is None:
            self.current_target = np.array(
                [
                    self.local_pos[0],
                    self.local_pos[1],
                    self.takeoff_alt,
                ],
                dtype=float,
            )

        self.publish_position_setpoint(self.current_target)

        self.get_logger().info(
            f"TAKEOFF_CHECK: z={self.local_pos[2]:.2f}, "
            f"target={self.takeoff_alt:.2f}"
        )

        if abs(self.local_pos[2] - self.takeoff_alt) < 0.25:
            self.current_target = None
            self.state = MissionState.SCAN
            self.get_logger().info("STATE -> SCAN")

    def handle_scan(self):
        hover = np.array(
            [
                self.local_pos[0],
                self.local_pos[1],
                self.flight_alt,
            ],
            dtype=float,
        )

        self.publish_position_setpoint(hover)

        if self.scanned_count >= self.max_scanned_trees:
            self.state = MissionState.FINISHED
            self.get_logger().info("STATE -> FINISHED")
            return

        target = self.latest_target

        if target is not None:
            self.get_logger().info(
                f"DEPTH_TARGET_FOUND: cx={target['cx']:.1f}, "
                f"cy={target['cy']:.1f}, error_x={target['error_x']:.1f}, "
                f"median_depth={target['median_depth']:.2f}, "
                f"mean_depth={target['mean_depth']:.2f}, "
                f"min_depth={target['min_depth']:.2f}, "
                f"area_ratio={target['area_ratio']:.4f}, "
                f"pixels={target['pixels']}"
            )

            self.previous_point = self.local_pos.copy()
            self.lost_target_counter = 0
            self.scan_fail_counter = 0
            self.approach_step_counter = 0

            self.state = MissionState.APPROACH_DEPTH_TARGET
            self.get_logger().info("STATE -> APPROACH_DEPTH_TARGET")
            return

        self.scan_fail_counter += 1

        self.get_logger().warn(
            f"NO_DEPTH_TARGET. scan_fail_counter={self.scan_fail_counter}"
        )

        if self.scan_fail_counter <= self.scan_fail_limit:
            self.target_yaw = self.yaw + self.rotate_angle
            self.rotate_start_time = time.time()
            self.state = MissionState.ROTATE_SEARCH
            self.get_logger().info("STATE -> ROTATE_SEARCH 180 deg")
        else:
            self.current_target = self.make_forward_target(self.search_forward_step)
            self.scan_fail_counter = 0
            self.state = MissionState.FORWARD_SEARCH
            self.get_logger().info("STATE -> FORWARD_SEARCH because no target")

    def handle_approach_depth_target(self):
        target_info = self.latest_target

        if target_info is None:
            self.lost_target_counter += 1

            hover = np.array(
                [
                    self.local_pos[0],
                    self.local_pos[1],
                    self.flight_alt,
                ],
                dtype=float,
            )

            self.publish_position_setpoint(hover)

            self.get_logger().warn(
                f"DEPTH_TARGET_LOST. lost={self.lost_target_counter}"
            )

            if self.lost_target_counter >= self.lost_target_limit:
                self.target_yaw = self.yaw + self.rotate_angle
                self.rotate_start_time = time.time()
                self.state = MissionState.ROTATE_SEARCH
                self.get_logger().info("STATE -> ROTATE_SEARCH because target lost")
            return

        self.lost_target_counter = 0

        img_w = target_info["width"]
        normalized_error = target_info["error_x"] / (img_w / 2.0)

        max_yaw_correction = math.radians(self.max_yaw_correction_deg)

        # Kalau arah yaw salah, ubah tanda minus jadi plus.
        yaw_correction = -normalized_error * max_yaw_correction
        target_yaw = self.yaw + yaw_correction

        centered = self.is_target_centered(target_info)
        close_enough = self.is_target_close_enough(target_info)
        danger_close = self.is_target_danger_close(target_info)

        if danger_close:
            move_target = np.array(
                [
                    self.local_pos[0],
                    self.local_pos[1],
                    self.flight_alt,
                ],
                dtype=float,
            )

        elif centered and not close_enough:
            move_target = self.make_forward_target(self.forward_step)
            self.approach_step_counter += 1

        else:
            move_target = np.array(
                [
                    self.local_pos[0],
                    self.local_pos[1],
                    self.flight_alt,
                ],
                dtype=float,
            )

        self.publish_position_setpoint(move_target, yaw=target_yaw)

        can_scan_by_depth = (
            centered
            and close_enough
            and self.approach_step_counter >= self.min_approach_steps_before_scan
        )

        can_scan_by_danger = danger_close

        # Fallback step scan dimatikan.
        can_scan_by_steps = False

        self.get_logger().info(
            f"APPROACH_DEPTH: error_x={target_info['error_x']:.1f}, "
            f"cy={target_info['cy']:.1f}, "
            f"median_depth={target_info['median_depth']:.2f}, "
            f"mean_depth={target_info['mean_depth']:.2f}, "
            f"min_depth={target_info['min_depth']:.2f}, "
            f"centered={centered}, close={close_enough}, danger={danger_close}, "
            f"steps={self.approach_step_counter}, "
            f"yaw_corr_deg={math.degrees(yaw_correction):.1f}"
        )

        if can_scan_by_depth or can_scan_by_danger or can_scan_by_steps:
            self.state = MissionState.MARK_SCANNED
            self.get_logger().info("STATE -> MARK_SCANNED")

    def handle_mark_scanned(self):
        self.scanned_count += 1
        self.visited_points.append(self.local_pos.copy())

        self.get_logger().info(
            f"SCANNED_DEPTH_TARGET: scanned={self.scanned_count}/{self.max_scanned_trees}"
        )

        if self.scanned_count >= self.max_scanned_trees:
            self.state = MissionState.FINISHED
            self.get_logger().info("STATE -> FINISHED")
            return

        self.previous_point = self.local_pos.copy()
        self.current_target = self.make_back_target(self.backtrack_step)

        self.state = MissionState.BACKTRACK
        self.get_logger().info("STATE -> BACKTRACK after scanned")

    def handle_rotate_search(self):
        hover = np.array(
            [
                self.local_pos[0],
                self.local_pos[1],
                self.flight_alt,
            ],
            dtype=float,
        )

        self.publish_position_setpoint(hover, yaw=self.target_yaw)

        if time.time() - self.rotate_start_time > self.rotate_duration:
            self.state = MissionState.SCAN
            self.get_logger().info("Rotate finished. STATE -> SCAN")

    def handle_backtrack(self):
        if self.current_target is None:
            self.state = MissionState.SCAN
            return

        dist = self.distance_to(self.current_target)

        self.publish_position_setpoint(self.current_target)

        self.get_logger().info(
            f"BACKTRACK: dist={dist:.2f}, "
            f"target=({self.current_target[0]:.2f}, "
            f"{self.current_target[1]:.2f}, {self.current_target[2]:.2f})"
        )

        if dist < 0.6:
            self.current_target = None
            self.target_yaw = self.yaw + self.rotate_angle
            self.rotate_start_time = time.time()
            self.state = MissionState.ROTATE_SEARCH
            self.get_logger().info("BACKTRACK reached. STATE -> ROTATE_SEARCH")

    def handle_forward_search(self):
        if self.current_target is None:
            self.current_target = self.make_forward_target(self.search_forward_step)

        dist = self.distance_to(self.current_target)

        self.publish_position_setpoint(self.current_target)

        self.get_logger().info(
            f"FORWARD_SEARCH: dist={dist:.2f}, "
            f"target=({self.current_target[0]:.2f}, "
            f"{self.current_target[1]:.2f}, {self.current_target[2]:.2f})"
        )

        # Kalau sambil maju target muncul, langsung approach.
        if self.latest_target is not None:
            self.get_logger().info(
                "DEPTH_TARGET found during FORWARD_SEARCH. STATE -> APPROACH_DEPTH_TARGET"
            )
            self.previous_point = self.local_pos.copy()
            self.lost_target_counter = 0
            self.scan_fail_counter = 0
            self.approach_step_counter = 0
            self.current_target = None
            self.state = MissionState.APPROACH_DEPTH_TARGET
            return

        if dist < self.forward_search_reached_radius:
            self.current_target = None
            self.target_yaw = self.yaw + self.rotate_angle
            self.rotate_start_time = time.time()
            self.state = MissionState.ROTATE_SEARCH
            self.get_logger().info("FORWARD_SEARCH reached. STATE -> ROTATE_SEARCH")

    def handle_finished(self):
        self.get_logger().info(
            f"MISSION FINISHED. scanned={self.scanned_count}/{self.max_scanned_trees}. LANDING..."
        )

        if not self.land_sent:
            self.land()
            self.land_sent = True


def main(args=None):
    rclpy.init(args=args)

    node = SawitDepthNavigator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopped by user")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()