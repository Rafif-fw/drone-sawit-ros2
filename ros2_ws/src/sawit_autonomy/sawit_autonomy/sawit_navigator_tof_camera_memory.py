#!/usr/bin/env python3
"""
sawit_navigator_tof_camera_memory.py

FULL NAVIGATION NODE sesuai revisi konsep dosen:
- Drone TIDAK diberi informasi awal letak pohon.
- Drone eksplorasi area menggunakan grid virtual sebagai lintasan pencarian, bukan titik pohon.
- Pohon ditemukan ketika ToF mendeteksi obstacle.
- Setelah obstacle terdeteksi, kamera/depth camera + point cloud dipakai untuk verifikasi.
- Jika obstacle valid sebagai pohon sawit, koordinat pohon dicatat ke memory.
- Sistem menghitung jarak titik grid eksplorasi terdekat ke titik pohon terdeteksi.
- Pada tugas kedua, drone membaca memory dan mengunjungi ulang pohon yang sudah ditemukan.
"""

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import LaserScan, PointCloud2, Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import String, Float32

try:
    import sensor_msgs_py.point_cloud2 as pc2
except Exception:
    pc2 = None

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def dist2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class SawitToFCameraMemoryNavigator(Node):
    def __init__(self):
        super().__init__("sawit_navigator_tof_camera_memory")

        # =====================================================
        # PARAMETER MISI
        # =====================================================
        self.declare_parameter("mission_mode", "mapping")  # mapping / recall
        self.declare_parameter("reset_memory_on_start", True)
        self.declare_parameter("memory_file", str(Path.home() / "ros2_ws" / "sawit_tree_memory.json"))
        self.declare_parameter("target_tree_count", 16)

        self.declare_parameter("takeoff_height", 1.5)
        self.declare_parameter("height_tolerance", 0.18)
        self.declare_parameter("prestream_count", 25)
        self.declare_parameter("setpoint_rate_hz", 20.0)

        # Grid BUKAN posisi pohon. Grid hanya lintasan eksplorasi dan bahan evaluasi jarak grid-pohon.
        self.declare_parameter("grid_xs", [-16.0, -8.0, 0.0, 8.0])
        self.declare_parameter("grid_ys", [8.0, 16.0, 24.0, 32.0])
        self.declare_parameter("grid_reach_distance", 1.0)
        self.declare_parameter("use_lawnmower_order", True)

        # ToF + rem bertahap. 2 m bukan awal rem, tapi batas akhir kunjungan/stop.
        self.declare_parameter("tof_trigger_distance", 6.0)
        self.declare_parameter("tof_slow_distance", 4.0)
        self.declare_parameter("tof_stop_distance", 2.0)
        self.declare_parameter("max_speed", 0.45)
        self.declare_parameter("min_speed", 0.08)
        self.declare_parameter("reaction_delay_s", 0.40)
        self.declare_parameter("brake_accel_mps2", 0.35)
        self.declare_parameter("safe_margin_m", 0.35)

        # Point cloud verification. Asumsi camera point cloud: x=depan, y=kiri, z=atas.
        self.declare_parameter("require_image_for_verify", False)
        self.declare_parameter("image_fresh_timeout_s", 1.0)
        self.declare_parameter("cloud_stride", 4)
        self.declare_parameter("max_cloud_points", 14000)
        self.declare_parameter("tof_cloud_tolerance", 1.25)
        self.declare_parameter("filter_lateral", 1.35)
        self.declare_parameter("filter_min_z", -0.30)
        self.declare_parameter("filter_max_z", 2.30)
        self.declare_parameter("min_cluster_points", 45)
        self.declare_parameter("min_cluster_height", 0.45)
        self.declare_parameter("max_cluster_width", 2.20)
        self.declare_parameter("max_cluster_depth", 2.60)

        # Memory dan visit
        self.declare_parameter("merge_radius", 2.5)
        self.declare_parameter("confirm_count_required", 2)
        self.declare_parameter("visit_distance", 3.0)

        # =====================================================
        # LOAD PARAMETER
        # =====================================================
        self.mission_mode = str(self.get_parameter("mission_mode").value)
        self.reset_memory_on_start = bool(self.get_parameter("reset_memory_on_start").value)
        self.memory_file = Path(str(self.get_parameter("memory_file").value)).expanduser()
        self.target_tree_count = int(self.get_parameter("target_tree_count").value)

        self.takeoff_height = float(self.get_parameter("takeoff_height").value)
        self.height_tolerance = float(self.get_parameter("height_tolerance").value)
        self.prestream_count = int(self.get_parameter("prestream_count").value)
        self.rate_hz = float(self.get_parameter("setpoint_rate_hz").value)

        self.grid_xs = [float(v) for v in self.get_parameter("grid_xs").value]
        self.grid_ys = [float(v) for v in self.get_parameter("grid_ys").value]
        self.grid_reach_distance = float(self.get_parameter("grid_reach_distance").value)
        self.use_lawnmower_order = bool(self.get_parameter("use_lawnmower_order").value)

        self.tof_trigger_distance = float(self.get_parameter("tof_trigger_distance").value)
        self.tof_slow_distance = float(self.get_parameter("tof_slow_distance").value)
        self.tof_stop_distance = float(self.get_parameter("tof_stop_distance").value)
        self.max_speed = float(self.get_parameter("max_speed").value)
        self.min_speed = float(self.get_parameter("min_speed").value)
        self.reaction_delay_s = float(self.get_parameter("reaction_delay_s").value)
        self.brake_accel_mps2 = float(self.get_parameter("brake_accel_mps2").value)
        self.safe_margin_m = float(self.get_parameter("safe_margin_m").value)

        self.require_image_for_verify = bool(self.get_parameter("require_image_for_verify").value)
        self.image_fresh_timeout_s = float(self.get_parameter("image_fresh_timeout_s").value)
        self.cloud_stride = int(self.get_parameter("cloud_stride").value)
        self.max_cloud_points = int(self.get_parameter("max_cloud_points").value)
        self.tof_cloud_tolerance = float(self.get_parameter("tof_cloud_tolerance").value)
        self.filter_lateral = float(self.get_parameter("filter_lateral").value)
        self.filter_min_z = float(self.get_parameter("filter_min_z").value)
        self.filter_max_z = float(self.get_parameter("filter_max_z").value)
        self.min_cluster_points = int(self.get_parameter("min_cluster_points").value)
        self.min_cluster_height = float(self.get_parameter("min_cluster_height").value)
        self.max_cluster_width = float(self.get_parameter("max_cluster_width").value)
        self.max_cluster_depth = float(self.get_parameter("max_cluster_depth").value)
        self.merge_radius = float(self.get_parameter("merge_radius").value)
        self.confirm_count_required = int(self.get_parameter("confirm_count_required").value)
        self.visit_distance = float(self.get_parameter("visit_distance").value)

        # =====================================================
        # STATE
        # =====================================================
        self.nav_state = "INIT"
        self.offboard_counter = 0
        self.armed = False
        self.vehicle_status = None

        self.have_local_pos = False
        self.x = 0.0
        self.y = 0.0
        self.z_down = 0.0
        self.yaw = 0.0

        self.latest_tof: Optional[float] = None
        self.latest_image_time: Optional[float] = None
        self.grid_nodes = self._build_grid_nodes()
        self.grid_index = 0

        self.trees: List[Dict] = []
        self.active_tree_id: Optional[int] = None

        self.last_cloud_verify_time = 0.0
        self.last_log_time = 0.0
        self.last_memory_save_time = 0.0

        if self.reset_memory_on_start and self.mission_mode == "mapping":
            self._reset_memory()
        self._load_memory()

        # PX4 QoS
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # Publishers PX4
        self.offboard_pub = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", 10)
        self.traj_pub = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", 10)
        self.cmd_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", 10)

        # Debug publishers
        self.marker_pub = self.create_publisher(MarkerArray, "/sawit/tof_camera_memory_markers", 10)
        self.memory_json_pub = self.create_publisher(String, "/sawit/tree_memory_json", 10)
        self.speed_limit_pub = self.create_publisher(Float32, "/sawit/speed_limit_mps", 10)

        # Subscribers
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1", self.on_local_pos, px4_qos)
        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status_v4", self.on_vehicle_status, px4_qos)
        self.create_subscription(LaserScan, "/tof_front", self.on_tof, sensor_qos)
        self.create_subscription(PointCloud2, "/camera/points", self.on_cloud, sensor_qos)
        self.create_subscription(Image, "/camera/image", self.on_image, sensor_qos)

        self.timer = self.create_timer(1.0 / self.rate_hz, self.on_timer)

        self.get_logger().info(
            f"START full navigator mode={self.mission_mode}, grid_nodes={len(self.grid_nodes)}, "
            f"loaded_trees={len(self.trees)}, memory={self.memory_file}"
        )

    # =====================================================
    # SENSOR CALLBACKS
    # =====================================================
    def on_local_pos(self, msg: VehicleLocalPosition):
        self.have_local_pos = True
        self.x = float(msg.x)
        self.y = float(msg.y)
        self.z_down = float(msg.z)
        if math.isfinite(float(msg.heading)):
            self.yaw = float(msg.heading)

    def on_vehicle_status(self, msg: VehicleStatus):
        self.vehicle_status = msg
        self.armed = msg.arming_state == VehicleStatus.ARMING_STATE_ARMED

    def on_image(self, msg: Image):
        self.latest_image_time = time.time()

    def on_tof(self, msg: LaserScan):
        if len(msg.ranges) > 5:
            mid = len(msg.ranges) // 2
            win = max(1, int(len(msg.ranges) * 0.08))
            ranges = msg.ranges[mid - win: mid + win + 1]
        else:
            ranges = msg.ranges

        vals = [float(r) for r in ranges if math.isfinite(r) and msg.range_min <= r <= msg.range_max]
        if not vals:
            return

        self.latest_tof = min(vals)
        speed_limit = self._speed_limit_from_tof(self.latest_tof)
        out = Float32()
        out.data = float(speed_limit)
        self.speed_limit_pub.publish(out)

    def on_cloud(self, msg: PointCloud2):
        # Point cloud dipakai untuk verifikasi pohon hanya setelah ToF melihat obstacle.
        if self.mission_mode != "mapping":
            return
        if self.nav_state not in ("EXPLORE_GRID", "APPROACH_TREE"):
            return
        if self.latest_tof is None or self.latest_tof > self.tof_trigger_distance:
            return
        if not self.have_local_pos:
            return
        if self.require_image_for_verify and not self._image_fresh():
            return

        now = time.time()
        if now - self.last_cloud_verify_time < 0.25:
            return
        self.last_cloud_verify_time = now

        candidate = self._verify_tree_from_cloud(msg, self.latest_tof)
        if candidate is None:
            return

        forward, left, metrics = candidate
        tree_x, tree_y = self._body_to_map(forward, left)
        gid, gx, gy, gd = self._nearest_grid(tree_x, tree_y)

        tree_id, status = self._add_or_update_tree(
            tree_x, tree_y, gid, gx, gy, gd, self.latest_tof, metrics
        )

        t = self._get_tree_by_id(tree_id)
        confirm_count = t.get("confirm_count", 0) if t else 0
        self.get_logger().info(
            f"TREE_{status.upper()} id={tree_id} pos=({tree_x:.2f},{tree_y:.2f}) "
            f"tof={self.latest_tof:.2f} grid={gid} grid_dist={gd:.2f} confirm={confirm_count}"
        )

        # Jika sudah valid sebagai pohon sawit, jadikan target kunjungan.
        if t and t.get("status") == "sawit" and not t.get("visited", False):
            self.active_tree_id = tree_id
            self.nav_state = "APPROACH_TREE"

    # =====================================================
    # MAIN TIMER
    # =====================================================
    def on_timer(self):
        self._publish_offboard_mode()
        self._publish_debug()

        if not self.have_local_pos:
            self._publish_hold_setpoint(0.0, 0.0, -self.takeoff_height, 0.0)
            self._log_throttle("Menunggu local position PX4...")
            return

        if self.nav_state == "INIT":
            self._state_init()
        elif self.nav_state == "TAKEOFF":
            self._state_takeoff()
        elif self.nav_state == "EXPLORE_GRID":
            self._state_explore_grid()
        elif self.nav_state == "APPROACH_TREE":
            self._state_approach_tree()
        elif self.nav_state == "RECALL_VISIT":
            self._state_recall_visit()
        elif self.nav_state == "LAND":
            self._state_land()
        elif self.nav_state == "DONE":
            self._publish_hold_setpoint(self.x, self.y, self.z_down, self.yaw)
        else:
            self.get_logger().warn(f"Unknown state {self.nav_state}")
            self.nav_state = "DONE"

    # =====================================================
    # STATE MACHINE
    # =====================================================
    def _state_init(self):
        self._publish_hold_setpoint(self.x, self.y, -self.takeoff_height, self.yaw)
        self.offboard_counter += 1
        if self.offboard_counter < self.prestream_count:
            self._log_throttle(f"Prestream offboard {self.offboard_counter}/{self.prestream_count}")
            return

        self._send_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self._arm()
        self.nav_state = "TAKEOFF"
        self.get_logger().info("STATE -> TAKEOFF")

    def _state_takeoff(self):
        self._publish_hold_setpoint(self.x, self.y, -self.takeoff_height, self.yaw)
        alt = -self.z_down
        if abs(alt - self.takeoff_height) <= self.height_tolerance:
            if self.mission_mode == "recall":
                self._prepare_recall_targets()
                self.nav_state = "RECALL_VISIT"
                self.get_logger().info("STATE -> RECALL_VISIT")
            else:
                self.grid_index = 0
                self.nav_state = "EXPLORE_GRID"
                self.get_logger().info("STATE -> EXPLORE_GRID")

    def _state_explore_grid(self):
        visited_count = sum(1 for t in self.trees if t.get("visited", False))
        sawit_count = sum(1 for t in self.trees if t.get("status") == "sawit")

        if visited_count >= self.target_tree_count or sawit_count >= self.target_tree_count:
            self.get_logger().info(f"MISSION complete sawit={sawit_count} visited={visited_count}; LAND")
            self._save_memory(force=True)
            self.nav_state = "LAND"
            return

        # Jika sudah ada sawit valid yang belum visited, kunjungi dulu.
        next_tree = self._nearest_unvisited_sawit()
        if next_tree is not None:
            self.active_tree_id = int(next_tree["id"])
            self.nav_state = "APPROACH_TREE"
            self.get_logger().info(f"Switch APPROACH_TREE id={self.active_tree_id}")
            return

        if self.grid_index >= len(self.grid_nodes):
            self.get_logger().info(f"GRID exploration finished sawit={sawit_count} visited={visited_count}; LAND")
            self._save_memory(force=True)
            self.nav_state = "LAND"
            return

        gid, gx, gy = self.grid_nodes[self.grid_index]
        speed_limit = self._speed_limit_from_tof(self.latest_tof)
        yaw_to_goal = math.atan2(gy - self.y, gx - self.x)
        sp_x, sp_y = self._step_toward(gx, gy, speed_limit)
        self._publish_hold_setpoint(sp_x, sp_y, -self.takeoff_height, yaw_to_goal)

        d = dist2d((self.x, self.y), (gx, gy))
        self._log_throttle(
            f"EXPLORE grid={gid} d={d:.2f} tof={self.latest_tof if self.latest_tof is not None else -1:.2f} "
            f"sawit={sawit_count}/{self.target_tree_count} visited={visited_count}"
        )

        if d <= self.grid_reach_distance:
            self.grid_index += 1
            self.get_logger().info(f"GRID_REACHED {gid}; next={self.grid_index}")

    def _state_approach_tree(self):
        if self.active_tree_id is None:
            self.nav_state = "EXPLORE_GRID"
            return

        t = self._get_tree_by_id(self.active_tree_id)
        if t is None:
            self.active_tree_id = None
            self.nav_state = "EXPLORE_GRID"
            return

        tx = float(t["x"])
        ty = float(t["y"])
        d = dist2d((self.x, self.y), (tx, ty))
        tof_close = self.latest_tof is not None and self.latest_tof <= self.tof_stop_distance

        if d <= self.visit_distance or tof_close:
            t["visited"] = True
            t["visited_time"] = time.time()
            t["visit_distance_actual_m"] = float(d)
            t["tof_at_visit_m"] = float(self.latest_tof) if self.latest_tof is not None else None
            self._save_memory(force=True)
            self.get_logger().info(
                f"TREE_VISITED id={self.active_tree_id} d={d:.2f} tof={self.latest_tof if self.latest_tof is not None else -1:.2f}"
            )
            self.active_tree_id = None
            self.nav_state = "EXPLORE_GRID"
            return

        yaw_to_tree = math.atan2(ty - self.y, tx - self.x)
        speed_limit = self._speed_limit_from_tof(self.latest_tof)
        sp_x, sp_y = self._step_toward(tx, ty, speed_limit)
        self._publish_hold_setpoint(sp_x, sp_y, -self.takeoff_height, yaw_to_tree)
        self._log_throttle(
            f"APPROACH_TREE id={self.active_tree_id} d={d:.2f} tof={self.latest_tof if self.latest_tof is not None else -1:.2f} speed={speed_limit:.2f}"
        )

    def _state_recall_visit(self):
        target = self._nearest_unvisited_sawit()
        if target is None:
            self.get_logger().info("RECALL complete; LAND")
            self._save_memory(force=True)
            self.nav_state = "LAND"
            return

        self.active_tree_id = int(target["id"])
        tx = float(target["x"])
        ty = float(target["y"])
        d = dist2d((self.x, self.y), (tx, ty))
        yaw_to_tree = math.atan2(ty - self.y, tx - self.x)

        if d <= self.visit_distance:
            target["visited"] = True
            target["recall_visited_time"] = time.time()
            target["recall_visit_distance_m"] = float(d)
            self._save_memory(force=True)
            self.get_logger().info(f"RECALL_TREE_VISITED id={self.active_tree_id} d={d:.2f}")
            return

        speed_limit = self._speed_limit_from_tof(self.latest_tof)
        sp_x, sp_y = self._step_toward(tx, ty, speed_limit)
        self._publish_hold_setpoint(sp_x, sp_y, -self.takeoff_height, yaw_to_tree)
        self._log_throttle(f"RECALL_VISIT id={self.active_tree_id} d={d:.2f}")

    def _state_land(self):
        self._send_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.nav_state = "DONE"
        self.get_logger().info("LAND command sent. STATE -> DONE")

    # =====================================================
    # PX4 HELPERS
    # =====================================================
    def _timestamp_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def _publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = self._timestamp_us()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_pub.publish(msg)

    def _publish_hold_setpoint(self, x: float, y: float, z_down: float, yaw: float):
        msg = TrajectorySetpoint()
        msg.timestamp = self._timestamp_us()
        msg.position = [float(x), float(y), float(z_down)]
        msg.yaw = float(yaw)
        self.traj_pub.publish(msg)

    def _send_vehicle_command(self, command: int, **params):
        msg = VehicleCommand()
        msg.timestamp = self._timestamp_us()
        msg.param1 = float(params.get("param1", 0.0))
        msg.param2 = float(params.get("param2", 0.0))
        msg.param3 = float(params.get("param3", 0.0))
        msg.param4 = float(params.get("param4", 0.0))
        msg.param5 = float(params.get("param5", 0.0))
        msg.param6 = float(params.get("param6", 0.0))
        msg.param7 = float(params.get("param7", 0.0))
        msg.command = int(command)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)

    def _arm(self):
        self._send_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)

    # =====================================================
    # NAVIGATION HELPERS
    # =====================================================
    def _step_toward(self, gx: float, gy: float, speed_limit: float) -> Tuple[float, float]:
        dx = gx - self.x
        dy = gy - self.y
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return self.x, self.y
        step = clamp(speed_limit / self.rate_hz, 0.02, 0.35)
        step = min(step, d)
        return self.x + dx / d * step, self.y + dy / d * step

    def _speed_limit_from_tof(self, tof: Optional[float]) -> float:
        if tof is None or not math.isfinite(tof):
            return self.max_speed
        if tof <= self.tof_stop_distance:
            return 0.0

        v = self.max_speed
        a = max(self.brake_accel_mps2, 0.05)
        d_stop = v * self.reaction_delay_s + (v * v) / (2.0 * a) + self.safe_margin_m
        slow_start = max(self.tof_slow_distance, self.tof_stop_distance + d_stop)

        if tof >= slow_start:
            return self.max_speed

        ratio = (tof - self.tof_stop_distance) / max(slow_start - self.tof_stop_distance, 0.01)
        speed = self.max_speed * clamp(ratio, 0.0, 1.0)
        if speed > 0.0:
            speed = max(speed, self.min_speed)
        return speed

    # =====================================================
    # POINT CLOUD VERIFICATION
    # =====================================================
    def _verify_tree_from_cloud(self, msg: PointCloud2, tof_range: float):
        if pc2 is None:
            self.get_logger().error("sensor_msgs_py.point_cloud2 tidak tersedia")
            return None

        pts = []
        count = 0
        for i, p in enumerate(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)):
            if i % self.cloud_stride != 0:
                continue
            try:
                x = float(p[0])
                y = float(p[1])
                z = float(p[2])
            except Exception:
                continue

            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            if x < 0.25:
                continue
            if abs(x - tof_range) > self.tof_cloud_tolerance:
                continue
            if abs(y) > self.filter_lateral:
                continue
            if z < self.filter_min_z or z > self.filter_max_z:
                continue

            pts.append((x, y, z))
            count += 1
            if count >= self.max_cloud_points:
                break

        if len(pts) < self.min_cluster_points:
            return None

        arr = np.array(pts, dtype=np.float32)
        xs = arr[:, 0]
        ys = arr[:, 1]
        zs = arr[:, 2]
        x5, x95 = np.percentile(xs, [5, 95])
        y5, y95 = np.percentile(ys, [5, 95])
        z5, z95 = np.percentile(zs, [5, 95])
        width = float(y95 - y5)
        height = float(z95 - z5)
        depth = float(x95 - x5)

        verified = (
            len(pts) >= self.min_cluster_points
            and height >= self.min_cluster_height
            and width <= self.max_cluster_width
            and depth <= self.max_cluster_depth
        )
        if not verified:
            return None

        forward = float(np.median(xs))
        left = float(np.median(ys))
        metrics = {
            "n_points": int(len(pts)),
            "width_m": width,
            "height_m": height,
            "depth_m": depth,
            "median_forward_m": forward,
            "median_left_m": left,
            "tof_range_m": float(tof_range),
        }
        return forward, left, metrics

    def _image_fresh(self) -> bool:
        if self.latest_image_time is None:
            return False
        return (time.time() - self.latest_image_time) <= self.image_fresh_timeout_s

    def _body_to_map(self, forward: float, left: float) -> Tuple[float, float]:
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        mx = self.x + c * forward - s * left
        my = self.y + s * forward + c * left
        return mx, my

    # =====================================================
    # GRID AND MEMORY
    # =====================================================
    def _build_grid_nodes(self) -> List[Tuple[str, float, float]]:
        nodes = []
        idx = 0
        if self.use_lawnmower_order:
            for row, y in enumerate(self.grid_ys):
                xs = self.grid_xs if row % 2 == 0 else list(reversed(self.grid_xs))
                for x in xs:
                    idx += 1
                    nodes.append((f"G{idx:02d}", float(x), float(y)))
        else:
            for y in self.grid_ys:
                for x in self.grid_xs:
                    idx += 1
                    nodes.append((f"G{idx:02d}", float(x), float(y)))
        return nodes

    def _nearest_grid(self, x: float, y: float) -> Tuple[str, float, float, float]:
        best = None
        for gid, gx, gy in self.grid_nodes:
            d = math.hypot(x - gx, y - gy)
            if best is None or d < best[3]:
                best = (gid, gx, gy, d)
        return best

    def _add_or_update_tree(self, x: float, y: float, grid_id: str, grid_x: float, grid_y: float, grid_dist: float, tof_range: float, metrics: Dict) -> Tuple[int, str]:
        nearest_tree = None
        nearest_d = 999999.0
        for t in self.trees:
            d = dist2d((float(t["x"]), float(t["y"])), (x, y))
            if d < self.merge_radius and d < nearest_d:
                nearest_d = d
                nearest_tree = t

        now = time.time()
        if nearest_tree is None:
            tree_id = self._next_tree_id()
            t = {
                "id": tree_id,
                "status": "candidate_sawit",
                "visited": False,
                "x": float(x),
                "y": float(y),
                "z_down": float(self.z_down),
                "confirm_count": 1,
                "first_seen_time": now,
                "last_seen_time": now,
                "detection_method": "ToF obstacle + camera/depth pointcloud verification",
                "tof_range_m": float(tof_range),
                "nearest_grid_id": grid_id,
                "nearest_grid_x": float(grid_x),
                "nearest_grid_y": float(grid_y),
                "grid_distance_m": float(grid_dist),
                "metrics": metrics,
            }
            if t["confirm_count"] >= self.confirm_count_required:
                t["status"] = "sawit"
            self.trees.append(t)
            self._save_memory()
            return tree_id, "new"

        n = int(nearest_tree.get("confirm_count", 1))
        old_w = n / (n + 1.0)
        new_w = 1.0 / (n + 1.0)
        nearest_tree["x"] = old_w * float(nearest_tree["x"]) + new_w * float(x)
        nearest_tree["y"] = old_w * float(nearest_tree["y"]) + new_w * float(y)
        nearest_tree["z_down"] = old_w * float(nearest_tree.get("z_down", self.z_down)) + new_w * float(self.z_down)
        nearest_tree["confirm_count"] = n + 1
        nearest_tree["last_seen_time"] = now
        nearest_tree["tof_range_m"] = float(tof_range)
        nearest_tree["metrics"] = metrics

        gid, gx, gy, gd = self._nearest_grid(float(nearest_tree["x"]), float(nearest_tree["y"]))
        nearest_tree["nearest_grid_id"] = gid
        nearest_tree["nearest_grid_x"] = float(gx)
        nearest_tree["nearest_grid_y"] = float(gy)
        nearest_tree["grid_distance_m"] = float(gd)
        if int(nearest_tree["confirm_count"]) >= self.confirm_count_required:
            nearest_tree["status"] = "sawit"

        self._save_memory()
        return int(nearest_tree["id"]), "updated"

    def _next_tree_id(self) -> int:
        if not self.trees:
            return 1
        return max(int(t.get("id", 0)) for t in self.trees) + 1

    def _get_tree_by_id(self, tree_id: int) -> Optional[Dict]:
        for t in self.trees:
            if int(t.get("id", -1)) == int(tree_id):
                return t
        return None

    def _nearest_unvisited_sawit(self) -> Optional[Dict]:
        candidates = [t for t in self.trees if t.get("status") == "sawit" and not bool(t.get("visited", False))]
        if not candidates:
            return None
        candidates.sort(key=lambda t: dist2d((self.x, self.y), (float(t["x"]), float(t["y"]))))
        return candidates[0]

    def _prepare_recall_targets(self):
        for t in self.trees:
            if t.get("status") == "sawit":
                t["visited"] = False
        self._save_memory(force=True)

    def _reset_memory(self):
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": 1, "description": "Memory pohon sawit reset untuk mapping baru", "trees": []}
        self.memory_file.write_text(json.dumps(data, indent=2))

    def _load_memory(self):
        if not self.memory_file.exists():
            self.trees = []
            return
        try:
            data = json.loads(self.memory_file.read_text())
            self.trees = data.get("trees", [])
        except Exception as e:
            self.get_logger().warn(f"Gagal load memory: {e}")
            self.trees = []

    def _save_memory(self, force: bool = False):
        now = time.time()
        if not force and now - self.last_memory_save_time < 0.5:
            return
        self.last_memory_save_time = now
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "description": "Memory lokasi pohon dari ToF obstacle + camera/depth pointcloud verification",
            "mission_mode": self.mission_mode,
            "grid_nodes": [{"id": gid, "x": gx, "y": gy} for gid, gx, gy in self.grid_nodes],
            "parameters": {
                "tof_trigger_distance": self.tof_trigger_distance,
                "tof_stop_distance": self.tof_stop_distance,
                "visit_distance": self.visit_distance,
                "confirm_count_required": self.confirm_count_required,
            },
            "trees": self.trees,
        }
        tmp = self.memory_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.memory_file)

    # =====================================================
    # DEBUG MARKERS
    # =====================================================
    def _publish_debug(self):
        self._publish_memory_json()
        self._publish_markers()

    def _publish_memory_json(self):
        msg = String()
        msg.data = json.dumps({
            "state": self.nav_state,
            "mode": self.mission_mode,
            "grid_index": self.grid_index,
            "tof": self.latest_tof,
            "tree_count": len(self.trees),
            "sawit_count": sum(1 for t in self.trees if t.get("status") == "sawit"),
            "visited_count": sum(1 for t in self.trees if t.get("visited", False)),
            "trees": self.trees,
        })
        self.memory_json_pub.publish(msg)

    def _publish_markers(self):
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        clear = Marker()
        clear.header.frame_id = "map"
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        for i, (gid, gx, gy) in enumerate(self.grid_nodes):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = stamp
            m.ns = "exploration_grid"
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(gx)
            m.pose.position.y = float(gy)
            m.pose.position.z = 0.1
            m.pose.orientation.w = 1.0
            m.scale.x = 0.35
            m.scale.y = 0.35
            m.scale.z = 0.12
            m.color.r = 1.0
            m.color.g = 1.0
            m.color.b = 1.0
            m.color.a = 0.65
            ma.markers.append(m)

            txt = Marker()
            txt.header.frame_id = "map"
            txt.header.stamp = stamp
            txt.ns = "grid_text"
            txt.id = 1000 + i
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = float(gx)
            txt.pose.position.y = float(gy)
            txt.pose.position.z = 0.8
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.42
            txt.color.r = 1.0
            txt.color.g = 1.0
            txt.color.b = 1.0
            txt.color.a = 0.85
            txt.text = gid
            ma.markers.append(txt)

        for t in self.trees:
            tid = int(t.get("id", 0))
            confirmed = t.get("status") == "sawit"
            visited = bool(t.get("visited", False))

            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = stamp
            m.ns = "detected_sawit_tree"
            m.id = 2000 + tid
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(t["x"])
            m.pose.position.y = float(t["y"])
            m.pose.position.z = 1.1
            m.pose.orientation.w = 1.0
            m.scale.x = 0.75
            m.scale.y = 0.75
            m.scale.z = 0.75
            if visited:
                m.color.r = 0.0; m.color.g = 0.75; m.color.b = 0.0; m.color.a = 0.95
            elif confirmed:
                m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.25; m.color.a = 0.95
            else:
                m.color.r = 1.0; m.color.g = 0.82; m.color.b = 0.0; m.color.a = 0.90
            ma.markers.append(m)

            txt = Marker()
            txt.header.frame_id = "map"
            txt.header.stamp = stamp
            txt.ns = "tree_text"
            txt.id = 3000 + tid
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = float(t["x"])
            txt.pose.position.y = float(t["y"])
            txt.pose.position.z = 2.0
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.40
            txt.color.r = 0.0; txt.color.g = 1.0; txt.color.b = 0.25; txt.color.a = 0.95
            txt.text = f"T{tid} {t.get('status')}\n{t.get('nearest_grid_id','?')} d={float(t.get('grid_distance_m', 0.0)):.2f}m"
            ma.markers.append(txt)

            if "nearest_grid_x" in t and "nearest_grid_y" in t:
                line = Marker()
                line.header.frame_id = "map"
                line.header.stamp = stamp
                line.ns = "grid_to_tree_distance"
                line.id = 4000 + tid
                line.type = Marker.LINE_STRIP
                line.action = Marker.ADD
                line.scale.x = 0.05
                line.color.r = 0.2; line.color.g = 0.8; line.color.b = 1.0; line.color.a = 0.85
                p1 = Point(); p1.x = float(t["nearest_grid_x"]); p1.y = float(t["nearest_grid_y"]); p1.z = 0.25
                p2 = Point(); p2.x = float(t["x"]); p2.y = float(t["y"]); p2.z = 0.25
                line.points = [p1, p2]
                ma.markers.append(line)

        self.marker_pub.publish(ma)

    def _log_throttle(self, text: str, period: float = 1.0):
        now = time.time()
        if now - self.last_log_time >= period:
            self.last_log_time = now
            self.get_logger().info(text)


def main(args=None):
    rclpy.init(args=args)
    node = SawitToFCameraMemoryNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._save_memory(force=True)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
