#!/usr/bin/env python3
"""
sawit_tree_memory_mapper.py

Node pendamping untuk kasus skripsi:
1. Drone tidak diberi titik pohon dari awal.
2. ToF mendeteksi obstacle awal.
3. Kamera/depth camera + point cloud memverifikasi obstacle sebagai pohon sawit.
4. Jika valid, posisi pohon dicatat ke memory JSON.
5. Pada tugas kedua, memory JSON bisa dibaca lagi supaya drone "ingat" letak pohon.
6. Menghitung jarak antara titik grid eksplorasi dan titik pohon terdeteksi.
7. Mengeluarkan speed limit untuk rem bertahap berdasarkan jarak ToF.

Topic utama:
- Subscribe:
    /tof_front       sensor_msgs/LaserScan
    /camera/points   sensor_msgs/PointCloud2
    /camera/image    sensor_msgs/Image         optional, bukti kamera aktif
    /fmu/out/vehicle_odometry px4_msgs/VehicleOdometry jika tersedia
    /odom            nav_msgs/Odometry fallback
- Publish:
    /sawit/tree_memory_markers visualization_msgs/MarkerArray
    /sawit/tree_memory_json    std_msgs/String
    /sawit/speed_limit_mps     std_msgs/Float32
"""

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan, PointCloud2, Image
from std_msgs.msg import String, Float32
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry

try:
    from px4_msgs.msg import VehicleOdometry
    PX4_AVAILABLE = True
except Exception:
    VehicleOdometry = None
    PX4_AVAILABLE = False

try:
    import sensor_msgs_py.point_cloud2 as pc2
except Exception:
    pc2 = None


def yaw_from_quat_wxyz(qw: float, qx: float, qy: float, qz: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_from_quat_xyzw(qx: float, qy: float, qz: float, qw: float) -> float:
    return yaw_from_quat_wxyz(qw, qx, qy, qz)


class SawitTreeMemoryMapper(Node):
    def __init__(self):
        super().__init__("sawit_tree_memory_mapper")

        # Misi
        self.declare_parameter("mission_mode", "mapping")  # mapping / recall
        self.declare_parameter("reset_memory_on_start", False)
        self.declare_parameter("memory_file", str(Path.home() / "ros2_ws" / "sawit_tree_memory.json"))

        # Grid ini BUKAN posisi pohon.
        # Grid hanya acuan eksplorasi dan evaluasi jarak grid-pohon.
        self.declare_parameter("grid_xs", [-16.0, -8.0, 0.0, 8.0])
        self.declare_parameter("grid_ys", [8.0, 16.0, 24.0, 32.0])

        # ToF dan rem bertahap
        self.declare_parameter("tof_trigger_distance", 6.0)   # obstacle mulai dicek
        self.declare_parameter("tof_visit_distance", 2.0)     # batas akhir, bukan awal pengereman
        self.declare_parameter("tof_slow_distance", 4.0)      # mulai pelan sebelum 2 m
        self.declare_parameter("max_speed_mps", 0.45)
        self.declare_parameter("min_approach_speed_mps", 0.08)
        self.declare_parameter("reaction_delay_s", 0.40)
        self.declare_parameter("brake_accel_mps2", 0.35)
        self.declare_parameter("stop_margin_m", 0.35)

        # Camera/point cloud verification
        # Frame cloud: x = depan, y = kiri, z = atas.
        self.declare_parameter("require_image_for_verify", False)
        self.declare_parameter("image_timeout_s", 1.0)

        self.declare_parameter("cloud_stride", 5)
        self.declare_parameter("max_cloud_points", 12000)
        self.declare_parameter("tof_cloud_tolerance_m", 1.25)
        self.declare_parameter("lateral_filter_m", 1.40)
        self.declare_parameter("min_z_m", -0.30)
        self.declare_parameter("max_z_m", 2.20)
        self.declare_parameter("min_cluster_points", 45)
        self.declare_parameter("min_cluster_height_m", 0.45)
        self.declare_parameter("max_cluster_width_m", 2.20)
        self.declare_parameter("max_cluster_depth_m", 2.60)

        # Memory / merge
        self.declare_parameter("merge_radius_m", 2.50)
        self.declare_parameter("confirm_count_required", 2)
        self.declare_parameter("visit_distance_m", 3.0)

        self.mission_mode = self.get_parameter("mission_mode").value
        self.memory_file = Path(self.get_parameter("memory_file").value).expanduser()
        self.grid_xs = [float(v) for v in self.get_parameter("grid_xs").value]
        self.grid_ys = [float(v) for v in self.get_parameter("grid_ys").value]

        self.tof_trigger_distance = float(self.get_parameter("tof_trigger_distance").value)
        self.tof_visit_distance = float(self.get_parameter("tof_visit_distance").value)
        self.tof_slow_distance = float(self.get_parameter("tof_slow_distance").value)
        self.max_speed_mps = float(self.get_parameter("max_speed_mps").value)
        self.min_approach_speed_mps = float(self.get_parameter("min_approach_speed_mps").value)
        self.reaction_delay_s = float(self.get_parameter("reaction_delay_s").value)
        self.brake_accel_mps2 = float(self.get_parameter("brake_accel_mps2").value)
        self.stop_margin_m = float(self.get_parameter("stop_margin_m").value)

        self.require_image_for_verify = bool(self.get_parameter("require_image_for_verify").value)
        self.image_timeout_s = float(self.get_parameter("image_timeout_s").value)

        self.cloud_stride = int(self.get_parameter("cloud_stride").value)
        self.max_cloud_points = int(self.get_parameter("max_cloud_points").value)
        self.tof_cloud_tolerance_m = float(self.get_parameter("tof_cloud_tolerance_m").value)
        self.lateral_filter_m = float(self.get_parameter("lateral_filter_m").value)
        self.min_z_m = float(self.get_parameter("min_z_m").value)
        self.max_z_m = float(self.get_parameter("max_z_m").value)
        self.min_cluster_points = int(self.get_parameter("min_cluster_points").value)
        self.min_cluster_height_m = float(self.get_parameter("min_cluster_height_m").value)
        self.max_cluster_width_m = float(self.get_parameter("max_cluster_width_m").value)
        self.max_cluster_depth_m = float(self.get_parameter("max_cluster_depth_m").value)

        self.merge_radius_m = float(self.get_parameter("merge_radius_m").value)
        self.confirm_count_required = int(self.get_parameter("confirm_count_required").value)
        self.visit_distance_m = float(self.get_parameter("visit_distance_m").value)

        self.grid_centers = self._build_grid_centers()

        self.pose_xy = (0.0, 0.0)
        self.pose_z = 0.0
        self.yaw = 0.0
        self.have_pose = False

        self.latest_tof: Optional[float] = None
        self.latest_image_time: Optional[float] = None
        self.last_detection_log_time = 0.0

        self.trees: List[Dict] = []

        if bool(self.get_parameter("reset_memory_on_start").value):
            self._reset_memory_file()

        self._load_memory()

        self.create_subscription(LaserScan, "/tof_front", self.on_tof, 10)
        self.create_subscription(PointCloud2, "/camera/points", self.on_cloud, 10)
        self.create_subscription(Image, "/camera/image", self.on_image, 10)

        if PX4_AVAILABLE:
            self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self.on_px4_odom, 10)
        self.create_subscription(Odometry, "/odom", self.on_nav_odom, 10)

        self.marker_pub = self.create_publisher(MarkerArray, "/sawit/tree_memory_markers", 10)
        self.json_pub = self.create_publisher(String, "/sawit/tree_memory_json", 10)
        self.speed_pub = self.create_publisher(Float32, "/sawit/speed_limit_mps", 10)

        self.create_timer(0.5, self.on_timer)

        self.get_logger().info(
            f"START mode={self.mission_mode}, memory={str(self.memory_file)}, "
            f"grid={len(self.grid_centers)} node, trees_loaded={len(self.trees)}"
        )

    # -----------------------------
    # Sensor callbacks
    # -----------------------------
    def on_tof(self, msg: LaserScan):
        vals = [r for r in msg.ranges if math.isfinite(r) and msg.range_min <= r <= msg.range_max]
        if not vals:
            return

        if len(msg.ranges) > 5:
            mid = len(msg.ranges) // 2
            win = max(1, int(len(msg.ranges) * 0.08))
            center_vals = [
                r for r in msg.ranges[mid - win: mid + win + 1]
                if math.isfinite(r) and msg.range_min <= r <= msg.range_max
            ]
            if center_vals:
                vals = center_vals

        self.latest_tof = float(min(vals))

        speed_limit = self._compute_speed_limit(self.latest_tof)
        out = Float32()
        out.data = float(speed_limit)
        self.speed_pub.publish(out)

    def on_image(self, msg: Image):
        self.latest_image_time = time.time()

    def on_px4_odom(self, msg):
        try:
            self.pose_xy = (float(msg.position[0]), float(msg.position[1]))
            self.pose_z = float(msg.position[2])
            q = msg.q  # PX4 umumnya [w, x, y, z]
            self.yaw = yaw_from_quat_wxyz(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            self.have_pose = True
        except Exception as e:
            self.get_logger().warn(f"PX4 odom parse failed: {e}")

    def on_nav_odom(self, msg: Odometry):
        if self.have_pose:
            return
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.pose_xy = (float(p.x), float(p.y))
        self.pose_z = float(p.z)
        self.yaw = yaw_from_quat_xyzw(float(q.x), float(q.y), float(q.z), float(q.w))
        self.have_pose = True

    def on_cloud(self, msg: PointCloud2):
        if pc2 is None:
            self.get_logger().error("sensor_msgs_py.point_cloud2 tidak tersedia")
            return

        if self.latest_tof is None:
            return

        # Point cloud dipakai definitif SETELAH ToF mendeteksi obstacle.
        if self.latest_tof > self.tof_trigger_distance:
            return

        if self.require_image_for_verify and not self._image_is_recent():
            self.get_logger().warn("ToF obstacle ada, tapi image belum fresh; verifikasi ditunda")
            return

        if not self.have_pose:
            self.get_logger().warn("Pose drone belum tersedia; belum bisa catat lokasi pohon")
            return

        candidate = self._extract_tree_candidate_from_cloud(msg, self.latest_tof)
        if candidate is None:
            return

        local_fwd, local_left, local_z, metrics = candidate
        tree_x, tree_y = self._local_camera_to_map(local_fwd, local_left)

        grid_id, gx, gy, grid_dist = self._nearest_grid(tree_x, tree_y)

        tree_id, status = self._add_or_update_tree(
            x=tree_x,
            y=tree_y,
            z=self.pose_z,
            grid_id=grid_id,
            grid_x=gx,
            grid_y=gy,
            grid_dist=grid_dist,
            tof_range=self.latest_tof,
            metrics=metrics,
        )

        now = time.time()
        if now - self.last_detection_log_time > 0.4:
            self.last_detection_log_time = now
            self.get_logger().info(
                f"TREE_{status.upper()} id={tree_id} "
                f"pos=({tree_x:.2f},{tree_y:.2f}) "
                f"tof={self.latest_tof:.2f}m "
                f"grid={grid_id} grid_dist={grid_dist:.2f}m "
                f"pts={metrics['n']} width={metrics['width']:.2f} height={metrics['height']:.2f}"
            )

    # -----------------------------
    # Camera/depth verification
    # -----------------------------
    def _extract_tree_candidate_from_cloud(self, msg: PointCloud2, tof_range: float):
        points = []
        used = 0

        for i, p in enumerate(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)):
            if i % self.cloud_stride != 0:
                continue
            try:
                x, y, z = float(p[0]), float(p[1]), float(p[2])
            except Exception:
                continue

            if not math.isfinite(x) or not math.isfinite(y) or not math.isfinite(z):
                continue

            # Filter titik sekitar obstacle ToF.
            if x < 0.25:
                continue
            if abs(x - tof_range) > self.tof_cloud_tolerance_m:
                continue
            if abs(y) > self.lateral_filter_m:
                continue
            if z < self.min_z_m or z > self.max_z_m:
                continue

            points.append((x, y, z))
            used += 1
            if used >= self.max_cloud_points:
                break

        if len(points) < self.min_cluster_points:
            return None

        arr = np.array(points, dtype=np.float32)
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
            len(points) >= self.min_cluster_points
            and height >= self.min_cluster_height_m
            and width <= self.max_cluster_width_m
            and depth <= self.max_cluster_depth_m
        )

        if not verified:
            return None

        fwd = float(np.median(xs))
        left = float(np.median(ys))
        zz = float(np.median(zs))

        metrics = {
            "n": int(len(points)),
            "width": width,
            "height": height,
            "depth": depth,
            "median_forward": fwd,
            "median_left": left,
            "median_z": zz,
        }
        return fwd, left, zz, metrics

    def _image_is_recent(self) -> bool:
        if self.latest_image_time is None:
            return False
        return (time.time() - self.latest_image_time) <= self.image_timeout_s

    # -----------------------------
    # Transform dan grid
    # -----------------------------
    def _local_camera_to_map(self, forward: float, left: float) -> Tuple[float, float]:
        px, py = self.pose_xy
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)

        # body x depan, body y kiri.
        mx = px + c * forward - s * left
        my = py + s * forward + c * left
        return mx, my

    def _build_grid_centers(self) -> List[Tuple[str, float, float]]:
        out = []
        idx = 0
        for y in self.grid_ys:
            for x in self.grid_xs:
                idx += 1
                out.append((f"G{idx:02d}", float(x), float(y)))
        return out

    def _nearest_grid(self, x: float, y: float) -> Tuple[str, float, float, float]:
        best = None
        for gid, gx, gy in self.grid_centers:
            d = math.hypot(x - gx, y - gy)
            if best is None or d < best[3]:
                best = (gid, gx, gy, d)
        return best

    # -----------------------------
    # Memory pohon
    # -----------------------------
    def _add_or_update_tree(
        self,
        x: float,
        y: float,
        z: float,
        grid_id: str,
        grid_x: float,
        grid_y: float,
        grid_dist: float,
        tof_range: float,
        metrics: Dict,
    ) -> Tuple[int, str]:
        now = time.time()

        nearest = None
        for t in self.trees:
            d = math.hypot(float(t["x"]) - x, float(t["y"]) - y)
            if d <= self.merge_radius_m:
                if nearest is None or d < nearest[0]:
                    nearest = (d, t)

        if nearest is None:
            new_id = len(self.trees) + 1
            tree = {
                "id": new_id,
                "x": x,
                "y": y,
                "z": z,
                "status": "candidate_sawit",
                "visited": False,
                "camera_verified": True,
                "confirm_count": 1,
                "first_seen_time": now,
                "last_seen_time": now,
                "nearest_grid_id": grid_id,
                "nearest_grid_x": grid_x,
                "nearest_grid_y": grid_y,
                "grid_distance_m": grid_dist,
                "tof_range_m": tof_range,
                "metrics": metrics,
            }
            if tree["confirm_count"] >= self.confirm_count_required:
                tree["status"] = "sawit"

            self.trees.append(tree)
            self._save_memory()
            return new_id, "new"

        _, tree = nearest
        n = int(tree.get("confirm_count", 1))

        # Weighted average supaya marker pohon tidak jitter dan tidak dobel.
        alpha_old = n / (n + 1.0)
        alpha_new = 1.0 / (n + 1.0)

        tree["x"] = float(alpha_old * float(tree["x"]) + alpha_new * x)
        tree["y"] = float(alpha_old * float(tree["y"]) + alpha_new * y)
        tree["z"] = float(alpha_old * float(tree.get("z", z)) + alpha_new * z)
        tree["confirm_count"] = n + 1
        tree["last_seen_time"] = now
        tree["camera_verified"] = True

        gid, gx, gy, gd = self._nearest_grid(float(tree["x"]), float(tree["y"]))
        tree["nearest_grid_id"] = gid
        tree["nearest_grid_x"] = gx
        tree["nearest_grid_y"] = gy
        tree["grid_distance_m"] = gd
        tree["tof_range_m"] = tof_range
        tree["metrics"] = metrics

        if int(tree["confirm_count"]) >= self.confirm_count_required:
            tree["status"] = "sawit"

        self._save_memory()
        return int(tree["id"]), "updated"

    def _load_memory(self):
        if not self.memory_file.exists():
            self.trees = []
            return

        try:
            data = json.loads(self.memory_file.read_text())
            self.trees = data.get("trees", [])
            self.get_logger().info(f"Loaded memory: {len(self.trees)} trees")
        except Exception as e:
            self.get_logger().warn(f"Gagal load memory {self.memory_file}: {e}")
            self.trees = []

    def _save_memory(self):
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "description": "Memory pohon sawit hasil ToF obstacle + camera/depth verification + point cloud localization",
            "mission_mode": self.mission_mode,
            "grid_xs": self.grid_xs,
            "grid_ys": self.grid_ys,
            "visit_distance_m": self.visit_distance_m,
            "tof_visit_distance_m": self.tof_visit_distance,
            "trees": self.trees,
        }
        tmp = self.memory_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.memory_file)

    def _reset_memory_file(self):
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "description": "Memory direset saat start mapping",
            "trees": [],
        }
        self.memory_file.write_text(json.dumps(data, indent=2))

    # -----------------------------
    # Rem bertahap
    # -----------------------------
    def _compute_speed_limit(self, tof_range: float) -> float:
        if tof_range <= self.tof_visit_distance:
            return 0.0

        # d_aman = v*t_delay + v^2/(2a) + margin
        v = self.max_speed_mps
        a = max(self.brake_accel_mps2, 0.05)
        d_stop = v * self.reaction_delay_s + (v * v) / (2.0 * a) + self.stop_margin_m

        slow_start = max(self.tof_slow_distance, self.tof_visit_distance + d_stop)

        if tof_range >= slow_start:
            return self.max_speed_mps

        ratio = (tof_range - self.tof_visit_distance) / max((slow_start - self.tof_visit_distance), 0.01)
        speed = self.max_speed_mps * max(0.0, min(1.0, ratio))
        if speed > 0.0:
            speed = max(speed, self.min_approach_speed_mps)
        return float(speed)

    # -----------------------------
    # Publish
    # -----------------------------
    def on_timer(self):
        self._publish_json()
        self._publish_markers()

    def _publish_json(self):
        msg = String()
        msg.data = json.dumps({
            "mission_mode": self.mission_mode,
            "tree_count": len(self.trees),
            "sawit_count": sum(1 for t in self.trees if t.get("status") == "sawit"),
            "trees": self.trees,
        })
        self.json_pub.publish(msg)

    def _publish_markers(self):
        ma = MarkerArray()

        clear = Marker()
        clear.header.frame_id = "map"
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        for i, (gid, gx, gy) in enumerate(self.grid_centers):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "grid"
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = gx
            m.pose.position.y = gy
            m.pose.position.z = 0.05
            m.pose.orientation.w = 1.0
            m.scale.x = 0.25
            m.scale.y = 0.25
            m.scale.z = 0.10
            m.color.r = 1.0
            m.color.g = 1.0
            m.color.b = 1.0
            m.color.a = 0.55
            ma.markers.append(m)

            txt = Marker()
            txt.header.frame_id = "map"
            txt.header.stamp = self.get_clock().now().to_msg()
            txt.ns = "grid_text"
            txt.id = 1000 + i
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = gx
            txt.pose.position.y = gy
            txt.pose.position.z = 0.9
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.45
            txt.color.r = 1.0
            txt.color.g = 1.0
            txt.color.b = 1.0
            txt.color.a = 0.85
            txt.text = gid
            ma.markers.append(txt)

        for i, t in enumerate(self.trees):
            status = t.get("status", "candidate_sawit")
            confirmed = status == "sawit"

            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "tree_memory"
            m.id = 2000 + int(t.get("id", i + 1))
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(t["x"])
            m.pose.position.y = float(t["y"])
            m.pose.position.z = 1.1
            m.pose.orientation.w = 1.0
            m.scale.x = 0.75
            m.scale.y = 0.75
            m.scale.z = 0.75

            if confirmed:
                m.color.r = 0.0
                m.color.g = 1.0
                m.color.b = 0.2
                m.color.a = 0.95
            else:
                m.color.r = 1.0
                m.color.g = 0.85
                m.color.b = 0.0
                m.color.a = 0.90
            ma.markers.append(m)

            txt = Marker()
            txt.header.frame_id = "map"
            txt.header.stamp = self.get_clock().now().to_msg()
            txt.ns = "tree_memory_text"
            txt.id = 3000 + int(t.get("id", i + 1))
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = float(t["x"])
            txt.pose.position.y = float(t["y"])
            txt.pose.position.z = 2.0
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.42
            txt.color.r = 0.0
            txt.color.g = 1.0
            txt.color.b = 0.2
            txt.color.a = 0.95
            txt.text = (
                f"T{int(t.get('id', i + 1))} {status}\n"
                f"{t.get('nearest_grid_id','?')} d={float(t.get('grid_distance_m', 0.0)):.2f}m"
            )
            ma.markers.append(txt)

            line = Marker()
            line.header.frame_id = "map"
            line.header.stamp = self.get_clock().now().to_msg()
            line.ns = "grid_to_tree_distance"
            line.id = 4000 + int(t.get("id", i + 1))
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.05
            line.color.r = 0.2
            line.color.g = 0.8
            line.color.b = 1.0
            line.color.a = 0.8

            p1 = Point()
            p1.x = float(t.get("nearest_grid_x", 0.0))
            p1.y = float(t.get("nearest_grid_y", 0.0))
            p1.z = 0.25

            p2 = Point()
            p2.x = float(t["x"])
            p2.y = float(t["y"])
            p2.z = 0.25

            line.points = [p1, p2]
            ma.markers.append(line)

        self.marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = SawitTreeMemoryMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._save_memory()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
