#!/usr/bin/env python3

import math
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Point, TransformStamped
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
)
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from sklearn.cluster import DBSCAN
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


class State(Enum):
    WAIT_DATA = 0
    PRESTREAM = 1
    TAKEOFF = 2
    SEARCH = 3
    APPROACH = 4
    HOLD_SCAN = 5
    BACKUP = 6
    FINISHED = 7


@dataclass
class TreeTrack:
    tree_id: int
    x: float
    y: float
    z: float
    confirmations: int
    confirmed: bool
    visited: bool
    last_seen: float


class SawitRandomPointCloudNavigator(Node):
    def __init__(self):
        super().__init__("sawit_random_pointcloud_navigator")

        # ============================================================
        # FLIGHT CONFIG
        # ============================================================

        self.flight_altitude = -1.8
        self.max_visited = 16

        self.visit_distance = 3.0
        self.scan_hold = 1.5
        self.backup_time = 1.2

        self.forward_step = 0.18
        self.backward_step = 0.15
        self.max_step = 0.55
        self.max_yaw_step = math.radians(4.0)

        self.home_radius = 32.0
        self.mission_timeout = 900.0

        # ============================================================
        # SAFE SEARCH CONFIG
        # Tidak pakai random waypoint jauh.
        # Pola: rotate pelan -> rotate pelan -> rotate pelan -> maju pendek.
        # ============================================================

        self.search_yaw_step = math.radians(22.5)
        self.search_forward_step = 0.70
        self.search_forward_every = 4
        self.search_counter = 0
        self.search_yaw_target = 0.0
        self.search_action_started = 0.0
        self.search_action_duration = 1.2

        # ============================================================
        # POINT CLOUD CONFIG
        # ============================================================

        self.cloud_topic = "/camera/points"
        self.cloud_period = 0.35

        self.max_raw_points = 16000
        self.voxel_size = 0.12

        # Camera convention:
        # x = forward
        # y = left/right
        # z = vertical
        self.min_forward = 1.7
        self.max_forward = 14.0
        self.max_abs_left = 7.0
        self.min_z = -1.25
        self.max_z = 1.20

        self.dbscan_eps = 0.48
        self.dbscan_min_samples = 9

        self.min_cluster_points = 28
        self.max_cluster_points = 4500

        self.max_x_span = 2.1
        self.max_y_span = 2.1
        self.min_z_span = 0.25
        self.max_z_span = 3.2

        # ============================================================
        # TRACKING / DOUBLE SCAN CONFIG
        # ============================================================

        self.match_radius = 3.2
        self.duplicate_radius = 4.2
        self.min_confirmations = 2
        self.ema_alpha = 0.35
        self.unconfirmed_timeout = 8.0

        self.target_timeout = 5.0
        self.max_select_age = 4.0

        self.min_target_distance = 2.2
        self.max_target_distance = 20.0

        # ============================================================
        # RUNTIME STATE
        # ============================================================

        self.state = State.WAIT_DATA

        self.position = np.array([np.nan, np.nan, np.nan], dtype=float)
        self.heading = 0.0
        self.position_valid = False

        self.home_xy: Optional[np.ndarray] = None
        self.takeoff_target: Optional[np.ndarray] = None

        self.prestream_count = 0
        self.land_sent = False
        self.mission_started: Optional[float] = None

        self.tracks: List[TreeTrack] = []
        self.next_tree_id = 0
        self.active_target_id: Optional[int] = None

        self.scan_started = 0.0
        self.backup_started = 0.0

        self.last_cloud_time = 0.0
        self.last_log_time = 0.0
        self.last_no_cloud_log = 0.0

        self.route: List[np.ndarray] = []
        self.last_route: Optional[np.ndarray] = None

        # ============================================================
        # QOS
        # ============================================================

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ============================================================
        # SUBSCRIBERS
        # ============================================================

        self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self.position_callback,
            px4_qos,
        )

        self.create_subscription(
            PointCloud2,
            self.cloud_topic,
            self.cloud_callback,
            sensor_qos,
        )

        # ============================================================
        # PX4 PUBLISHERS
        # ============================================================

        self.offboard_pub = self.create_publisher(
            OffboardControlMode,
            "/fmu/in/offboard_control_mode",
            10,
        )

        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint,
            "/fmu/in/trajectory_setpoint",
            10,
        )

        self.command_pub = self.create_publisher(
            VehicleCommand,
            "/fmu/in/vehicle_command",
            10,
        )

        # ============================================================
        # RVIZ PUBLISHERS
        # ============================================================

        self.tree_pub = self.create_publisher(
            MarkerArray,
            "/sawit/tree_markers",
            10,
        )

        self.route_pub = self.create_publisher(
            Marker,
            "/sawit/route_marker",
            10,
        )

        # Non-SLAM RViz frame.
        self.static_tf = StaticTransformBroadcaster(self)

        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = "map"
        tf.child_frame_id = "px4_local"
        tf.transform.rotation.w = 1.0
        self.static_tf.sendTransform(tf)

        self.create_timer(0.1, self.control_loop)

        self.get_logger().info("Random PointCloud Navigator started")
        self.get_logger().info("Mode: NON-SLAM SAFE random point cloud")
        self.get_logger().info("Search mode: rotate + short forward step")
        self.get_logger().info("One confirmed tree = one RViz sphere")

    # ============================================================
    # CALLBACKS
    # ============================================================

    def position_callback(self, msg: VehicleLocalPosition):
        p = np.array([msg.x, msg.y, msg.z], dtype=float)

        if not np.all(np.isfinite(p)):
            return

        if hasattr(msg, "xy_valid") and not msg.xy_valid:
            return

        if hasattr(msg, "z_valid") and not msg.z_valid:
            return

        self.position = p
        self.position_valid = True

        if math.isfinite(msg.heading):
            self.heading = self.normalize(float(msg.heading))

        if self.home_xy is None:
            self.home_xy = p[:2].copy()
            self.takeoff_target = np.array(
                [p[0], p[1], self.flight_altitude],
                dtype=float,
            )

            self.search_yaw_target = self.heading

            self.get_logger().info(
                f"PX4 ready x={p[0]:.2f}, y={p[1]:.2f}, z={p[2]:.2f}"
            )

        enu = self.ned_to_enu(p)

        if (
            self.last_route is None
            or np.linalg.norm(enu[:2] - self.last_route[:2]) >= 0.20
        ):
            self.route.append(enu)
            self.last_route = enu
            self.route = self.route[-5000:]

    def cloud_callback(self, msg: PointCloud2):
        if not self.position_valid:
            return

        now = time.monotonic()

        if now - self.last_cloud_time < self.cloud_period:
            return

        self.last_cloud_time = now

        points = []

        for raw in point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
        ):
            x = float(raw[0])
            y = float(raw[1])
            z = float(raw[2])

            if not (
                math.isfinite(x)
                and math.isfinite(y)
                and math.isfinite(z)
            ):
                continue

            if x < self.min_forward or x > self.max_forward:
                continue

            if abs(y) > self.max_abs_left:
                continue

            if z < self.min_z or z > self.max_z:
                continue

            points.append((x, y, z))

            if len(points) >= self.max_raw_points:
                break

        if len(points) < self.min_cluster_points:
            self.remove_stale(now)

            if now - self.last_no_cloud_log >= 1.5:
                self.last_no_cloud_log = now
                self.get_logger().info(
                    f"CLOUD too few points={len(points)}, "
                    f"confirmed={len(self.confirmed_tracks())}, "
                    f"visited={self.count_visited()}/{self.max_visited}"
                )

            return

        cloud = np.asarray(points, dtype=np.float32)

        voxels = np.floor(cloud / self.voxel_size).astype(np.int32)

        _, ids = np.unique(
            voxels,
            axis=0,
            return_index=True,
        )

        cloud = cloud[np.sort(ids)]

        if len(cloud) < self.min_cluster_points:
            self.remove_stale(now)
            return

        labels = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            n_jobs=1,
        ).fit_predict(cloud[:, :2])

        local_candidates: List[Tuple[float, float, float, int]] = []

        for label in set(labels.tolist()):
            if label == -1:
                continue

            cluster = cloud[labels == label]
            count = len(cluster)

            if count < self.min_cluster_points:
                continue

            if count > self.max_cluster_points:
                continue

            span = np.ptp(cluster, axis=0)

            if float(span[0]) > self.max_x_span:
                continue

            if float(span[1]) > self.max_y_span:
                continue

            if float(span[2]) < self.min_z_span:
                continue

            if float(span[2]) > self.max_z_span:
                continue

            center = np.median(cluster, axis=0)

            local_candidates.append(
                (
                    float(center[0]),
                    float(center[1]),
                    float(center[2]),
                    count,
                )
            )

        # One local cluster should become only one candidate.
        merged = []

        for c in sorted(
            local_candidates,
            key=lambda a: math.hypot(a[0], a[1]),
        ):
            duplicate = False

            for m in merged:
                if math.hypot(c[0] - m[0], c[1] - m[1]) < 0.9:
                    duplicate = True
                    break

            if not duplicate:
                merged.append(c)

        updated = set()

        for forward, left, up, _ in merged:
            obs = self.local_to_ned(forward, left, up)
            self.update_track(obs, now, updated)

        self.remove_stale(now)

        if now - self.last_log_time >= 1.0:
            self.last_log_time = now

            self.get_logger().info(
                f"CLOUD points={len(cloud)}, candidates={len(merged)}, "
                f"confirmed={len(self.confirmed_tracks())}, "
                f"visited={self.count_visited()}/{self.max_visited}"
            )

    # ============================================================
    # COORDINATE AND TRACKING
    # ============================================================

    def local_to_ned(self, forward: float, left: float, up: float) -> np.ndarray:
        x = (
            self.position[0]
            + forward * math.cos(self.heading)
            - left * math.sin(self.heading)
        )

        y = (
            self.position[1]
            + forward * math.sin(self.heading)
            + left * math.cos(self.heading)
        )

        z = self.position[2] - up

        return np.array([x, y, z], dtype=float)

    def update_track(self, obs: np.ndarray, now: float, updated: set):
        nearest = None
        nearest_d = float("inf")

        for track in self.tracks:
            if track.tree_id in updated:
                continue

            d = math.hypot(
                obs[0] - track.x,
                obs[1] - track.y,
            )

            if d < nearest_d:
                nearest_d = d
                nearest = track

        # Case 1: observasi cocok kuat dengan track lama.
        if nearest is not None and nearest_d <= self.match_radius:
            a = self.ema_alpha

            nearest.x = (1.0 - a) * nearest.x + a * obs[0]
            nearest.y = (1.0 - a) * nearest.y + a * obs[1]
            nearest.z = (1.0 - a) * nearest.z + a * obs[2]

            nearest.confirmations += 1
            nearest.last_seen = now

            updated.add(nearest.tree_id)

            if (
                not nearest.confirmed
                and nearest.confirmations >= self.min_confirmations
            ):
                nearest.confirmed = True

                self.get_logger().info(
                    f"TREE_CONFIRMED id={nearest.tree_id} "
                    f"x={nearest.x:.2f}, y={nearest.y:.2f}"
                )

            return

        # Case 2: observasi tidak cukup dekat untuk match kuat,
        # tetapi masih dekat dengan pohon confirmed.
        # Jangan buat ID baru. Update pelan dan refresh last_seen.
        for track in self.tracks:
            if not track.confirmed:
                continue

            d = math.hypot(
                obs[0] - track.x,
                obs[1] - track.y,
            )

            if d <= self.duplicate_radius:
                a = 0.12

                track.x = (1.0 - a) * track.x + a * obs[0]
                track.y = (1.0 - a) * track.y + a * obs[1]
                track.z = (1.0 - a) * track.z + a * obs[2]

                track.last_seen = now
                track.confirmations += 1

                updated.add(track.tree_id)

                return

        # Case 3: benar-benar jauh dari semua track lama.
        self.tracks.append(
            TreeTrack(
                tree_id=self.next_tree_id,
                x=float(obs[0]),
                y=float(obs[1]),
                z=float(obs[2]),
                confirmations=1,
                confirmed=False,
                visited=False,
                last_seen=now,
            )
        )

        self.next_tree_id += 1

    def remove_stale(self, now: float):
        self.tracks = [
            t
            for t in self.tracks
            if t.confirmed or now - t.last_seen <= self.unconfirmed_timeout
        ]

    def confirmed_tracks(self):
        return [t for t in self.tracks if t.confirmed]

    def count_visited(self):
        return sum(1 for t in self.tracks if t.visited)

    def get_track(self, tree_id: Optional[int]) -> Optional[TreeTrack]:
        if tree_id is None:
            return None

        for track in self.tracks:
            if track.tree_id == tree_id:
                return track

        return None

    def select_target(self) -> bool:
        choices = []
        now = time.monotonic()

        for track in self.tracks:
            if not track.confirmed:
                continue

            if track.visited:
                continue

            age = now - track.last_seen

            # Jangan pilih target lama yang tidak sedang terlihat.
            if age > self.max_select_age:
                continue

            d = math.hypot(
                track.x - self.position[0],
                track.y - self.position[1],
            )

            if d < self.min_target_distance:
                continue

            if d > self.max_target_distance:
                continue

            score = d + 0.8 * age
            choices.append((score, track))

        if not choices:
            return False

        choices.sort(key=lambda item: item[0])

        # Pilih salah satu dari 3 target terbaik.
        _, selected = random.choice(choices[: min(3, len(choices))])

        self.active_target_id = selected.tree_id

        self.get_logger().info(
            f"TARGET_SELECTED id={selected.tree_id}, "
            f"distance={math.hypot(selected.x - self.position[0], selected.y - self.position[1]):.2f}, "
            f"age={now - selected.last_seen:.2f}s"
        )

        return True

    # ============================================================
    # SAFE SEARCH
    # ============================================================

    def handle_exploration(self):
        now = time.monotonic()

        if self.search_action_started == 0.0:
            self.search_action_started = now
            self.search_yaw_target = self.heading

        # Tahan setiap aksi 1.2 detik.
        if now - self.search_action_started < self.search_action_duration:
            self.hold(self.search_yaw_target)
            return

        self.search_action_started = now
        self.search_counter += 1

        # Setiap beberapa rotasi, maju pendek.
        if self.search_counter % self.search_forward_every == 0:
            target = self.body_step(self.search_forward_step)
            self.publish_setpoint(target, self.heading)

            self.get_logger().info(
                f"SEARCH_FORWARD step={self.search_forward_step:.2f}"
            )
            return

        # Rotate kiri-kanan pelan.
        direction = 1.0 if self.search_counter % 2 == 1 else -1.0

        self.search_yaw_target = self.normalize(
            self.heading + direction * self.search_yaw_step
        )

        self.hold(self.search_yaw_target)

        self.get_logger().info("SEARCH_ROTATE")

    # ============================================================
    # PX4 CONTROL
    # ============================================================

    def timestamp(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    def heartbeat(self):
        msg = OffboardControlMode()
        msg.timestamp = self.timestamp()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False

        self.offboard_pub.publish(msg)

    def publish_setpoint(self, requested: np.ndarray, yaw: float):
        if not self.position_valid:
            return

        target = np.asarray(requested, dtype=float).copy()

        if target.shape != (3,) or not np.all(np.isfinite(target)):
            self.get_logger().error("Invalid setpoint rejected")
            return

        delta = target[:2] - self.position[:2]
        d = float(np.linalg.norm(delta))

        if d > self.max_step:
            target[:2] = self.position[:2] + delta / d * self.max_step

        target[2] = self.flight_altitude

        if (
            self.home_xy is not None
            and np.linalg.norm(target[:2] - self.home_xy) > self.home_radius
        ):
            self.get_logger().error("Boundary reached; landing")
            self.state = State.FINISHED
            return

        msg = TrajectorySetpoint()
        msg.timestamp = self.timestamp()

        msg.position = [
            float(target[0]),
            float(target[1]),
            float(target[2]),
        ]

        msg.velocity = [math.nan, math.nan, math.nan]
        msg.acceleration = [math.nan, math.nan, math.nan]
        msg.jerk = [math.nan, math.nan, math.nan]
        msg.yaw = float(self.normalize(yaw))
        msg.yawspeed = math.nan

        self.setpoint_pub.publish(msg)

    def command(self, command: int, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.timestamp = self.timestamp()

        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.command = int(command)

        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True

        self.command_pub.publish(msg)

    def hold(self, yaw: Optional[float] = None):
        self.publish_setpoint(
            np.array(
                [
                    self.position[0],
                    self.position[1],
                    self.flight_altitude,
                ],
                dtype=float,
            ),
            self.heading if yaw is None else yaw,
        )

    def body_step(self, distance: float):
        return np.array(
            [
                self.position[0] + distance * math.cos(self.heading),
                self.position[1] + distance * math.sin(self.heading),
                self.flight_altitude,
            ],
            dtype=float,
        )

    # ============================================================
    # STATE MACHINE
    # ============================================================

    def control_loop(self):
        self.heartbeat()
        self.publish_markers()
        self.publish_route()

        if self.state == State.WAIT_DATA:
            if self.position_valid and self.takeoff_target is not None:
                self.state = State.PRESTREAM
                self.get_logger().info("STATE -> PRESTREAM")
            return

        if self.state == State.PRESTREAM:
            self.publish_setpoint(self.takeoff_target, self.heading)
            self.prestream_count += 1

            if self.prestream_count == 20:
                self.command(
                    VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                    1.0,
                    6.0,
                )
                self.command(
                    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                    1.0,
                )
                self.get_logger().info("OFFBOARD + ARM sent")

            if self.prestream_count >= 35:
                self.state = State.TAKEOFF
                self.get_logger().info("STATE -> TAKEOFF")

            return

        if self.state == State.TAKEOFF:
            self.publish_setpoint(self.takeoff_target, self.heading)

            if abs(self.position[2] - self.flight_altitude) <= 0.25:
                self.mission_started = time.monotonic()
                self.state = State.SEARCH
                self.search_action_started = 0.0
                self.get_logger().info("TAKEOFF complete; STATE -> SEARCH")

            return

        if (
            self.mission_started is not None
            and time.monotonic() - self.mission_started >= self.mission_timeout
        ):
            self.state = State.FINISHED

        if self.state == State.SEARCH:
            if self.count_visited() >= self.max_visited:
                self.state = State.FINISHED
                return

            if self.select_target():
                self.state = State.APPROACH
                return

            self.handle_exploration()
            return

        if self.state == State.APPROACH:
            track = self.get_track(self.active_target_id)

            if track is None or track.visited:
                self.active_target_id = None
                self.search_action_started = 0.0
                self.state = State.SEARCH
                return

            age = time.monotonic() - track.last_seen

            if age > self.target_timeout:
                self.get_logger().warn(
                    f"TARGET_LOST id={track.tree_id}, age={age:.2f}s. "
                    "Hold and return to SEARCH."
                )

                self.hold()
                self.active_target_id = None
                self.search_action_started = 0.0
                self.state = State.SEARCH

                return

            dx = track.x - self.position[0]
            dy = track.y - self.position[1]
            distance = math.hypot(dx, dy)

            if distance <= self.visit_distance:
                self.scan_started = time.monotonic()
                self.state = State.HOLD_SCAN
                self.get_logger().info(
                    f"TARGET_REACHED id={track.tree_id}, distance={distance:.2f}"
                )
                return

            target_yaw = math.atan2(dy, dx)
            yaw_error = self.normalize(target_yaw - self.heading)

            if abs(yaw_error) > math.radians(10.0):
                correction = max(
                    -self.max_yaw_step,
                    min(self.max_yaw_step, yaw_error),
                )

                self.hold(
                    self.normalize(self.heading + correction)
                )
            else:
                self.publish_setpoint(
                    np.array(
                        [
                            track.x,
                            track.y,
                            self.flight_altitude,
                        ],
                        dtype=float,
                    ),
                    target_yaw,
                )

            return

        if self.state == State.HOLD_SCAN:
            self.hold()

            if time.monotonic() - self.scan_started >= self.scan_hold:
                track = self.get_track(self.active_target_id)

                if track is not None:
                    track.visited = True

                    self.get_logger().info(
                        f"TREE_VISITED id={track.tree_id}: "
                        f"{self.count_visited()}/{self.max_visited}"
                    )

                self.backup_started = time.monotonic()
                self.state = State.BACKUP

            return

        if self.state == State.BACKUP:
            if time.monotonic() - self.backup_started < self.backup_time:
                self.publish_setpoint(
                    self.body_step(-self.backward_step),
                    self.heading,
                )
            else:
                self.active_target_id = None
                self.search_action_started = 0.0
                self.state = State.SEARCH

            return

        if self.state == State.FINISHED and not self.land_sent:
            self.land_sent = True
            self.command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.get_logger().info("MISSION FINISHED; LAND sent")

    # ============================================================
    # RVIZ
    # ============================================================

    def ned_to_enu(self, p: np.ndarray):
        return np.array(
            [
                p[0],
                -p[1],
                -p[2],
            ],
            dtype=float,
        )

    def publish_markers(self):
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()

        for track in self.confirmed_tracks():
            enu = self.ned_to_enu(
                np.array(
                    [
                        track.x,
                        track.y,
                        track.z,
                    ],
                    dtype=float,
                )
            )

            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = now
            marker.ns = "trees"
            marker.id = int(track.tree_id)
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = float(enu[0])
            marker.pose.position.y = float(enu[1])
            marker.pose.position.z = float(max(0.25, enu[2]))
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.85
            marker.scale.y = 0.85
            marker.scale.z = 0.85

            if track.visited:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0

            marker.color.a = 1.0

            arr.markers.append(marker)

        active = self.get_track(self.active_target_id)

        if active is not None:
            enu = self.ned_to_enu(
                np.array(
                    [
                        active.x,
                        active.y,
                        active.z,
                    ],
                    dtype=float,
                )
            )

            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = now
            marker.ns = "active_target"
            marker.id = 50000
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = float(enu[0])
            marker.pose.position.y = float(enu[1])
            marker.pose.position.z = float(max(0.3, enu[2] + 0.3))
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.65
            marker.scale.y = 0.65
            marker.scale.z = 0.65

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            arr.markers.append(marker)

        self.tree_pub.publish(arr)

    def publish_route(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "route"
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.10

        marker.color.r = 0.0
        marker.color.g = 0.25
        marker.color.b = 1.0
        marker.color.a = 1.0

        for xyz in self.route:
            p = Point()
            p.x = float(xyz[0])
            p.y = float(xyz[1])
            p.z = float(xyz[2])

            marker.points.append(p)

        self.route_pub.publish(marker)

    @staticmethod
    def normalize(angle: float):
        return math.atan2(
            math.sin(angle),
            math.cos(angle),
        )


def main(args=None):
    rclpy.init(args=args)

    node = SawitRandomPointCloudNavigator()

    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)

    try:
        executor.spin()

    except KeyboardInterrupt:
        pass

    finally:
        executor.shutdown()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
