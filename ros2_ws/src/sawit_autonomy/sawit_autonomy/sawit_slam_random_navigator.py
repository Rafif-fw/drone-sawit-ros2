#!/usr/bin/env python3

import math
import random
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)
from rclpy.time import Time

from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray

from sklearn.cluster import DBSCAN

from tf2_ros import Buffer, TransformException, TransformListener

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
)


class MissionState(Enum):
    WAIT_DATA = 0
    PRESTREAM = 1
    TAKEOFF = 2
    SEARCH = 3
    APPROACH = 4
    HOLD_SCAN = 5
    BACKUP = 6
    FINISHED = 7


class SawitSlamRandomNavigator(Node):
    """
    Random oil-palm navigator using RTAB-Map.

    Perception:
        /rtabmap/cloud_map

    SLAM pose and route:
        /rtabmap/odom

    Flight control:
        PX4 Offboard using VehicleLocalPosition and TrajectorySetpoint.

    Target positions remain in the RTAB-Map 'map' frame.
    During approach, the target is transformed from map to camera frame
    using TF. This avoids manually projecting targets with only drone yaw.
    """

    def __init__(self) -> None:
        super().__init__("sawit_slam_random_navigator")

        # =========================================================
        # SIMULATION SAFETY
        # =========================================================

        self.simulation_only = True

        # PX4 NED: negative z means upward.
        self.flight_altitude = -1.70

        # Maximum number of unique trees to visit.
        self.maximum_visited_trees = 16

        # Operating boundary in PX4 local coordinates.
        self.maximum_distance_from_home = 45.0

        # =========================================================
        # TOPICS AND FRAMES
        # =========================================================

        self.cloud_topic = "/rtabmap/cloud_map"
        self.odom_topic = "/rtabmap/odom"

        self.map_frame = "map"
        self.camera_frame = (
            "x500_0/front_sensor_link/camera_front_depth"
        )

        self.px4_position_topic = (
            "/fmu/out/vehicle_local_position_v1"
        )

        # =========================================================
        # CLOUD PROCESSING
        # =========================================================

        # Cloud map may be very large.
        self.maximum_cloud_points = 30000

        # Process map only every few seconds.
        self.cloud_process_interval = 2.0

        # Voxel downsampling.
        self.voxel_size = 0.15

        # Horizontal detection range relative to current SLAM position.
        self.local_map_radius = 18.0

        # Detect approximately lower-middle portions of trees.
        # Values are relative to current SLAM camera altitude.
        self.minimum_relative_z = -1.55
        self.maximum_relative_z = 0.35

        # DBSCAN on map X-Y.
        self.dbscan_eps = 0.55
        self.dbscan_min_samples = 10

        self.minimum_cluster_points = 25
        self.maximum_cluster_points = 4000

        # Candidate trunk shape.
        self.maximum_x_span = 1.70
        self.maximum_y_span = 1.70
        self.minimum_z_span = 0.35
        self.maximum_z_span = 2.30

        # Track confirmation.
        self.minimum_confirmations = 3
        self.track_match_radius = 1.75
        self.duplicate_tree_radius = 2.80
        self.unconfirmed_timeout = 8.0

        # =========================================================
        # RANDOM TARGET SELECTION
        # =========================================================

        # Random is chosen from nearest candidates, not whole map.
        self.random_candidate_top_k = 5

        # Do not pick targets too close or too far.
        self.minimum_target_distance = 3.5
        self.maximum_target_distance = 20.0

        # =========================================================
        # APPROACH CONTROL
        # =========================================================

        # Target in camera coordinates.
        self.center_tolerance = 0.70

        # Target considered reached at this horizontal distance.
        self.visit_distance = 3.0

        # Emergency stop distance.
        self.danger_distance = 2.0

        # Small PX4 local setpoint movement.
        self.forward_step = 0.12
        self.backward_step = 0.10

        # Maximum yaw correction per control cycle.
        self.maximum_yaw_step_deg = 4.0

        self.scan_hold_duration = 1.5
        self.backup_duration = 1.2

        # Search behaviour.
        self.search_yaw_step_deg = 20.0
        self.search_yaw_interval = 1.4

        # After several rotations, move into a new area.
        self.search_forward_every = 6
        self.search_forward_step = 0.30

        # If target cannot be transformed for this duration, abandon it.
        self.target_tf_timeout = 3.0

        # =========================================================
        # SETPOINT SAFETY
        # =========================================================

        self.maximum_setpoint_step = 0.60

        # =========================================================
        # RUNTIME STATE
        # =========================================================

        self.state = MissionState.WAIT_DATA

        self.px4_position = np.array(
            [float("nan"), float("nan"), float("nan")],
            dtype=float,
        )
        self.px4_yaw = 0.0
        self.px4_position_valid = False
        self.px4_home_xy: Optional[np.ndarray] = None

        self.slam_position = np.array(
            [float("nan"), float("nan"), float("nan")],
            dtype=float,
        )
        self.slam_yaw = 0.0
        self.slam_odom_valid = False

        self.last_cloud_time = 0.0

        # Tree track:
        # {
        #   id,
        #   position_map,
        #   confirmations,
        #   confirmed,
        #   visited,
        #   last_seen
        # }
        self.tree_tracks: List[Dict] = []
        self.next_tree_id = 0

        self.current_target_id: Optional[int] = None
        self.last_target_tf_success = 0.0

        self.prestream_counter = 0
        self.takeoff_target: Optional[np.ndarray] = None

        self.search_yaw_target = 0.0
        self.last_search_action = 0.0
        self.search_action_count = 0

        self.scan_start_time: Optional[float] = None
        self.backup_start_time: Optional[float] = None

        self.route_points: List[np.ndarray] = []
        self.last_route_point: Optional[np.ndarray] = None
        self.route_record_distance = 0.20

        self.last_log_time = 0.0
        self.land_command_sent = False

        # =========================================================
        # TF
        # =========================================================

        self.tf_buffer = Buffer(
            cache_time=Duration(seconds=20.0)
        )
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        # =========================================================
        # QOS
        # =========================================================

        self.sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.px4_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # =========================================================
        # SUBSCRIBERS
        # =========================================================

        self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            self.sensor_qos,
        )

        self.create_subscription(
            PointCloud2,
            self.cloud_topic,
            self.cloud_callback,
            self.sensor_qos,
        )

        self.create_subscription(
            VehicleLocalPosition,
            self.px4_position_topic,
            self.px4_position_callback,
            self.px4_qos,
        )

        # =========================================================
        # PX4 PUBLISHERS
        # =========================================================

        self.offboard_publisher = self.create_publisher(
            OffboardControlMode,
            "/fmu/in/offboard_control_mode",
            10,
        )

        self.setpoint_publisher = self.create_publisher(
            TrajectorySetpoint,
            "/fmu/in/trajectory_setpoint",
            10,
        )

        self.command_publisher = self.create_publisher(
            VehicleCommand,
            "/fmu/in/vehicle_command",
            10,
        )

        # =========================================================
        # RVIZ PUBLISHERS
        # =========================================================

        self.tree_marker_publisher = self.create_publisher(
            MarkerArray,
            "/sawit/tree_markers",
            10,
        )

        self.route_marker_publisher = self.create_publisher(
            Marker,
            "/sawit/route_marker",
            10,
        )

        self.clear_markers()

        self.create_timer(0.1, self.control_loop)

        self.get_logger().info(
            "Sawit SLAM Random Navigator started"
        )
        self.get_logger().info(
            "SIMULATION ONLY: Gazebo/PX4 SITL"
        )
        self.get_logger().info(
            f"SLAM odom: {self.odom_topic}"
        )
        self.get_logger().info(
            f"SLAM cloud: {self.cloud_topic}"
        )
        self.get_logger().info(
            f"Camera frame: {self.camera_frame}"
        )

    # =============================================================
    # CALLBACKS
    # =============================================================

    def px4_position_callback(
        self,
        message: VehicleLocalPosition,
    ) -> None:
        position = np.array(
            [message.x, message.y, message.z],
            dtype=float,
        )

        if not np.all(np.isfinite(position)):
            return

        if not bool(getattr(message, "xy_valid", True)):
            return

        if not bool(getattr(message, "z_valid", True)):
            return

        self.px4_position = position
        self.px4_position_valid = True

        if math.isfinite(message.heading):
            self.px4_yaw = self.normalize_angle(
                float(message.heading)
            )

        if self.px4_home_xy is None:
            self.px4_home_xy = position[:2].copy()

            self.takeoff_target = np.array(
                [
                    position[0],
                    position[1],
                    self.flight_altitude,
                ],
                dtype=float,
            )

            self.search_yaw_target = self.px4_yaw

            self.get_logger().info(
                f"PX4 position ready: "
                f"x={position[0]:.2f}, "
                f"y={position[1]:.2f}, "
                f"z={position[2]:.2f}"
            )

    def odom_callback(
        self,
        message: Odometry,
    ) -> None:
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation

        slam_position = np.array(
            [position.x, position.y, position.z],
            dtype=float,
        )

        if not np.all(np.isfinite(slam_position)):
            return

        self.slam_position = slam_position
        self.slam_yaw = self.quaternion_to_yaw(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        self.slam_odom_valid = True

        self.record_route()

    def cloud_callback(
        self,
        message: PointCloud2,
    ) -> None:
        if not self.slam_odom_valid:
            return

        now = time.time()

        if (
            now - self.last_cloud_time
            < self.cloud_process_interval
        ):
            return

        self.last_cloud_time = now

        points = self.read_cloud(message)

        if len(points) < self.minimum_cluster_points:
            self.remove_stale_unconfirmed_tracks(now)
            return

        points = self.voxel_downsample(points)

        candidates = self.detect_tree_candidates(points)

        updated_track_ids = set()

        for candidate in candidates:
            self.update_or_create_track(
                candidate,
                now,
                updated_track_ids,
            )

        self.remove_stale_unconfirmed_tracks(now)

    # =============================================================
    # POINT CLOUD PROCESSING
    # =============================================================

    def read_cloud(
        self,
        message: PointCloud2,
    ) -> np.ndarray:
        points: List[List[float]] = []

        current_x = float(self.slam_position[0])
        current_y = float(self.slam_position[1])
        current_z = float(self.slam_position[2])

        for raw_point in point_cloud2.read_points(
            message,
            field_names=("x", "y", "z"),
            skip_nans=True,
        ):
            x = float(raw_point[0])
            y = float(raw_point[1])
            z = float(raw_point[2])

            if not (
                math.isfinite(x)
                and math.isfinite(y)
                and math.isfinite(z)
            ):
                continue

            horizontal_distance = math.hypot(
                x - current_x,
                y - current_y,
            )

            if horizontal_distance > self.local_map_radius:
                continue

            relative_z = z - current_z

            if relative_z < self.minimum_relative_z:
                continue

            if relative_z > self.maximum_relative_z:
                continue

            points.append([x, y, z])

            if len(points) >= self.maximum_cloud_points:
                break

        if not points:
            return np.empty((0, 3), dtype=float)

        return np.asarray(points, dtype=float)

    def voxel_downsample(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        if len(points) == 0:
            return points

        voxel_indices = np.floor(
            points / self.voxel_size
        ).astype(np.int32)

        _, unique_indices = np.unique(
            voxel_indices,
            axis=0,
            return_index=True,
        )

        return points[np.sort(unique_indices)]

    def detect_tree_candidates(
        self,
        points: np.ndarray,
    ) -> List[np.ndarray]:
        if len(points) < self.minimum_cluster_points:
            return []

        labels = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
        ).fit_predict(points[:, :2])

        candidates: List[np.ndarray] = []

        raw_clusters = 0
        rejected_clusters = 0

        for label in set(labels.tolist()):
            if label == -1:
                continue

            raw_clusters += 1

            cluster = points[labels == label]

            if len(cluster) < self.minimum_cluster_points:
                rejected_clusters += 1
                continue

            if len(cluster) > self.maximum_cluster_points:
                rejected_clusters += 1
                continue

            x_span = float(
                np.max(cluster[:, 0])
                - np.min(cluster[:, 0])
            )

            y_span = float(
                np.max(cluster[:, 1])
                - np.min(cluster[:, 1])
            )

            z_span = float(
                np.max(cluster[:, 2])
                - np.min(cluster[:, 2])
            )

            if x_span > self.maximum_x_span:
                rejected_clusters += 1
                continue

            if y_span > self.maximum_y_span:
                rejected_clusters += 1
                continue

            if z_span < self.minimum_z_span:
                rejected_clusters += 1
                continue

            if z_span > self.maximum_z_span:
                rejected_clusters += 1
                continue

            center = np.median(cluster, axis=0)

            candidates.append(center)

        candidates = self.merge_close_candidates(
            candidates
        )

        self.get_logger().info(
            f"SLAM_CLOUD: points={len(points)}, "
            f"raw_clusters={raw_clusters}, "
            f"rejected={rejected_clusters}, "
            f"candidates={len(candidates)}"
        )

        return candidates

    def merge_close_candidates(
        self,
        candidates: List[np.ndarray],
    ) -> List[np.ndarray]:
        merged: List[np.ndarray] = []

        for candidate in candidates:
            duplicate_index = None

            for index, existing in enumerate(merged):
                distance = float(
                    np.linalg.norm(
                        candidate[:2] - existing[:2]
                    )
                )

                if distance < self.duplicate_tree_radius:
                    duplicate_index = index
                    break

            if duplicate_index is None:
                merged.append(candidate.copy())
            else:
                merged[duplicate_index] = (
                    0.5 * merged[duplicate_index]
                    + 0.5 * candidate
                )

        return merged

    # =============================================================
    # TRACKING
    # =============================================================

    def update_or_create_track(
        self,
        position_map: np.ndarray,
        now: float,
        updated_track_ids: set,
    ) -> None:
        nearest_track = None
        nearest_distance = float("inf")

        for track in self.tree_tracks:
            if track["id"] in updated_track_ids:
                continue

            distance = float(
                np.linalg.norm(
                    position_map[:2]
                    - track["position_map"][:2]
                )
            )

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_track = track

        if (
            nearest_track is not None
            and nearest_distance <= self.track_match_radius
        ):
            if not nearest_track["confirmed"]:
                alpha = 0.15

                nearest_track["position_map"] = (
                    (1.0 - alpha)
                    * nearest_track["position_map"]
                    + alpha
                    * position_map
                )

            nearest_track["confirmations"] += 1
            nearest_track["last_seen"] = now

            updated_track_ids.add(
                nearest_track["id"]
            )

            if (
                not nearest_track["confirmed"]
                and nearest_track["confirmations"]
                >= self.minimum_confirmations
            ):
                nearest_track["confirmed"] = True

                self.get_logger().info(
                    f"TREE_CONFIRMED id={nearest_track['id']}: "
                    f"x={nearest_track['position_map'][0]:.2f}, "
                    f"y={nearest_track['position_map'][1]:.2f}"
                )

            return

        for track in self.tree_tracks:
            if not track["confirmed"]:
                continue

            distance = float(
                np.linalg.norm(
                    position_map[:2]
                    - track["position_map"][:2]
                )
            )

            if distance < self.duplicate_tree_radius:
                return

        track = {
            "id": self.next_tree_id,
            "position_map": position_map.copy(),
            "confirmations": 1,
            "confirmed": False,
            "visited": False,
            "last_seen": now,
        }

        self.next_tree_id += 1
        self.tree_tracks.append(track)
        updated_track_ids.add(track["id"])

    def remove_stale_unconfirmed_tracks(
        self,
        now: float,
    ) -> None:
        kept_tracks = []

        for track in self.tree_tracks:
            if track["confirmed"]:
                kept_tracks.append(track)
                continue

            age = now - float(track["last_seen"])

            if age <= self.unconfirmed_timeout:
                kept_tracks.append(track)

        self.tree_tracks = kept_tracks

    def get_track(
        self,
        track_id: Optional[int],
    ) -> Optional[Dict]:
        if track_id is None:
            return None

        for track in self.tree_tracks:
            if track["id"] == track_id:
                return track

        return None

    def confirmed_unvisited_tracks(
        self,
    ) -> List[Dict]:
        return [
            track
            for track in self.tree_tracks
            if track["confirmed"] and not track["visited"]
        ]

    def select_random_target(self) -> bool:
        candidates = []

        for track in self.confirmed_unvisited_tracks():
            distance = float(
                np.linalg.norm(
                    track["position_map"][:2]
                    - self.slam_position[:2]
                )
            )

            if distance < self.minimum_target_distance:
                continue

            if distance > self.maximum_target_distance:
                continue

            candidates.append((distance, track))

        if not candidates:
            return False

        candidates.sort(key=lambda item: item[0])

        nearest_candidates = candidates[
            : self.random_candidate_top_k
        ]

        _, selected_track = random.choice(
            nearest_candidates
        )

        self.current_target_id = selected_track["id"]
        self.last_target_tf_success = time.time()

        self.get_logger().info(
            f"RANDOM_TARGET id={selected_track['id']}: "
            f"map=("
            f"{selected_track['position_map'][0]:.2f},"
            f"{selected_track['position_map'][1]:.2f})"
        )

        return True

    # =============================================================
    # TF TARGET TRANSFORMATION
    # =============================================================

    def transform_target_to_camera(
        self,
        target_map: np.ndarray,
    ) -> Optional[np.ndarray]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.camera_frame,
                self.map_frame,
                Time(),
                timeout=Duration(seconds=0.15),
            )

        except TransformException as error:
            self.get_logger().warn(
                f"TF map->camera unavailable: {error}",
                throttle_duration_sec=2.0,
            )
            return None

        translation = transform.transform.translation
        rotation = transform.transform.rotation

        point = target_map - np.array(
            [
                translation.x,
                translation.y,
                translation.z,
            ],
            dtype=float,
        )

        rotated = self.rotate_vector_by_quaternion(
            point,
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        )

        return rotated

    @staticmethod
    def rotate_vector_by_quaternion(
        vector: np.ndarray,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
    ) -> np.ndarray:
        quaternion_vector = np.array(
            [qx, qy, qz],
            dtype=float,
        )

        uv = np.cross(
            quaternion_vector,
            vector,
        )

        uuv = np.cross(
            quaternion_vector,
            uv,
        )

        return (
            vector
            + 2.0 * (
                qw * uv + uuv
            )
        )

    # =============================================================
    # PX4 CONTROL
    # =============================================================

    def timestamp(self) -> int:
        return int(
            self.get_clock().now().nanoseconds / 1000
        )

    def publish_offboard_heartbeat(self) -> None:
        message = OffboardControlMode()
        message.timestamp = self.timestamp()

        message.position = True
        message.velocity = False
        message.acceleration = False
        message.attitude = False
        message.body_rate = False

        self.offboard_publisher.publish(message)

    def publish_setpoint(
        self,
        requested_target: np.ndarray,
        requested_yaw: Optional[float] = None,
    ) -> None:
        if not self.px4_position_valid:
            return

        target = np.asarray(
            requested_target,
            dtype=float,
        ).copy()

        if target.shape != (3,):
            return

        if not np.all(np.isfinite(target)):
            return

        delta_xy = target[:2] - self.px4_position[:2]
        distance = float(np.linalg.norm(delta_xy))

        if distance > self.maximum_setpoint_step:
            direction = delta_xy / distance

            target[0] = (
                self.px4_position[0]
                + direction[0]
                * self.maximum_setpoint_step
            )

            target[1] = (
                self.px4_position[1]
                + direction[1]
                * self.maximum_setpoint_step
            )

        target[2] = self.flight_altitude

        if not self.within_home_boundary(target):
            self.get_logger().error(
                "Safety boundary reached. Landing."
            )
            self.state = MissionState.FINISHED
            return

        yaw = (
            self.px4_yaw
            if requested_yaw is None
            else self.normalize_angle(requested_yaw)
        )

        message = TrajectorySetpoint()
        message.timestamp = self.timestamp()

        message.position = [
            float(target[0]),
            float(target[1]),
            float(target[2]),
        ]

        message.velocity = [
            float("nan"),
            float("nan"),
            float("nan"),
        ]

        message.acceleration = [
            float("nan"),
            float("nan"),
            float("nan"),
        ]

        message.jerk = [
            float("nan"),
            float("nan"),
            float("nan"),
        ]

        message.yaw = float(yaw)
        message.yawspeed = float("nan")

        self.setpoint_publisher.publish(message)

    def send_vehicle_command(
        self,
        command: int,
        param1: float = 0.0,
        param2: float = 0.0,
    ) -> None:
        message = VehicleCommand()
        message.timestamp = self.timestamp()

        message.param1 = float(param1)
        message.param2 = float(param2)
        message.param3 = 0.0
        message.param4 = 0.0
        message.param5 = 0.0
        message.param6 = 0.0
        message.param7 = 0.0

        message.command = int(command)

        message.target_system = 1
        message.target_component = 1
        message.source_system = 1
        message.source_component = 1
        message.from_external = True

        self.command_publisher.publish(message)

    def set_offboard(self) -> None:
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            1.0,
            6.0,
        )

    def arm(self) -> None:
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            1.0,
        )

    def land(self) -> None:
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_NAV_LAND,
        )

    def hold_position(
        self,
        yaw: Optional[float] = None,
    ) -> None:
        target = np.array(
            [
                self.px4_position[0],
                self.px4_position[1],
                self.flight_altitude,
            ],
            dtype=float,
        )

        self.publish_setpoint(target, yaw)

    def forward_target(
        self,
        distance: float,
    ) -> np.ndarray:
        return np.array(
            [
                self.px4_position[0]
                + distance * math.cos(self.px4_yaw),

                self.px4_position[1]
                + distance * math.sin(self.px4_yaw),

                self.flight_altitude,
            ],
            dtype=float,
        )

    def backward_target(
        self,
        distance: float,
    ) -> np.ndarray:
        return np.array(
            [
                self.px4_position[0]
                - distance * math.cos(self.px4_yaw),

                self.px4_position[1]
                - distance * math.sin(self.px4_yaw),

                self.flight_altitude,
            ],
            dtype=float,
        )

    # =============================================================
    # CONTROL LOOP
    # =============================================================

    def control_loop(self) -> None:
        self.publish_offboard_heartbeat()
        self.publish_markers()
        self.publish_route()

        if self.state == MissionState.WAIT_DATA:
            self.handle_wait_data()

        elif self.state == MissionState.PRESTREAM:
            self.handle_prestream()

        elif self.state == MissionState.TAKEOFF:
            self.handle_takeoff()

        elif self.state == MissionState.SEARCH:
            self.handle_search()

        elif self.state == MissionState.APPROACH:
            self.handle_approach()

        elif self.state == MissionState.HOLD_SCAN:
            self.handle_hold_scan()

        elif self.state == MissionState.BACKUP:
            self.handle_backup()

        elif self.state == MissionState.FINISHED:
            self.handle_finished()

    def handle_wait_data(self) -> None:
        if not self.px4_position_valid:
            return

        if not self.slam_odom_valid:
            return

        if self.takeoff_target is None:
            return

        self.prestream_counter = 0
        self.state = MissionState.PRESTREAM

        self.get_logger().info(
            "PX4 + SLAM ready. STATE -> PRESTREAM"
        )

    def handle_prestream(self) -> None:
        self.publish_setpoint(
            self.takeoff_target,
            self.px4_yaw,
        )

        self.prestream_counter += 1

        if self.prestream_counter == 20:
            self.set_offboard()
            self.arm()

            self.get_logger().info(
                "OFFBOARD + ARM sent"
            )

        if self.prestream_counter >= 35:
            self.state = MissionState.TAKEOFF
            self.get_logger().info(
                "STATE -> TAKEOFF"
            )

    def handle_takeoff(self) -> None:
        self.publish_setpoint(
            self.takeoff_target,
            self.px4_yaw,
        )

        altitude_error = abs(
            self.px4_position[2]
            - self.flight_altitude
        )

        if altitude_error <= 0.25:
            self.search_yaw_target = self.px4_yaw
            self.last_search_action = time.time()
            self.state = MissionState.SEARCH

            self.get_logger().info(
                "TAKEOFF complete. STATE -> SEARCH"
            )

    def handle_search(self) -> None:
        if (
            self.count_visited()
            >= self.maximum_visited_trees
        ):
            self.state = MissionState.FINISHED
            return

        if self.select_random_target():
            self.state = MissionState.APPROACH
            self.get_logger().info(
                "STATE -> APPROACH"
            )
            return

        now = time.time()

        self.hold_position(
            self.search_yaw_target
        )

        if (
            now - self.last_search_action
            < self.search_yaw_interval
        ):
            return

        self.last_search_action = now
        self.search_action_count += 1

        if (
            self.search_action_count
            % self.search_forward_every
            == 0
        ):
            target = self.forward_target(
                self.search_forward_step
            )
            self.publish_setpoint(
                target,
                self.px4_yaw,
            )

            self.get_logger().info(
                "SEARCH: small forward step"
            )
            return

        direction = random.choice(
            [-1.0, 1.0]
        )

        self.search_yaw_target = self.normalize_angle(
            self.search_yaw_target
            + direction
            * math.radians(
                self.search_yaw_step_deg
            )
        )

        self.get_logger().info(
            f"SEARCH_ROTATE: "
            f"confirmed={len(self.confirmed_unvisited_tracks())}, "
            f"visited={self.count_visited()}"
        )

    def handle_approach(self) -> None:
        target_track = self.get_track(
            self.current_target_id
        )

        if target_track is None:
            self.current_target_id = None
            self.state = MissionState.SEARCH
            return

        target_map = target_track["position_map"]

        target_camera = self.transform_target_to_camera(
            target_map
        )

        if target_camera is None:
            self.hold_position()

            if (
                time.time() - self.last_target_tf_success
                > self.target_tf_timeout
            ):
                self.get_logger().warn(
                    "Target TF timeout. Returning to SEARCH."
                )
                self.current_target_id = None
                self.state = MissionState.SEARCH

            return

        self.last_target_tf_success = time.time()

        # Current camera convention from previous point cloud:
        # x = forward
        # y = left
        forward = float(target_camera[0])
        left = float(target_camera[1])

        horizontal_distance = math.hypot(
            forward,
            left,
        )

        if horizontal_distance <= self.visit_distance:
            self.scan_start_time = time.time()
            self.state = MissionState.HOLD_SCAN

            self.get_logger().info(
                f"TARGET_REACHED id={target_track['id']}, "
                f"distance={horizontal_distance:.2f}"
            )
            return

        if horizontal_distance <= self.danger_distance:
            self.hold_position()
            self.scan_start_time = time.time()
            self.state = MissionState.HOLD_SCAN
            return

        bearing_left = math.atan2(
            left,
            max(forward, 0.01),
        )

        # PX4 NED heading increases clockwise.
        # Positive camera-left therefore decreases PX4 yaw.
        yaw_correction = -bearing_left

        max_yaw_step = math.radians(
            self.maximum_yaw_step_deg
        )

        yaw_correction = max(
            -max_yaw_step,
            min(max_yaw_step, yaw_correction),
        )

        target_yaw = self.normalize_angle(
            self.px4_yaw + yaw_correction
        )

        centered = abs(left) <= self.center_tolerance

        if not centered or forward <= 0.0:
            self.hold_position(target_yaw)

        else:
            movement_target = self.forward_target(
                self.forward_step
            )

            self.publish_setpoint(
                movement_target,
                target_yaw,
            )

        now = time.time()

        if now - self.last_log_time >= 0.8:
            self.last_log_time = now

            self.get_logger().info(
                f"APPROACH id={target_track['id']}: "
                f"forward={forward:.2f}, "
                f"left={left:.2f}, "
                f"distance={horizontal_distance:.2f}, "
                f"centered={centered}, "
                f"visited={self.count_visited()}/"
                f"{self.maximum_visited_trees}"
            )

    def handle_hold_scan(self) -> None:
        self.hold_position()

        if self.scan_start_time is None:
            self.scan_start_time = time.time()

        if (
            time.time() - self.scan_start_time
            < self.scan_hold_duration
        ):
            return

        target = self.get_track(
            self.current_target_id
        )

        if target is not None:
            target["visited"] = True

            self.get_logger().info(
                f"TREE_VISITED id={target['id']}: "
                f"{self.count_visited()}/"
                f"{self.maximum_visited_trees}"
            )

        self.scan_start_time = None
        self.backup_start_time = time.time()
        self.state = MissionState.BACKUP

    def handle_backup(self) -> None:
        if self.backup_start_time is None:
            self.backup_start_time = time.time()

        elapsed = (
            time.time() - self.backup_start_time
        )

        if elapsed < self.backup_duration:
            target = self.backward_target(
                self.backward_step
            )

            self.publish_setpoint(
                target,
                self.px4_yaw,
            )
            return

        self.current_target_id = None
        self.backup_start_time = None

        self.search_yaw_target = self.normalize_angle(
            self.px4_yaw
            + random.choice([-1.0, 1.0])
            * math.radians(30.0)
        )

        self.last_search_action = time.time()
        self.state = MissionState.SEARCH

    def handle_finished(self) -> None:
        if self.land_command_sent:
            return

        self.land_command_sent = True

        self.get_logger().info(
            f"MISSION FINISHED: "
            f"visited={self.count_visited()}. LANDING"
        )

        self.land()

    # =============================================================
    # RVIZ
    # =============================================================

    def clear_markers(self) -> None:
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = (
            self.get_clock().now().to_msg()
        )
        marker.action = Marker.DELETEALL

        marker_array = MarkerArray()
        marker_array.markers.append(marker)

        self.tree_marker_publisher.publish(
            marker_array
        )
        self.route_marker_publisher.publish(
            marker
        )

    def publish_markers(self) -> None:
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        for track in self.tree_tracks:
            if not track["confirmed"]:
                continue

            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = now
            marker.ns = "slam_trees"
            marker.id = int(track["id"])
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = float(
                track["position_map"][0]
            )
            marker.pose.position.y = float(
                track["position_map"][1]
            )
            marker.pose.position.z = float(
                track["position_map"][2]
            )
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.85
            marker.scale.y = 0.85
            marker.scale.z = 0.85

            if track["visited"]:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
            else:
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0

            marker_array.markers.append(marker)

        target = self.get_track(
            self.current_target_id
        )

        if target is not None:
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = now
            marker.ns = "active_target"
            marker.id = 50000
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = float(
                target["position_map"][0]
            )
            marker.pose.position.y = float(
                target["position_map"][1]
            )
            marker.pose.position.z = float(
                target["position_map"][2] + 0.3
            )
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.70
            marker.scale.y = 0.70
            marker.scale.z = 0.70

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)

        self.tree_marker_publisher.publish(
            marker_array
        )

    def record_route(self) -> None:
        point = self.slam_position.copy()

        if self.last_route_point is None:
            self.route_points.append(point)
            self.last_route_point = point
            return

        distance = float(
            np.linalg.norm(
                point[:2]
                - self.last_route_point[:2]
            )
        )

        if distance >= self.route_record_distance:
            self.route_points.append(point)
            self.last_route_point = point

        if len(self.route_points) > 5000:
            self.route_points = self.route_points[-5000:]

    def publish_route(self) -> None:
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = (
            self.get_clock().now().to_msg()
        )

        marker.ns = "slam_route"
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.10

        marker.color.r = 0.0
        marker.color.g = 0.30
        marker.color.b = 1.0
        marker.color.a = 1.0

        for route_point in self.route_points:
            point = Point()
            point.x = float(route_point[0])
            point.y = float(route_point[1])
            point.z = float(route_point[2])

            marker.points.append(point)

        self.route_marker_publisher.publish(
            marker
        )

    # =============================================================
    # UTILITIES
    # =============================================================

    def count_visited(self) -> int:
        return sum(
            1
            for track in self.tree_tracks
            if track["confirmed"] and track["visited"]
        )

    def within_home_boundary(
        self,
        target: np.ndarray,
    ) -> bool:
        if self.px4_home_xy is None:
            return False

        distance = float(
            np.linalg.norm(
                target[:2] - self.px4_home_xy
            )
        )

        return (
            distance
            <= self.maximum_distance_from_home
        )

    @staticmethod
    def normalize_angle(
        angle: float,
    ) -> float:
        return math.atan2(
            math.sin(angle),
            math.cos(angle),
        )

    @staticmethod
    def quaternion_to_yaw(
        x: float,
        y: float,
        z: float,
        w: float,
    ) -> float:
        sin_yaw = 2.0 * (
            w * z + x * y
        )

        cos_yaw = 1.0 - 2.0 * (
            y * y + z * z
        )

        return math.atan2(
            sin_yaw,
            cos_yaw,
        )


def main(args=None) -> None:
    rclpy.init(args=args)

    node = SawitSlamRandomNavigator()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info(
            "Stopped by user"
        )

    finally:
        node.clear_markers()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
