#!/usr/bin/env python3
import math
import time
from enum import Enum
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker, MarkerArray

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
)


class State(Enum):
    WAIT_DATA = 0
    PRESTREAM = 1
    TAKEOFF = 2
    SEARCH = 3
    APPROACH = 4
    HOLD_SCAN = 5
    BACKUP = 6
    FINISHED = 7


class ReactiveNavigatorBase(Node):
    """
    Base controller for Gazebo/PX4 SITL.

    Important:
    - Detection is local/reactive.
    - PX4 local position is used for flight control and route logging.
    - This is not SLAM and does not create a globally consistent object map.
    """

    def __init__(self, node_name: str):
        super().__init__(node_name)

        # ---------------- User-tunable parameters ----------------
        self.declare_parameter("flight_altitude_ned", -1.8)
        self.declare_parameter("maximum_visits", 16)
        self.declare_parameter("visit_distance", 3.0)
        self.declare_parameter("danger_distance", 1.8)
        self.declare_parameter("forward_step", 0.22)
        self.declare_parameter("backward_step", 0.16)
        self.declare_parameter("search_forward_step", 0.60)
        self.declare_parameter("search_forward_every", 4)
        self.declare_parameter("search_yaw_step_deg", 22.5)
        self.declare_parameter("maximum_yaw_step_deg", 5.0)
        self.declare_parameter("scan_hold_seconds", 1.5)
        self.declare_parameter("backup_seconds", 1.2)
        self.declare_parameter("target_timeout_seconds", 2.0)
        self.declare_parameter("maximum_home_radius", 35.0)
        self.declare_parameter(
            "px4_position_topic",
            "/fmu/out/vehicle_local_position_v1",
        )

        self.flight_altitude = float(self.get_parameter("flight_altitude_ned").value)
        self.maximum_visits = int(self.get_parameter("maximum_visits").value)
        self.visit_distance = float(self.get_parameter("visit_distance").value)
        self.danger_distance = float(self.get_parameter("danger_distance").value)
        self.forward_step = float(self.get_parameter("forward_step").value)
        self.backward_step = float(self.get_parameter("backward_step").value)
        self.search_forward_step = float(self.get_parameter("search_forward_step").value)
        self.search_forward_every = int(self.get_parameter("search_forward_every").value)
        self.search_yaw_step = math.radians(
            float(self.get_parameter("search_yaw_step_deg").value)
        )
        self.maximum_yaw_step = math.radians(
            float(self.get_parameter("maximum_yaw_step_deg").value)
        )
        self.scan_hold_seconds = float(self.get_parameter("scan_hold_seconds").value)
        self.backup_seconds = float(self.get_parameter("backup_seconds").value)
        self.target_timeout_seconds = float(
            self.get_parameter("target_timeout_seconds").value
        )
        self.maximum_home_radius = float(
            self.get_parameter("maximum_home_radius").value
        )
        self.px4_position_topic = str(
            self.get_parameter("px4_position_topic").value
        )

        # ---------------- Runtime ----------------
        self.state = State.WAIT_DATA
        self.position = np.array([np.nan, np.nan, np.nan], dtype=float)
        self.heading = 0.0
        self.position_valid = False
        self.home_xy: Optional[np.ndarray] = None
        self.takeoff_target: Optional[np.ndarray] = None

        self.target_forward = math.nan
        self.target_left = math.nan
        self.target_distance = math.nan
        self.target_stamp = 0.0
        self.target_valid = False

        self.prestream_count = 0
        self.search_count = 0
        self.search_yaw_target = 0.0
        self.last_search_action = 0.0
        self.scan_started = 0.0
        self.backup_started = 0.0
        self.visited_count = 0
        self.land_sent = False

        self.route_points = []
        self.last_route_point: Optional[np.ndarray] = None
        self.visited_positions = []

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(
            VehicleLocalPosition,
            self.px4_position_topic,
            self._position_callback,
            sensor_qos,
        )

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

        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/sawit/tree_markers",
            10,
        )
        self.route_pub = self.create_publisher(
            Marker,
            "/sawit/route_marker",
            10,
        )

        # Provide a simple map frame for non-SLAM visualization.
        self.static_tf = StaticTransformBroadcaster(self)
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = "map"
        tf.child_frame_id = "px4_local"
        tf.transform.rotation.w = 1.0
        self.static_tf.sendTransform(tf)

        self.create_timer(0.1, self._control_loop)
        self.get_logger().info(f"{node_name} started (non-SLAM reactive mode)")

    # Child node must call this whenever it has a detection.
    def update_detection(self, forward_m: float, left_m: float) -> None:
        if not (math.isfinite(forward_m) and math.isfinite(left_m)):
            return
        distance = math.hypot(forward_m, left_m)
        if distance <= 0.0:
            return

        self.target_forward = float(forward_m)
        self.target_left = float(left_m)
        self.target_distance = float(distance)
        self.target_stamp = time.monotonic()
        self.target_valid = True

    def clear_detection(self) -> None:
        self.target_valid = False

    def _position_callback(self, msg: VehicleLocalPosition) -> None:
        values = np.array([msg.x, msg.y, msg.z], dtype=float)
        if not np.all(np.isfinite(values)):
            return
        if hasattr(msg, "xy_valid") and not msg.xy_valid:
            return
        if hasattr(msg, "z_valid") and not msg.z_valid:
            return

        self.position = values
        if math.isfinite(msg.heading):
            self.heading = self._normalize(float(msg.heading))
        self.position_valid = True

        if self.home_xy is None:
            self.home_xy = values[:2].copy()
            self.takeoff_target = np.array(
                [values[0], values[1], self.flight_altitude],
                dtype=float,
            )
            self.search_yaw_target = self.heading

        # Route is shown in ENU-like RViz coordinates:
        # map x = PX4 NED x, map y = -PX4 NED y, map z = -PX4 NED z.
        enu = np.array([values[0], -values[1], -values[2]], dtype=float)
        if self.last_route_point is None or np.linalg.norm(
            enu[:2] - self.last_route_point[:2]
        ) >= 0.20:
            self.route_points.append(enu)
            self.last_route_point = enu
            self.route_points = self.route_points[-4000:]

    def _timestamp(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def _publish_heartbeat(self) -> None:
        msg = OffboardControlMode()
        msg.timestamp = self._timestamp()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_pub.publish(msg)

    def _publish_setpoint(self, target: np.ndarray, yaw: float) -> None:
        if not self.position_valid:
            return

        target = np.asarray(target, dtype=float).copy()
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            self.get_logger().error("Rejected invalid trajectory target")
            return

        # Limit every control step to prevent sudden jumps.
        delta = target[:2] - self.position[:2]
        norm = float(np.linalg.norm(delta))
        if norm > 0.70:
            target[:2] = self.position[:2] + delta / norm * 0.70

        target[2] = self.flight_altitude

        if self.home_xy is not None:
            radius = float(np.linalg.norm(target[:2] - self.home_xy))
            if radius > self.maximum_home_radius:
                self.get_logger().error("Home boundary reached; landing")
                self.state = State.FINISHED
                return

        msg = TrajectorySetpoint()
        msg.timestamp = self._timestamp()
        msg.position = [float(target[0]), float(target[1]), float(target[2])]
        msg.velocity = [math.nan, math.nan, math.nan]
        msg.acceleration = [math.nan, math.nan, math.nan]
        msg.jerk = [math.nan, math.nan, math.nan]
        msg.yaw = float(self._normalize(yaw))
        msg.yawspeed = math.nan
        self.setpoint_pub.publish(msg)

    def _vehicle_command(
        self,
        command: int,
        param1: float = 0.0,
        param2: float = 0.0,
    ) -> None:
        msg = VehicleCommand()
        msg.timestamp = self._timestamp()
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.command = int(command)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.command_pub.publish(msg)

    def _hold(self, yaw: Optional[float] = None) -> None:
        self._publish_setpoint(
            np.array(
                [self.position[0], self.position[1], self.flight_altitude],
                dtype=float,
            ),
            self.heading if yaw is None else yaw,
        )

    def _body_step(self, distance: float) -> np.ndarray:
        return np.array(
            [
                self.position[0] + distance * math.cos(self.heading),
                self.position[1] + distance * math.sin(self.heading),
                self.flight_altitude,
            ],
            dtype=float,
        )

    def _target_is_fresh(self) -> bool:
        return self.target_valid and (
            time.monotonic() - self.target_stamp <= self.target_timeout_seconds
        )

    def _control_loop(self) -> None:
        self._publish_heartbeat()
        self._publish_visualization()

        if self.state == State.WAIT_DATA:
            if self.position_valid and self.takeoff_target is not None:
                self.state = State.PRESTREAM
                self.get_logger().info("STATE -> PRESTREAM")
            return

        if self.state == State.PRESTREAM:
            self._publish_setpoint(self.takeoff_target, self.heading)
            self.prestream_count += 1
            if self.prestream_count == 20:
                self._vehicle_command(
                    VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0
                )
                self._vehicle_command(
                    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0
                )
                self.get_logger().info("OFFBOARD + ARM sent")
            if self.prestream_count >= 35:
                self.state = State.TAKEOFF
                self.get_logger().info("STATE -> TAKEOFF")
            return

        if self.state == State.TAKEOFF:
            self._publish_setpoint(self.takeoff_target, self.heading)
            if abs(self.position[2] - self.flight_altitude) <= 0.25:
                self.state = State.SEARCH
                self.last_search_action = time.monotonic()
                self.search_yaw_target = self.heading
                self.get_logger().info("TAKEOFF complete; STATE -> SEARCH")
            return

        if self.state == State.SEARCH:
            if self.visited_count >= self.maximum_visits:
                self.state = State.FINISHED
                return

            if self._target_is_fresh():
                self.state = State.APPROACH
                self.get_logger().info(
                    f"Target acquired: distance={self.target_distance:.2f} m"
                )
                return

            self._hold(self.search_yaw_target)
            now = time.monotonic()
            if now - self.last_search_action < 1.25:
                return

            self.last_search_action = now
            self.search_count += 1
            if self.search_count % self.search_forward_every == 0:
                self._publish_setpoint(
                    self._body_step(self.search_forward_step),
                    self.heading,
                )
                self.get_logger().info("SEARCH: forward exploration step")
            else:
                direction = -1.0 if self.search_count % 2 == 0 else 1.0
                self.search_yaw_target = self._normalize(
                    self.search_yaw_target + direction * self.search_yaw_step
                )
                self.get_logger().info("SEARCH: rotate")
            return

        if self.state == State.APPROACH:
            if not self._target_is_fresh():
                self._hold()
                self.state = State.SEARCH
                self.get_logger().warn("Target lost; STATE -> SEARCH")
                return

            if self.target_distance <= self.visit_distance:
                self.scan_started = time.monotonic()
                self.state = State.HOLD_SCAN
                self.get_logger().info("Target reached; STATE -> HOLD_SCAN")
                return

            if self.target_distance <= self.danger_distance:
                self.scan_started = time.monotonic()
                self.state = State.HOLD_SCAN
                return

            bearing_left = math.atan2(
                self.target_left,
                max(0.05, self.target_forward),
            )
            # Positive camera-left means counter-clockwise in camera view.
            # PX4 NED heading increases clockwise, so subtract the bearing.
            correction = max(
                -self.maximum_yaw_step,
                min(self.maximum_yaw_step, -bearing_left),
            )
            target_yaw = self._normalize(self.heading + correction)

            centered = abs(self.target_left) <= 0.65 and self.target_forward > 0.0
            if centered:
                self._publish_setpoint(
                    self._body_step(self.forward_step),
                    target_yaw,
                )
            else:
                self._hold(target_yaw)
            return

        if self.state == State.HOLD_SCAN:
            self._hold()
            if time.monotonic() - self.scan_started >= self.scan_hold_seconds:
                self.visited_count += 1
                self.visited_positions.append(
                    np.array(
                        [self.position[0], -self.position[1], -self.position[2]],
                        dtype=float,
                    )
                )
                self.backup_started = time.monotonic()
                self.clear_detection()
                self.state = State.BACKUP
                self.get_logger().info(
                    f"VISITED {self.visited_count}/{self.maximum_visits}"
                )
            return

        if self.state == State.BACKUP:
            if time.monotonic() - self.backup_started < self.backup_seconds:
                self._publish_setpoint(
                    self._body_step(-self.backward_step),
                    self.heading,
                )
            else:
                self.search_yaw_target = self._normalize(
                    self.heading + self.search_yaw_step
                )
                self.last_search_action = time.monotonic()
                self.state = State.SEARCH
            return

        if self.state == State.FINISHED:
            if not self.land_sent:
                self.land_sent = True
                self._vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.get_logger().info("MISSION FINISHED; LAND sent")

    def _publish_visualization(self) -> None:
        now = self.get_clock().now().to_msg()

        route = Marker()
        route.header.frame_id = "map"
        route.header.stamp = now
        route.ns = "route"
        route.id = 1
        route.type = Marker.LINE_STRIP
        route.action = Marker.ADD
        route.pose.orientation.w = 1.0
        route.scale.x = 0.10
        route.color.b = 1.0
        route.color.a = 1.0
        for xyz in self.route_points:
            p = Point()
            p.x, p.y, p.z = map(float, xyz)
            route.points.append(p)
        self.route_pub.publish(route)

        array = MarkerArray()
        for index, xyz in enumerate(self.visited_positions):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = now
            marker.ns = "visited"
            marker.id = index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(xyz[0])
            marker.pose.position.y = float(xyz[1])
            marker.pose.position.z = float(max(0.2, xyz[2]))
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.75
            marker.color.g = 1.0
            marker.color.a = 1.0
            array.markers.append(marker)

        if self._target_is_fresh() and self.position_valid:
            # Approximate current target position only for display.
            target_ned_x = self.position[0] + (
                self.target_forward * math.cos(self.heading)
                - self.target_left * math.sin(self.heading)
            )
            target_ned_y = self.position[1] + (
                self.target_forward * math.sin(self.heading)
                + self.target_left * math.cos(self.heading)
            )
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = now
            marker.ns = "active_target"
            marker.id = 50000
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(target_ned_x)
            marker.pose.position.y = float(-target_ned_y)
            marker.pose.position.z = float(-self.position[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.85
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.a = 1.0
            array.markers.append(marker)

        self.marker_pub.publish(array)

    @staticmethod
    def _normalize(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))
