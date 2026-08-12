#!/usr/bin/env python3
"""
Sawit Random Navigator V21K - Three Layer Discovery
====================================================

Random-only exploration layered policy:
  Layer 3 m : detect there is an object ahead and approach slowly.
  Layer 2 m : stop and verify that the position/range is safe and stable.
  Layer 1 m : map-association radius, NOT physical flight distance.
              - no visited tree within 1 m -> mark/create tree as VISITED (green)
              - visited tree within 1 m    -> old tree, choose another random node

The drone never intentionally flies to 1 m from the obstacle. PointCloud tree
classification is performed while the drone is stationary near 2 m.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
import rclpy

from sawit_autonomy import sawit_navigator_fast as base


# Compatible with the V21 family while preferring the user's active V21H1 base.
_BASE_CLASS = None
for _name in (
    "SawitNavigatorV21H1",
    "SawitNavigatorV21J",
    "SawitNavigatorV21I",
    "SawitNavigatorV21H",
    "SawitNavigatorV21G",
    "SawitNavigatorV21F",
    "SawitNavigatorV21E",
    "SawitNavigatorV21C",
    "SawitNavigatorV21B",
):
    _candidate = getattr(base, _name, None)
    if _candidate is not None:
        _BASE_CLASS = _candidate
        break

if _BASE_CLASS is None:
    raise ImportError(
        "Base navigator V21 tidak ditemukan di sawit_navigator_fast.py. "
        "Pasang V21H1/V21J terlebih dahulu."
    )

NavState = base.NavState
TrackState = base.TrackState
TreeTrack = base.TreeTrack


class SawitRandomThreeLayerV21K(_BASE_CLASS):
    """Random-only navigator with 3 m / 2 m / 1 m layered decisions."""

    def __init__(self) -> None:
        super().__init__()

        self.declare_parameter("random_layer3_object_distance", 3.00)
        self.declare_parameter("random_layer2_stop_distance", 2.05)
        self.declare_parameter("random_layer2_hard_unsafe", 1.45)
        self.declare_parameter("random_layer2_safe_min", 1.60)
        self.declare_parameter("random_layer2_safe_max", 2.25)
        self.declare_parameter("random_layer2_hold_time", 1.30)
        self.declare_parameter("random_layer2_timeout", 3.00)
        self.declare_parameter("random_layer2_min_samples", 5)
        self.declare_parameter("random_layer2_max_mad", 0.32)
        self.declare_parameter("random_layer1_assoc_radius", 1.00)
        self.declare_parameter("random_layer1_min_cloud_frames", 3)
        self.declare_parameter("random_layer1_front_window_deg", 16.0)
        self.declare_parameter("random_layer1_range_tolerance", 1.30)
        self.declare_parameter("random_next_node_min_distance", 4.00)

        self.layer3_distance = float(
            self.get_parameter("random_layer3_object_distance").value
        )
        self.layer2_stop = float(
            self.get_parameter("random_layer2_stop_distance").value
        )
        self.layer2_hard_unsafe = float(
            self.get_parameter("random_layer2_hard_unsafe").value
        )
        self.layer2_safe_min = float(
            self.get_parameter("random_layer2_safe_min").value
        )
        self.layer2_safe_max = float(
            self.get_parameter("random_layer2_safe_max").value
        )
        self.layer2_hold_time = float(
            self.get_parameter("random_layer2_hold_time").value
        )
        self.layer2_timeout = float(
            self.get_parameter("random_layer2_timeout").value
        )
        self.layer2_min_samples = int(
            self.get_parameter("random_layer2_min_samples").value
        )
        self.layer2_max_mad = float(
            self.get_parameter("random_layer2_max_mad").value
        )
        self.layer1_assoc_radius = float(
            self.get_parameter("random_layer1_assoc_radius").value
        )
        self.layer1_min_cloud_frames = int(
            self.get_parameter("random_layer1_min_cloud_frames").value
        )
        self.layer1_front_window = math.radians(
            float(self.get_parameter("random_layer1_front_window_deg").value)
        )
        self.layer1_range_tolerance = float(
            self.get_parameter("random_layer1_range_tolerance").value
        )
        self.random_next_node_min_distance = float(
            self.get_parameter("random_next_node_min_distance").value
        )

        self._layer_stage = 0
        self._layer_hold_xy: Optional[Tuple[float, float]] = None
        self._layer_hold_yaw = 0.0
        self._layer_hold_started = 0.0
        self._layer_ranges: Deque[float] = deque(maxlen=20)
        self._layer_candidates: Deque[Tuple[int, float, float, bool, float]] = deque(
            maxlen=20
        )
        self._layer_last_cloud_seq = -1

        self.get_logger().info(
            "START V21K RANDOM THREE-LAYER "
            "random_only=1 layer3_object=3.00m layer2_safe_stop=2.00m "
            "layer1_map_assoc=1.00m new_tree=green old_tree=next_random_node "
            "physical_1m_approach=0 stationary_pointcloud_verify=1"
        )

    # ------------------------------------------------------------------
    # Random-only policy: never select a mapped tree as an approach target.
    # The tree is validated only when random motion encounters it.
    # ------------------------------------------------------------------
    def _select_target(self) -> bool:
        self.active_target_id = None
        self.active_standoff_goal = None
        self._log_throttle(
            "random_only_target_bypass_v21k",
            2.0,
            "info",
            "RANDOM_ONLY_V21K mapped_target_queue=bypassed action=random_explore",
        )
        return False

    def _reset_layer_state(self) -> None:
        self._layer_stage = 0
        self._layer_hold_xy = None
        self._layer_hold_yaw = 0.0
        self._layer_hold_started = 0.0
        self._layer_ranges.clear()
        self._layer_candidates.clear()
        self._layer_last_cloud_seq = -1
        self.motion_waypoint_xy = None
        self.motion_goal_xy = None
        self.motion_brake_anchor_xy = None

    def _front_statistics(self) -> Tuple[float, float, int]:
        if not self._layer_ranges:
            return math.inf, math.inf, 0
        arr = np.asarray(self._layer_ranges, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return math.inf, math.inf, 0
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        return median, mad, int(arr.size)

    def _nearest_track_any_state(
        self,
        x: float,
        y: float,
    ) -> Tuple[Optional[object], float]:
        nearest = None
        nearest_distance = math.inf
        for track in self.tracks.values():
            if track.state == TrackState.REJECTED:
                continue
            distance = math.hypot(track.x - x, track.y - y)
            if distance <= self.layer1_assoc_radius and distance < nearest_distance:
                nearest = track
                nearest_distance = distance
        return nearest, nearest_distance

    def _collect_stationary_front_candidate(self, front_median: float) -> None:
        if self.pose is None or self.last_cloud_msg is None:
            return
        if self.cloud_seq == self._layer_last_cloud_seq:
            return
        if time.monotonic() - self.last_cloud_receipt_mono > 0.65:
            return
        if self.pose.speed_xy > 0.18 or abs(self.pose.vz_up) > 0.25:
            return

        self._layer_last_cloud_seq = self.cloud_seq
        try:
            raw_points = self._pointcloud_to_xyz(self.last_cloud_msg)
            if raw_points.size == 0:
                return
            candidates, _filtered = self._extract_trunk_candidates(raw_points)
        except Exception as exc:
            self._log_throttle(
                "layer_cloud_error_v21k",
                1.0,
                "warning",
                f"RANDOM_LAYER1_CLOUD_ERROR_V21K type={type(exc).__name__} "
                f"error={exc}",
            )
            return

        valid = []
        for candidate in candidates:
            if abs(float(candidate.bearing)) > self.layer1_front_window:
                continue
            if not (1.20 <= float(candidate.range_m) <= 3.40):
                continue
            if math.isfinite(front_median) and (
                abs(float(candidate.range_m) - front_median)
                > self.layer1_range_tolerance
            ):
                continue
            valid.append(candidate)

        if not valid:
            return

        selected = min(
            valid,
            key=lambda candidate: (
                abs(float(candidate.bearing)),
                abs(float(candidate.range_m) - front_median)
                if math.isfinite(front_median)
                else float(candidate.range_m),
                -int(candidate.point_count),
            ),
        )

        map_x, map_y = self._body_to_map(
            float(selected.forward),
            float(selected.left),
            (self.pose.x_enu, self.pose.y_enu),
            self.pose.yaw_enu,
        )
        self._layer_candidates.append(
            (
                int(self.cloud_seq),
                float(map_x),
                float(map_y),
                bool(selected.strong),
                float(selected.score),
            )
        )

    def _robust_candidate_center(
        self,
    ) -> Tuple[Optional[Tuple[float, float]], int, int, float, float]:
        if not self._layer_candidates:
            return None, 0, 0, math.inf, 0.0

        # Keep one observation per cloud frame.
        by_frame = {}
        for frame_id, x, y, strong, score in self._layer_candidates:
            by_frame[int(frame_id)] = (float(x), float(y), bool(strong), float(score))

        values = list(by_frame.values())
        if not values:
            return None, 0, 0, math.inf, 0.0

        xy = np.asarray([(v[0], v[1]) for v in values], dtype=np.float64)
        center = np.median(xy, axis=0)
        residual = np.linalg.norm(xy - center, axis=1)
        mad = float(np.median(residual)) if residual.size else math.inf
        strong_hits = sum(1 for v in values if v[2])
        mean_score = float(np.mean([v[3] for v in values]))
        return (
            (float(center[0]), float(center[1])),
            len(values),
            strong_hits,
            mad,
            mean_score,
        )

    def _mark_new_or_existing_tree_green(
        self,
        center: Tuple[float, float],
        frame_count: int,
        strong_hits: int,
        position_mad: float,
        mean_score: float,
        range_samples: int,
    ) -> None:
        now = time.monotonic()
        map_x, map_y = center
        nearest, nearest_distance = self._nearest_track_any_state(map_x, map_y)

        if nearest is not None and nearest.state == TrackState.VISITED:
            self.get_logger().info(
                f"RANDOM_LAYER1_OLD_TREE_V21K id={nearest.tree_id} "
                f"distance={nearest_distance:.2f}m radius={self.layer1_assoc_radius:.2f}m "
                "marker=already_green action=next_random_node"
            )
            self._choose_next_random_node(map_x, map_y, "old_tree")
            return

        if nearest is not None:
            # Existing yellow/orange candidate becomes visited green, but no new ID.
            nearest.x = float(map_x)
            nearest.y = float(map_y)
            nearest.state = TrackState.VISITED
            nearest.hits = max(int(nearest.hits), frame_count)
            nearest.strong_hits = max(int(nearest.strong_hits), strong_hits)
            nearest.tof_hits = max(int(nearest.tof_hits), range_samples)
            nearest.updated_mono = now
            nearest.last_score = max(float(nearest.last_score), mean_score)
            nearest.position_mad = float(position_mad)
            for _frame_id, x, y, _strong, _score in self._layer_candidates:
                nearest.observations.append((float(x), float(y)))
            self._save_memory()
            self.get_logger().info(
                f"RANDOM_LAYER1_NEW_TREE_GREEN_V21K id={nearest.tree_id} "
                f"mode=promote_existing_candidate map=({map_x:.2f},{map_y:.2f}) "
                f"assoc={nearest_distance:.2f}m frames={frame_count} "
                f"strong={strong_hits} mad={position_mad:.2f} marker=green "
                "action=next_random_node"
            )
        else:
            tree_id = int(self.next_tree_id)
            self.next_tree_id += 1
            track = TreeTrack(
                tree_id=tree_id,
                x=float(map_x),
                y=float(map_y),
                state=TrackState.VISITED,
            )
            track.hits = int(frame_count)
            track.strong_hits = int(strong_hits)
            track.tof_hits = int(range_samples)
            track.created_mono = now
            track.updated_mono = now
            track.last_score = float(mean_score)
            track.position_mad = float(position_mad)
            for _frame_id, x, y, _strong, _score in self._layer_candidates:
                track.observations.append((float(x), float(y)))
            self.tracks[tree_id] = track
            self._save_memory()
            self.get_logger().info(
                f"RANDOM_LAYER1_NEW_TREE_GREEN_V21K id={tree_id} "
                f"mode=create_new map=({map_x:.2f},{map_y:.2f}) "
                f"assoc_radius={self.layer1_assoc_radius:.2f}m "
                f"frames={frame_count} strong={strong_hits} "
                f"mad={position_mad:.2f} marker=green action=next_random_node"
            )

        if self._visited_count() >= self.target_tree_count:
            self._reset_layer_state()
            self._set_state(NavState.COMPLETE)
        else:
            self._choose_next_random_node(map_x, map_y, "new_tree_green")

    def _choose_next_random_node(
        self,
        obstacle_x: float,
        obstacle_y: float,
        reason: str,
    ) -> None:
        min_x = self.orchard_min_x + self.explore_margin
        max_x = self.orchard_max_x - self.explore_margin
        min_y = self.orchard_min_y + self.explore_margin
        max_y = self.orchard_max_y - self.explore_margin

        selected = None
        for _ in range(50):
            x = self.random.uniform(min_x, max_x)
            y = self.random.uniform(min_y, max_y)
            if math.hypot(x - obstacle_x, y - obstacle_y) < self.random_next_node_min_distance:
                continue
            if self.pose is not None and math.hypot(
                x - self.pose.x_enu,
                y - self.pose.y_enu,
            ) < 2.5:
                continue
            selected = (x, y)
            break

        if selected is None:
            selected = (
                self.random.uniform(min_x, max_x),
                self.random.uniform(min_y, max_y),
            )

        self.explore_goal = selected
        self.explore_count += 1
        self._reset_layer_state()
        self.get_logger().info(
            f"RANDOM_NEXT_NODE_V21K reason={reason} count={self.explore_count} "
            f"goal=({selected[0]:.2f},{selected[1]:.2f})"
        )
        self._set_state(NavState.EXPLORE_ALIGN)

    def _run_random_three_layer(self) -> None:
        assert self.pose is not None
        now = time.monotonic()

        if self.explore_goal is None:
            self._choose_explore_goal()
            self._reset_layer_state()
            self._set_state(NavState.EXPLORE_ALIGN)
            return

        goal_distance = math.hypot(
            self.explore_goal[0] - self.pose.x_enu,
            self.explore_goal[1] - self.pose.y_enu,
        )
        desired_yaw = math.atan2(
            self.explore_goal[1] - self.pose.y_enu,
            self.explore_goal[0] - self.pose.x_enu,
        )

        if goal_distance <= self.explore_arrival_distance:
            self.get_logger().info(
                f"RANDOM_NODE_REACHED_V21K goal=({self.explore_goal[0]:.2f},"
                f"{self.explore_goal[1]:.2f}) action=stationary_scan"
            )
            self._reset_layer_state()
            self._start_scan("random_node_reached_v21k")
            return

        if not self._horizontal_motion_ready(desired_yaw):
            return

        front_range, front_source, front_tof, cloud_front = (
            self._combined_front_range()
        )

        # No object inside 3 m: ordinary random motion.
        if not math.isfinite(front_range) or front_range > self.layer3_distance:
            if self._layer_stage != 0:
                self._reset_layer_state()
            self._command_toward(
                self.explore_goal[0],
                self.explore_goal[1],
                desired_yaw,
            )
            self._log_throttle(
                "random_clear_v21k",
                0.8,
                "info",
                f"RANDOM_CLEAR_MOVE_V21K goal_dist={goal_distance:.2f} "
                f"front={front_range if math.isfinite(front_range) else float('inf'):.2f} "
                f"source={front_source}",
            )
            return

        # Layer 3 m: object exists, keep controlled approach only until 2 m.
        if front_range > self.layer2_stop:
            if self._layer_stage < 1:
                self._layer_stage = 1
                self.get_logger().info(
                    f"RANDOM_LAYER3_OBJECT_V21K range={front_range:.2f} "
                    f"source={front_source} action=slow_approach_to_2m"
                )
            self._command_toward(
                self.explore_goal[0],
                self.explore_goal[1],
                desired_yaw,
            )
            self._log_throttle(
                "random_layer3_progress_v21k",
                0.6,
                "info",
                f"RANDOM_LAYER3_PROGRESS_V21K range={front_range:.2f} "
                f"tof={front_tof if math.isfinite(front_tof) else float('inf'):.2f} "
                f"cloud={cloud_front if math.isfinite(cloud_front) else float('inf'):.2f}",
            )
            return

        # Layer 2 m: fixed hold. The anchor must not follow the drifting pose.
        if self._layer_stage < 2 or self._layer_hold_xy is None:
            self._layer_stage = 2
            self._layer_hold_xy = (self.pose.x_enu, self.pose.y_enu)
            self._layer_hold_yaw = self.pose.yaw_enu
            self._layer_hold_started = now
            self._layer_ranges.clear()
            self._layer_candidates.clear()
            self._layer_last_cloud_seq = -1
            self.motion_waypoint_xy = None
            self.motion_goal_xy = None
            self.motion_brake_anchor_xy = None
            self.get_logger().warning(
                f"RANDOM_LAYER2_STOP_V21K range={front_range:.2f} "
                f"source={front_source} hold=({self._layer_hold_xy[0]:.2f},"
                f"{self._layer_hold_xy[1]:.2f}) action=safety_check"
            )

        self._publish_position_enu(
            self._layer_hold_xy[0],
            self._layer_hold_xy[1],
            self.flight_altitude,
            self._layer_hold_yaw,
        )

        if math.isfinite(front_range):
            self._layer_ranges.append(float(front_range))

        median, mad, sample_count = self._front_statistics()
        self._collect_stationary_front_candidate(median)
        center, frame_count, strong_hits, position_mad, mean_score = (
            self._robust_candidate_center()
        )

        hold_elapsed = now - self._layer_hold_started
        stationary = self.pose.speed_xy <= 0.18 and abs(self.pose.vz_up) <= 0.25

        if math.isfinite(front_range) and front_range < self.layer2_hard_unsafe:
            self.get_logger().warning(
                f"RANDOM_LAYER2_UNSAFE_V21K reason=too_close "
                f"range={front_range:.2f} limit={self.layer2_hard_unsafe:.2f} "
                "action=next_random_node"
            )
            self._choose_next_random_node(
                self.pose.x_enu,
                self.pose.y_enu,
                "layer2_too_close",
            )
            return

        range_safe = (
            sample_count >= self.layer2_min_samples
            and self.layer2_safe_min <= median <= self.layer2_safe_max
            and mad <= self.layer2_max_mad
        )

        if hold_elapsed >= self.layer2_hold_time and stationary and range_safe:
            self._log_throttle(
                "random_layer2_safe_v21k",
                0.7,
                "info",
                f"RANDOM_LAYER2_SAFE_V21K median={median:.2f} mad={mad:.2f} "
                f"samples={sample_count} stationary=1 "
                f"tree_frames={frame_count}/{self.layer1_min_cloud_frames} "
                "action=layer1_map_check",
            )

            if frame_count >= self.layer1_min_cloud_frames and center is not None:
                self._mark_new_or_existing_tree_green(
                    center=center,
                    frame_count=frame_count,
                    strong_hits=strong_hits,
                    position_mad=position_mad,
                    mean_score=mean_score,
                    range_samples=sample_count,
                )
                return

        if hold_elapsed >= self.layer2_timeout:
            if not range_safe:
                self.get_logger().warning(
                    f"RANDOM_LAYER2_UNSAFE_V21K reason=unstable_or_not_2m "
                    f"median={median:.2f} mad={mad:.2f} samples={sample_count} "
                    f"stationary={int(stationary)} action=next_random_node"
                )
                reason = "layer2_unstable"
            else:
                self.get_logger().warning(
                    f"RANDOM_LAYER1_NOT_TREE_V21K reason=no_stable_trunk "
                    f"frames={frame_count}/{self.layer1_min_cloud_frames} "
                    f"range={median:.2f} action=next_random_node"
                )
                reason = "layer1_not_tree"
            self._choose_next_random_node(
                self.pose.x_enu,
                self.pose.y_enu,
                reason,
            )
            return

        self._log_throttle(
            "random_layer2_wait_v21k",
            0.5,
            "info",
            f"RANDOM_LAYER2_CHECK_V21K elapsed={hold_elapsed:.2f}/"
            f"{self.layer2_timeout:.2f}s median={median:.2f} mad={mad:.2f} "
            f"samples={sample_count} stationary={int(stationary)} "
            f"tree_frames={frame_count}",
        )

    def _control_loop(self) -> None:
        # Let the trusted V21 base handle all states except random movement.
        if self.state != NavState.EXPLORE_MOVE:
            super()._control_loop()
            return

        now = time.monotonic()
        self._publish_offboard_mode()

        if self.pose is None:
            return

        if now < self.pose_fault_until:
            self._publish_position_enu(
                self.pose.x_enu,
                self.pose.y_enu,
                self.flight_altitude,
                self.pose.yaw_enu,
            )
            self._log_throttle(
                "pose_fault_hold_v21k",
                0.6,
                "error",
                f"POSE_FAULT_HOLD_V21K remaining={self.pose_fault_until - now:.2f}s",
            )
            return

        if now - self.pose.receipt_mono > 0.70:
            self._publish_position_enu(
                self.pose.x_enu,
                self.pose.y_enu,
                self.flight_altitude,
                self.pose.yaw_enu,
            )
            self._log_throttle(
                "pose_stale_hold_v21k",
                0.7,
                "error",
                f"POSE_STALE_HOLD_V21K age={now - self.pose.receipt_mono:.2f}s",
            )
            return

        self._run_random_three_layer()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SawitRandomThreeLayerV21K()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
