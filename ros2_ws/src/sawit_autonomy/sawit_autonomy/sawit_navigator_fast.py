#!/usr/bin/env python3
"""
Sawit Navigator V21N
===================

ROS 2 Humble + PX4 SITL + Gazebo.

Tujuan utama:
- PointCloud hanya dipetakan ketika drone benar-benar diam.
- Setiap sektor: TURN -> SETTLE -> FLUSH frame lama -> COLLECT -> POST HOLD.
- Selama COLLECT, transform memakai pose/yaw anchor yang dibekukan.
- Kandidat satu pohon digabung dengan association radius.
- Kandidat satu sudut tidak langsung menjadi landmark confirmed.
- Kandidat provisional boleh dikunjungi untuk verifikasi dekat agar eksplorasi tidak mandek.
- Titik di luar batas kebun ditolak setelah transform.
- Terbang rendah stabil (default 1.20 m); marker RViz memakai tinggi aktual.
- Pergerakan memakai micro-waypoint terkunci, governor kecepatan, dan yaw slew agar drone tidak goyang.
- ToF dan PointCloud koridor sempit dipakai sebagai rem keselamatan, bukan sebagai pembentuk landmark.
- Actual/SDF tidak dipakai sebagai target atau gate.

Frame internal:
- PX4 VehicleLocalPosition: NED.
- Peta internal/RViz: ENU relatif terhadap posisi home.
- PointCloud default: x=forward, y=left, z=up.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)

try:
    from sawit_autonomy.actual_tree_positions import get_actual_tree_positions_gazebo
except Exception:
    get_actual_tree_positions_gazebo = None


# =============================================================================
# Utilitas matematika
# =============================================================================

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def angle_diff(target: float, current: float) -> float:
    return wrap_pi(target - current)


def finite(*values: float) -> bool:
    return all(math.isfinite(float(v)) for v in values)


def robust_median(values: Sequence[float], default: float = 0.0) -> float:
    if not values:
        return float(default)
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.median(arr))


def robust_xy(points: Sequence[Tuple[float, float]]) -> Optional[Tuple[float, float, float]]:
    if not points:
        return None
    arr = np.asarray(points, dtype=np.float64)
    good = np.isfinite(arr).all(axis=1)
    arr = arr[good]
    if arr.shape[0] == 0:
        return None
    center = np.median(arr, axis=0)
    residual = np.linalg.norm(arr - center, axis=1)
    mad = float(np.median(residual))
    return float(center[0]), float(center[1]), mad


def circular_span(angles: Sequence[float]) -> float:
    """Span maksimum pasangan sudut dalam radian, dibatasi pi."""
    if len(angles) < 2:
        return 0.0
    result = 0.0
    for i in range(len(angles)):
        for j in range(i + 1, len(angles)):
            result = max(result, abs(angle_diff(angles[i], angles[j])))
    return min(math.pi, result)


# =============================================================================
# Data model
# =============================================================================

class NavState(str, Enum):
    WAIT_DATA = "WAIT_DATA"
    PRESTREAM = "PRESTREAM"
    TAKEOFF = "TAKEOFF"
    STABILIZE = "STABILIZE"

    SCAN_TURN = "SCAN_TURN"
    SCAN_SETTLE = "SCAN_SETTLE"
    SCAN_FLUSH = "SCAN_FLUSH"
    SCAN_COLLECT = "SCAN_COLLECT"
    SCAN_POST = "SCAN_POST"

    SELECT_TARGET = "SELECT_TARGET"
    ALIGN_TARGET = "ALIGN_TARGET"
    APPROACH = "APPROACH"
    AVOID_OBSTACLE = "AVOID_OBSTACLE"
    RETRY_VERIFY = "RETRY_VERIFY"
    REACQUIRE_FINAL = "REACQUIRE_FINAL"
    RETREAT_VISITED = "RETREAT_VISITED"
    CLOSE_SETTLE = "CLOSE_SETTLE"
    CLOSE_FLUSH = "CLOSE_FLUSH"
    CLOSE_COLLECT = "CLOSE_COLLECT"

    EXPLORE_ALIGN = "EXPLORE_ALIGN"
    EXPLORE_MOVE = "EXPLORE_MOVE"

    BRAKE_HOLD = "BRAKE_HOLD"
    HOLD = "HOLD"
    COMPLETE = "COMPLETE"


class TrackState(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    VISITED = "visited"
    REJECTED = "rejected"


@dataclass
class PoseSample:
    receipt_mono: float
    x_enu: float
    y_enu: float
    altitude: float
    yaw_enu: float
    vx_enu: float
    vy_enu: float
    vz_up: float

    @property
    def speed_xy(self) -> float:
        return math.hypot(self.vx_enu, self.vy_enu)


@dataclass
class TrunkCandidate:
    forward: float
    left: float
    z_center: float
    range_m: float
    bearing: float
    point_count: int
    height: float
    radius: float
    occupied_bins: int
    axis_wander: float
    score: float
    strong: bool
    tof_range: float = math.inf
    tof_support: bool = False


@dataclass
class ScanAccumulator:
    positions: List[Tuple[float, float]] = field(default_factory=list)
    frame_ids: Set[int] = field(default_factory=set)
    sectors: Set[int] = field(default_factory=set)
    view_yaws: List[float] = field(default_factory=list)
    strong_hits: int = 0
    tof_hits: int = 0
    scores: List[float] = field(default_factory=list)
    ranges: List[float] = field(default_factory=list)


@dataclass
class TreeTrack:
    tree_id: int
    x: float
    y: float
    state: TrackState = TrackState.TENTATIVE

    hits: int = 0
    strong_hits: int = 0
    tof_hits: int = 0
    sectors: Set[int] = field(default_factory=set)
    view_yaws: List[float] = field(default_factory=list)
    observations: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=40)
    )

    created_mono: float = 0.0
    updated_mono: float = 0.0
    last_target_mono: float = 0.0
    cooldown_until: float = 0.0
    verify_failures: int = 0
    last_score: float = 0.0
    position_mad: float = math.inf

    def targetable(self) -> bool:
        return self.state in (TrackState.TENTATIVE, TrackState.CONFIRMED)

    def rank(self) -> Tuple[int, int, int, float]:
        state_rank = {
            TrackState.REJECTED: 0,
            TrackState.TENTATIVE: 1,
            TrackState.CONFIRMED: 2,
            TrackState.VISITED: 3,
        }[self.state]
        return state_rank, self.hits, self.strong_hits, self.last_score


# =============================================================================
# Node utama
# =============================================================================

class SawitNavigatorV21H(Node):
    def __init__(self) -> None:
        super().__init__("sawit_navigator_fast_v21h")

        # ---------------------------------------------------------------------
        # Parameter topic
        # ---------------------------------------------------------------------
        self.declare_parameter(
            "local_position_topic",
            "/fmu/out/vehicle_local_position_v1",
        )
        self.declare_parameter(
            "vehicle_status_topic",
            "/fmu/out/vehicle_status_v4",
        )
        self.declare_parameter(
            "offboard_control_topic",
            "/fmu/in/offboard_control_mode",
        )
        self.declare_parameter(
            "trajectory_setpoint_topic",
            "/fmu/in/trajectory_setpoint",
        )
        self.declare_parameter(
            "vehicle_command_topic",
            "/fmu/in/vehicle_command",
        )
        self.declare_parameter("cloud_topic", "/camera/points")
        self.declare_parameter("tof_topic", "/tof_front")

        # ---------------------------------------------------------------------
        # Parameter misi
        # ---------------------------------------------------------------------
        self.declare_parameter("target_tree_count", 16)
        self.declare_parameter("flight_altitude", 0.72)
        self.declare_parameter("reset_memory_on_start", True)
        # Visualisasi RViz memakai koordinat dunia Gazebo ENU.
        # Navigasi tetap memakai ENU relatif home; offset ini VISUAL ONLY.
        self.declare_parameter("visual_spawn_x", -25.0)
        self.declare_parameter("visual_spawn_y", 0.0)
        self.declare_parameter("publish_actual_visual", True)
        self.declare_parameter(
            "memory_path",
            str(
                Path.home()
                / "ros2_ws/src/sawit_autonomy/data/sawit_tree_memory_v21h.json"
            ),
        )

        # ---------------------------------------------------------------------
        # Batas kebun relatif terhadap home, frame ENU
        # ---------------------------------------------------------------------
        self.declare_parameter("orchard_min_x", 9.5)
        self.declare_parameter("orchard_max_x", 40.5)
        self.declare_parameter("orchard_min_y", -15.0)
        self.declare_parameter("orchard_max_y", 15.0)
        self.declare_parameter("orchard_gate_margin", 0.75)

        # ---------------------------------------------------------------------
        # PointCloud dan geometri batang
        # ---------------------------------------------------------------------
        self.declare_parameter(
            "cloud_axes_mode",
            "x_forward_y_left_z_up",
        )
        self.declare_parameter("cloud_left_sign", 1.0)
        self.declare_parameter("cloud_point_stride", 4)
        self.declare_parameter("cloud_min_forward", 1.4)
        self.declare_parameter("cloud_max_forward", 24.0)
        self.declare_parameter("cloud_max_abs_left", 12.0)
        self.declare_parameter("cloud_min_z", -0.95)
        self.declare_parameter("cloud_max_z", 2.60)

        self.declare_parameter("voxel_xy", 0.28)
        self.declare_parameter("seed_min_points", 7)
        self.declare_parameter("cluster_radius", 0.68)
        self.declare_parameter("cluster_min_points", 28)
        self.declare_parameter("trunk_min_height", 0.80)
        self.declare_parameter("trunk_min_radius", 0.05)
        self.declare_parameter("trunk_max_radius", 0.62)
        self.declare_parameter("trunk_min_vertical_bins", 4)
        self.declare_parameter("trunk_max_axis_wander", 0.42)
        self.declare_parameter("candidate_nms_radius", 1.05)

        # ---------------------------------------------------------------------
        # Association dan konsensus
        # ---------------------------------------------------------------------
        self.declare_parameter("scan_assoc_radius", 1.35)
        self.declare_parameter("tree_assoc_radius", 1.60)
        self.declare_parameter("tree_merge_radius", 1.65)
        self.declare_parameter("new_track_min_frames", 3)
        self.declare_parameter("new_track_min_strong", 2)
        self.declare_parameter("confirm_min_total_hits", 5)
        self.declare_parameter("confirm_min_view_span_deg", 20.0)
        self.declare_parameter("confirm_min_tof_hits", 2)

        # ---------------------------------------------------------------------
        # Scan stabil
        # ---------------------------------------------------------------------
        self.declare_parameter("scan_sector_count", 8)
        self.declare_parameter("scan_yaw_tolerance_deg", 3.0)
        self.declare_parameter("scan_settle_time", 1.50)
        self.declare_parameter("scan_flush_time", 1.30)
        self.declare_parameter("scan_flush_fresh_frames", 3)
        self.declare_parameter("scan_collect_time", 3.80)
        self.declare_parameter("scan_collect_min_frames", 8)
        self.declare_parameter("scan_post_hold", 0.80)
        self.declare_parameter("scan_max_speed", 0.16)
        self.declare_parameter("scan_max_drift", 0.20)
        self.declare_parameter("scan_max_alt_error", 0.16)

        # ---------------------------------------------------------------------
        # Navigasi dan keselamatan
        # ---------------------------------------------------------------------
        self.declare_parameter("standoff_distance", 4.20)
        self.declare_parameter("max_command_ahead", 0.32)
        self.declare_parameter("target_align_tolerance_deg", 7.0)
        self.declare_parameter("target_realign_deg", 28.0)
        self.declare_parameter("goal_arrival_distance", 0.65)
        self.declare_parameter("random_near_band", 2.50)
        self.declare_parameter("explore_arrival_distance", 0.70)
        self.declare_parameter("explore_margin", 2.0)

        self.declare_parameter("tof_hard_stop", 1.85)
        self.declare_parameter("tof_unmapped_stop", 2.35)
        self.declare_parameter("tof_candidate_tolerance", 1.40)
        self.declare_parameter("tof_direction_window_deg", 7.0)
        self.declare_parameter("tof_slow_distance", 5.50)
        self.declare_parameter("tof_target_match_margin", 2.20)
        self.declare_parameter("tof_history_size", 9)
        self.declare_parameter("tof_history_window", 0.90)
        self.declare_parameter("tof_min_stable_samples", 5)
        self.declare_parameter("tof_max_mad", 0.45)
        self.declare_parameter("tof_unstable_percentile", 30.0)
        self.declare_parameter("max_altitude_command_rate", 0.45)
        self.declare_parameter("altitude_recovery_error", 0.28)
        self.declare_parameter("min_motion_lookahead", 0.12)

        # V21H flight balance governor. Position setpoint is latched instead
        # of being moved ahead on every 20 Hz control tick.
        self.declare_parameter("motion_step_distance", 0.32)
        self.declare_parameter("motion_waypoint_reach", 0.13)
        self.declare_parameter("motion_advance_speed", 0.55)
        self.declare_parameter("motion_brake_speed", 0.95)
        self.declare_parameter("motion_release_speed", 0.30)
        self.declare_parameter("motion_waypoint_timeout", 1.20)
        self.declare_parameter("yaw_slew_rate_deg", 28.0)
        self.declare_parameter("vertical_speed_hold_threshold", 1.10)
        self.declare_parameter("vertical_speed_hold_min_error", 0.10)

        # Pose estimator guard. V21B logs showed impossible altitude/XY jumps;
        # those samples must never enter the route or setpoint state machine.
        self.declare_parameter("pose_min_altitude", -0.30)
        self.declare_parameter("pose_max_altitude", 4.00)
        self.declare_parameter("pose_max_xy_speed", 5.00)
        self.declare_parameter("pose_max_vz", 3.00)
        self.declare_parameter("pose_xy_jump_base", 0.45)
        self.declare_parameter("pose_xy_jump_speed", 5.00)
        self.declare_parameter("pose_z_jump_base", 0.25)
        self.declare_parameter("pose_z_jump_speed", 3.00)
        self.declare_parameter("pose_fault_hold_time", 2.00)

        # Lightweight moving PointCloud safety. It is NOT mapping; it only
        # detects an obstacle in the flight corridor while APPROACH/EXPLORE.
        self.declare_parameter("moving_cloud_safety_enabled", True)
        self.declare_parameter("moving_cloud_corridor_half_width", 0.55)
        self.declare_parameter("moving_cloud_min_z", -0.75)
        self.declare_parameter("moving_cloud_max_z", 0.75)
        self.declare_parameter("moving_cloud_min_forward", 0.45)
        self.declare_parameter("moving_cloud_max_forward", 6.00)
        self.declare_parameter("moving_cloud_min_points", 24)
        self.declare_parameter("moving_cloud_fresh_age", 0.45)
        self.declare_parameter("moving_cloud_hard_stop", 1.90)
        self.declare_parameter("moving_cloud_unmapped_stop", 3.40)
        self.declare_parameter("brake_hold_time", 1.50)

        self.declare_parameter("close_settle_time", 1.30)
        self.declare_parameter("close_collect_time", 3.20)
        self.declare_parameter("close_min_hits", 3)
        self.declare_parameter("close_match_radius", 1.70)
        self.declare_parameter("failed_target_cooldown", 14.0)
        self.declare_parameter("reject_after_failures", 2)

        self.declare_parameter("debug_pc_enabled", True)
        self.declare_parameter("debug_publish_period", 0.80)

        self._read_parameters()

        # V21N navigation-only policy. The complete V21H PointCloud detector,
        # stationary scan, clustering, filtering, accumulation, and scan
        # finalization methods remain byte-for-byte unchanged.
        #
        # Physical approach uses the temporal-median ToF only:
        # 3 m = stop and classify new/old with stationary PointCloud,
        # 2 m = stationary safety hold,
        # 1 m = final safe hold then VISITED.
        self.layer3_object_distance = 3.00
        self.layer2_stop_distance = 2.00
        self.layer1_visit_distance = 1.00
        self.layer1_old_tree_radius = 1.00
        self.tof_layer_tolerance = 0.10
        self.tof_layer_max_mad = 0.25
        self.tof_layer_hold_time = 0.80
        self.tof_layer_timeout = 4.00
        self.tof_final_min_safe = 0.65
        self.tof_cloud_emergency_distance = 0.60

        # V21N2: ToF target tracker. Narrow-center and nearest robust
        # wide-front clusters are evaluated together. Stable background
        # jumps are rejected after a nearer trunk has been tracked.
        self.tof_recovery_window = math.radians(24.0)
        self.tof_recovery_history: Deque[Tuple[float, float]] = deque(maxlen=30)
        self.tof_dropout_started = 0.0
        self.tof_last_valid_distance = math.inf
        self.tof_last_valid_mono = 0.0
        self.tof_last_valid_target_distance = math.inf
        self.tof_selected_source_v21n2 = "none"
        self.tof_far_jump_margin_v21n2 = 1.10
        self.tof_track_recent_time_v21n2 = 2.50
        self.tof3_dropout_gate_timeout = 0.80
        self.tof3_dropout_gate_target_max = 4.20
        self.tof3_dropout_gate_last_tof_max = 3.20
        self.tof_recovery_sweep = math.radians(6.0)
        self.tof_map_collision_guard = 0.85

        # V21N3: after the stationary 3 m classification, freeze the physical
        # approach bearing. The close PointCloud centroid is retained only for
        # map refinement at final VISITED; it must not steer the drone sideways.
        self.tof_final_yaw_v21n3 = math.nan
        self.tof_final_target_id_v21n3: Optional[int] = None
        self.tof_refined_centers_v21n3: Dict[int, Tuple[float, float, float]] = {}
        self.tof_final_forward_distance_v21n3 = 2.50
        self.tof_stage2_step_v21n3 = 0.050
        self.tof_stage1_step_v21n3 = 0.038
        self.tof_near_realign_block_distance_v21n3 = 4.50

        # V21N14: 3 m re-verification uses one frozen XY anchor and one
        # monotonic 60 degree arc: -30 -> 0 -> +30 degrees. Every view uses
        # exactly the same TURN/SETTLE/FLUSH/COLLECT sequence as the 360 scan.
        self.tof3_sweep_offsets_v21n4 = (
            math.radians(-30.0),
            0.0,
            math.radians(30.0),
        )
        self.tof3_sweep_index_v21n4 = 0
        self.tof3_sweep_nominal_yaw_v21n4 = 0.0
        self.tof3_sweep_frame_view_v21n4: Dict[int, int] = {}
        self.tof3_sweep_settle_time_v21n4 = 1.50
        self.tof3_sweep_flush_time_v21n10 = 1.30
        self.tof3_sweep_flush_fresh_v21n10 = 3
        self.tof3_sweep_flush_start_seq_v21n10 = 0
        self.tof3_sweep_flush_started_v21n10 = 0.0
        self.tof3_sweep_collect_time_v21n4 = 2.20
        self.tof3_sweep_collect_min_frames_v21n10 = 6
        self.tof3_sweep_collect_timeout_v21n10 = 3.60
        self.tof3_sweep_match_radius_v21n10 = 2.40
        self.tof3_sweep_view_assoc_radius_v21n10 = 1.10
        self.tof3_sweep_max_mad_v21n4 = 0.55
        self.tof3_sweep_max_shift_v21n4 = 2.00
        self.tof3_sweep_range_min_v21n10 = 1.80
        self.tof3_sweep_range_max_v21n10 = 4.50

        # V21N5: VISITED coordinates are immutable. A later scan may only
        # deduplicate against an existing green tree inside a tight radius;
        # it may never drag the green marker toward another distant tree.
        self.visited_merge_radius_v21n5 = 0.55
        self.old_tree_center_radius_v21n5 = 0.70
        self.old_tree_node_radius_v21n5 = 1.00

        # Independent ToF-only collision guard. It tracks the nearest robust
        # cluster inside +/-30 degrees and is checked in addition to the
        # target ToF tracker. This catches a trunk even if the selected beam
        # accidentally switches to a far background return.
        self.tof_front_guard_window_v21n5 = math.radians(30.0)
        self.tof_front_guard_history_v21n5: Deque[
            Tuple[float, float]
        ] = deque(maxlen=24)
        self.tof_front_guard_history_window_v21n5 = 0.65
        self.tof_front_guard_max_mad_v21n5 = 0.22
        self.tof_front_guard_hard_stop_v21n5 = 0.88
        self.tof_front_guard_layer3_v21n5 = 3.10
        self.tof_front_guard_layer2_v21n5 = 2.10
        self.tof_front_guard_layer1_v21n5 = 1.10

        # V21N12: final approach remains conservative near the trunk, but no
        # longer crawls at 0.03-0.04 m/s for many minutes.
        self.tof_stage2_step_v21n3 = 0.160
        self.tof_stage1_step_v21n3 = 0.075

        # Close-return-only latch. The old guard mixed near trunk returns with
        # far background values, making MAD unstable and allowing the target
        # ToF to switch to a 9-10 m background while the map target was near.
        self.tof_near_gate_window_v21n12 = 1.20
        self.tof_near_gate_min_samples_v21n12 = 3
        self.tof_near_gate_max_mad_v21n12 = 0.28
        self.tof_layer2_near_max_v21n12 = 2.42
        self.tof_layer1_near_max_v21n12 = 1.28

        # Progress watchdog for TO_2M / TO_1M. It watches both map progress and
        # physical ToF progress. A stopped or lost-beam approach is recovered
        # instead of holding forever.
        self.final_progress_target_id_v21n12: Optional[int] = None
        self.final_progress_stage_v21n12 = ""
        self.final_progress_best_map_v21n12 = math.inf
        self.final_progress_best_tof_v21n12 = math.inf
        self.final_progress_last_mono_v21n12 = 0.0
        self.final_progress_timeout_v21n12 = 12.0
        self.final_progress_map_delta_v21n12 = 0.10
        self.final_progress_tof_delta_v21n12 = 0.10

        # Back-away and re-align recovery for a lost trunk return.
        self.final_reacquire_target_id_v21n12: Optional[int] = None
        self.final_reacquire_stage_v21n12 = ""
        self.final_reacquire_goal_v21n12: Optional[
            Tuple[float, float]
        ] = None
        self.final_reacquire_yaw_v21n12 = 0.0
        self.final_reacquire_started_v21n12 = 0.0
        self.final_reacquire_timeout_v21n12 = 7.0
        self.final_reacquire_attempts_v21n12: Dict[int, int] = {}

        # The selected target remains locked. active_standoff_goal is visual;
        # APPROACH commands directly toward the target and stops by ToF layers.
        self.standoff_distance = 1.00
        self.goal_arrival_distance = 0.30

        # Moving cloud is kept for debug and last-resort collision hold only.
        # It is never used as a 3/2/1 m distance or to switch targets.
        self.tof_hard_stop = 0.55
        self.tof_unmapped_stop = 0.70
        self.moving_cloud_hard_stop = self.tof_cloud_emergency_distance
        self.moving_cloud_unmapped_stop = 0.80

        # ---------------------------------------------------------------------
        # QoS
        # ---------------------------------------------------------------------
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        cloud_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ---------------------------------------------------------------------
        # Subscriber
        # ---------------------------------------------------------------------
        self.create_subscription(
            VehicleLocalPosition,
            self.local_position_topic,
            self._on_local_position,
            px4_qos,
        )
        self.create_subscription(
            VehicleStatus,
            self.vehicle_status_topic,
            self._on_vehicle_status,
            px4_qos,
        )
        self.create_subscription(
            PointCloud2,
            self.cloud_topic,
            self._on_cloud,
            cloud_qos,
        )
        self.create_subscription(
            LaserScan,
            self.tof_topic,
            self._on_tof,
            sensor_qos,
        )

        # ---------------------------------------------------------------------
        # Publisher
        # ---------------------------------------------------------------------
        self.offboard_pub = self.create_publisher(
            OffboardControlMode,
            self.offboard_control_topic,
            10,
        )
        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint,
            self.trajectory_setpoint_topic,
            10,
        )
        self.command_pub = self.create_publisher(
            VehicleCommand,
            self.vehicle_command_topic,
            10,
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/sawit/tree_markers",
            10,
        )
        self.actual_pub = self.create_publisher(
            MarkerArray,
            "/sawit/actual_tree_markers",
            10,
        )
        self.nav_pub = self.create_publisher(
            MarkerArray,
            "/sawit/navigation_markers",
            10,
        )
        self.route_pub = self.create_publisher(
            Marker,
            "/sawit/route_marker",
            10,
        )
        self.drone_marker_pub = self.create_publisher(
            Marker,
            "/sawit/drone_marker",
            10,
        )
        self.trunk_models_pub = self.create_publisher(
            MarkerArray,
            "/sawit/trunk_models",
            10,
        )
        self.comparison_pub = self.create_publisher(
            MarkerArray,
            "/sawit/comparison_markers",
            10,
        )
        self.debug_stationary_pub = self.create_publisher(
            PointCloud2,
            "/sawit/debug_pc/stationary_raw",
            2,
        )
        self.debug_roi_pub = self.create_publisher(
            PointCloud2,
            "/sawit/debug_pc/roi",
            2,
        )
        self.debug_rejected_pub = self.create_publisher(
            PointCloud2,
            "/sawit/debug_pc/rejected",
            2,
        )
        self.debug_accepted_pub = self.create_publisher(
            PointCloud2,
            "/sawit/debug_pc/accepted",
            2,
        )
        self.debug_landmark_pub = self.create_publisher(
            PointCloud2,
            "/sawit/debug_pc/landmark_memory",
            2,
        )

        # ---------------------------------------------------------------------
        # Runtime state
        # ---------------------------------------------------------------------
        self.state = NavState.WAIT_DATA
        self.state_enter_mono = time.monotonic()

        self.have_pose = False
        self.have_cloud = False
        self.have_status = False

        self.home_ned_x = 0.0
        self.home_ned_y = 0.0
        self.home_ned_z = 0.0
        self.flight_ned_z = -self.flight_altitude
        self.commanded_altitude = 0.10
        self.last_altitude_command_mono = time.monotonic()

        self.pose: Optional[PoseSample] = None
        self.pose_history: Deque[PoseSample] = deque(maxlen=300)
        self.vehicle_status: Optional[VehicleStatus] = None

        self.last_cloud_msg: Optional[PointCloud2] = None
        self.last_cloud_receipt_mono = 0.0
        self.cloud_seq = 0
        self.last_processed_cloud_seq = -1

        self.tof_ranges: Optional[np.ndarray] = None
        self.tof_angle_min = 0.0
        self.tof_angle_increment = 0.0
        self.tof_range_min = 0.0
        self.tof_range_max = 0.0
        self.tof_receipt_mono = 0.0
        self.tof_front_history: Deque[Tuple[float, float]] = deque(
            maxlen=self.tof_history_size
        )

        self.scan_generation = 0
        self.scan_reason = ""
        self.scan_anchor_xy = (0.0, 0.0)
        self.scan_anchor_altitude = self.flight_altitude
        self.scan_base_yaw = 0.0
        self.scan_sector = 0
        self.scan_target_yaw = 0.0
        self.scan_phase_seq = 0
        self.scan_collect_frames = 0

        self.sector_anchor_xy = (0.0, 0.0)
        self.sector_anchor_altitude = self.flight_altitude
        self.sector_anchor_yaw = 0.0

        self.scan_accumulators: List[ScanAccumulator] = []

        self.tracks: Dict[int, TreeTrack] = {}
        self.next_tree_id = 0
        self.active_target_id: Optional[int] = None
        self.active_standoff_goal: Optional[Tuple[float, float]] = None
        self.active_target_selected_mono = 0.0
        self.first_target_selected = False

        # V21N6: each completed 360deg/8-sector scan creates one fixed
        # random FIFO queue. The order is not recalculated after each visit.
        self.random_batch_queue_v21n6: Deque[int] = deque()
        self.random_batch_snapshot_v21n6: List[int] = []
        self.random_batch_generation_v21n6 = 0
        self.random_batch_processed_v21n6 = 0
        self.random_batch_active_v21n6 = False
        self.random_batch_scan_started_mono_v21n6 = 0.0

        # V21N7: do not fly through a nearer obstacle to reach a far node.
        # A stable front return substantially nearer than the selected map
        # target means the straight route is blocked.
        self.route_block_target_id_v21n7: Optional[int] = None
        self.route_block_started_v21n7 = 0.0
        self.route_block_margin_v21n7 = 3.20
        self.route_block_min_target_distance_v21n7 = 6.00
        self.route_block_confirm_time_v21n7 = 0.85

        # Latch repeated impossible PX4 poses as a simulation collision.
        # Navigation is stopped instead of continuing with stale coordinates.
        self.pose_reject_window_started_v21n7 = 0.0
        self.pose_reject_streak_v21n7 = 0
        self.collision_abort_latched_v21n7 = False

        # V21N8: after a successful 1 m visit, back away while still facing
        # the trunk. This prevents the next yaw turn / 360 scan from starting
        # only one metre from the leaves.
        self.post_visit_retreat_goal_v21n8: Optional[
            Tuple[float, float]
        ] = None
        self.post_visit_retreat_yaw_v21n8 = 0.0
        self.post_visit_retreat_started_v21n8 = 0.0
        self.post_visit_retreat_id_v21n8: Optional[int] = None
        self.post_visit_retreat_distance_v21n8 = 1.00
        self.post_visit_retreat_timeout_v21n8 = 10.0

        # A detected-tree marker may turn green only after the explicit
        # ToF 1 m stationary safety hold succeeds in this run.
        self.visited_proof_ids_v21n8: Set[int] = set()

        # V21N11: every successful ToF-1m visit is followed by a complete
        # stationary 360 degree / 8-sector scan. The fixed random order that
        # still remains is preserved while that scan refreshes the map.
        self.post_visit_rescan_pending_v21n11 = False
        self.post_visit_rescan_visited_id_v21n11: Optional[int] = None
        self.post_visit_rescan_count_v21n11 = 0

        # A scan can physically see an already visited trunk again, but that
        # observation must be absorbed into the immutable green landmark,
        # never promoted as a new yellow candidate.
        self.visited_rescan_suppress_radius_v21n11 = 3.70

        # V21N13: a second post-scan dedupe pass joins yellow candidates that
        # came from different sectors/views of the same trunk. The orchard
        # spacing is much larger than this radius, while current visual error
        # can exceed the old 1.65 m merge threshold.
        self.post_scan_dedupe_radius_v21n13 = 3.45

        # A map node that is already close but has no ToF return and no usable
        # target frame must not hold forever. It is marked as a ghost estimate,
        # a full stationary 360/8 scan is run at the current position, and the
        # preserved random remainder continues. A newly rediscovered trunk is
        # deferred to the next batch rather than inserted into the frozen order.
        self.ghost_rescan_pending_v21n13 = False
        self.ghost_rescan_target_id_v21n13: Optional[int] = None
        self.ghost_rescan_old_xy_v21n13: Optional[Tuple[float, float]] = None
        self.tof3_ghost_target_max_distance_v21n13 = 3.45
        self.tof3_ghost_dropout_timeout_v21n13 = 8.0
        self.ghost_rescan_count_v21n13 = 0

        # Confidence limits for the 60 degree close verification. Multi-view
        # fusion may update more strongly; a single-view result only nudges the
        # old node so one off-axis trunk/leaf cluster cannot drag it far away.
        self.tof3_multiview_max_update_v21n13 = 1.35
        self.tof3_singleview_max_update_v21n13 = 0.75
        self.tof3_singleview_alpha_v21n13 = 0.45

        # V21N9: fixed random order is mandatory for distinct trees.
        # A blocked route does not consume the queue item. The active target
        # stays locked while the drone backs up, sidesteps, and retries it.
        self.avoid_target_id_v21n9: Optional[int] = None
        self.avoid_phase_v21n9 = ""
        self.avoid_goal_v21n9: Optional[Tuple[float, float]] = None
        self.avoid_yaw_v21n9 = 0.0
        self.avoid_side_v21n9 = 1
        self.avoid_phase_started_v21n9 = 0.0
        self.avoid_attempts_v21n9: Dict[int, int] = {}
        self.avoid_trigger_distance_v21n9 = 3.25
        self.avoid_target_gap_v21n9 = 4.00
        self.avoid_backup_distance_v21n9 = 0.65
        self.avoid_sidestep_distance_v21n9 = 2.20
        self.avoid_phase_timeout_v21n9 = 8.0

        # A failed 60 degree PointCloud verification also keeps the same ID.
        # Back away slightly, face the same target, and retry the 3 m check.
        self.verify_retry_target_id_v21n9: Optional[int] = None
        self.verify_retry_goal_v21n9: Optional[Tuple[float, float]] = None
        self.verify_retry_yaw_v21n9 = 0.0
        self.verify_retry_started_v21n9 = 0.0
        self.verify_retry_distance_v21n9 = 0.60
        self.verify_retry_timeout_v21n9 = 8.0
        self.verify_retry_count_v21n9: Dict[int, int] = {}

        self.close_anchor_xy = (0.0, 0.0)
        self.close_anchor_altitude = self.flight_altitude
        self.close_anchor_yaw = 0.0
        self.close_collect_frames = 0
        self.close_matches: List[Tuple[float, float, int, float]] = []

        # V21N staged ToF approach state.
        self.tof_approach_stage = "TO_3M"
        self.tof_stage_hold_xy = (0.0, 0.0)
        self.tof_stage_hold_yaw = 0.0
        self.tof_stage_hold_started = 0.0
        self.close_verify_purpose = ""
        self.tof_stage_target_id: Optional[int] = None

        self.explore_goal: Optional[Tuple[float, float]] = None
        self.explore_count = 0

        # Latched horizontal waypoint and fixed alignment anchor.
        self.motion_waypoint_xy: Optional[Tuple[float, float]] = None
        self.motion_goal_xy: Optional[Tuple[float, float]] = None
        self.motion_waypoint_since = 0.0
        self.motion_brake_anchor_xy: Optional[Tuple[float, float]] = None
        self.align_anchor_xy = (0.0, 0.0)

        # Slew-limited yaw command.
        self.commanded_yaw_enu = 0.0
        self.last_yaw_command_mono = time.monotonic()

        self.path_points: Deque[Tuple[float, float]] = deque(maxlen=3000)
        self.last_path_append_mono = 0.0

        self.last_debug_publish_mono = 0.0
        self.last_marker_publish_mono = 0.0
        self.last_log_mono: Dict[str, float] = {}

        self.last_debug_stationary_points: Optional[np.ndarray] = None
        self.last_debug_candidate_points: Optional[np.ndarray] = None
        self.last_debug_moving_safety_points: Optional[np.ndarray] = None

        self.moving_cloud_front_min = math.inf
        self.moving_cloud_front_receipt_mono = 0.0

        self.pose_fault_until = 0.0
        self.pose_fault_count = 0

        self.brake_reason = ""
        self.brake_next_action = "select"
        self.brake_scan_reason = ""
        self.brake_hold_xy = (0.0, 0.0)
        self.brake_hold_yaw = 0.0

        self.prestream_started_mono = 0.0
        self.last_arm_command_mono = 0.0
        self.hold_xy = (0.0, 0.0)
        self.hold_yaw = 0.0

        self.random = random.Random(20260714)

        self._load_or_reset_memory()

        self.control_timer = self.create_timer(0.05, self._control_loop)
        self.marker_timer = self.create_timer(0.25, self._publish_markers)

        self.get_logger().info(
            "START V21H VISUAL COMPARE SAFE FROZEN-ANCHOR MAP "
            "stationary_only=1 delayed_frame_flush=1 one_tree_one_id=1 "
            "actual_visual_only=1 altitude_guard=1 tof_corridor_brake=1 "
            "cycle=one_scan_one_visit visible_drone_beacon=1 "
            "lower_altitude_0p72=1 micro_waypoint=1 speed_governor=1 "
            "yaw_slew=1 moving_cloud_brake=1 "
            "legacy_topics_only=1"
        )
        self.get_logger().info(
            "V21H transform: PX4 NED -> internal ENU-relative-home; "
            f"cloud_axes={self.cloud_axes_mode}; visual_map="
            f"({self.visual_spawn_x:+.1f}+East,"
            f"{self.visual_spawn_y:+.1f}+North)"
        )
        self.get_logger().info(
            "SPAWN_ORCHARD_CONFIG_AUTO_V1 "
            f"visual_spawn=({self.visual_spawn_x:.2f},{self.visual_spawn_y:.2f}) "
            f"orchard_local_x=({self.orchard_min_x:.2f},{self.orchard_max_x:.2f}) "
            f"orchard_local_y=({self.orchard_min_y:.2f},{self.orchard_max_y:.2f}) "
            "actual_used_as_gate=0 actual_used_for_control=0"
        )
        self.get_logger().info(
            "V21N13 route: original fixed random batch remains frozen; new rescan candidates wait for next batch; ghost targets trigger local 360 rescan; "
            f"altitude={self.flight_altitude:.2f}m "
            f"standoff={self.standoff_distance:.2f}m "
            f"node_step<={self.motion_step_distance:.2f}m "
            f"speed_brake={self.motion_brake_speed:.2f}m/s; "
            "actual/SDF visual-only"
        )
        self.get_logger().info(
            "START V21N14 MINISCAN60 EARLY BLOCK RELOCALIZE "
            "detector=v21h_exact stationary_scan=unchanged "
            "tof3=classify_new_or_old tof2=safety_hold "
            "tof1=safe_visited moving_cloud=emergency_only "
            "target_lock=until_old_or_visited"
        )
        self.get_logger().info(
            "V21N4 RVIZ ToF beam gates: 3m=FACE+45DEG REVERIFY+UPDATE, "
            "2m=SAFETY HOLD, 1m=VISITED; no radius circles"
        )
        self.get_logger().info(
            "V21N4 tracker: at ToF 3m face target, scan 45deg from three "
            "stationary views, robustly update tree position, then use "
            "the updated frozen bearing for ToF 2m and 1m"
        )
        self.get_logger().info(
            "V21N6 random: build one shuffled FIFO order after each full "
            "8-sector scan; finish the whole fixed order before next scan"
        )

    # =========================================================================
    # Parameter
    # =========================================================================

    def _p(self, name: str):
        return self.get_parameter(name).value

    def _read_parameters(self) -> None:
        self.local_position_topic = str(self._p("local_position_topic"))
        self.vehicle_status_topic = str(self._p("vehicle_status_topic"))
        self.offboard_control_topic = str(self._p("offboard_control_topic"))
        self.trajectory_setpoint_topic = str(self._p("trajectory_setpoint_topic"))
        self.vehicle_command_topic = str(self._p("vehicle_command_topic"))
        self.cloud_topic = str(self._p("cloud_topic"))
        self.tof_topic = str(self._p("tof_topic"))

        self.target_tree_count = int(self._p("target_tree_count"))
        self.flight_altitude = float(self._p("flight_altitude"))
        self.reset_memory_on_start = bool(self._p("reset_memory_on_start"))
        self.visual_spawn_x = float(self._p("visual_spawn_x"))
        self.visual_spawn_y = float(self._p("visual_spawn_y"))
        self.publish_actual_visual = bool(self._p("publish_actual_visual"))
        self.memory_path = Path(str(self._p("memory_path"))).expanduser()

        self.orchard_min_x = float(self._p("orchard_min_x"))
        self.orchard_max_x = float(self._p("orchard_max_x"))
        self.orchard_min_y = float(self._p("orchard_min_y"))
        self.orchard_max_y = float(self._p("orchard_max_y"))
        self.orchard_gate_margin = float(self._p("orchard_gate_margin"))

        self.cloud_axes_mode = str(self._p("cloud_axes_mode"))
        self.cloud_left_sign = float(self._p("cloud_left_sign"))
        self.cloud_point_stride = max(1, int(self._p("cloud_point_stride")))
        self.cloud_min_forward = float(self._p("cloud_min_forward"))
        self.cloud_max_forward = float(self._p("cloud_max_forward"))
        self.cloud_max_abs_left = float(self._p("cloud_max_abs_left"))
        self.cloud_min_z = float(self._p("cloud_min_z"))
        self.cloud_max_z = float(self._p("cloud_max_z"))

        self.voxel_xy = float(self._p("voxel_xy"))
        self.seed_min_points = int(self._p("seed_min_points"))
        self.cluster_radius = float(self._p("cluster_radius"))
        self.cluster_min_points = int(self._p("cluster_min_points"))
        self.trunk_min_height = float(self._p("trunk_min_height"))
        self.trunk_min_radius = float(self._p("trunk_min_radius"))
        self.trunk_max_radius = float(self._p("trunk_max_radius"))
        self.trunk_min_vertical_bins = int(self._p("trunk_min_vertical_bins"))
        self.trunk_max_axis_wander = float(self._p("trunk_max_axis_wander"))
        self.candidate_nms_radius = float(self._p("candidate_nms_radius"))

        self.scan_assoc_radius = float(self._p("scan_assoc_radius"))
        self.tree_assoc_radius = float(self._p("tree_assoc_radius"))
        self.tree_merge_radius = float(self._p("tree_merge_radius"))
        self.new_track_min_frames = int(self._p("new_track_min_frames"))
        self.new_track_min_strong = int(self._p("new_track_min_strong"))
        self.confirm_min_total_hits = int(self._p("confirm_min_total_hits"))
        self.confirm_min_view_span = math.radians(
            float(self._p("confirm_min_view_span_deg"))
        )
        self.confirm_min_tof_hits = int(self._p("confirm_min_tof_hits"))

        self.scan_sector_count = max(4, int(self._p("scan_sector_count")))
        self.scan_yaw_tolerance = math.radians(
            float(self._p("scan_yaw_tolerance_deg"))
        )
        self.scan_settle_time = float(self._p("scan_settle_time"))
        self.scan_flush_time = float(self._p("scan_flush_time"))
        self.scan_flush_fresh_frames = int(self._p("scan_flush_fresh_frames"))
        self.scan_collect_time = float(self._p("scan_collect_time"))
        self.scan_collect_min_frames = int(self._p("scan_collect_min_frames"))
        self.scan_post_hold = float(self._p("scan_post_hold"))
        self.scan_max_speed = float(self._p("scan_max_speed"))
        self.scan_max_drift = float(self._p("scan_max_drift"))
        self.scan_max_alt_error = float(self._p("scan_max_alt_error"))

        self.standoff_distance = float(self._p("standoff_distance"))
        self.max_command_ahead = float(self._p("max_command_ahead"))
        self.target_align_tolerance = math.radians(
            float(self._p("target_align_tolerance_deg"))
        )
        self.target_realign = math.radians(
            float(self._p("target_realign_deg"))
        )
        self.goal_arrival_distance = float(self._p("goal_arrival_distance"))
        self.random_near_band = float(self._p("random_near_band"))
        self.explore_arrival_distance = float(self._p("explore_arrival_distance"))
        self.explore_margin = float(self._p("explore_margin"))

        self.tof_hard_stop = float(self._p("tof_hard_stop"))
        self.tof_unmapped_stop = float(self._p("tof_unmapped_stop"))
        self.tof_candidate_tolerance = float(
            self._p("tof_candidate_tolerance")
        )
        self.tof_direction_window = math.radians(
            float(self._p("tof_direction_window_deg"))
        )
        self.tof_slow_distance = float(self._p("tof_slow_distance"))
        self.tof_target_match_margin = float(
            self._p("tof_target_match_margin")
        )
        self.tof_history_size = int(self._p("tof_history_size"))
        self.tof_history_window = float(self._p("tof_history_window"))
        self.tof_min_stable_samples = int(
            self._p("tof_min_stable_samples")
        )
        self.tof_max_mad = float(self._p("tof_max_mad"))
        self.tof_unstable_percentile = float(
            self._p("tof_unstable_percentile")
        )
        self.max_altitude_command_rate = float(
            self._p("max_altitude_command_rate")
        )
        self.altitude_recovery_error = float(
            self._p("altitude_recovery_error")
        )
        self.min_motion_lookahead = float(
            self._p("min_motion_lookahead")
        )
        self.motion_step_distance = float(
            self._p("motion_step_distance")
        )
        self.motion_waypoint_reach = float(
            self._p("motion_waypoint_reach")
        )
        self.motion_advance_speed = float(
            self._p("motion_advance_speed")
        )
        self.motion_brake_speed = float(
            self._p("motion_brake_speed")
        )
        self.motion_release_speed = float(
            self._p("motion_release_speed")
        )
        self.motion_waypoint_timeout = float(
            self._p("motion_waypoint_timeout")
        )
        self.yaw_slew_rate = math.radians(
            float(self._p("yaw_slew_rate_deg"))
        )
        self.vertical_speed_hold_threshold = float(
            self._p("vertical_speed_hold_threshold")
        )
        self.vertical_speed_hold_min_error = float(
            self._p("vertical_speed_hold_min_error")
        )

        self.pose_min_altitude = float(self._p("pose_min_altitude"))
        self.pose_max_altitude = float(self._p("pose_max_altitude"))
        self.pose_max_xy_speed = float(self._p("pose_max_xy_speed"))
        self.pose_max_vz = float(self._p("pose_max_vz"))
        self.pose_xy_jump_base = float(self._p("pose_xy_jump_base"))
        self.pose_xy_jump_speed = float(self._p("pose_xy_jump_speed"))
        self.pose_z_jump_base = float(self._p("pose_z_jump_base"))
        self.pose_z_jump_speed = float(self._p("pose_z_jump_speed"))
        self.pose_fault_hold_time = float(self._p("pose_fault_hold_time"))

        self.moving_cloud_safety_enabled = bool(
            self._p("moving_cloud_safety_enabled")
        )
        self.moving_cloud_corridor_half_width = float(
            self._p("moving_cloud_corridor_half_width")
        )
        self.moving_cloud_min_z = float(self._p("moving_cloud_min_z"))
        self.moving_cloud_max_z = float(self._p("moving_cloud_max_z"))
        self.moving_cloud_min_forward = float(
            self._p("moving_cloud_min_forward")
        )
        self.moving_cloud_max_forward = float(
            self._p("moving_cloud_max_forward")
        )
        self.moving_cloud_min_points = int(
            self._p("moving_cloud_min_points")
        )
        self.moving_cloud_fresh_age = float(
            self._p("moving_cloud_fresh_age")
        )
        self.moving_cloud_hard_stop = float(
            self._p("moving_cloud_hard_stop")
        )
        self.moving_cloud_unmapped_stop = float(
            self._p("moving_cloud_unmapped_stop")
        )
        self.brake_hold_time = float(self._p("brake_hold_time"))

        self.close_settle_time = float(self._p("close_settle_time"))
        self.close_collect_time = float(self._p("close_collect_time"))
        self.close_min_hits = int(self._p("close_min_hits"))
        self.close_match_radius = float(self._p("close_match_radius"))
        self.failed_target_cooldown = float(
            self._p("failed_target_cooldown")
        )
        self.reject_after_failures = int(self._p("reject_after_failures"))

        self.debug_pc_enabled = bool(self._p("debug_pc_enabled"))
        self.debug_publish_period = float(self._p("debug_publish_period"))

    # =========================================================================
    # Callback sensor
    # =========================================================================

    def _on_local_position(self, msg: VehicleLocalPosition) -> None:
        now = time.monotonic()

        x_ned = float(msg.x)
        y_ned = float(msg.y)
        z_ned = float(msg.z)
        heading_ned = float(msg.heading)

        if not finite(x_ned, y_ned, z_ned, heading_ned):
            self._log_throttle(
                "invalid_pose",
                1.0,
                "warning",
                "INVALID_POSE_DROP_V21H non-finite VehicleLocalPosition",
            )
            return

        xy_valid = bool(getattr(msg, "xy_valid", True))
        z_valid = bool(getattr(msg, "z_valid", True))
        if not xy_valid or not z_valid:
            self._log_throttle(
                "invalid_pose_flags",
                1.0,
                "warning",
                f"INVALID_POSE_DROP_V21H xy_valid={int(xy_valid)} "
                f"z_valid={int(z_valid)}",
            )
            return

        if not self.have_pose:
            self.home_ned_x = x_ned
            self.home_ned_y = y_ned
            self.home_ned_z = z_ned
            self.flight_ned_z = self.home_ned_z - self.flight_altitude

        # NED -> ENU relatif home
        x_enu = y_ned - self.home_ned_y
        y_enu = x_ned - self.home_ned_x
        altitude = -(z_ned - self.home_ned_z)
        yaw_enu = wrap_pi((math.pi / 2.0) - heading_ned)

        vx_ned = float(getattr(msg, "vx", 0.0))
        vy_ned = float(getattr(msg, "vy", 0.0))
        vz_ned = float(getattr(msg, "vz", 0.0))

        vx_enu = vy_ned if math.isfinite(vy_ned) else 0.0
        vy_enu = vx_ned if math.isfinite(vx_ned) else 0.0
        vz_up = -vz_ned if math.isfinite(vz_ned) else 0.0

        sample = PoseSample(
            receipt_mono=now,
            x_enu=x_enu,
            y_enu=y_enu,
            altitude=altitude,
            yaw_enu=yaw_enu,
            vx_enu=vx_enu,
            vy_enu=vy_enu,
            vz_up=vz_up,
        )

        reject_reason = ""
        if not (
            self.pose_min_altitude
            <= sample.altitude
            <= self.pose_max_altitude
        ):
            reject_reason = (
                f"altitude_out_of_bounds z={sample.altitude:.2f} "
                f"bounds=({self.pose_min_altitude:.2f},"
                f"{self.pose_max_altitude:.2f})"
            )
        elif sample.speed_xy > self.pose_max_xy_speed:
            reject_reason = (
                f"xy_speed={sample.speed_xy:.2f}>"
                f"{self.pose_max_xy_speed:.2f}"
            )
        elif abs(sample.vz_up) > self.pose_max_vz:
            reject_reason = (
                f"vz={sample.vz_up:+.2f}>"
                f"{self.pose_max_vz:.2f}"
            )

        if self.pose is not None and not reject_reason:
            dt = max(1e-3, now - self.pose.receipt_mono)
            jump_xy = math.hypot(
                sample.x_enu - self.pose.x_enu,
                sample.y_enu - self.pose.y_enu,
            )
            jump_z = abs(sample.altitude - self.pose.altitude)
            allowed_xy = self.pose_xy_jump_base + self.pose_xy_jump_speed * dt
            allowed_z = self.pose_z_jump_base + self.pose_z_jump_speed * dt

            if jump_xy > max(6.0, allowed_xy):
                reject_reason = (
                    f"global_xy_jump={jump_xy:.2f}m dt={dt:.3f}s"
                )
            elif jump_z > max(2.0, allowed_z):
                reject_reason = (
                    f"global_z_jump={jump_z:.2f}m dt={dt:.3f}s"
                )
            elif dt < 0.80 and jump_xy > allowed_xy:
                reject_reason = (
                    f"xy_jump={jump_xy:.2f}m dt={dt:.3f}s "
                    f"allowed={allowed_xy:.2f}m"
                )
            elif dt < 0.80 and jump_z > allowed_z:
                reject_reason = (
                    f"z_jump={jump_z:.2f}m dt={dt:.3f}s "
                    f"allowed={allowed_z:.2f}m"
                )

        if reject_reason:
            if (
                self.pose_reject_window_started_v21n7 <= 0.0
                or now - self.pose_reject_window_started_v21n7 > 1.00
            ):
                self.pose_reject_window_started_v21n7 = now
                self.pose_reject_streak_v21n7 = 0
            self.pose_reject_streak_v21n7 += 1

            if self.pose_reject_streak_v21n7 >= 3:
                self.collision_abort_latched_v21n7 = True
                self._log_throttle(
                    "v21n7_collision_abort_latch",
                    0.50,
                    "error",
                    f"SIM_COLLISION_ABORT_LATCH_V21N7 "
                    f"streak={self.pose_reject_streak_v21n7} "
                    f"reason={reject_reason} "
                    "action=stop_navigation_reset_gazebo_required",
                )

            self.pose_fault_count += 1
            self.pose_fault_until = max(
                self.pose_fault_until,
                now + self.pose_fault_hold_time,
            )
            self._log_throttle(
                "pose_spike",
                0.35,
                "error",
                f"POSE_GUARD_REJECT_V21H reason={reject_reason} "
                f"count={self.pose_fault_count} action=hold_last_good",
            )
            return

        self.pose_reject_streak_v21n7 = 0
        self.pose_reject_window_started_v21n7 = 0.0
        self.pose = sample
        self.pose_history.append(sample)

        if not self.have_pose:
            self.commanded_altitude = clamp(
                sample.altitude,
                0.05,
                self.flight_altitude,
            )
            self.last_altitude_command_mono = now
            self.commanded_yaw_enu = sample.yaw_enu
            self.last_yaw_command_mono = now
            self.align_anchor_xy = (sample.x_enu, sample.y_enu)
            self.have_pose = True
            self.get_logger().info(
                f"PX4_READY_V21H home_ned=({self.home_ned_x:.2f},"
                f"{self.home_ned_y:.2f},{self.home_ned_z:.2f}) "
                f"yaw_enu={math.degrees(yaw_enu):.1f}deg"
            )

        if now - self.last_path_append_mono >= 0.20:
            if not self.path_points:
                self.path_points.append((x_enu, y_enu))
            else:
                px, py = self.path_points[-1]
                if math.hypot(x_enu - px, y_enu - py) >= 0.10:
                    self.path_points.append((x_enu, y_enu))
            self.last_path_append_mono = now

    def _on_vehicle_status(self, msg: VehicleStatus) -> None:
        self.vehicle_status = msg
        self.have_status = True

    def _on_tof(self, msg: LaserScan) -> None:
        arr = np.asarray(msg.ranges, dtype=np.float32)
        self.tof_ranges = arr
        self.tof_angle_min = float(msg.angle_min)
        self.tof_angle_increment = float(msg.angle_increment)
        self.tof_range_min = float(msg.range_min)
        self.tof_range_max = float(msg.range_max)
        now = time.monotonic()
        self.tof_receipt_mono = now

        raw_front = self._tof_at_bearing(0.0)
        if math.isfinite(raw_front):
            self.tof_front_history.append((now, raw_front))

    def _on_cloud(self, msg: PointCloud2) -> None:
        now = time.monotonic()
        self.last_cloud_msg = msg
        self.last_cloud_receipt_mono = now
        self.have_cloud = True
        self.cloud_seq += 1

        capture_scan = self.state == NavState.SCAN_COLLECT
        capture_close = self.state == NavState.CLOSE_COLLECT
        capture_safety = (
            self.moving_cloud_safety_enabled
            and self.state in (
                NavState.APPROACH,
                NavState.EXPLORE_MOVE,
            )
        )

        # Mapping remains stationary-only. During motion, the cloud is parsed
        # only for a narrow front-corridor brake and never enters tree memory.
        if not capture_scan and not capture_close and not capture_safety:
            return

        if self.cloud_seq == self.last_processed_cloud_seq:
            return

        try:
            raw_points = self._pointcloud_to_xyz(msg)
        except Exception as exc:
            self.get_logger().error(
                f"CLOUD_PARSE_ERROR_V21H type={type(exc).__name__} error={exc}"
            )
            return

        if capture_safety:
            self._update_moving_cloud_safety(raw_points, now)
            if not capture_scan and not capture_close:
                self.last_processed_cloud_seq = self.cloud_seq
                return

        if capture_scan and not self._scan_capture_stable():
            self._log_throttle(
                "scan_unstable_cloud",
                0.8,
                "warning",
                "SCAN_CAPTURE_BLOCK_V21H reason=not_stable",
            )
            return

        if capture_close and not self._close_capture_stable():
            self._log_throttle(
                "close_unstable_cloud",
                0.8,
                "warning",
                "CLOSE_CAPTURE_BLOCK_V21H reason=not_stable",
            )
            return

        self.last_processed_cloud_seq = self.cloud_seq

        try:
            if raw_points.size == 0:
                return

            candidates, filtered_points = self._extract_trunk_candidates(raw_points)
        except Exception as exc:
            self.get_logger().error(
                f"CLOUD_PROCESS_ERROR_V21H type={type(exc).__name__} error={exc}"
            )
            return

        if capture_scan:
            anchor_xy = self.sector_anchor_xy
            anchor_yaw = self.sector_anchor_yaw
            anchor_alt = self.sector_anchor_altitude
            self.scan_collect_frames += 1
            frame_id = self.cloud_seq

            self._accumulate_candidates(
                candidates=candidates,
                anchor_xy=anchor_xy,
                anchor_yaw=anchor_yaw,
                sector=self.scan_sector,
                frame_id=frame_id,
            )
        else:
            anchor_xy = self.close_anchor_xy
            anchor_yaw = self.close_anchor_yaw
            anchor_alt = self.close_anchor_altitude
            self.close_collect_frames += 1
            self._accumulate_close_matches(
                candidates=candidates,
                anchor_xy=anchor_xy,
                anchor_yaw=anchor_yaw,
                frame_id=self.cloud_seq,
            )

        if self.debug_pc_enabled:
            transformed = self._transform_points_to_map(
                filtered_points,
                anchor_xy=anchor_xy,
                anchor_yaw=anchor_yaw,
                max_points=15000,
            )
            self.last_debug_stationary_points = transformed

            centers = []
            for c in candidates:
                mx, my = self._body_to_map(
                    c.forward,
                    c.left,
                    anchor_xy,
                    anchor_yaw,
                )
                centers.append((mx, my, anchor_alt))
            self.last_debug_candidate_points = (
                np.asarray(centers, dtype=np.float32)
                if centers
                else np.empty((0, 3), dtype=np.float32)
            )
            if (
                now - self.last_debug_publish_mono
                >= self.debug_publish_period
            ):
                self._publish_debug_clouds()
                self.last_debug_publish_mono = now


    # =========================================================================
    # PointCloud
    # =========================================================================

    def _pointcloud_to_xyz(self, msg: PointCloud2) -> np.ndarray:
        offsets = {field.name: int(field.offset) for field in msg.fields}
        for name in ("x", "y", "z"):
            if name not in offsets:
                raise ValueError(f"PointCloud field '{name}' tidak ditemukan")

        endian = ">" if msg.is_bigendian else "<"
        dtype = np.dtype(
            {
                "names": ["x", "y", "z"],
                "formats": [
                    endian + "f4",
                    endian + "f4",
                    endian + "f4",
                ],
                "offsets": [
                    offsets["x"],
                    offsets["y"],
                    offsets["z"],
                ],
                "itemsize": int(msg.point_step),
            }
        )

        count = int(msg.width) * int(msg.height)
        if count <= 0:
            return np.empty((0, 3), dtype=np.float32)

        arr = np.frombuffer(msg.data, dtype=dtype, count=count)
        xyz = np.column_stack((arr["x"], arr["y"], arr["z"])).astype(
            np.float32,
            copy=False,
        )

        if self.cloud_point_stride > 1:
            xyz = xyz[:: self.cloud_point_stride]

        good = np.isfinite(xyz).all(axis=1)
        xyz = xyz[good]
        return xyz

    def _convert_cloud_axes(self, xyz: np.ndarray) -> np.ndarray:
        if xyz.size == 0:
            return xyz

        mode = self.cloud_axes_mode.strip().lower()
        if mode == "x_forward_y_left_z_up":
            forward = xyz[:, 0]
            left = self.cloud_left_sign * xyz[:, 1]
            up = xyz[:, 2]
        elif mode in ("optical", "x_right_y_down_z_forward"):
            # Optical -> body FLU
            forward = xyz[:, 2]
            left = -xyz[:, 0]
            up = -xyz[:, 1]
        else:
            raise ValueError(
                f"cloud_axes_mode tidak dikenal: {self.cloud_axes_mode}"
            )

        return np.column_stack((forward, left, up)).astype(
            np.float32,
            copy=False,
        )

    def _extract_trunk_candidates(
        self,
        xyz_raw: np.ndarray,
    ) -> Tuple[List[TrunkCandidate], np.ndarray]:
        points = self._convert_cloud_axes(xyz_raw)
        if points.size == 0:
            return [], points

        forward = points[:, 0]
        left = points[:, 1]
        up = points[:, 2]

        mask = (
            (forward >= self.cloud_min_forward)
            & (forward <= self.cloud_max_forward)
            & (np.abs(left) <= self.cloud_max_abs_left)
            & (up >= self.cloud_min_z)
            & (up <= self.cloud_max_z)
        )
        points = points[mask]

        if points.shape[0] < self.cluster_min_points:
            return [], points

        forward = points[:, 0]
        left = points[:, 1]
        up = points[:, 2]

        nx = int(
            math.ceil(
                (self.cloud_max_forward - self.cloud_min_forward)
                / self.voxel_xy
            )
        ) + 1
        ny = int(
            math.ceil((2.0 * self.cloud_max_abs_left) / self.voxel_xy)
        ) + 1

        ix = np.floor(
            (forward - self.cloud_min_forward) / self.voxel_xy
        ).astype(np.int32)
        iy = np.floor(
            (left + self.cloud_max_abs_left) / self.voxel_xy
        ).astype(np.int32)

        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        if not np.any(valid):
            return [], points

        ix = ix[valid]
        iy = iy[valid]
        p = points[valid]
        keys = ix.astype(np.int64) * ny + iy.astype(np.int64)

        order = np.argsort(keys)
        keys_sorted = keys[order]
        z_sorted = p[order, 2]

        starts = np.r_[
            0,
            np.flatnonzero(np.diff(keys_sorted)) + 1,
        ]
        ends = np.r_[starts[1:], len(keys_sorted)]
        counts = ends - starts
        unique_keys = keys_sorted[starts]

        z_min = np.minimum.reduceat(z_sorted, starts)
        z_max = np.maximum.reduceat(z_sorted, starts)
        z_span = z_max - z_min

        seed_mask = (
            (counts >= self.seed_min_points)
            & (z_span >= 0.55 * self.trunk_min_height)
        )

        seed_indices = np.flatnonzero(seed_mask)
        if seed_indices.size == 0:
            return [], points

        seed_scores = counts[seed_indices] * np.maximum(
            z_span[seed_indices],
            0.01,
        )
        seed_order = seed_indices[np.argsort(seed_scores)[::-1]]
        seed_order = seed_order[:60]

        candidates: List[TrunkCandidate] = []
        accepted_centers: List[Tuple[float, float]] = []

        for seed_idx in seed_order:
            key = int(unique_keys[seed_idx])
            seed_ix = key // ny
            seed_iy = key % ny

            sx = (
                self.cloud_min_forward
                + (float(seed_ix) + 0.5) * self.voxel_xy
            )
            sy = (
                -self.cloud_max_abs_left
                + (float(seed_iy) + 0.5) * self.voxel_xy
            )

            if any(
                math.hypot(sx - ax, sy - ay) < self.candidate_nms_radius
                for ax, ay in accepted_centers
            ):
                continue

            radial = np.hypot(forward - sx, left - sy)
            cluster = points[radial <= self.cluster_radius]

            if cluster.shape[0] < self.cluster_min_points:
                continue

            z10 = float(np.percentile(cluster[:, 2], 10))
            z90 = float(np.percentile(cluster[:, 2], 90))
            height = z90 - z10
            if height < self.trunk_min_height:
                continue

            # Pusat batang diambil dari 15-65% rentang tinggi,
            # bukan centroid seluruh daun.
            lower_high = z10 + 0.65 * height
            lower_low = z10 + 0.15 * height
            trunk_band = cluster[
                (cluster[:, 2] >= lower_low)
                & (cluster[:, 2] <= lower_high)
            ]
            if trunk_band.shape[0] < max(12, self.cluster_min_points // 3):
                continue

            cx = float(np.median(trunk_band[:, 0]))
            cy = float(np.median(trunk_band[:, 1]))
            cz = float(np.median(trunk_band[:, 2]))

            radii = np.hypot(
                trunk_band[:, 0] - cx,
                trunk_band[:, 1] - cy,
            )
            radius = float(np.percentile(radii, 75))
            if not (self.trunk_min_radius <= radius <= self.trunk_max_radius):
                continue

            bins = np.linspace(z10, z90, 8)
            occupied_bins = 0
            bin_centers: List[Tuple[float, float]] = []
            for low, high in zip(bins[:-1], bins[1:]):
                in_bin = cluster[
                    (cluster[:, 2] >= low)
                    & (cluster[:, 2] < high)
                ]
                if in_bin.shape[0] >= 5:
                    occupied_bins += 1
                    bin_centers.append(
                        (
                            float(np.median(in_bin[:, 0])),
                            float(np.median(in_bin[:, 1])),
                        )
                    )

            if occupied_bins < self.trunk_min_vertical_bins:
                continue

            if bin_centers:
                axis_wander = max(
                    math.hypot(bx - cx, by - cy)
                    for bx, by in bin_centers
                )
            else:
                axis_wander = math.inf

            if axis_wander > self.trunk_max_axis_wander:
                continue

            range_m = math.hypot(cx, cy)
            bearing = math.atan2(cy, cx)
            tof_range = self._tof_at_bearing(bearing)
            tof_support = (
                math.isfinite(tof_range)
                and abs(tof_range - range_m) <= self.tof_candidate_tolerance
            )

            compactness = 1.0 - clamp(
                radius / max(self.trunk_max_radius, 1e-3),
                0.0,
                1.0,
            )
            straightness = 1.0 - clamp(
                axis_wander / max(self.trunk_max_axis_wander, 1e-3),
                0.0,
                1.0,
            )

            score = (
                min(cluster.shape[0], 250) * 0.35
                + height * 45.0
                + occupied_bins * 15.0
                + compactness * 45.0
                + straightness * 55.0
                + (30.0 if tof_support else 0.0)
            )

            strong = (
                height >= 1.00
                and occupied_bins >= max(4, self.trunk_min_vertical_bins)
                and axis_wander <= 0.32
                and radius <= 0.50
                and score >= 165.0
            )

            candidate = TrunkCandidate(
                forward=cx,
                left=cy,
                z_center=cz,
                range_m=range_m,
                bearing=bearing,
                point_count=int(cluster.shape[0]),
                height=height,
                radius=radius,
                occupied_bins=occupied_bins,
                axis_wander=axis_wander,
                score=score,
                strong=strong,
                tof_range=tof_range,
                tof_support=tof_support,
            )
            candidates.append(candidate)
            accepted_centers.append((cx, cy))

            if len(candidates) >= 8:
                break

        return candidates, points

    # =========================================================================
    # Transform dan association
    # =========================================================================

    def _body_to_map(
        self,
        forward: float,
        left: float,
        anchor_xy: Tuple[float, float],
        anchor_yaw: float,
    ) -> Tuple[float, float]:
        c = math.cos(anchor_yaw)
        s = math.sin(anchor_yaw)
        dx = c * forward - s * left
        dy = s * forward + c * left
        return anchor_xy[0] + dx, anchor_xy[1] + dy

    def _transform_points_to_map(
        self,
        points_body: np.ndarray,
        anchor_xy: Tuple[float, float],
        anchor_yaw: float,
        max_points: int,
    ) -> np.ndarray:
        if points_body.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        p = points_body
        if p.shape[0] > max_points:
            stride = max(1, p.shape[0] // max_points)
            p = p[::stride]

        c = math.cos(anchor_yaw)
        s = math.sin(anchor_yaw)

        mx = anchor_xy[0] + c * p[:, 0] - s * p[:, 1]
        my = anchor_xy[1] + s * p[:, 0] + c * p[:, 1]
        mz = np.full_like(mx, self.flight_altitude) + p[:, 2]

        return np.column_stack((mx, my, mz)).astype(np.float32)

    def _inside_orchard(self, x: float, y: float) -> bool:
        m = self.orchard_gate_margin
        return (
            self.orchard_min_x - m <= x <= self.orchard_max_x + m
            and self.orchard_min_y - m <= y <= self.orchard_max_y + m
        )

    def _accumulate_candidates(
        self,
        candidates: Sequence[TrunkCandidate],
        anchor_xy: Tuple[float, float],
        anchor_yaw: float,
        sector: int,
        frame_id: int,
    ) -> None:
        for candidate in candidates:
            mx, my = self._body_to_map(
                candidate.forward,
                candidate.left,
                anchor_xy,
                anchor_yaw,
            )

            if not self._inside_orchard(mx, my):
                self._log_throttle(
                    "outside_candidate",
                    0.5,
                    "warning",
                    f"OUTSIDE_CANDIDATE_REJECT_V21H map=({mx:.2f},{my:.2f}) "
                    f"range={candidate.range_m:.2f}",
                )
                continue

            best: Optional[ScanAccumulator] = None
            best_d = math.inf
            for accumulator in self.scan_accumulators:
                center = robust_xy(accumulator.positions)
                if center is None:
                    continue
                d = math.hypot(mx - center[0], my - center[1])
                if d < best_d:
                    best_d = d
                    best = accumulator

            if best is None or best_d > self.scan_assoc_radius:
                best = ScanAccumulator()
                self.scan_accumulators.append(best)

            best.positions.append((mx, my))
            best.frame_ids.add(frame_id)
            best.sectors.add(sector)
            best.view_yaws.append(anchor_yaw)
            best.strong_hits += int(candidate.strong)
            best.tof_hits += int(candidate.tof_support)
            best.scores.append(candidate.score)
            best.ranges.append(candidate.range_m)

    def _accumulate_close_matches(
        self,
        candidates: Sequence[TrunkCandidate],
        anchor_xy: Tuple[float, float],
        anchor_yaw: float,
        frame_id: int,
    ) -> None:
        target = self._active_track()
        if target is None:
            return

        best: Optional[Tuple[float, float, TrunkCandidate]] = None
        best_d = math.inf

        for candidate in candidates:
            mx, my = self._body_to_map(
                candidate.forward,
                candidate.left,
                anchor_xy,
                anchor_yaw,
            )
            d = math.hypot(mx - target.x, my - target.y)
            if d < best_d:
                best_d = d
                best = (mx, my, candidate)

        match_radius = (
            self.tof3_sweep_match_radius_v21n10
            if self.close_verify_purpose == "TOF3_MINISCAN60_V21N14"
            else self.close_match_radius
        )
        if best is None or best_d > match_radius:
            return

        mx, my, candidate = best
        if self.close_verify_purpose in (
            "TOF3_SWEEP45_V21N4",
            "TOF3_MINISCAN60_V21N14",
        ):
            self.tof3_sweep_frame_view_v21n4[frame_id] = (
                self.tof3_sweep_index_v21n4
            )
        self.close_matches.append(
            (mx, my, frame_id, candidate.score)
        )

    def _nearest_proven_visited_v21n11(
        self,
        x: float,
        y: float,
    ) -> Tuple[Optional[TreeTrack], float]:
        best: Optional[TreeTrack] = None
        best_distance = math.inf

        for track in self.tracks.values():
            if track.state != TrackState.VISITED:
                continue
            if track.tree_id not in self.visited_proof_ids_v21n8:
                continue

            distance = math.hypot(
                x - track.x,
                y - track.y,
            )
            if (
                distance
                <= self.visited_rescan_suppress_radius_v21n11
                and distance < best_distance
            ):
                best = track
                best_distance = distance

        return best, best_distance

    def _suppress_scan_accumulators_near_visited_v21n11(
        self,
    ) -> None:
        """Absorb visited-tree observations before normal scan association."""
        if not self.scan_accumulators:
            return

        kept: List[ScanAccumulator] = []
        suppressed = 0

        for accumulator in self.scan_accumulators:
            center = robust_xy(accumulator.positions)
            if center is None:
                kept.append(accumulator)
                continue

            x, y, _ = center
            visited, distance = self._nearest_proven_visited_v21n11(
                x,
                y,
            )
            if visited is None:
                kept.append(accumulator)
                continue

            frame_count = len(accumulator.frame_ids)
            visited.hits += frame_count
            visited.strong_hits += accumulator.strong_hits
            visited.tof_hits += accumulator.tof_hits
            visited.sectors.update(accumulator.sectors)
            visited.view_yaws.extend(accumulator.view_yaws)
            visited.updated_mono = time.monotonic()
            suppressed += 1

            self.get_logger().info(
                f"SCAN_VISITED_ACCUMULATOR_SUPPRESS_V21N11 "
                f"visited_id={visited.tree_id} "
                f"center=({x:.2f},{y:.2f}) "
                f"distance={distance:.2f} "
                f"frames={frame_count} "
                "action=absorb_evidence_keep_green_position"
            )

        self.scan_accumulators = kept

        if suppressed:
            self.get_logger().info(
                f"SCAN_VISITED_SUPPRESS_SUMMARY_V21N11 "
                f"suppressed_accumulators={suppressed} "
                f"remaining_accumulators={len(kept)}"
            )

    def _suppress_tracks_near_visited_v21n11(self) -> None:
        """Delete yellow/tentative duplicates near proven green landmarks."""
        remove_ids: List[Tuple[int, int, float]] = []

        for track in list(self.tracks.values()):
            if track.state not in (
                TrackState.TENTATIVE,
                TrackState.CONFIRMED,
            ):
                continue

            visited, distance = self._nearest_proven_visited_v21n11(
                track.x,
                track.y,
            )
            if visited is None:
                continue

            remove_ids.append(
                (track.tree_id, visited.tree_id, distance)
            )

        for duplicate_id, visited_id, distance in remove_ids:
            duplicate = self.tracks.get(duplicate_id)
            visited = self.tracks.get(visited_id)
            if duplicate is None or visited is None:
                continue

            visited.hits += duplicate.hits
            visited.strong_hits += duplicate.strong_hits
            visited.tof_hits += duplicate.tof_hits
            visited.sectors.update(duplicate.sectors)
            visited.view_yaws.extend(duplicate.view_yaws)
            visited.updated_mono = time.monotonic()

            self.tracks.pop(duplicate_id, None)
            self.tof_refined_centers_v21n3.pop(
                duplicate_id,
                None,
            )

            if self.active_target_id == duplicate_id:
                self.active_target_id = None
                self.active_standoff_goal = None

            self.get_logger().warning(
                f"POST_SCAN_VISITED_TRACK_SUPPRESS_V21N11 "
                f"duplicate={duplicate_id} visited_id={visited_id} "
                f"distance={distance:.2f} "
                f"green_fixed=({visited.x:.2f},{visited.y:.2f}) "
                "action=remove_duplicate_candidate"
            )

    def _merge_nonvisited_duplicates_v21n13(self) -> None:
        """Strong post-scan one-trunk-one-node dedupe without using SDF."""
        changed = True
        merged = 0

        while changed:
            changed = False
            ids = sorted(self.tracks)
            for index, first_id in enumerate(ids):
                first = self.tracks.get(first_id)
                if first is None or first.state not in (
                    TrackState.TENTATIVE,
                    TrackState.CONFIRMED,
                ):
                    continue

                for second_id in ids[index + 1 :]:
                    second = self.tracks.get(second_id)
                    if second is None or second.state not in (
                        TrackState.TENTATIVE,
                        TrackState.CONFIRMED,
                    ):
                        continue

                    distance = math.hypot(
                        first.x - second.x,
                        first.y - second.y,
                    )
                    if distance > self.post_scan_dedupe_radius_v21n13:
                        continue

                    keep, remove = (
                        (first, second)
                        if first.rank() >= second.rank()
                        else (second, first)
                    )

                    for observation in remove.observations:
                        keep.observations.append(observation)

                    center = robust_xy(list(keep.observations))
                    if center is not None:
                        cx, cy, cmad = center
                        keep.x = cx
                        keep.y = cy
                        keep.position_mad = cmad

                    keep.hits += remove.hits
                    keep.strong_hits += remove.strong_hits
                    keep.tof_hits += remove.tof_hits
                    keep.sectors.update(remove.sectors)
                    keep.view_yaws.extend(remove.view_yaws)
                    keep.updated_mono = time.monotonic()
                    if (
                        remove.state == TrackState.CONFIRMED
                        and keep.state == TrackState.TENTATIVE
                    ):
                        keep.state = TrackState.CONFIRMED

                    self.tracks.pop(remove.tree_id, None)
                    self.tof_refined_centers_v21n3.pop(
                        remove.tree_id,
                        None,
                    )
                    merged += 1
                    changed = True

                    self.get_logger().warning(
                        f"POST_SCAN_STRONG_DEDUPE_V21N13 "
                        f"keep={keep.tree_id} remove={remove.tree_id} "
                        f"distance={distance:.2f} "
                        f"fixed_random_current_batch_unchanged=1"
                    )
                    break

                if changed:
                    break

        if merged:
            self.get_logger().info(
                f"POST_SCAN_STRONG_DEDUPE_SUMMARY_V21N13 "
                f"merged={merged} remaining_tracks={len(self.tracks)}"
            )

    def _start_ghost_target_rescan_v21n13(
        self,
        target: TreeTrack,
        dropout_elapsed: float,
        target_distance: float,
    ) -> None:
        """Invalidate an unobservable map point and remap from current pose."""
        ghost_id = target.tree_id
        old_xy = (float(target.x), float(target.y))

        target.state = TrackState.REJECTED
        target.verify_failures += 1
        target.cooldown_until = 0.0
        target.updated_mono = time.monotonic()

        self.ghost_rescan_pending_v21n13 = True
        self.ghost_rescan_target_id_v21n13 = ghost_id
        self.ghost_rescan_old_xy_v21n13 = old_xy

        self.active_target_id = None
        self.active_standoff_goal = None
        self.tof_stage_target_id = None
        self.tof_approach_stage = "TO_3M"
        self.tof_stage_hold_started = 0.0
        self.tof_final_target_id_v21n3 = None
        self.tof_final_yaw_v21n3 = math.nan
        self.close_verify_purpose = ""
        self.motion_waypoint_xy = None
        self.motion_goal_xy = None
        self.motion_brake_anchor_xy = None
        self.tof_front_history.clear()
        self.tof_recovery_history.clear()
        self.tof_front_guard_history_v21n5.clear()

        self._save_memory()
        self.get_logger().error(
            f"GHOST_TARGET_REJECT_RESCAN_V21N13 "
            f"id={ghost_id} old=({old_xy[0]:.2f},{old_xy[1]:.2f}) "
            f"target_dist={target_distance:.2f} "
            f"dropout={dropout_elapsed:.2f}s "
            "reason=no_tof_no_target_frame "
            "action=full_360_rescan_preserve_random_remainder"
        )
        self._start_scan("ghost_target_relocalize_v21n13")

    def _resume_after_ghost_rescan_v21n13(self) -> bool:
        """Continue old random remainder; rediscovered trees wait next batch."""
        ghost_id = self.ghost_rescan_target_id_v21n13
        old_xy = self.ghost_rescan_old_xy_v21n13

        self.ghost_rescan_pending_v21n13 = False
        self.ghost_rescan_target_id_v21n13 = None
        self.ghost_rescan_old_xy_v21n13 = None
        self.ghost_rescan_count_v21n13 += 1
        self.random_batch_processed_v21n6 += 1

        deferred_new = sorted(
            track.tree_id
            for track in self.tracks.values()
            if (
                track.targetable()
                and track.tree_id
                not in self.random_batch_snapshot_v21n6
            )
        )
        remaining = self._valid_random_batch_ids_v21n6()

        self.get_logger().info(
            f"GHOST_TARGET_RELOCALIZE_DONE_V21N13 "
            f"rejected_id={ghost_id} old={old_xy} "
            f"remaining_fixed_order={remaining} "
            f"new_candidates_deferred={deferred_new}"
        )

        if remaining:
            return self._select_target()

        self.random_batch_active_v21n6 = False
        self.get_logger().info(
            f"GHOST_RESCAN_BATCH_EXHAUSTED_V21N13 "
            f"rejected_id={ghost_id} "
            "action=build_next_random_batch_from_current_scan"
        )
        self._build_random_batch_queue_v21n6()
        return self._select_target()

    def _resume_batch_after_post_visit_scan_v21n11(
        self,
    ) -> bool:
        """Resume the preserved fixed order after the mandatory rescan."""
        visited_id = self.post_visit_rescan_visited_id_v21n11
        self.post_visit_rescan_pending_v21n11 = False
        self.post_visit_rescan_visited_id_v21n11 = None
        self.post_visit_rescan_count_v21n11 += 1

        self.random_batch_processed_v21n6 += 1
        remaining = self._valid_random_batch_ids_v21n6()
        deferred_new_v21n13 = sorted(
            track.tree_id
            for track in self.tracks.values()
            if (
                track.targetable()
                and track.tree_id
                not in self.random_batch_snapshot_v21n6
            )
        )
        if deferred_new_v21n13:
            self.get_logger().info(
                f"POST_VISIT_NEW_CANDIDATES_DEFERRED_V21N13 "
                f"ids={deferred_new_v21n13} "
                "action=wait_until_original_fixed_batch_complete"
            )

        if remaining:
            self.get_logger().info(
                f"POST_VISIT_RESCAN360_DONE_V21N11 "
                f"visited_id={visited_id} "
                f"scan_generation={self.scan_generation} "
                f"processed={self.random_batch_processed_v21n6}/"
                f"{len(self.random_batch_snapshot_v21n6)} "
                f"preserved_remaining={remaining} "
                "action=continue_same_fixed_random_order"
            )
            return self._select_target()

        # The old frozen order is complete. Reuse the scan that has just
        # finished to create the next fixed order; do not rotate twice.
        previous_order = list(self.random_batch_snapshot_v21n6)
        self.random_batch_active_v21n6 = False
        self.get_logger().info(
            f"POST_VISIT_BATCH_EXHAUSTED_V21N11 "
            f"visited_id={visited_id} previous_order={previous_order} "
            "action=build_next_batch_from_current_360_scan"
        )
        self._build_random_batch_queue_v21n6()
        return self._select_target()

    def _finalize_scan(self) -> None:
        promoted: List[int] = []
        tentative: List[int] = []
        dropped = 0

        for accumulator in self.scan_accumulators:
            frame_count = len(accumulator.frame_ids)
            if (
                frame_count < self.new_track_min_frames
                or accumulator.strong_hits < self.new_track_min_strong
            ):
                dropped += 1
                continue

            center = robust_xy(accumulator.positions)
            if center is None:
                dropped += 1
                continue

            x, y, mad = center
            if not self._inside_orchard(x, y):
                dropped += 1
                continue

            track = self._find_nearest_track(
                x,
                y,
                radius=self.tree_assoc_radius,
                include_rejected=False,
            )

            if track is None:
                track = TreeTrack(
                    tree_id=self.next_tree_id,
                    x=x,
                    y=y,
                    state=TrackState.TENTATIVE,
                    created_mono=time.monotonic(),
                    updated_mono=time.monotonic(),
                )
                self.tracks[track.tree_id] = track
                self.next_tree_id += 1

            for px, py in accumulator.positions:
                track.observations.append((px, py))

            robust = robust_xy(list(track.observations))
            if robust is not None:
                rx, ry, track_mad = robust

                if track.state == TrackState.CONFIRMED:
                    shift = math.hypot(rx - track.x, ry - track.y)
                    max_shift = 0.45
                    if shift > max_shift:
                        scale = max_shift / max(shift, 1e-6)
                        rx = track.x + (rx - track.x) * scale
                        ry = track.y + (ry - track.y) * scale

                track.x = rx
                track.y = ry
                track.position_mad = track_mad

            track.hits += frame_count
            track.strong_hits += accumulator.strong_hits
            track.tof_hits += accumulator.tof_hits
            track.sectors.update(accumulator.sectors)
            track.view_yaws.extend(accumulator.view_yaws)
            track.updated_mono = time.monotonic()
            track.last_score = robust_median(accumulator.scores, 0.0)

            view_span = circular_span(track.view_yaws)
            multi_view_ok = (
                len(track.sectors) >= 2
                and view_span >= self.confirm_min_view_span
                and track.hits >= self.confirm_min_total_hits
            )
            tof_geometry_ok = (
                track.tof_hits >= self.confirm_min_tof_hits
                and track.strong_hits >= self.new_track_min_strong
                and track.hits >= self.new_track_min_frames
            )

            if multi_view_ok or tof_geometry_ok:
                if track.state != TrackState.VISITED:
                    track.state = TrackState.CONFIRMED
                promoted.append(track.tree_id)
            else:
                tentative.append(track.tree_id)

        self._merge_duplicate_tracks()
        self._save_memory()

        targetable = [
            t.tree_id
            for t in self.tracks.values()
            if t.targetable()
        ]

        self.get_logger().info(
            f"FAST_SCAN_DONE_V21H reason={self.scan_reason} "
            f"targetable={sorted(targetable)} "
            f"confirmed={sorted(promoted)} tentative={sorted(tentative)} "
            f"dropped={dropped} visited={self._visited_count()}/"
            f"{self.target_tree_count}"
        )

        self.scan_accumulators = []

    def _find_nearest_track(
        self,
        x: float,
        y: float,
        radius: float,
        include_rejected: bool,
    ) -> Optional[TreeTrack]:
        best = None
        best_d = math.inf
        for track in self.tracks.values():
            if not include_rejected and track.state == TrackState.REJECTED:
                continue

            # A green/visited landmark is immutable. A new scan is accumulated
            # into a fresh track first; tight duplicate removal is handled by
            # _merge_duplicate_tracks without moving the visited coordinate.
            if track.state == TrackState.VISITED:
                continue

            d = math.hypot(x - track.x, y - track.y)
            if d <= radius and d < best_d:
                best_d = d
                best = track
        return best

    def _merge_duplicate_tracks(self) -> None:
        changed = True
        while changed:
            changed = False
            ids = sorted(self.tracks)

            for i, first_id in enumerate(ids):
                if first_id not in self.tracks:
                    continue
                first = self.tracks[first_id]

                for second_id in ids[i + 1 :]:
                    if second_id not in self.tracks:
                        continue
                    second = self.tracks[second_id]

                    if (
                        first.state == TrackState.REJECTED
                        or second.state == TrackState.REJECTED
                    ):
                        continue

                    d = math.hypot(
                        first.x - second.x,
                        first.y - second.y,
                    )

                    first_visited = first.state == TrackState.VISITED
                    second_visited = second.state == TrackState.VISITED

                    # Never transfer VISITED across the old 1.65 m merge radius.
                    # Only a very tight duplicate may be removed, while the
                    # original green coordinate remains exactly unchanged.
                    if first_visited or second_visited:
                        if d > self.visited_merge_radius_v21n5:
                            continue

                        if first_visited and not second_visited:
                            keep, remove = first, second
                        elif second_visited and not first_visited:
                            keep, remove = second, first
                        else:
                            keep, remove = (
                                (first, second)
                                if first.tree_id < second.tree_id
                                else (second, first)
                            )

                        keep.hits += remove.hits
                        keep.strong_hits += remove.strong_hits
                        keep.tof_hits += remove.tof_hits
                        keep.sectors.update(remove.sectors)
                        keep.view_yaws.extend(remove.view_yaws)
                        keep.updated_mono = time.monotonic()

                        if self.active_target_id == remove.tree_id:
                            self.active_target_id = None
                            self.active_standoff_goal = None

                        del self.tracks[remove.tree_id]
                        changed = True

                        self.get_logger().info(
                            f"VISITED_POSITION_FROZEN_MERGE_V21N5 "
                            f"keep={keep.tree_id} remove={remove.tree_id} "
                            f"distance={d:.2f} "
                            f"fixed=({keep.x:.2f},{keep.y:.2f})"
                        )
                        break

                    if d > self.tree_merge_radius:
                        continue

                    keep, remove = (
                        (first, second)
                        if first.rank() >= second.rank()
                        else (second, first)
                    )

                    for obs in remove.observations:
                        keep.observations.append(obs)

                    center = robust_xy(list(keep.observations))
                    if center is not None:
                        keep.x, keep.y, keep.position_mad = center

                    keep.hits += remove.hits
                    keep.strong_hits += remove.strong_hits
                    keep.tof_hits += remove.tof_hits
                    keep.sectors.update(remove.sectors)
                    keep.view_yaws.extend(remove.view_yaws)
                    keep.verify_failures = min(
                        keep.verify_failures,
                        remove.verify_failures,
                    )

                    if (
                        remove.state == TrackState.CONFIRMED
                        and keep.state == TrackState.TENTATIVE
                    ):
                        keep.state = TrackState.CONFIRMED

                    if self.active_target_id == remove.tree_id:
                        self.active_target_id = keep.tree_id

                    del self.tracks[remove.tree_id]
                    changed = True

                    self.get_logger().info(
                        f"LANDMARK_DUPLICATE_MERGE_V21H "
                        f"keep={keep.tree_id} remove={remove.tree_id} "
                        f"distance={d:.2f}"
                    )
                    break

                if changed:
                    break

    # =========================================================================
    # ToF
    # =========================================================================

    def _tof_at_bearing(self, bearing: float) -> float:
        if self.tof_ranges is None:
            return math.inf
        if time.monotonic() - self.tof_receipt_mono > 0.60:
            return math.inf
        if abs(self.tof_angle_increment) < 1e-9:
            return math.inf

        angles = (
            self.tof_angle_min
            + np.arange(self.tof_ranges.size) * self.tof_angle_increment
        )
        delta = np.abs(
            np.arctan2(
                np.sin(angles - bearing),
                np.cos(angles - bearing),
            )
        )

        mask = delta <= self.tof_direction_window
        values = self.tof_ranges[mask]
        if values.size == 0:
            return math.inf

        valid = np.isfinite(values)
        valid &= values >= max(self.tof_range_min, 0.05)
        if self.tof_range_max > 0.0:
            valid &= values <= self.tof_range_max

        values = values[valid]
        if values.size == 0:
            return math.inf
        return float(np.median(values))

    def _front_tof(self) -> float:
        """Temporal median; ignore one-frame ToF spikes and branches."""
        now = time.monotonic()
        values = [
            value
            for stamp, value in self.tof_front_history
            if now - stamp <= self.tof_history_window
        ]
        if len(values) < self.tof_min_stable_samples:
            return math.inf

        arr = np.asarray(values, dtype=np.float64)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        if mad > self.tof_max_mad:
            conservative = float(
                np.percentile(
                    arr,
                    clamp(self.tof_unstable_percentile, 5.0, 50.0),
                )
            )
            self._log_throttle(
                "tof_unstable",
                0.8,
                "warning",
                f"TOF_UNSTABLE_CONSERVATIVE_V21H samples={len(values)} "
                f"median={median:.2f} mad={mad:.2f} "
                f"used={conservative:.2f}",
            )
            return conservative
        return median

    def _update_moving_cloud_safety(
        self,
        xyz_raw: np.ndarray,
        now: float,
    ) -> None:
        """Fast corridor-only obstacle detector; never writes tree memory."""
        if (
            not self.moving_cloud_safety_enabled
            or xyz_raw.size == 0
            or self.pose is None
        ):
            self.moving_cloud_front_min = math.inf
            self.last_debug_moving_safety_points = np.empty(
                (0, 3),
                dtype=np.float32,
            )
            return

        points = self._convert_cloud_axes(xyz_raw)
        if points.size == 0:
            self.moving_cloud_front_min = math.inf
            return

        forward = points[:, 0]
        left = points[:, 1]
        up = points[:, 2]
        mask = (
            (forward >= self.moving_cloud_min_forward)
            & (forward <= self.moving_cloud_max_forward)
            & (np.abs(left) <= self.moving_cloud_corridor_half_width)
            & (up >= self.moving_cloud_min_z)
            & (up <= self.moving_cloud_max_z)
        )
        corridor = points[mask]

        if corridor.shape[0] < self.moving_cloud_min_points:
            self.moving_cloud_front_min = math.inf
            self.moving_cloud_front_receipt_mono = now
            self.last_debug_moving_safety_points = np.empty(
                (0, 3),
                dtype=np.float32,
            )
            return

        # Robust nearest range: median of the nearest N points, not one pixel.
        n = min(
            corridor.shape[0],
            max(self.moving_cloud_min_points, 18),
        )
        nearest = np.partition(corridor[:, 0], n - 1)[:n]
        self.moving_cloud_front_min = float(np.median(nearest))
        self.moving_cloud_front_receipt_mono = now

        debug_corridor = corridor
        if debug_corridor.shape[0] > 3500:
            step = int(math.ceil(debug_corridor.shape[0] / 3500.0))
            debug_corridor = debug_corridor[::step]
        self.last_debug_moving_safety_points = self._transform_points_to_map(
            debug_corridor,
            anchor_xy=(self.pose.x_enu, self.pose.y_enu),
            anchor_yaw=self.pose.yaw_enu,
            max_points=3500,
        )

    def _moving_cloud_front(self) -> float:
        if not self.moving_cloud_safety_enabled:
            return math.inf
        if (
            time.monotonic() - self.moving_cloud_front_receipt_mono
            > self.moving_cloud_fresh_age
        ):
            return math.inf
        return float(self.moving_cloud_front_min)

    def _combined_front_range(self) -> Tuple[float, str, float, float]:
        tof = self._front_tof()
        cloud = self._moving_cloud_front()

        if math.isfinite(tof) and math.isfinite(cloud):
            if cloud < tof:
                return cloud, "pointcloud", tof, cloud
            return tof, "tof", tof, cloud
        if math.isfinite(cloud):
            return cloud, "pointcloud", tof, cloud
        return tof, "tof", tof, cloud

    def _enter_brake_hold(
        self,
        reason: str,
        *,
        next_action: str = "select",
        scan_reason: str = "",
        cooldown_target: Optional[TreeTrack] = None,
    ) -> None:
        if self.pose is None:
            return

        if cooldown_target is not None:
            cooldown_target.cooldown_until = (
                time.monotonic() + self.failed_target_cooldown
            )

        self.brake_reason = reason
        self.brake_next_action = next_action
        self.brake_scan_reason = scan_reason
        self.brake_hold_xy = (self.pose.x_enu, self.pose.y_enu)
        self.brake_hold_yaw = self.pose.yaw_enu
        self.hold_xy = self.brake_hold_xy
        self.hold_yaw = self.brake_hold_yaw

        self.active_target_id = None
        self.active_standoff_goal = None
        self._set_state(NavState.BRAKE_HOLD)

        self.get_logger().warning(
            f"BRAKE_HOLD_START_V21H reason={reason} "
            f"next={next_action} hold=({self.brake_hold_xy[0]:.2f},"
            f"{self.brake_hold_xy[1]:.2f})"
        )

    # =========================================================================
    # Scan state machine
    # =========================================================================

    def _start_scan(self, reason: str) -> None:
        if self.pose is None:
            return

        preserve_batch = (
            (
                reason == "after_every_visited_v21n11"
                and self.post_visit_rescan_pending_v21n11
            )
            or (
                reason == "ghost_target_relocalize_v21n13"
                and self.ghost_rescan_pending_v21n13
            )
        )

        if preserve_batch:
            self.get_logger().info(
                f"RANDOM_BATCH_PRESERVE_FOR_RESCAN_V21N11 "
                f"generation={self.random_batch_generation_v21n6} "
                f"processed={self.random_batch_processed_v21n6}/"
                f"{len(self.random_batch_snapshot_v21n6)} "
                f"remaining={list(self.random_batch_queue_v21n6)}"
            )
        else:
            if self.random_batch_queue_v21n6:
                self.get_logger().warning(
                    f"RANDOM_BATCH_ABORT_FOR_SCAN_V21N6 "
                    f"generation={self.random_batch_generation_v21n6} "
                    f"remaining={list(self.random_batch_queue_v21n6)} "
                    f"reason={reason}"
                )
            self.random_batch_queue_v21n6.clear()
            self.random_batch_snapshot_v21n6 = []
            self.random_batch_active_v21n6 = False
            self.random_batch_processed_v21n6 = 0

        self.random_batch_scan_started_mono_v21n6 = time.monotonic()

        self.scan_generation += 1
        self.scan_reason = reason
        self.scan_anchor_xy = (self.pose.x_enu, self.pose.y_enu)
        self.scan_anchor_altitude = self.pose.altitude
        self.scan_base_yaw = self.pose.yaw_enu
        self.scan_sector = 0
        self.scan_accumulators = []
        self.scan_collect_frames = 0

        self._start_scan_sector(0)

        self.get_logger().info(
            f"SCAN_START_V21H reason={reason} generation={self.scan_generation} "
            f"anchor=({self.scan_anchor_xy[0]:.2f},"
            f"{self.scan_anchor_xy[1]:.2f}) "
            f"sectors={self.scan_sector_count} "
            f"settle={self.scan_settle_time:.2f}s "
            f"flush={self.scan_flush_time:.2f}s/"
            f"{self.scan_flush_fresh_frames}fresh "
            f"collect={self.scan_collect_time:.2f}s/"
            f"{self.scan_collect_min_frames}frames"
        )

    def _start_scan_sector(self, sector: int) -> None:
        self.scan_sector = sector
        step = 2.0 * math.pi / float(self.scan_sector_count)
        self.scan_target_yaw = wrap_pi(self.scan_base_yaw + sector * step)
        self.scan_phase_seq = self.cloud_seq
        self._set_state(NavState.SCAN_TURN)

    def _scan_capture_stable(self) -> bool:
        if self.pose is None:
            return False

        drift = math.hypot(
            self.pose.x_enu - self.sector_anchor_xy[0],
            self.pose.y_enu - self.sector_anchor_xy[1],
        )
        alt_error = abs(
            self.pose.altitude - self.sector_anchor_altitude
        )
        yaw_error = abs(
            angle_diff(self.sector_anchor_yaw, self.pose.yaw_enu)
        )
        return (
            self.pose.speed_xy <= self.scan_max_speed
            and drift <= self.scan_max_drift
            and alt_error <= self.scan_max_alt_error
            and yaw_error <= self.scan_yaw_tolerance
        )

    def _close_capture_stable(self) -> bool:
        if self.pose is None:
            return False

        drift = math.hypot(
            self.pose.x_enu - self.close_anchor_xy[0],
            self.pose.y_enu - self.close_anchor_xy[1],
        )
        alt_error = abs(self.pose.altitude - self.close_anchor_altitude)
        yaw_error = abs(
            angle_diff(self.close_anchor_yaw, self.pose.yaw_enu)
        )
        return (
            self.pose.speed_xy <= self.scan_max_speed
            and drift <= 0.25
            and alt_error <= self.scan_max_alt_error
            and yaw_error <= math.radians(5.0)
        )

    def _freeze_sector_anchor(self) -> None:
        samples = [
            sample
            for sample in self.pose_history
            if time.monotonic() - sample.receipt_mono <= 0.80
        ]
        if not samples and self.pose is not None:
            samples = [self.pose]

        self.sector_anchor_xy = (
            robust_median([p.x_enu for p in samples], self.scan_anchor_xy[0]),
            robust_median([p.y_enu for p in samples], self.scan_anchor_xy[1]),
        )
        self.sector_anchor_altitude = robust_median(
            [p.altitude for p in samples],
            self.flight_altitude,
        )

        yaws = [p.yaw_enu for p in samples]
        if yaws:
            sin_med = robust_median([math.sin(y) for y in yaws], 0.0)
            cos_med = robust_median([math.cos(y) for y in yaws], 1.0)
            self.sector_anchor_yaw = math.atan2(sin_med, cos_med)
        else:
            self.sector_anchor_yaw = self.scan_target_yaw

        self.get_logger().info(
            f"SCAN_ANCHOR_FREEZE_V21H sector={self.scan_sector + 1}/"
            f"{self.scan_sector_count} "
            f"xy=({self.sector_anchor_xy[0]:.2f},"
            f"{self.sector_anchor_xy[1]:.2f}) "
            f"yaw={math.degrees(self.sector_anchor_yaw):.1f}deg"
        )

    # =========================================================================
    # Target dan verifikasi
    # =========================================================================

    def _active_track(self) -> Optional[TreeTrack]:
        if self.active_target_id is None:
            return None
        return self.tracks.get(self.active_target_id)

    def _build_random_batch_queue_v21n6(self) -> None:
        """Build a frozen FIFO order from the latest completed scan.

        V21N7 policy:
        - navigation uses CONFIRMED tracks only;
        - the first item is the nearest confirmed track;
        - all remaining confirmed tracks are shuffled exactly once;
        - TENTATIVE tracks stay visible/debuggable but are deferred until a
          later full scan promotes them to CONFIRMED.
        """
        if self.pose is None:
            self.random_batch_queue_v21n6.clear()
            self.random_batch_snapshot_v21n6 = []
            self.random_batch_active_v21n6 = False
            return

        now = time.monotonic()
        confirmed = [
            track
            for track in self.tracks.values()
            if (
                track.targetable()
                and track.cooldown_until <= now
                and track.state == TrackState.CONFIRMED
            )
        ]
        tentative_ids = sorted(
            track.tree_id
            for track in self.tracks.values()
            if (
                track.targetable()
                and track.cooldown_until <= now
                and track.state == TrackState.TENTATIVE
            )
        )

        if confirmed:
            confirmed.sort(
                key=lambda track: (
                    math.hypot(
                        track.x - self.pose.x_enu,
                        track.y - self.pose.y_enu,
                    ),
                    track.tree_id,
                )
            )
            nearest = confirmed[0]
            remainder = [track.tree_id for track in confirmed[1:]]
            self.random.shuffle(remainder)
            ids = [nearest.tree_id] + remainder
            nearest_distance = math.hypot(
                nearest.x - self.pose.x_enu,
                nearest.y - self.pose.y_enu,
            )
        else:
            ids = []
            nearest_distance = math.inf

        self.random_batch_queue_v21n6 = deque(ids)
        self.random_batch_snapshot_v21n6 = list(ids)
        self.random_batch_generation_v21n6 = self.scan_generation
        self.random_batch_processed_v21n6 = 0
        self.random_batch_active_v21n6 = bool(ids)

        self.get_logger().info(
            f"RANDOM_BATCH_BUILD_V21N7 "
            f"scan_generation={self.scan_generation} "
            f"scan_reason={self.scan_reason} "
            f"confirmed_count={len(confirmed)} "
            f"nearest_first={ids[0] if ids else None} "
            f"nearest_dist={nearest_distance:.2f} "
            f"fixed_order={ids} "
            f"tentative_deferred={tentative_ids}"
        )
        if tentative_ids:
            self.get_logger().warning(
                f"INITIAL_SCAN_TENTATIVE_DEFER_V21N7 ids={tentative_ids} "
                "reason=not_confirmed_do_not_navigate_until_next_scan"
            )

    def _valid_random_batch_ids_v21n6(self) -> List[int]:
        now = time.monotonic()
        valid: List[int] = []
        for tree_id in self.random_batch_queue_v21n6:
            track = self.tracks.get(tree_id)
            if (
                track is not None
                and track.targetable()
                and track.cooldown_until <= now
            ):
                valid.append(tree_id)
        return valid

    def _continue_random_batch_or_rescan_v21n6(
        self,
        processed_id: int,
        outcome: str,
    ) -> None:
        """Continue the frozen queue; rescan only after it is exhausted."""
        self.active_target_id = None
        self.active_standoff_goal = None
        self.tof_stage_target_id = None
        self.tof_approach_stage = "TO_3M"
        self.tof_stage_hold_started = 0.0
        self.tof_final_target_id_v21n3 = None
        self.tof_final_yaw_v21n3 = math.nan
        self.close_verify_purpose = ""
        self.motion_waypoint_xy = None
        self.motion_goal_xy = None
        self.motion_brake_anchor_xy = None
        self.tof_front_guard_history_v21n5.clear()

        self.random_batch_processed_v21n6 += 1
        remaining = self._valid_random_batch_ids_v21n6()

        if remaining:
            self.get_logger().info(
                f"RANDOM_BATCH_CONTINUE_V21N7 "
                f"generation={self.random_batch_generation_v21n6} "
                f"processed_id={processed_id} outcome={outcome} "
                f"processed={self.random_batch_processed_v21n6}/"
                f"{len(self.random_batch_snapshot_v21n6)} "
                f"remaining_fixed_order={remaining}"
            )
            self._set_state(NavState.SELECT_TARGET)
            return

        self.random_batch_active_v21n6 = False
        self.get_logger().info(
            f"RANDOM_BATCH_COMPLETE_V21N7 "
            f"generation={self.random_batch_generation_v21n6} "
            f"processed_id={processed_id} outcome={outcome} "
            f"original_order={self.random_batch_snapshot_v21n6} "
            "action=stationary_360deg_8_sector_scan_for_next_batch"
        )
        self._start_scan("fixed_random_batch_complete_v21n6")

    def _select_target(self) -> bool:
        if self.pose is None:
            return False

        now = time.monotonic()

        while self.random_batch_queue_v21n6:
            chosen_id = self.random_batch_queue_v21n6.popleft()
            chosen = self.tracks.get(chosen_id)

            if chosen is None:
                self.random_batch_processed_v21n6 += 1
                self.get_logger().warning(
                    f"RANDOM_BATCH_SKIP_V21N7 id={chosen_id} "
                    "reason=track_removed counted_as_processed=1"
                )
                continue

            if (
                not chosen.targetable()
                or chosen.cooldown_until > now
            ):
                self.random_batch_processed_v21n6 += 1
                self.get_logger().warning(
                    f"RANDOM_BATCH_SKIP_V21N7 id={chosen_id} "
                    f"state={chosen.state.value} "
                    f"cooldown={max(0.0, chosen.cooldown_until - now):.1f}s "
                    "counted_as_processed=1"
                )
                continue

            chosen_distance = math.hypot(
                chosen.x - self.pose.x_enu,
                chosen.y - self.pose.y_enu,
            )

            self.active_target_id = chosen.tree_id
            self.active_target_selected_mono = now
            chosen.last_target_mono = now

            self.tof_approach_stage = "TO_3M"
            self.tof_stage_target_id = chosen.tree_id
            self.tof_stage_hold_started = 0.0
            self.close_verify_purpose = ""
            self.tof3_sweep_index_v21n4 = 0
            self.tof3_sweep_frame_view_v21n4.clear()
            self.motion_waypoint_xy = None
            self.motion_goal_xy = None
            self.motion_brake_anchor_xy = None
            self.route_block_target_id_v21n7 = chosen.tree_id
            self.route_block_started_v21n7 = 0.0
            self.avoid_target_id_v21n9 = None
            self.avoid_phase_v21n9 = ""
            self.avoid_goal_v21n9 = None
            self.verify_retry_target_id_v21n9 = None
            self.verify_retry_goal_v21n9 = None
            self.final_reacquire_target_id_v21n12 = None
            self.final_reacquire_goal_v21n12 = None
            self.final_progress_target_id_v21n12 = None
            self.final_progress_stage_v21n12 = ""
            self.final_progress_last_mono_v21n12 = 0.0

            dx = chosen.x - self.pose.x_enu
            dy = chosen.y - self.pose.y_enu
            distance = max(1e-6, math.hypot(dx, dy))
            ux = dx / distance
            uy = dy / distance

            goal_x = chosen.x - ux * self.standoff_distance
            goal_y = chosen.y - uy * self.standoff_distance
            goal_x, goal_y = self._clamp_goal_to_orchard(
                goal_x,
                goal_y,
            )
            self.active_standoff_goal = (goal_x, goal_y)

            remaining = self._valid_random_batch_ids_v21n6()
            position = self.random_batch_processed_v21n6 + 1
            total = len(self.random_batch_snapshot_v21n6)

            self.get_logger().info(
                f"TARGET_FIXED_RANDOM_BATCH_V21N7 "
                f"generation={self.random_batch_generation_v21n6} "
                f"position={position}/{total} id={chosen.tree_id} "
                f"state={chosen.state.value} dist={chosen_distance:.2f} "
                f"remaining_fixed_order={remaining} "
                f"target=({chosen.x:.2f},{chosen.y:.2f}) "
                f"goal=({goal_x:.2f},{goal_y:.2f})"
            )
            self.get_logger().info(
                f"DETECTED_QUEUE_VISIT_V21N7 id={chosen.tree_id} "
                f"policy=NEAREST_FIRST_THEN_FIXED_RANDOM_CONFIRMED "
                f"scan_generation={self.random_batch_generation_v21n6} "
                f"full_order={self.random_batch_snapshot_v21n6}"
            )
            return True

        self.active_target_id = None
        self.active_standoff_goal = None
        return False

    def _start_verify_retry_v21n9(
        self,
        target: TreeTrack,
        reason: str,
    ) -> None:
        """Retry the same target after a failed stationary verification."""
        assert self.pose is not None

        yaw_to_target = math.atan2(
            target.y - self.pose.y_enu,
            target.x - self.pose.x_enu,
        )
        away_x = self.pose.x_enu - target.x
        away_y = self.pose.y_enu - target.y
        away_norm = math.hypot(away_x, away_y)
        if away_norm < 0.20:
            away_x = -math.cos(yaw_to_target)
            away_y = -math.sin(yaw_to_target)
            away_norm = 1.0

        goal = (
            self.pose.x_enu
            + away_x / away_norm * self.verify_retry_distance_v21n9,
            self.pose.y_enu
            + away_y / away_norm * self.verify_retry_distance_v21n9,
        )
        goal = self._clamp_goal_to_orchard(*goal)

        target.verify_failures += 1
        target.cooldown_until = 0.0
        retries = self.verify_retry_count_v21n9.get(
            target.tree_id,
            0,
        ) + 1
        self.verify_retry_count_v21n9[target.tree_id] = retries

        self.verify_retry_target_id_v21n9 = target.tree_id
        self.verify_retry_goal_v21n9 = goal
        self.verify_retry_yaw_v21n9 = yaw_to_target
        self.verify_retry_started_v21n9 = time.monotonic()

        self.close_matches = []
        self.tof3_sweep_frame_view_v21n4.clear()
        self.tof_approach_stage = "TO_3M"
        self.tof_stage_target_id = target.tree_id
        self.tof_stage_hold_started = 0.0
        self.tof_final_target_id_v21n3 = None
        self.tof_final_yaw_v21n3 = math.nan
        self.tof_front_history.clear()
        self.tof_recovery_history.clear()
        self.tof_front_guard_history_v21n5.clear()
        self.motion_waypoint_xy = None
        self.motion_goal_xy = None
        self.motion_brake_anchor_xy = None

        self.get_logger().warning(
            f"TOF3_VERIFY_RETRY_SAME_TARGET_V21N9 "
            f"id={target.tree_id} retry={retries} reason={reason} "
            f"goal=({goal[0]:.2f},{goal[1]:.2f}) "
            "action=back_away_then_repeat_45deg_same_id"
        )
        self._set_state(NavState.RETRY_VERIFY)

    def _choose_avoid_side_v21n9(
        self,
        target_id: int,
    ) -> Tuple[int, float, float]:
        """Choose the clearer side; alternate when side ToF is unavailable."""
        left = self._tof_at_bearing(math.radians(55.0))
        right = self._tof_at_bearing(math.radians(-55.0))

        if math.isfinite(left) and math.isfinite(right):
            side = 1 if left >= right else -1
        elif math.isfinite(left):
            side = 1
        elif math.isfinite(right):
            side = -1
        else:
            attempts = self.avoid_attempts_v21n9.get(target_id, 0)
            side = 1 if attempts % 2 == 0 else -1

        return side, left, right

    def _start_obstacle_avoid_v21n9(
        self,
        target: TreeTrack,
        target_yaw: float,
        front_distance: float,
    ) -> None:
        """Start backup + sidestep without releasing the active target."""
        assert self.pose is not None

        attempts = self.avoid_attempts_v21n9.get(
            target.tree_id,
            0,
        ) + 1
        self.avoid_attempts_v21n9[target.tree_id] = attempts

        side, left, right = self._choose_avoid_side_v21n9(
            target.tree_id
        )

        backup_goal = (
            self.pose.x_enu
            - math.cos(target_yaw) * self.avoid_backup_distance_v21n9,
            self.pose.y_enu
            - math.sin(target_yaw) * self.avoid_backup_distance_v21n9,
        )
        backup_goal = self._clamp_goal_to_orchard(*backup_goal)

        self.avoid_target_id_v21n9 = target.tree_id
        self.avoid_phase_v21n9 = "BACKUP"
        self.avoid_goal_v21n9 = backup_goal
        self.avoid_yaw_v21n9 = target_yaw
        self.avoid_side_v21n9 = side
        self.avoid_phase_started_v21n9 = time.monotonic()

        self.motion_waypoint_xy = None
        self.motion_goal_xy = None
        self.motion_brake_anchor_xy = None

        self.get_logger().warning(
            f"EARLY_OBSTACLE_AVOID_START_V21N14 "
            f"id={target.tree_id} attempt={attempts} "
            f"front={front_distance:.2f} "
            f"side={'LEFT' if side > 0 else 'RIGHT'} "
            f"left={left:.2f} right={right:.2f} "
            "action=backup_then_sidestep_retry_same_target"
        )
        self._set_state(NavState.AVOID_OBSTACLE)

    def _freeze_close_anchor_v21n10(self) -> None:
        """Freeze verification anchor exactly like the 360 sector anchor."""
        samples = [
            sample
            for sample in self.pose_history
            if time.monotonic() - sample.receipt_mono <= 0.80
        ]
        if not samples and self.pose is not None:
            samples = [self.pose]

        self.close_anchor_xy = (
            robust_median(
                [p.x_enu for p in samples],
                self.close_anchor_xy[0],
            ),
            robust_median(
                [p.y_enu for p in samples],
                self.close_anchor_xy[1],
            ),
        )
        self.close_anchor_altitude = robust_median(
            [p.altitude for p in samples],
            self.flight_altitude,
        )
        yaws = [p.yaw_enu for p in samples]
        if yaws:
            sin_med = robust_median(
                [math.sin(yaw) for yaw in yaws],
                0.0,
            )
            cos_med = robust_median(
                [math.cos(yaw) for yaw in yaws],
                1.0,
            )
            self.close_anchor_yaw = math.atan2(sin_med, cos_med)

        self.get_logger().info(
            f"TOF3_MINISCAN_ANCHOR_V21N14 id={self.active_target_id} "
            f"view={self.tof3_sweep_index_v21n4 + 1}/3 "
            f"xy=({self.close_anchor_xy[0]:.2f},"
            f"{self.close_anchor_xy[1]:.2f}) "
            f"yaw={math.degrees(self.close_anchor_yaw):.1f}deg"
        )

    def _finish_close_verify(self) -> None:
        target = self._active_track()
        if target is None:
            self.close_verify_purpose = ""
            self._set_state(NavState.SELECT_TARGET)
            return

        by_frame: Dict[int, Tuple[float, float, float]] = {}
        for x, y, frame_id, score in self.close_matches:
            existing = by_frame.get(frame_id)
            if existing is None or score > existing[2]:
                by_frame[frame_id] = (x, y, score)

        # Build one robust center for each verification heading.
        per_view_frames: Dict[
            int,
            List[Tuple[int, float, float, float]],
        ] = {0: [], 1: [], 2: []}
        for frame_id, (x, y, score) in by_frame.items():
            view = self.tof3_sweep_frame_view_v21n4.get(
                frame_id,
                -1,
            )
            if view in per_view_frames:
                per_view_frames[view].append(
                    (frame_id, x, y, score)
                )

        per_view_center: Dict[
            int,
            Tuple[float, float, float],
        ] = {}
        for view, rows in per_view_frames.items():
            center = robust_xy(
                [(x, y) for _, x, y, _ in rows]
            )
            if center is not None:
                per_view_center[view] = center

        # Center heading is the synchronization seed. If it has no return,
        # use the view with the most matched frames.
        if 1 in per_view_center:
            seed_view = 1
        elif per_view_center:
            seed_view = max(
                per_view_center,
                key=lambda view: len(per_view_frames[view]),
            )
        else:
            seed_view = -1

        accepted_views: List[int] = []
        if seed_view >= 0:
            seed_x, seed_y, _ = per_view_center[seed_view]
            for view, (vx, vy, _) in per_view_center.items():
                if math.hypot(vx - seed_x, vy - seed_y) <= (
                    self.tof3_sweep_view_assoc_radius_v21n10
                ):
                    accepted_views.append(view)

        accepted_rows: List[
            Tuple[int, float, float, float]
        ] = []
        for view in accepted_views:
            accepted_rows.extend(per_view_frames[view])

        fused = robust_xy(
            [(x, y) for _, x, y, _ in accepted_rows]
        )
        if fused is not None:
            x, y, mad = fused
            shift = math.hypot(
                x - target.x,
                y - target.y,
            )
            observed_range = math.hypot(
                x - self.close_anchor_xy[0],
                y - self.close_anchor_xy[1],
            )
        else:
            x = float(target.x)
            y = float(target.y)
            mad = math.inf
            shift = 0.0
            observed_range = math.hypot(
                x - self.close_anchor_xy[0],
                y - self.close_anchor_xy[1],
            )

        hit_count = len(accepted_rows)
        view_count = len(accepted_views)
        geometry_ok = (
            fused is not None
            and hit_count >= 3
            and shift <= self.tof3_sweep_max_shift_v21n4
            and mad <= self.tof3_sweep_max_mad_v21n4
            and self.tof3_sweep_range_min_v21n10
            <= observed_range
            <= self.tof3_sweep_range_max_v21n10
        )

        if geometry_ok and view_count >= 2:
            sync_mode = "MULTI_VIEW"
        elif geometry_ok:
            sync_mode = "BEST_SINGLE_VIEW"
        else:
            # Hasil mini-scan tidak cukup untuk sinkronisasi posisi.
            # Nilai lama hanya dipakai sebagai penanda mode; pemanggil akan
            # menolak target ini dan melanjutkan fixed random queue.
            sync_mode = "KEEP_OLD_NO_CLOUD_SYNC"
            x = float(target.x)
            y = float(target.y)
            mad = float(target.position_mad)
            shift = 0.0
            observed_range = math.hypot(
                x - self.close_anchor_xy[0],
                y - self.close_anchor_xy[1],
            )

        raw_sync_x_v21n13 = float(x)
        raw_sync_y_v21n13 = float(y)
        raw_shift_v21n13 = math.hypot(
            raw_sync_x_v21n13 - target.x,
            raw_sync_y_v21n13 - target.y,
        )

        if sync_mode == "MULTI_VIEW":
            max_update_v21n13 = self.tof3_multiview_max_update_v21n13
            if raw_shift_v21n13 > max_update_v21n13:
                scale_v21n13 = (
                    max_update_v21n13
                    / max(raw_shift_v21n13, 1e-6)
                )
                x = target.x + (
                    raw_sync_x_v21n13 - target.x
                ) * scale_v21n13
                y = target.y + (
                    raw_sync_y_v21n13 - target.y
                ) * scale_v21n13
                shift = max_update_v21n13
                self.get_logger().warning(
                    f"TOF3_SYNC_CLAMP_V21N13 id={target.tree_id} "
                    f"mode=MULTI_VIEW raw_shift={raw_shift_v21n13:.2f} "
                    f"applied_shift={shift:.2f}"
                )
        elif sync_mode == "BEST_SINGLE_VIEW":
            dx_v21n13 = (
                raw_sync_x_v21n13 - target.x
            ) * self.tof3_singleview_alpha_v21n13
            dy_v21n13 = (
                raw_sync_y_v21n13 - target.y
            ) * self.tof3_singleview_alpha_v21n13
            proposed_v21n13 = math.hypot(
                dx_v21n13,
                dy_v21n13,
            )
            if proposed_v21n13 > self.tof3_singleview_max_update_v21n13:
                scale_v21n13 = (
                    self.tof3_singleview_max_update_v21n13
                    / max(proposed_v21n13, 1e-6)
                )
                dx_v21n13 *= scale_v21n13
                dy_v21n13 *= scale_v21n13

            x = target.x + dx_v21n13
            y = target.y + dy_v21n13
            shift = math.hypot(dx_v21n13, dy_v21n13)
            self.get_logger().warning(
                f"TOF3_SINGLE_VIEW_LIMIT_V21N13 id={target.tree_id} "
                f"raw_shift={raw_shift_v21n13:.2f} "
                f"applied_shift={shift:.2f} "
                f"alpha={self.tof3_singleview_alpha_v21n13:.2f}"
            )

        purpose = self.close_verify_purpose
        self.close_verify_purpose = ""

        if purpose not in (
            "TOF3_CLASSIFY",
            "TOF3_SWEEP45_V21N4",
            "TOF3_MINISCAN60_V21N14",
        ):
            self.get_logger().warning(
                f"CLOSE_VERIFY_PURPOSE_UNKNOWN_V21N10 "
                f"id={target.tree_id} purpose={purpose!r} "
                "action=keep_target"
            )

        self.get_logger().info(
            f"TOF3_MINISCAN_FUSION_V21N14 id={target.tree_id} "
            f"mode={sync_mode} seed_view={seed_view + 1 if seed_view >= 0 else 0} "
            f"accepted_views={[view + 1 for view in accepted_views]} "
            f"view_hits={{1:{len(per_view_frames[0])},"
            f"2:{len(per_view_frames[1])},"
            f"3:{len(per_view_frames[2])}}} "
            f"hits={hit_count} shift={shift:.2f} mad={mad:.2f} "
            f"range={observed_range:.2f}"
        )

        if hit_count <= 0 or seed_view < 0:
            target_distance_v21n14 = math.hypot(
                target.x - self.pose.x_enu,
                target.y - self.pose.y_enu,
            )
            self.close_matches = []
            self.tof3_sweep_frame_view_v21n4.clear()
            self.close_verify_purpose = ""
            self.get_logger().error(
                f"TOF3_NO_FRAME_RELOCALIZE_V21N14 "
                f"id={target.tree_id} "
                f"target_dist={target_distance_v21n14:.2f} "
                "views=0/3 arc=60deg "
                "action=reject_bad_coordinate_full_360_rescan"
            )
            self._start_ghost_target_rescan_v21n13(
                target,
                0.0,
                target_distance_v21n14,
            )
            return

        old_tree = None
        old_distance = math.inf
        old_node_distance = math.inf
        for other in self.tracks.values():
            if (
                other.tree_id == target.tree_id
                or other.state != TrackState.VISITED
            ):
                continue

            d_center = math.hypot(
                x - other.x,
                y - other.y,
            )
            d_node = math.hypot(
                target.x - other.x,
                target.y - other.y,
            )
            if (
                d_center <= self.visited_rescan_suppress_radius_v21n11
                and d_node <= self.visited_rescan_suppress_radius_v21n11
                and d_center < old_distance
            ):
                old_tree = other
                old_distance = d_center
                old_node_distance = d_node

        if old_tree is not None:
            duplicate_id = target.tree_id
            old_tree.updated_mono = time.monotonic()
            old_tree.hits += target.hits
            old_tree.strong_hits += target.strong_hits
            old_tree.tof_hits += target.tof_hits
            self.tracks.pop(duplicate_id, None)
            self.get_logger().info(
                f"TOF_LAYER3_OLD_TREE_V21N14 "
                f"duplicate={duplicate_id} old_id={old_tree.tree_id} "
                f"center_assoc={old_distance:.2f}m "
                f"node_assoc={old_node_distance:.2f}m "
                "action=remove_duplicate_continue_fixed_random_id"
            )
            self.active_target_id = None
            self.active_standoff_goal = None
            self.close_matches = []
            self.tof_stage_target_id = None
            self.tof_final_target_id_v21n3 = None
            self.tof_final_yaw_v21n3 = math.nan
            self.tof_refined_centers_v21n3.pop(
                duplicate_id,
                None,
            )
            self.tof3_sweep_frame_view_v21n4.clear()
            self._save_memory()
            self._continue_random_batch_or_rescan_v21n6(
                duplicate_id,
                "old_tree_duplicate",
            )
            return

        original_x = float(target.x)
        original_y = float(target.y)

        # FINAL_FROZEN_V1_KEEP_OLD_REJECT
        # Mini-scan yang tidak menghasilkan sinkronisasi cloud baru tidak
        # boleh dianggap sebagai verifikasi sukses. Target lama ditolak,
        # antrean random lama dilanjutkan, dan target dapat ditemukan lagi
        # sebagai track baru pada scan 360 derajat berikutnya.
        if sync_mode == "KEEP_OLD_NO_CLOUD_SYNC":
            failed_id = int(target.tree_id)

            target.state = TrackState.REJECTED
            target.verify_failures += 1
            target.cooldown_until = 0.0
            target.updated_mono = time.monotonic()

            self.active_target_id = None
            self.active_standoff_goal = None
            self.close_matches = []
            self.tof_stage_target_id = None
            self.tof_approach_stage = "TO_3M"
            self.tof_stage_hold_started = 0.0
            self.tof_final_target_id_v21n3 = None
            self.tof_final_yaw_v21n3 = math.nan
            self.tof_refined_centers_v21n3.pop(failed_id, None)
            self.tof3_sweep_frame_view_v21n4.clear()

            self.tof_front_history.clear()
            self.tof_recovery_history.clear()
            self.tof_front_guard_history_v21n5.clear()
            self.tof_last_valid_distance = math.inf
            self.tof_last_valid_mono = 0.0
            self.tof_last_valid_target_distance = math.inf
            self.tof_selected_source_v21n2 = "none"
            self.tof_dropout_started = 0.0

            self.motion_waypoint_xy = None
            self.motion_goal_xy = None
            self.motion_brake_anchor_xy = None

            self.get_logger().error(
                f"TOF3_NO_CLOUD_SYNC_REJECT_FINAL id={failed_id} "
                f"old=({original_x:.2f},{original_y:.2f}) "
                "kalman_update=blocked "
                "action=reject_continue_fixed_queue_rescan_after_batch"
            )

            self._save_memory()
            self._continue_random_batch_or_rescan_v21n6(
                failed_id,
                "tof3_no_cloud_sync_rejected",
            )
            return

        target.x = float(x)
        target.y = float(y)

        target.state = TrackState.CONFIRMED
        target.verify_failures = 0
        target.updated_mono = time.monotonic()

        self.tof_refined_centers_v21n3[target.tree_id] = (
            target.x,
            target.y,
            float(mad),
        )

        updated_yaw = math.atan2(
            target.y - self.pose.y_enu,
            target.x - self.pose.x_enu,
        )

        self.close_matches = []
        self.tof3_sweep_frame_view_v21n4.clear()
        self.tof_approach_stage = "TO_2M"
        self.tof_stage_target_id = target.tree_id
        self.tof_final_target_id_v21n3 = target.tree_id
        self.tof_final_yaw_v21n3 = float(updated_yaw)

        self.tof_front_history.clear()
        self.tof_recovery_history.clear()
        self.tof_last_valid_distance = math.inf
        self.tof_last_valid_mono = 0.0
        self.tof_last_valid_target_distance = math.inf
        self.tof_selected_source_v21n2 = "none"
        self.tof_front_guard_history_v21n5.clear()
        self.tof_dropout_started = 0.0

        self.motion_waypoint_xy = None
        self.motion_goal_xy = None
        self.motion_brake_anchor_xy = None

        self.get_logger().info(
            f"TOF3_POSITION_SYNC_V21N14 id={target.tree_id} "
            f"mode={sync_mode} "
            f"old=({original_x:.2f},{original_y:.2f}) "
            f"new=({target.x:.2f},{target.y:.2f}) "
            f"yaw={math.degrees(updated_yaw):.1f}deg "
            "action=tof2_then_tof1"
        )
        self._save_memory()
        self._set_state(NavState.APPROACH)

    def _tof_layer_stats(
        self,
        window: float = 0.85,
    ) -> Tuple[float, float, int]:
        now = time.monotonic()
        values = [
            float(value)
            for stamp, value in self.tof_front_history
            if now - stamp <= window and math.isfinite(value)
        ]
        if not values:
            return math.inf, math.inf, 0
        arr = np.asarray(values, dtype=np.float64)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        return median, mad, int(arr.size)

    def _tof_wide_front_sample_v21n1(self) -> float:
        """Nearest robust ToF cluster inside a wider front window."""
        if self.tof_ranges is None:
            return math.inf
        if time.monotonic() - self.tof_receipt_mono > 0.70:
            return math.inf
        if abs(self.tof_angle_increment) < 1e-9:
            return math.inf

        angles = (
            self.tof_angle_min
            + np.arange(self.tof_ranges.size) * self.tof_angle_increment
        )
        delta = np.abs(np.arctan2(np.sin(angles), np.cos(angles)))
        mask = delta <= self.tof_recovery_window
        indexes = np.flatnonzero(mask)
        if indexes.size == 0:
            return math.inf

        ranges = self.tof_ranges[indexes].astype(np.float64, copy=False)
        valid = np.isfinite(ranges)
        valid &= ranges >= max(self.tof_range_min, 0.05)
        if self.tof_range_max > 0.0:
            valid &= ranges <= self.tof_range_max

        valid_indexes = indexes[valid]
        valid_ranges = ranges[valid]
        if valid_ranges.size == 0:
            return math.inf

        clusters = []
        start = 0
        for i in range(1, valid_indexes.size + 1):
            boundary = (
                i == valid_indexes.size
                or valid_indexes[i] != valid_indexes[i - 1] + 1
            )
            if boundary:
                values = valid_ranges[start:i]
                if values.size >= 2:
                    clusters.append(float(np.median(values)))
                start = i

        if clusters:
            return float(min(clusters))
        return float(np.min(valid_ranges))

    def _tof_front_guard_sample_v21n5(self) -> float:
        """Nearest robust current ToF cluster inside +/-30 degrees."""
        if self.tof_ranges is None:
            return math.inf
        if time.monotonic() - self.tof_receipt_mono > 0.70:
            return math.inf
        if abs(self.tof_angle_increment) < 1e-9:
            return math.inf

        angles = (
            self.tof_angle_min
            + np.arange(self.tof_ranges.size) * self.tof_angle_increment
        )
        delta = np.abs(
            np.arctan2(np.sin(angles), np.cos(angles))
        )
        indexes = np.flatnonzero(
            delta <= self.tof_front_guard_window_v21n5
        )
        if indexes.size == 0:
            return math.inf

        ranges = self.tof_ranges[indexes].astype(
            np.float64,
            copy=False,
        )
        valid = np.isfinite(ranges)
        valid &= ranges >= max(self.tof_range_min, 0.05)
        if self.tof_range_max > 0.0:
            valid &= ranges <= self.tof_range_max

        valid_indexes = indexes[valid]
        valid_ranges = ranges[valid]
        if valid_ranges.size == 0:
            return math.inf

        cluster_medians: List[float] = []
        start = 0
        for i in range(1, valid_indexes.size + 1):
            boundary = (
                i == valid_indexes.size
                or valid_indexes[i] != valid_indexes[i - 1] + 1
            )
            if boundary:
                values = valid_ranges[start:i]
                if values.size >= 2:
                    cluster_medians.append(
                        float(np.median(values))
                    )
                start = i

        if cluster_medians:
            return float(min(cluster_medians))
        return float(np.min(valid_ranges))

    def _tof_front_guard_stable_v21n5(
        self,
    ) -> Tuple[bool, float, float, int, float]:
        now = time.monotonic()
        current = self._tof_front_guard_sample_v21n5()
        if math.isfinite(current):
            self.tof_front_guard_history_v21n5.append(
                (now, current)
            )

        values = [
            float(value)
            for stamp, value in self.tof_front_guard_history_v21n5
            if (
                now - stamp
                <= self.tof_front_guard_history_window_v21n5
                and math.isfinite(value)
            )
        ]
        if not values:
            return False, math.inf, math.inf, 0, current

        arr = np.asarray(values, dtype=np.float64)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        count = int(arr.size)
        stable = (
            count >= 3
            and mad <= self.tof_front_guard_max_mad_v21n5
        )
        return stable, median, mad, count, current

    def _tof_layer_stable(self) -> Tuple[bool, float, float, int]:
        """Return the accepted physical ToF for 3/2/1 transitions.

        The center beam can miss the trunk and read a stable far background.
        V21N2 therefore evaluates the narrow temporal history and the nearest
        robust cluster in a wider front window on every cycle.
        """
        now = time.monotonic()

        narrow_median, narrow_mad, narrow_count = self._tof_layer_stats()
        narrow_stable = (
            narrow_count >= max(3, self.tof_min_stable_samples)
            and math.isfinite(narrow_median)
            and narrow_mad <= self.tof_layer_max_mad
        )

        wide_now = self._tof_wide_front_sample_v21n1()
        if math.isfinite(wide_now):
            self.tof_recovery_history.append((now, wide_now))

        wide_values = [
            float(value)
            for stamp, value in self.tof_recovery_history
            if now - stamp <= 0.70 and math.isfinite(value)
        ]
        wide_median = math.inf
        wide_mad = math.inf
        wide_count = 0
        wide_stable = False
        if wide_values:
            arr = np.asarray(wide_values, dtype=np.float64)
            wide_median = float(np.median(arr))
            wide_mad = float(np.median(np.abs(arr - wide_median)))
            wide_count = int(arr.size)
            wide_stable = (
                wide_count >= 3
                and wide_mad <= max(0.30, self.tof_layer_max_mad)
            )

        candidates = []
        if narrow_stable:
            candidates.append(
                ("narrow", narrow_median, narrow_mad, narrow_count)
            )
        if wide_stable:
            candidates.append(
                ("wide", wide_median, wide_mad, wide_count)
            )

        if not candidates:
            self.tof_selected_source_v21n2 = "none"
            if math.isfinite(wide_median):
                return False, wide_median, wide_mad, wide_count
            return False, narrow_median, narrow_mad, narrow_count

        # The nearest stable front return is the safe physical obstacle.
        source, selected, selected_mad, selected_count = min(
            candidates,
            key=lambda item: item[1],
        )

        recent_near_track = (
            math.isfinite(self.tof_last_valid_distance)
            and self.tof_last_valid_mono > 0.0
            and now - self.tof_last_valid_mono
            <= self.tof_track_recent_time_v21n2
            and self.tof_last_valid_distance <= 6.0
        )

        # Reject a switch from the nearby trunk to a far background return.
        if (
            recent_near_track
            and selected
            > self.tof_last_valid_distance + self.tof_far_jump_margin_v21n2
        ):
            self.tof_selected_source_v21n2 = "far_jump_rejected"
            self._log_throttle(
                "v21n2_far_jump",
                0.45,
                "warning",
                f"TOF_TARGET_JUMP_REJECT_V21N3 "
                f"last={self.tof_last_valid_distance:.2f} "
                f"candidate={selected:.2f} source={source} "
                "action=hold_reacquire_not_background",
            )
            return False, selected, selected_mad, selected_count

        self.tof_selected_source_v21n2 = source
        if source == "wide":
            self._log_throttle(
                "v21n2_wide_select",
                0.70,
                "warning",
                f"TOF_TARGET_WIDE_SELECTED_V21N3 "
                f"tof={selected:.2f} mad={selected_mad:.2f} "
                f"samples={selected_count} "
                f"window={math.degrees(self.tof_recovery_window):.0f}deg",
            )

        return True, selected, selected_mad, selected_count

    def _start_tof3_classification(
        self,
        target: TreeTrack,
        desired_yaw: float,
        tof_median: float,
        tof_mad: float,
        tof_count: int,
    ) -> None:
        assert self.pose is not None

        nominal_yaw = math.atan2(
            target.y - self.pose.y_enu,
            target.x - self.pose.x_enu,
        )

        self.close_anchor_xy = (self.pose.x_enu, self.pose.y_enu)
        self.close_anchor_altitude = self.pose.altitude
        self.close_anchor_yaw = nominal_yaw
        self.close_matches = []
        self.close_collect_frames = 0
        self.close_verify_purpose = "TOF3_MINISCAN60_V21N14"
        self.tof_approach_stage = "CHECK_3M_MINISCAN60"
        self.tof_stage_hold_xy = self.close_anchor_xy
        self.tof_stage_hold_yaw = nominal_yaw

        self.tof3_sweep_nominal_yaw_v21n4 = nominal_yaw
        self.tof3_sweep_index_v21n4 = 0
        self.tof3_sweep_frame_view_v21n4.clear()
        self.tof3_sweep_flush_start_seq_v21n10 = self.cloud_seq
        self.tof3_sweep_flush_started_v21n10 = 0.0

        self.motion_waypoint_xy = None
        self.motion_goal_xy = None
        self.motion_brake_anchor_xy = None

        first_offset = self.tof3_sweep_offsets_v21n4[0]
        self.get_logger().info(
            f"TOF_LAYER3_CHECK_START_V21N14 id={target.tree_id} "
            f"tof={tof_median:.2f} mad={tof_mad:.2f} samples={tof_count} "
            f"center_yaw={math.degrees(nominal_yaw):.1f}deg "
            "action=single_monotonic_60deg_miniscan"
        )
        self.get_logger().info(
            f"TOF3_MINISCAN_TURN_V21N14 id={target.tree_id} "
            f"view=1/3 offset={math.degrees(first_offset):+.1f}deg"
        )
        self._set_state(NavState.CLOSE_SETTLE)

    def _set_tof_hold_stage(
        self,
        stage: str,
        target: TreeTrack,
        desired_yaw: float,
        tof_median: float,
        tof_mad: float,
        tof_count: int,
    ) -> None:
        assert self.pose is not None
        self.tof_approach_stage = stage
        self.tof_stage_hold_xy = (self.pose.x_enu, self.pose.y_enu)
        self.tof_stage_hold_yaw = desired_yaw
        self.tof_stage_hold_started = time.monotonic()
        self.motion_waypoint_xy = None
        self.motion_goal_xy = None
        self.motion_brake_anchor_xy = None
        self.get_logger().info(
            f"TOF_{stage}_START_V21N id={target.tree_id} "
            f"tof={tof_median:.2f} mad={tof_mad:.2f} samples={tof_count} "
            "action=stationary_hold"
        )

    def _finish_tof_safe_visit(
        self,
        target: TreeTrack,
        tof_median: float,
        tof_mad: float,
        tof_count: int,
    ) -> None:
        visited_id = target.tree_id
        # Position was already updated once by the 3 m 60-degree sweep.
        # Do not move the green marker again at the 1 m visit.
        self.tof_refined_centers_v21n3.pop(visited_id, None)

        target.state = TrackState.VISITED
        target.verify_failures = 0
        target.updated_mono = time.monotonic()
        self.visited_proof_ids_v21n8.add(visited_id)

        self.get_logger().info(
            f"TOF_LAYER1_SAFE_VISITED_V21N11 id={visited_id} "
            f"tof={tof_median:.2f} mad={tof_mad:.2f} samples={tof_count} "
            f"local=({target.x:.2f},{target.y:.2f}) marker=green"
        )

        self.active_target_id = None
        self.active_standoff_goal = None
        self.tof_stage_target_id = None
        self.tof_approach_stage = "TO_3M"
        self.tof_stage_hold_started = 0.0
        self.tof_final_target_id_v21n3 = None
        self.tof_final_yaw_v21n3 = math.nan
        self.close_verify_purpose = ""
        self._save_memory()

        if self._visited_count() >= self.target_tree_count:
            self._set_state(NavState.COMPLETE)
            return

        assert self.pose is not None

        # Back away from the verified tree while preserving the final yaw
        # toward it. The vehicle therefore does not rotate its propellers
        # through nearby leaves at the 1 m point.
        away_x = self.pose.x_enu - target.x
        away_y = self.pose.y_enu - target.y
        away_norm = math.hypot(away_x, away_y)
        final_yaw = math.atan2(
            target.y - self.pose.y_enu,
            target.x - self.pose.x_enu,
        )

        if away_norm < 0.20:
            away_x = -math.cos(final_yaw)
            away_y = -math.sin(final_yaw)
            away_norm = 1.0

        ux = away_x / max(away_norm, 1e-6)
        uy = away_y / max(away_norm, 1e-6)
        retreat_goal = (
            self.pose.x_enu
            + ux * self.post_visit_retreat_distance_v21n8,
            self.pose.y_enu
            + uy * self.post_visit_retreat_distance_v21n8,
        )
        retreat_goal = self._clamp_goal_to_orchard(*retreat_goal)

        self.post_visit_retreat_goal_v21n8 = retreat_goal
        self.post_visit_retreat_yaw_v21n8 = final_yaw
        self.post_visit_retreat_started_v21n8 = time.monotonic()
        self.post_visit_retreat_id_v21n8 = visited_id
        self.motion_waypoint_xy = None
        self.motion_goal_xy = None
        self.motion_brake_anchor_xy = None

        self.get_logger().info(
            f"POST_VISIT_RETREAT_START_V21N9 id={visited_id} "
            f"from=({self.pose.x_enu:.2f},{self.pose.y_enu:.2f}) "
            f"goal=({retreat_goal[0]:.2f},{retreat_goal[1]:.2f}) "
            f"distance={self.post_visit_retreat_distance_v21n8:.2f} "
            "yaw=face_tree action=back_away_before_next_turn"
        )
        self._set_state(NavState.RETREAT_VISITED)

    def _command_toward_tof_only(
        self,
        goal_x: float,
        goal_y: float,
        yaw_enu: float,
        step_cap: float,
    ) -> None:
        """Use the V21H micro-waypoint governor with ToF-only distance scaling."""
        saved_enabled = self.moving_cloud_safety_enabled
        saved_step = self.motion_step_distance
        saved_ahead = self.max_command_ahead
        saved_min = self.min_motion_lookahead
        saved_reach = self.motion_waypoint_reach
        try:
            self.moving_cloud_safety_enabled = False
            cap = max(0.04, float(step_cap))
            self.motion_step_distance = min(saved_step, cap)
            self.max_command_ahead = min(saved_ahead, cap)
            self.min_motion_lookahead = min(saved_min, cap)
            self.motion_waypoint_reach = min(
                saved_reach,
                max(0.035, 0.55 * cap),
            )
            self._command_toward(goal_x, goal_y, yaw_enu)
        finally:
            self.moving_cloud_safety_enabled = saved_enabled
            self.motion_step_distance = saved_step
            self.max_command_ahead = saved_ahead
            self.min_motion_lookahead = saved_min
            self.motion_waypoint_reach = saved_reach

    def _command_forward_frozen_v21n3(
        self,
        yaw_enu: float,
        step_cap: float,
    ) -> None:
        """Move only along the bearing frozen at the 3 m gate.

        No target-coordinate correction and no lateral waypoint is allowed
        during the 3 m -> 2 m -> 1 m physical approach.
        """
        assert self.pose is not None
        distance = max(1.0, self.tof_final_forward_distance_v21n3)
        goal_x = self.pose.x_enu + math.cos(yaw_enu) * distance
        goal_y = self.pose.y_enu + math.sin(yaw_enu) * distance
        self._command_toward_tof_only(
            goal_x,
            goal_y,
            yaw_enu,
            step_cap,
        )

    def _tof_near_gate_stats_v21n12(
        self,
        max_distance: float,
    ) -> Tuple[bool, float, float, int]:
        """Stable statistics using only recent physically near returns."""
        now = time.monotonic()
        values = [
            float(value)
            for stamp, value in self.tof_front_guard_history_v21n5
            if (
                now - stamp <= self.tof_near_gate_window_v21n12
                and math.isfinite(value)
                and self.tof_final_min_safe <= value <= max_distance
            )
        ]
        if not values:
            return False, math.inf, math.inf, 0

        arr = np.asarray(values, dtype=np.float64)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        count = int(arr.size)
        stable = (
            count >= self.tof_near_gate_min_samples_v21n12
            and mad <= self.tof_near_gate_max_mad_v21n12
        )
        return stable, median, mad, count

    def _reset_final_progress_v21n12(
        self,
        target_id: int,
        stage: str,
        target_distance: float,
        physical_distance: float,
    ) -> None:
        self.final_progress_target_id_v21n12 = target_id
        self.final_progress_stage_v21n12 = stage
        self.final_progress_best_map_v21n12 = target_distance
        self.final_progress_best_tof_v21n12 = physical_distance
        self.final_progress_last_mono_v21n12 = time.monotonic()

    def _update_final_progress_v21n12(
        self,
        target_id: int,
        stage: str,
        target_distance: float,
        physical_distance: float,
    ) -> float:
        """Return seconds since meaningful map or ToF progress."""
        now = time.monotonic()
        if (
            self.final_progress_target_id_v21n12 != target_id
            or self.final_progress_stage_v21n12 != stage
            or self.final_progress_last_mono_v21n12 <= 0.0
        ):
            self._reset_final_progress_v21n12(
                target_id,
                stage,
                target_distance,
                physical_distance,
            )
            return 0.0

        progressed = False
        if (
            target_distance
            <= self.final_progress_best_map_v21n12
            - self.final_progress_map_delta_v21n12
        ):
            self.final_progress_best_map_v21n12 = target_distance
            progressed = True

        if (
            math.isfinite(physical_distance)
            and physical_distance
            <= self.final_progress_best_tof_v21n12
            - self.final_progress_tof_delta_v21n12
        ):
            self.final_progress_best_tof_v21n12 = physical_distance
            progressed = True

        if progressed:
            self.final_progress_last_mono_v21n12 = now

        return now - self.final_progress_last_mono_v21n12

    def _final_step_cap_v21n12(
        self,
        stage: str,
        physical_distance: float,
    ) -> float:
        """Dynamic step: fast while far, conservative near each ToF gate."""
        if stage == "TO_2M":
            if not math.isfinite(physical_distance):
                return 0.075
            if physical_distance > 4.00:
                return 0.160
            if physical_distance > 3.00:
                return 0.125
            if physical_distance > 2.45:
                return 0.085
            return 0.055

        if stage == "TO_1M":
            if not math.isfinite(physical_distance):
                return 0.045
            if physical_distance > 1.65:
                return 0.075
            if physical_distance > 1.30:
                return 0.055
            return 0.040

        return 0.080

    def _start_final_reacquire_v21n12(
        self,
        target: TreeTrack,
        stage: str,
        reason: str,
    ) -> None:
        """Back away slightly and re-align to the verified target position."""
        assert self.pose is not None

        attempts = self.final_reacquire_attempts_v21n12.get(
            target.tree_id,
            0,
        ) + 1
        self.final_reacquire_attempts_v21n12[target.tree_id] = attempts

        target_yaw = math.atan2(
            target.y - self.pose.y_enu,
            target.x - self.pose.x_enu,
        )
        side = 1.0 if attempts % 2 else -1.0
        back_distance = 0.50
        side_distance = 0.16 if attempts > 1 else 0.0

        goal_x = (
            self.pose.x_enu
            - math.cos(target_yaw) * back_distance
            - math.sin(target_yaw) * side * side_distance
        )
        goal_y = (
            self.pose.y_enu
            - math.sin(target_yaw) * back_distance
            + math.cos(target_yaw) * side * side_distance
        )
        goal_x, goal_y = self._clamp_goal_to_orchard(
            goal_x,
            goal_y,
        )

        self.final_reacquire_target_id_v21n12 = target.tree_id
        self.final_reacquire_stage_v21n12 = stage
        self.final_reacquire_goal_v21n12 = (goal_x, goal_y)
        self.final_reacquire_yaw_v21n12 = target_yaw
        self.final_reacquire_started_v21n12 = time.monotonic()

        self.tof_front_history.clear()
        self.tof_recovery_history.clear()
        self.tof_front_guard_history_v21n5.clear()
        self.tof_last_valid_distance = math.inf
        self.tof_last_valid_mono = 0.0
        self.tof_selected_source_v21n2 = "none"
        self.motion_waypoint_xy = None
        self.motion_goal_xy = None
        self.motion_brake_anchor_xy = None

        self._reset_final_progress_v21n12(
            target.tree_id,
            stage,
            math.hypot(
                target.x - self.pose.x_enu,
                target.y - self.pose.y_enu,
            ),
            math.inf,
        )

        self.get_logger().warning(
            f"FINAL_TOF_REACQUIRE_START_V21N12 "
            f"id={target.tree_id} stage={stage} attempt={attempts} "
            f"reason={reason} "
            f"goal=({goal_x:.2f},{goal_y:.2f}) "
            "action=back_away_realign_same_target"
        )
        self._set_state(NavState.REACQUIRE_FINAL)

    def _horizontal_motion_ready(self, yaw_enu: float) -> bool:
        """Blok gerak XY saat tinggi belum stabil; drone naik/turun dulu di tempat."""
        if self.pose is None:
            return False
        altitude_error = self.flight_altitude - self.pose.altitude
        altitude_bad = (
            abs(altitude_error) > self.altitude_recovery_error
        )
        vertical_motion_bad = (
            abs(self.pose.vz_up) > self.vertical_speed_hold_threshold
            and abs(altitude_error) > self.vertical_speed_hold_min_error
        )
        if altitude_bad or vertical_motion_bad:
            self._publish_position_enu(
                self.pose.x_enu,
                self.pose.y_enu,
                self.flight_altitude,
                yaw_enu,
            )
            self._log_throttle(
                "altitude_recovery",
                0.7,
                "warning",
                f"ALTITUDE_RECOVERY_HOLD_V21H current={self.pose.altitude:.2f} "
                f"target={self.flight_altitude:.2f} "
                f"error={altitude_error:+.2f} vz={self.pose.vz_up:+.2f}",
            )
            return False
        return True

    # =========================================================================
    # Control loop
    # =========================================================================

    def _control_loop(self) -> None:
        now = time.monotonic()

        # Offboard heartbeat harus terus dikirim.
        self._publish_offboard_mode()

        if self.state == NavState.WAIT_DATA:
            if not self.have_pose or not self.have_cloud:
                self._log_throttle(
                    "wait_data",
                    1.0,
                    "warning",
                    f"WAIT_DATA_V21H px4={int(self.have_pose)} "
                    f"cloud={int(self.have_cloud)} tof={int(self.tof_ranges is not None)}",
                )
                return

            assert self.pose is not None
            self.hold_xy = (self.pose.x_enu, self.pose.y_enu)
            self.hold_yaw = self.pose.yaw_enu
            self.prestream_started_mono = now
            self._set_state(NavState.PRESTREAM)
            return

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
                "pose_fault_hold",
                0.6,
                "error",
                f"POSE_FAULT_HOLD_V21H remaining="
                f"{self.pose_fault_until - now:.2f}s "
                f"count={self.pose_fault_count}",
            )
            return

        if self.collision_abort_latched_v21n7:
            self._publish_position_enu(
                self.pose.x_enu,
                self.pose.y_enu,
                self.flight_altitude,
                self.pose.yaw_enu,
            )
            self.motion_waypoint_xy = None
            self.motion_goal_xy = None
            self.motion_brake_anchor_xy = None
            self._log_throttle(
                "v21n8_collision_abort_hold",
                2.00,
                "error",
                "MISSION_COLLISION_ABORT_V21N8 action=hold_last_good_"
                "stop_all_targets_reset_gazebo_required",
            )
            return

        if now - self.pose.receipt_mono > 0.70:
            self._log_throttle(
                "pose_stale",
                0.8,
                "error",
                f"POSE_STALE_HOLD_V21H age={now - self.pose.receipt_mono:.2f}s",
            )
            self._publish_position_enu(
                self.pose.x_enu,
                self.pose.y_enu,
                self.flight_altitude,
                self.pose.yaw_enu,
            )
            return

        if self.state == NavState.PRESTREAM:
            self._publish_position_enu(
                self.hold_xy[0],
                self.hold_xy[1],
                max(self.pose.altitude, 0.10),
                self.hold_yaw,
            )

            if now - self.prestream_started_mono >= 2.0:
                if now - self.last_arm_command_mono >= 0.8:
                    self._send_offboard_and_arm()
                    self.last_arm_command_mono = now

                if now - self.prestream_started_mono >= 3.0:
                    self._set_state(NavState.TAKEOFF)
            return

        if self.state == NavState.TAKEOFF:
            self._publish_position_enu(
                0.0,
                0.0,
                self.flight_altitude,
                self.hold_yaw,
            )

            if (
                abs(self.pose.altitude - self.flight_altitude) <= 0.18
                and self.pose.speed_xy <= 0.25
                and abs(self.pose.vz_up) <= 0.20
            ):
                self.hold_xy = (self.pose.x_enu, self.pose.y_enu)
                self.hold_yaw = self.pose.yaw_enu
                self._set_state(NavState.STABILIZE)
            return

        if self.state == NavState.STABILIZE:
            self._hold_current(self.hold_yaw)
            if self._state_elapsed() >= 1.8:
                self._start_scan("initial_takeoff")
            return

        if self.state == NavState.SCAN_TURN:
            self._publish_position_enu(
                self.scan_anchor_xy[0],
                self.scan_anchor_xy[1],
                self.flight_altitude,
                self.scan_target_yaw,
            )

            yaw_error = abs(
                angle_diff(self.scan_target_yaw, self.pose.yaw_enu)
            )
            drift = math.hypot(
                self.pose.x_enu - self.scan_anchor_xy[0],
                self.pose.y_enu - self.scan_anchor_xy[1],
            )

            if (
                yaw_error <= self.scan_yaw_tolerance
                and self.pose.speed_xy <= self.scan_max_speed
                and drift <= self.scan_max_drift
            ):
                self._set_state(NavState.SCAN_SETTLE)
            else:
                self._log_throttle(
                    "scan_turn",
                    0.8,
                    "info",
                    f"SCAN_TURN_V21H sector={self.scan_sector + 1}/"
                    f"{self.scan_sector_count} "
                    f"yaw_err={math.degrees(yaw_error):.1f}deg "
                    f"drift={drift:.2f} speed={self.pose.speed_xy:.2f}",
                )
            return

        if self.state == NavState.SCAN_SETTLE:
            self._publish_position_enu(
                self.scan_anchor_xy[0],
                self.scan_anchor_xy[1],
                self.flight_altitude,
                self.scan_target_yaw,
            )

            yaw_error = abs(
                angle_diff(self.scan_target_yaw, self.pose.yaw_enu)
            )
            drift = math.hypot(
                self.pose.x_enu - self.scan_anchor_xy[0],
                self.pose.y_enu - self.scan_anchor_xy[1],
            )

            stable = (
                yaw_error <= self.scan_yaw_tolerance
                and drift <= self.scan_max_drift
                and self.pose.speed_xy <= self.scan_max_speed
                and abs(self.pose.altitude - self.flight_altitude)
                <= self.scan_max_alt_error
            )

            if not stable:
                self._set_state(NavState.SCAN_TURN)
                return

            if self._state_elapsed() >= self.scan_settle_time:
                self.scan_phase_seq = self.cloud_seq
                self._set_state(NavState.SCAN_FLUSH)
            return

        if self.state == NavState.SCAN_FLUSH:
            self._publish_position_enu(
                self.scan_anchor_xy[0],
                self.scan_anchor_xy[1],
                self.flight_altitude,
                self.scan_target_yaw,
            )

            fresh = self.cloud_seq - self.scan_phase_seq
            elapsed = self._state_elapsed()

            if (
                elapsed >= self.scan_flush_time
                and fresh >= self.scan_flush_fresh_frames
            ):
                self._freeze_sector_anchor()
                self.scan_collect_frames = 0
                self._set_state(NavState.SCAN_COLLECT)
            else:
                self._log_throttle(
                    "scan_flush",
                    0.8,
                    "info",
                    f"SCAN_FLUSH_WAIT_V21H sector={self.scan_sector + 1}/"
                    f"{self.scan_sector_count} fresh={fresh}/"
                    f"{self.scan_flush_fresh_frames} "
                    f"elapsed={elapsed:.2f}/{self.scan_flush_time:.2f}s",
                )
            return

        if self.state == NavState.SCAN_COLLECT:
            self._publish_position_enu(
                self.sector_anchor_xy[0],
                self.sector_anchor_xy[1],
                self.flight_altitude,
                self.sector_anchor_yaw,
            )

            elapsed = self._state_elapsed()
            if (
                elapsed >= self.scan_collect_time
                and self.scan_collect_frames >= self.scan_collect_min_frames
            ):
                self._set_state(NavState.SCAN_POST)
            else:
                self._log_throttle(
                    "scan_collect",
                    0.8,
                    "info",
                    f"SCAN_COLLECT_V21H sector={self.scan_sector + 1}/"
                    f"{self.scan_sector_count} "
                    f"frames={self.scan_collect_frames}/"
                    f"{self.scan_collect_min_frames} "
                    f"elapsed={elapsed:.2f}/{self.scan_collect_time:.2f}s "
                    f"accumulators={len(self.scan_accumulators)}",
                )
            return

        if self.state == NavState.SCAN_POST:
            self._publish_position_enu(
                self.sector_anchor_xy[0],
                self.sector_anchor_xy[1],
                self.flight_altitude,
                self.sector_anchor_yaw,
            )

            if self._state_elapsed() >= self.scan_post_hold:
                next_sector = self.scan_sector + 1
                if next_sector < self.scan_sector_count:
                    self._start_scan_sector(next_sector)
                else:
                    self._suppress_scan_accumulators_near_visited_v21n11()
                    self._finalize_scan()
                    self._merge_nonvisited_duplicates_v21n13()
                    self._suppress_tracks_near_visited_v21n11()
                    self._save_memory()

                    is_post_visit_scan = (
                        self.scan_reason == "after_every_visited_v21n11"
                        and self.post_visit_rescan_pending_v21n11
                    )
                    is_ghost_rescan = (
                        self.scan_reason == "ghost_target_relocalize_v21n13"
                        and self.ghost_rescan_pending_v21n13
                    )

                    if is_post_visit_scan:
                        selected = (
                            self._resume_batch_after_post_visit_scan_v21n11()
                        )
                    elif is_ghost_rescan:
                        selected = self._resume_after_ghost_rescan_v21n13()
                    else:
                        self._build_random_batch_queue_v21n6()
                        selected = self._select_target()

                    if selected:
                        self.get_logger().info(
                            f"SCAN_TO_TARGET_DIRECT_V21H "
                            f"id={self.active_target_id} "
                            f"reason={self.scan_reason} "
                            "action=align_then_visit_no_extra_scan"
                        )
                        self._set_state(NavState.ALIGN_TARGET)
                    else:
                        self._choose_explore_goal()
                        self._set_state(NavState.EXPLORE_ALIGN)
            return

        if self.state == NavState.REACQUIRE_FINAL:
            target = self._active_track()
            goal = self.final_reacquire_goal_v21n12
            target_id = self.final_reacquire_target_id_v21n12

            if (
                target is None
                or goal is None
                or target_id is None
                or target.tree_id != target_id
            ):
                self.get_logger().error(
                    "FINAL_TOF_REACQUIRE_STATE_INVALID_V21N12 "
                    "action=return_select_target"
                )
                self._set_state(NavState.SELECT_TARGET)
                return

            distance = math.hypot(
                goal[0] - self.pose.x_enu,
                goal[1] - self.pose.y_enu,
            )
            elapsed = now - self.final_reacquire_started_v21n12
            reached = (
                distance <= 0.12
                and self.pose.speed_xy <= 0.28
            )
            timed_out = elapsed >= self.final_reacquire_timeout_v21n12

            if reached or timed_out:
                stage = self.final_reacquire_stage_v21n12
                updated_yaw = math.atan2(
                    target.y - self.pose.y_enu,
                    target.x - self.pose.x_enu,
                )

                self.tof_approach_stage = stage
                self.tof_stage_target_id = target.tree_id
                self.tof_final_target_id_v21n3 = target.tree_id
                self.tof_final_yaw_v21n3 = updated_yaw
                self.final_reacquire_goal_v21n12 = None
                self.final_reacquire_target_id_v21n12 = None
                self.tof_front_history.clear()
                self.tof_recovery_history.clear()
                self.tof_front_guard_history_v21n5.clear()
                self.motion_waypoint_xy = None
                self.motion_goal_xy = None
                self.motion_brake_anchor_xy = None

                self.get_logger().info(
                    f"FINAL_TOF_REACQUIRE_DONE_V21N12 "
                    f"id={target.tree_id} stage={stage} "
                    f"remaining={distance:.2f} timeout={int(timed_out)} "
                    f"yaw={math.degrees(updated_yaw):.1f}deg "
                    "action=align_resume_same_stage"
                )
                self._set_state(NavState.ALIGN_TARGET)
                return

            self._command_toward(
                goal[0],
                goal[1],
                self.final_reacquire_yaw_v21n12,
            )
            self._log_throttle(
                "v21n12_final_reacquire_move",
                0.45,
                "info",
                f"FINAL_TOF_REACQUIRE_MOVE_V21N12 "
                f"id={target.tree_id} "
                f"stage={self.final_reacquire_stage_v21n12} "
                f"remaining={distance:.2f}",
            )
            return

        if self.state == NavState.RETRY_VERIFY:
            target = self._active_track()
            goal = self.verify_retry_goal_v21n9
            retry_id = self.verify_retry_target_id_v21n9

            if (
                target is None
                or retry_id is None
                or target.tree_id != retry_id
                or goal is None
            ):
                self.get_logger().error(
                    "VERIFY_RETRY_STATE_INVALID_V21N9 "
                    "action=hold_same_target_selection"
                )
                self._set_state(NavState.SELECT_TARGET)
                return

            distance = math.hypot(
                goal[0] - self.pose.x_enu,
                goal[1] - self.pose.y_enu,
            )
            elapsed = now - self.verify_retry_started_v21n9
            reached = (
                distance <= 0.12
                and self.pose.speed_xy <= 0.25
            )
            timed_out = elapsed >= self.verify_retry_timeout_v21n9

            if reached or timed_out:
                self._publish_position_enu(
                    self.pose.x_enu,
                    self.pose.y_enu,
                    self.flight_altitude,
                    self.verify_retry_yaw_v21n9,
                )
                self.verify_retry_goal_v21n9 = None
                self.verify_retry_target_id_v21n9 = None
                self.tof_approach_stage = "TO_3M"
                self.tof_stage_target_id = target.tree_id
                self.tof_front_history.clear()
                self.tof_recovery_history.clear()
                self.tof_front_guard_history_v21n5.clear()
                self.motion_waypoint_xy = None
                self.motion_goal_xy = None
                self.motion_brake_anchor_xy = None

                self.get_logger().info(
                    f"VERIFY_RETRY_READY_V21N9 id={target.tree_id} "
                    f"remaining={distance:.2f} timeout={int(timed_out)} "
                    "action=align_and_reverify_same_id"
                )
                self._set_state(NavState.ALIGN_TARGET)
                return

            self._command_toward(
                goal[0],
                goal[1],
                self.verify_retry_yaw_v21n9,
            )
            self._log_throttle(
                "v21n9_verify_retry_move",
                0.50,
                "info",
                f"VERIFY_RETRY_BACKAWAY_V21N9 id={target.tree_id} "
                f"remaining={distance:.2f} "
                "action=keep_same_fixed_random_id",
            )
            return

        if self.state == NavState.AVOID_OBSTACLE:
            target = self._active_track()
            goal = self.avoid_goal_v21n9
            avoid_id = self.avoid_target_id_v21n9

            if (
                target is None
                or avoid_id is None
                or target.tree_id != avoid_id
                or goal is None
            ):
                self.get_logger().error(
                    "OBSTACLE_AVOID_STATE_INVALID_V21N9 "
                    "action=retry_active_target"
                )
                self._set_state(NavState.ALIGN_TARGET)
                return

            distance = math.hypot(
                goal[0] - self.pose.x_enu,
                goal[1] - self.pose.y_enu,
            )
            elapsed = now - self.avoid_phase_started_v21n9
            reached = (
                distance <= 0.14
                and self.pose.speed_xy <= 0.28
            )
            timed_out = elapsed >= self.avoid_phase_timeout_v21n9

            if reached or timed_out:
                if self.avoid_phase_v21n9 == "BACKUP":
                    side = self.avoid_side_v21n9
                    attempts_v21n14 = self.avoid_attempts_v21n9.get(
                        target.tree_id,
                        1,
                    )
                    side_scale_v21n14 = min(
                        1.50,
                        1.0 + 0.25 * max(0, attempts_v21n14 - 1),
                    )
                    side_distance_v21n14 = (
                        self.avoid_sidestep_distance_v21n9
                        * side_scale_v21n14
                    )
                    side_goal = (
                        self.pose.x_enu
                        - math.sin(self.avoid_yaw_v21n9)
                        * side
                        * side_distance_v21n14,
                        self.pose.y_enu
                        + math.cos(self.avoid_yaw_v21n9)
                        * side
                        * side_distance_v21n14,
                    )
                    side_goal = self._clamp_goal_to_orchard(
                        *side_goal
                    )
                    self.avoid_phase_v21n9 = "SIDESTEP"
                    self.avoid_goal_v21n9 = side_goal
                    self.avoid_phase_started_v21n9 = now
                    self.motion_waypoint_xy = None
                    self.motion_goal_xy = None
                    self.motion_brake_anchor_xy = None

                    self.get_logger().info(
                        f"EARLY_OBSTACLE_AVOID_SIDESTEP_V21N14 "
                        f"id={target.tree_id} "
                        f"side={'LEFT' if side > 0 else 'RIGHT'} "
                        f"distance={side_distance_v21n14:.2f} "
                        f"goal=({side_goal[0]:.2f},{side_goal[1]:.2f}) "
                        "action=keep_same_target"
                    )
                    return

                # SIDESTEP complete: align to and retry exactly the same node.
                self.avoid_phase_v21n9 = ""
                self.avoid_goal_v21n9 = None
                self.avoid_target_id_v21n9 = None
                self.tof_approach_stage = "TO_3M"
                self.tof_stage_target_id = target.tree_id
                self.tof_front_history.clear()
                self.tof_recovery_history.clear()
                self.tof_front_guard_history_v21n5.clear()
                self.motion_waypoint_xy = None
                self.motion_goal_xy = None
                self.motion_brake_anchor_xy = None

                self.get_logger().info(
                    f"EARLY_OBSTACLE_AVOID_DONE_V21N14 "
                    f"id={target.tree_id} "
                    f"attempt={self.avoid_attempts_v21n9.get(target.tree_id, 0)} "
                    "action=align_retry_same_fixed_random_id"
                )
                self._set_state(NavState.ALIGN_TARGET)
                return

            self._command_toward(
                goal[0],
                goal[1],
                self.avoid_yaw_v21n9,
            )
            self._log_throttle(
                "v21n9_obstacle_avoid_move",
                0.45,
                "info",
                f"EARLY_OBSTACLE_AVOID_MOVE_V21N14 "
                f"id={target.tree_id} "
                f"phase={self.avoid_phase_v21n9} "
                f"remaining={distance:.2f} "
                "queue_consumed=0",
            )
            return

        if self.state == NavState.RETREAT_VISITED:
            goal = self.post_visit_retreat_goal_v21n8
            visited_id = self.post_visit_retreat_id_v21n8

            if goal is None or visited_id is None:
                self.get_logger().warning(
                    "POST_VISIT_RETREAT_STATE_INVALID_V21N8 "
                    "action=continue_batch"
                )
                self._continue_random_batch_or_rescan_v21n6(
                    -1,
                    "retreat_state_invalid",
                )
                return

            distance = math.hypot(
                goal[0] - self.pose.x_enu,
                goal[1] - self.pose.y_enu,
            )
            elapsed = now - self.post_visit_retreat_started_v21n8
            reached = (
                distance <= 0.13
                and self.pose.speed_xy <= 0.28
            )
            timed_out = elapsed >= self.post_visit_retreat_timeout_v21n8

            if reached or timed_out:
                self._publish_position_enu(
                    self.pose.x_enu,
                    self.pose.y_enu,
                    self.flight_altitude,
                    self.post_visit_retreat_yaw_v21n8,
                )
                self.motion_waypoint_xy = None
                self.motion_goal_xy = None
                self.motion_brake_anchor_xy = None
                self.post_visit_retreat_goal_v21n8 = None
                self.post_visit_retreat_id_v21n8 = None

                self.post_visit_rescan_pending_v21n11 = True
                self.post_visit_rescan_visited_id_v21n11 = visited_id

                self.get_logger().info(
                    f"POST_VISIT_RETREAT_DONE_V21N11 id={visited_id} "
                    f"remaining={distance:.2f} elapsed={elapsed:.2f}s "
                    f"timeout={int(timed_out)} "
                    "action=stationary_360deg_8sector_rescan_before_next_target"
                )
                self.get_logger().info(
                    f"POST_VISIT_RESCAN360_START_V21N11 "
                    f"visited_id={visited_id} "
                    f"preserved_remaining="
                    f"{list(self.random_batch_queue_v21n6)}"
                )
                self._start_scan("after_every_visited_v21n11")
                return

            self._command_toward(
                goal[0],
                goal[1],
                self.post_visit_retreat_yaw_v21n8,
            )
            self._log_throttle(
                "v21n8_post_visit_retreat",
                0.50,
                "info",
                f"POST_VISIT_RETREAT_MOVE_V21N9 id={visited_id} "
                f"remaining={distance:.2f} speed={self.pose.speed_xy:.2f} "
                "yaw=face_tree",
            )
            return

        if self.state == NavState.SELECT_TARGET:
            self._hold_current(self.pose.yaw_enu)

            if self._visited_count() >= self.target_tree_count:
                self._set_state(NavState.COMPLETE)
                return

            if self._select_target():
                self._set_state(NavState.ALIGN_TARGET)
            elif self.random_batch_active_v21n6:
                self.get_logger().info(
                    f"RANDOM_BATCH_EMPTY_ON_SELECT_V21N6 "
                    f"generation={self.random_batch_generation_v21n6} "
                    "action=full_rescan"
                )
                self.random_batch_active_v21n6 = False
                self._start_scan(
                    "fixed_random_batch_empty_on_select_v21n6"
                )
            else:
                self._choose_explore_goal()
                self._set_state(NavState.EXPLORE_ALIGN)
            return

        if self.state == NavState.ALIGN_TARGET:
            target = self._active_track()
            if target is None:
                self._set_state(NavState.SELECT_TARGET)
                return

            desired_yaw = math.atan2(
                target.y - self.pose.y_enu,
                target.x - self.pose.x_enu,
            )
            yaw_error = abs(angle_diff(desired_yaw, self.pose.yaw_enu))

            self._publish_position_enu(
                self.align_anchor_xy[0],
                self.align_anchor_xy[1],
                self.flight_altitude,
                desired_yaw,
            )

            if yaw_error <= self.target_align_tolerance:
                self._set_state(NavState.APPROACH)
            return

        if self.state == NavState.APPROACH:
            target = self._active_track()
            if target is None:
                self._set_state(NavState.SELECT_TARGET)
                return

            if self.tof_stage_target_id != target.tree_id:
                self.tof_approach_stage = "TO_3M"
                self.tof_stage_target_id = target.tree_id
                self.tof_stage_hold_started = 0.0
                self.tof_recovery_history.clear()
                self.tof_dropout_started = 0.0
                self.tof_last_valid_distance = math.inf
                self.tof_last_valid_mono = 0.0
                self.tof_last_valid_target_distance = math.inf
                self.tof_selected_source_v21n2 = "none"
                self.tof_front_guard_history_v21n5.clear()
                self.tof_final_yaw_v21n3 = math.nan
                self.tof_final_target_id_v21n3 = None

            target_distance = math.hypot(
                target.x - self.pose.x_enu,
                target.y - self.pose.y_enu,
            )
            stage = self.tof_approach_stage
            final_stage = stage in (
                "TO_2M",
                "HOLD_2M",
                "TO_1M",
                "HOLD_1M",
            )
            final_lock_valid = (
                final_stage
                and self.tof_final_target_id_v21n3 == target.tree_id
                and math.isfinite(self.tof_final_yaw_v21n3)
            )
            if final_lock_valid:
                desired_yaw = self.tof_final_yaw_v21n3
            else:
                desired_yaw = math.atan2(
                    target.y - self.pose.y_enu,
                    target.x - self.pose.x_enu,
                )
            yaw_error = abs(angle_diff(desired_yaw, self.pose.yaw_enu))

            if not self._horizontal_motion_ready(desired_yaw):
                return

            stable_tof, tof_median, tof_mad, tof_count = (
                self._tof_layer_stable()
            )
            raw_tof = self._tof_at_bearing(0.0)
            tof_for_log = (
                tof_median
                if math.isfinite(tof_median)
                else raw_tof
            )
            cloud_front = self._moving_cloud_front()
            (
                guard_stable,
                guard_median,
                guard_mad,
                guard_count,
                guard_now,
            ) = self._tof_front_guard_stable_v21n5()
            (
                near2_stable_v21n12,
                near2_median_v21n12,
                near2_mad_v21n12,
                near2_count_v21n12,
            ) = self._tof_near_gate_stats_v21n12(
                self.tof_layer2_near_max_v21n12
            )
            (
                near1_stable_v21n12,
                near1_median_v21n12,
                near1_mad_v21n12,
                near1_count_v21n12,
            ) = self._tof_near_gate_stats_v21n12(
                self.tof_layer1_near_max_v21n12
            )

            physical_progress_v21n12 = math.inf
            for candidate_v21n12 in (
                tof_median if stable_tof else math.inf,
                guard_median if guard_stable else math.inf,
                guard_now,
            ):
                if (
                    math.isfinite(candidate_v21n12)
                    and candidate_v21n12 < physical_progress_v21n12
                ):
                    physical_progress_v21n12 = candidate_v21n12

            final_stall_elapsed_v21n12 = 0.0
            if stage in ("TO_2M", "TO_1M"):
                final_stall_elapsed_v21n12 = (
                    self._update_final_progress_v21n12(
                        target.tree_id,
                        stage,
                        target_distance,
                        physical_progress_v21n12,
                    )
                )

            intervening_obstacle_v21n9 = (
                stage == "TO_3M"
                and guard_stable
                and math.isfinite(guard_median)
                and target_distance - guard_median
                >= self.avoid_target_gap_v21n9
            )

            if intervening_obstacle_v21n9:
                self._log_throttle(
                    "v21n9_intervening_obstacle",
                    0.50,
                    "warning",
                    f"INTERVENING_OBSTACLE_TRACK_V21N9 "
                    f"id={target.tree_id} target_dist={target_distance:.2f} "
                    f"front={guard_median:.2f} "
                    f"gap={target_distance - guard_median:.2f} "
                    f"failsafe={self.avoid_trigger_distance_v21n9:.2f} "
                    "action=keep_same_target_avoid_early_at_3p25m",
                )

                if (
                    guard_median
                    <= self.avoid_trigger_distance_v21n9
                ):
                    self._start_obstacle_avoid_v21n9(
                        target,
                        desired_yaw,
                        guard_median,
                    )
                    return

            if stable_tof and math.isfinite(tof_median):
                self.tof_last_valid_distance = float(tof_median)
                self.tof_last_valid_mono = now
                self.tof_last_valid_target_distance = float(target_distance)
                self.tof_dropout_started = 0.0
            elif self.tof_dropout_started <= 0.0:
                self.tof_dropout_started = now

            # Independent current-frame ToF collision guard. It never changes
            # target identity, but it can stop forward motion immediately.
            if (
                math.isfinite(guard_now)
                and guard_now
                <= self.tof_front_guard_hard_stop_v21n5
            ):
                self._publish_position_enu(
                    self.pose.x_enu,
                    self.pose.y_enu,
                    self.flight_altitude,
                    desired_yaw,
                )
                self.motion_waypoint_xy = None
                self.motion_goal_xy = None

                if stage == "TO_1M":
                    self._set_tof_hold_stage(
                        "HOLD_1M",
                        target,
                        desired_yaw,
                        guard_now,
                        0.0,
                        1,
                    )

                self._log_throttle(
                    "v21n5_tof_front_hard_stop",
                    0.30,
                    "error",
                    f"TOF_FRONT_HARD_STOP_V21N5 id={target.tree_id} "
                    f"stage={stage} front={guard_now:.2f} "
                    "action=hold_no_forward",
                )
                return

            # PointCloud never controls the 3/2/1 m layers. It is retained only
            # as a last-resort hold below 0.70 m and never switches target.
            if (
                math.isfinite(cloud_front)
                and cloud_front <= self.tof_cloud_emergency_distance
            ):
                self._publish_position_enu(
                    self.pose.x_enu,
                    self.pose.y_enu,
                    self.flight_altitude,
                    desired_yaw,
                )
                self.motion_waypoint_xy = None
                self.motion_goal_xy = None
                self._log_throttle(
                    "v21n_cloud_emergency",
                    0.45,
                    "error",
                    f"CLOUD_EMERGENCY_HOLD_V21N id={target.tree_id} "
                    f"cloud={cloud_front:.2f} tof={tof_for_log:.2f} "
                    f"stage={self.tof_approach_stage} action=hold_same_target",
                )
                return

            # ToF hard safety: never command closer below this range.
            if (
                stable_tof
                and tof_median <= self.tof_hard_stop
                and self.tof_approach_stage not in ("TO_1M", "HOLD_1M")
            ):
                self._publish_position_enu(
                    self.pose.x_enu,
                    self.pose.y_enu,
                    self.flight_altitude,
                    desired_yaw,
                )
                self.motion_waypoint_xy = None
                self.motion_goal_xy = None
                self._log_throttle(
                    "v21n_tof_hard_hold",
                    0.40,
                    "error",
                    f"TOF_HARD_HOLD_V21N id={target.tree_id} "
                    f"tof={tof_median:.2f} stage={self.tof_approach_stage} "
                    "action=hold_no_target_switch",
                )
                return

            if (
                stage == "TO_3M"
                and target_distance
                > self.tof_near_realign_block_distance_v21n3
                and yaw_error > self.target_realign
            ):
                self._set_state(NavState.ALIGN_TARGET)
                return

            if (
                stage != "TO_3M"
                and yaw_error > math.radians(8.0)
            ):
                self._publish_position_enu(
                    self.pose.x_enu,
                    self.pose.y_enu,
                    self.flight_altitude,
                    desired_yaw,
                )
                self.motion_waypoint_xy = None
                self.motion_goal_xy = None
                self._log_throttle(
                    "v21n3_frozen_yaw_hold",
                    0.40,
                    "warning",
                    f"TOF_FINAL_BEARING_HOLD_V21N5 id={target.tree_id} "
                    f"stage={stage} yaw_error={math.degrees(yaw_error):.1f}deg "
                    "action=rotate_in_place_no_forward",
                )
                return

            # Map distance is still not used to pass a ToF gate. It only
            # detects that the verified node has been reached while the sensor
            # is reading far background. V21N12 backs away and reacquires
            # instead of holding forever.
            if (
                target_distance <= self.tof_map_collision_guard
                and stage in ("TO_3M", "TO_2M", "TO_1M")
                and (
                    not stable_tof
                    or not math.isfinite(tof_median)
                    or tof_median > 3.50
                )
            ):
                if stage in ("TO_2M", "TO_1M"):
                    self.get_logger().error(
                        f"TOF_TRACK_LOST_MAP_RECOVER_V21N12 "
                        f"id={target.tree_id} stage={stage} "
                        f"target_dist={target_distance:.2f} "
                        f"tof={tof_for_log:.2f} "
                        "action=back_away_realign_same_target"
                    )
                    self._start_final_reacquire_v21n12(
                        target,
                        stage,
                        "map_close_tof_background",
                    )
                    return

                self._publish_position_enu(
                    self.pose.x_enu,
                    self.pose.y_enu,
                    self.flight_altitude,
                    desired_yaw,
                )
                self.motion_waypoint_xy = None
                self.motion_goal_xy = None
                self._log_throttle(
                    "v21n12_map_guard_tof3",
                    0.50,
                    "error",
                    f"TOF_TRACK_LOST_MAP_GUARD_V21N12 "
                    f"id={target.tree_id} stage={stage} "
                    f"target_dist={target_distance:.2f} "
                    f"tof={tof_for_log:.2f} action=hold_tof3",
                )
                return

            if stage == "TO_3M":
                if (
                    not intervening_obstacle_v21n9
                    and guard_stable
                    and guard_median
                    <= self.tof_front_guard_layer3_v21n5
                ):
                    self.get_logger().warning(
                        f"TOF_FRONT_GUARD_GATE3_V21N5 id={target.tree_id} "
                        f"front={guard_median:.2f} mad={guard_mad:.2f} "
                        f"samples={guard_count} "
                        "action=stationary_45deg_verify"
                    )
                    self._start_tof3_classification(
                        target,
                        desired_yaw,
                        guard_median,
                        guard_mad,
                        guard_count,
                    )
                    return

                if (
                    not intervening_obstacle_v21n9
                    and stable_tof
                    and tof_median
                    <= self.layer3_object_distance + self.tof_layer_tolerance
                ):
                    self._start_tof3_classification(
                        target,
                        desired_yaw,
                        tof_median,
                        tof_mad,
                        tof_count,
                    )
                    return

                if (
                    not stable_tof
                    and (
                        not math.isfinite(tof_for_log)
                        or tof_for_log
                        <= self.layer3_object_distance + 0.35
                        or (
                            math.isfinite(self.tof_last_valid_distance)
                            and self.tof_last_valid_distance <= 6.0
                        )
                    )
                ):
                    dropout_elapsed = (
                        now - self.tof_dropout_started
                        if self.tof_dropout_started > 0.0
                        else 0.0
                    )
                    last_age = (
                        now - self.tof_last_valid_mono
                        if self.tof_last_valid_mono > 0.0
                        else math.inf
                    )
                    near_gate_latched = (
                        target_distance <= self.tof3_dropout_gate_target_max
                        and self.tof_last_valid_distance
                        <= self.tof3_dropout_gate_last_tof_max
                        and last_age <= 4.0
                    )

                    if (
                        near_gate_latched
                        and dropout_elapsed >= self.tof3_dropout_gate_timeout
                    ):
                        self.get_logger().warning(
                            f"TOF3_LAST_VALID_GATE_V21N2 id={target.tree_id} "
                            f"last_tof={self.tof_last_valid_distance:.2f} "
                            f"last_age={last_age:.2f}s "
                            f"target_dist={target_distance:.2f} "
                            f"dropout={dropout_elapsed:.2f}s "
                            "action=tof3_last_valid_stationary_classify"
                        )
                        self._start_tof3_classification(
                            target,
                            desired_yaw,
                            self.tof_last_valid_distance,
                            max(0.0, tof_mad)
                            if math.isfinite(tof_mad)
                            else 0.0,
                            max(1, tof_count),
                        )
                        return

                    if (
                        target_distance
                        <= self.tof3_ghost_target_max_distance_v21n13
                        and dropout_elapsed
                        >= self.tof3_ghost_dropout_timeout_v21n13
                    ):
                        self._start_ghost_target_rescan_v21n13(
                            target,
                            dropout_elapsed,
                            target_distance,
                        )
                        return

                    sweep = self.tof_recovery_sweep * math.sin(
                        2.0 * math.pi * dropout_elapsed / 2.40
                    )
                    recovery_yaw = desired_yaw + sweep
                    self._publish_position_enu(
                        self.pose.x_enu,
                        self.pose.y_enu,
                        self.flight_altitude,
                        recovery_yaw,
                    )
                    self.motion_waypoint_xy = None
                    self.motion_goal_xy = None
                    self._log_throttle(
                        "v21n2_wait_tof3",
                        0.55,
                        "warning",
                        f"TOF_TARGET_REACQUIRE_V21N3 id={target.tree_id} "
                        f"tof={tof_for_log:.2f} mad={tof_mad:.2f} "
                        f"samples={tof_count} target_dist={target_distance:.2f} "
                        f"dropout={dropout_elapsed:.2f}s "
                        f"sweep={math.degrees(sweep):+.1f}deg "
                        "action=hold_and_reacquire",
                    )
                    return

                step_cap = 0.28 if tof_for_log > 5.0 else 0.16
                self._command_toward_tof_only(
                    target.x,
                    target.y,
                    desired_yaw,
                    step_cap,
                )

            elif stage == "TO_2M":
                if (
                    near2_stable_v21n12
                    and near2_median_v21n12
                    <= self.tof_layer2_near_max_v21n12
                ):
                    self.get_logger().warning(
                        f"TOF2_NEAR_LATCH_GATE_V21N12 "
                        f"id={target.tree_id} "
                        f"tof={near2_median_v21n12:.2f} "
                        f"mad={near2_mad_v21n12:.2f} "
                        f"samples={near2_count_v21n12} "
                        "action=stationary_hold2"
                    )
                    self._set_tof_hold_stage(
                        "HOLD_2M",
                        target,
                        desired_yaw,
                        near2_median_v21n12,
                        near2_mad_v21n12,
                        near2_count_v21n12,
                    )
                    return

                background_mismatch_v21n12 = (
                    target_distance <= 2.30
                    and stable_tof
                    and math.isfinite(tof_median)
                    and tof_median >= max(
                        4.50,
                        target_distance + 2.50,
                    )
                )
                if background_mismatch_v21n12:
                    self.get_logger().error(
                        f"TOF2_BACKGROUND_MISMATCH_V21N12 "
                        f"id={target.tree_id} "
                        f"target_dist={target_distance:.2f} "
                        f"tof={tof_median:.2f} "
                        f"front_now={guard_now:.2f} "
                        "action=reacquire_same_target"
                    )
                    self._start_final_reacquire_v21n12(
                        target,
                        "TO_2M",
                        "far_background_near_verified_node",
                    )
                    return

                if (
                    final_stall_elapsed_v21n12
                    >= self.final_progress_timeout_v21n12
                ):
                    self.get_logger().error(
                        f"TOF2_PROGRESS_STALL_V21N12 "
                        f"id={target.tree_id} "
                        f"stall={final_stall_elapsed_v21n12:.1f}s "
                        f"target_dist={target_distance:.2f} "
                        f"tof={tof_for_log:.2f} "
                        "action=reacquire_same_target"
                    )
                    self._start_final_reacquire_v21n12(
                        target,
                        "TO_2M",
                        "no_map_or_tof_progress",
                    )
                    return

                if (
                    guard_stable
                    and guard_median
                    <= self.tof_front_guard_layer2_v21n5
                ):
                    self.get_logger().warning(
                        f"TOF_FRONT_GUARD_GATE2_V21N5 id={target.tree_id} "
                        f"front={guard_median:.2f} mad={guard_mad:.2f} "
                        f"samples={guard_count} action=stationary_hold"
                    )
                    self._set_tof_hold_stage(
                        "HOLD_2M",
                        target,
                        desired_yaw,
                        guard_median,
                        guard_mad,
                        guard_count,
                    )
                    return

                if (
                    stable_tof
                    and tof_median
                    <= self.layer2_stop_distance + self.tof_layer_tolerance
                ):
                    self._set_tof_hold_stage(
                        "HOLD_2M",
                        target,
                        desired_yaw,
                        tof_median,
                        tof_mad,
                        tof_count,
                    )
                    return

                if (
                    not stable_tof
                    and math.isfinite(tof_for_log)
                    and tof_for_log <= self.layer2_stop_distance + 0.30
                ):
                    self._publish_position_enu(
                        self.pose.x_enu,
                        self.pose.y_enu,
                        self.flight_altitude,
                        desired_yaw,
                    )
                    self.motion_waypoint_xy = None
                    self.motion_goal_xy = None
                    self._log_throttle(
                        "v21n_wait_tof2",
                        0.45,
                        "warning",
                        f"TOF_LAYER2_WAIT_STABLE_V21N id={target.tree_id} "
                        f"tof={tof_for_log:.2f} mad={tof_mad:.2f} "
                        f"samples={tof_count} action=hold",
                    )
                    return

                if not stable_tof:
                    last_age = (
                        now - self.tof_last_valid_mono
                        if self.tof_last_valid_mono > 0.0
                        else math.inf
                    )
                    if (
                        self.tof_last_valid_distance
                        <= self.layer2_stop_distance + self.tof_layer_tolerance
                        and last_age <= 1.50
                    ):
                        self._set_tof_hold_stage(
                            "HOLD_2M",
                            target,
                            desired_yaw,
                            self.tof_last_valid_distance,
                            0.0,
                            max(1, tof_count),
                        )
                        self.get_logger().warning(
                            f"TOF2_LAST_VALID_GATE_V21N2 id={target.tree_id} "
                            f"last_tof={self.tof_last_valid_distance:.2f} "
                            "action=stationary_safety_hold"
                        )
                        return

                    self._publish_position_enu(
                        self.pose.x_enu,
                        self.pose.y_enu,
                        self.flight_altitude,
                        desired_yaw,
                    )
                    self.motion_waypoint_xy = None
                    self.motion_goal_xy = None
                    self._log_throttle(
                        "v21n2_reacquire2",
                        0.45,
                        "warning",
                        f"TOF2_REACQUIRE_V21N2 id={target.tree_id} "
                        f"candidate={tof_for_log:.2f} "
                        f"last={self.tof_last_valid_distance:.2f} "
                        "action=hold_same_target",
                    )
                    return

                final_step_v21n12 = self._final_step_cap_v21n12(
                    "TO_2M",
                    physical_progress_v21n12,
                )
                self._command_forward_frozen_v21n3(
                    desired_yaw,
                    min(
                        self.tof_stage2_step_v21n3,
                        final_step_v21n12,
                    ),
                )

            elif stage == "HOLD_2M":
                self._publish_position_enu(
                    self.tof_stage_hold_xy[0],
                    self.tof_stage_hold_xy[1],
                    self.flight_altitude,
                    self.tof_stage_hold_yaw,
                )
                elapsed = time.monotonic() - self.tof_stage_hold_started
                hold_tof = tof_median
                hold_stable = stable_tof
                hold_source = self.tof_selected_source_v21n2
                last_age = (
                    now - self.tof_last_valid_mono
                    if self.tof_last_valid_mono > 0.0
                    else math.inf
                )
                if (
                    not hold_stable
                    and last_age <= 1.50
                    and self.tof_final_min_safe
                    < self.tof_last_valid_distance
                    <= self.layer2_stop_distance + 0.28
                ):
                    hold_tof = self.tof_last_valid_distance
                    hold_stable = True
                    hold_source = "last_valid_tof"

                if (
                    guard_stable
                    and guard_median
                    <= self.layer2_stop_distance + 0.28
                    and (
                        not math.isfinite(hold_tof)
                        or guard_median < hold_tof
                    )
                ):
                    hold_tof = guard_median
                    hold_stable = True
                    hold_source = "front_guard_v21n5"

                safe = (
                    hold_stable
                    and self.tof_final_min_safe < hold_tof
                    <= max(
                        self.layer2_stop_distance + 0.28,
                        self.tof_layer2_near_max_v21n12,
                    )
                    and self.pose.speed_xy <= 0.14
                    and yaw_error <= math.radians(5.0)
                )
                self._log_throttle(
                    "v21n_hold2",
                    0.45,
                    "info",
                    f"TOF_LAYER2_SAFETY_CHECK_V21N2 id={target.tree_id} "
                    f"tof={hold_tof:.2f} source={hold_source} "
                    f"mad={tof_mad:.2f} samples={tof_count} "
                    f"elapsed={elapsed:.2f}s safe={int(safe)}",
                )
                if safe and elapsed >= self.tof_layer_hold_time:
                    self.tof_approach_stage = "TO_1M"
                    self.tof_stage_hold_started = 0.0
                    self.motion_waypoint_xy = None
                    self.motion_goal_xy = None
                    self.motion_brake_anchor_xy = None
                    self.get_logger().info(
                        f"TOF_LAYER2_SAFE_V21N5 id={target.tree_id} "
                        f"tof={hold_tof:.2f} source={hold_source} "
                        "action=continue_to_tof_1m"
                    )
                    return
                if elapsed >= self.tof_layer_timeout:
                    self.get_logger().warning(
                        f"TOF_LAYER2_UNSAFE_TIMEOUT_V21N id={target.tree_id} "
                        f"tof={tof_median:.2f} mad={tof_mad:.2f} "
                        "action=hold_and_stationary_rescan"
                    )
                    target.cooldown_until = (
                        time.monotonic() + self.failed_target_cooldown
                    )
                    failed_id = target.tree_id
                    self._continue_random_batch_or_rescan_v21n6(
                        failed_id,
                        "tof2_unsafe_timeout",
                    )
                return

            elif stage == "TO_1M":
                if (
                    near1_stable_v21n12
                    and near1_median_v21n12
                    <= self.tof_layer1_near_max_v21n12
                ):
                    self.get_logger().warning(
                        f"TOF1_NEAR_LATCH_GATE_V21N12 "
                        f"id={target.tree_id} "
                        f"tof={near1_median_v21n12:.2f} "
                        f"mad={near1_mad_v21n12:.2f} "
                        f"samples={near1_count_v21n12} "
                        "action=stationary_hold1"
                    )
                    self._set_tof_hold_stage(
                        "HOLD_1M",
                        target,
                        desired_yaw,
                        near1_median_v21n12,
                        near1_mad_v21n12,
                        near1_count_v21n12,
                    )
                    return

                background_mismatch1_v21n12 = (
                    target_distance <= 1.35
                    and stable_tof
                    and math.isfinite(tof_median)
                    and tof_median >= max(
                        3.50,
                        target_distance + 2.00,
                    )
                )
                if background_mismatch1_v21n12:
                    self.get_logger().error(
                        f"TOF1_BACKGROUND_MISMATCH_V21N12 "
                        f"id={target.tree_id} "
                        f"target_dist={target_distance:.2f} "
                        f"tof={tof_median:.2f} "
                        "action=reacquire_same_target"
                    )
                    self._start_final_reacquire_v21n12(
                        target,
                        "TO_1M",
                        "far_background_near_verified_node",
                    )
                    return

                if (
                    final_stall_elapsed_v21n12
                    >= self.final_progress_timeout_v21n12
                ):
                    self.get_logger().error(
                        f"TOF1_PROGRESS_STALL_V21N12 "
                        f"id={target.tree_id} "
                        f"stall={final_stall_elapsed_v21n12:.1f}s "
                        f"target_dist={target_distance:.2f} "
                        f"tof={tof_for_log:.2f} "
                        "action=reacquire_same_target"
                    )
                    self._start_final_reacquire_v21n12(
                        target,
                        "TO_1M",
                        "no_map_or_tof_progress",
                    )
                    return

                if (
                    guard_stable
                    and guard_median
                    <= self.tof_front_guard_layer1_v21n5
                ):
                    self.get_logger().warning(
                        f"TOF_FRONT_GUARD_GATE1_V21N5 id={target.tree_id} "
                        f"front={guard_median:.2f} mad={guard_mad:.2f} "
                        f"samples={guard_count} action=final_stationary_hold"
                    )
                    self._set_tof_hold_stage(
                        "HOLD_1M",
                        target,
                        desired_yaw,
                        guard_median,
                        guard_mad,
                        guard_count,
                    )
                    return

                if (
                    stable_tof
                    and tof_median
                    <= self.layer1_visit_distance + self.tof_layer_tolerance
                ):
                    self._set_tof_hold_stage(
                        "HOLD_1M",
                        target,
                        desired_yaw,
                        tof_median,
                        tof_mad,
                        tof_count,
                    )
                    return

                if (
                    not stable_tof
                    and math.isfinite(tof_for_log)
                    and tof_for_log <= self.layer1_visit_distance + 0.30
                ):
                    self._publish_position_enu(
                        self.pose.x_enu,
                        self.pose.y_enu,
                        self.flight_altitude,
                        desired_yaw,
                    )
                    self.motion_waypoint_xy = None
                    self.motion_goal_xy = None
                    self._log_throttle(
                        "v21n_wait_tof1",
                        0.40,
                        "warning",
                        f"TOF_LAYER1_WAIT_STABLE_V21N id={target.tree_id} "
                        f"tof={tof_for_log:.2f} mad={tof_mad:.2f} "
                        f"samples={tof_count} action=hold",
                    )
                    return

                if not stable_tof:
                    last_age = (
                        now - self.tof_last_valid_mono
                        if self.tof_last_valid_mono > 0.0
                        else math.inf
                    )
                    if (
                        self.tof_last_valid_distance
                        <= self.layer1_visit_distance + self.tof_layer_tolerance
                        and last_age <= 1.20
                    ):
                        self._set_tof_hold_stage(
                            "HOLD_1M",
                            target,
                            desired_yaw,
                            self.tof_last_valid_distance,
                            0.0,
                            max(1, tof_count),
                        )
                        self.get_logger().warning(
                            f"TOF1_LAST_VALID_GATE_V21N2 id={target.tree_id} "
                            f"last_tof={self.tof_last_valid_distance:.2f} "
                            "action=stationary_final_hold"
                        )
                        return

                    self._publish_position_enu(
                        self.pose.x_enu,
                        self.pose.y_enu,
                        self.flight_altitude,
                        desired_yaw,
                    )
                    self.motion_waypoint_xy = None
                    self.motion_goal_xy = None
                    self._log_throttle(
                        "v21n2_reacquire1",
                        0.40,
                        "warning",
                        f"TOF1_REACQUIRE_V21N2 id={target.tree_id} "
                        f"candidate={tof_for_log:.2f} "
                        f"last={self.tof_last_valid_distance:.2f} "
                        "action=hold_same_target",
                    )
                    return

                final_step_v21n12 = self._final_step_cap_v21n12(
                    "TO_1M",
                    physical_progress_v21n12,
                )
                self._command_forward_frozen_v21n3(
                    desired_yaw,
                    min(
                        self.tof_stage1_step_v21n3,
                        final_step_v21n12,
                    ),
                )

            elif stage == "HOLD_1M":
                self._publish_position_enu(
                    self.tof_stage_hold_xy[0],
                    self.tof_stage_hold_xy[1],
                    self.flight_altitude,
                    self.tof_stage_hold_yaw,
                )
                elapsed = time.monotonic() - self.tof_stage_hold_started
                hold_tof = tof_median
                hold_stable = stable_tof
                hold_source = self.tof_selected_source_v21n2
                last_age = (
                    now - self.tof_last_valid_mono
                    if self.tof_last_valid_mono > 0.0
                    else math.inf
                )
                if (
                    not hold_stable
                    and last_age <= 1.20
                    and self.tof_final_min_safe
                    <= self.tof_last_valid_distance
                    <= self.layer1_visit_distance + 0.28
                ):
                    hold_tof = self.tof_last_valid_distance
                    hold_stable = True
                    hold_source = "last_valid_tof"

                if (
                    guard_stable
                    and self.tof_final_min_safe
                    <= guard_median
                    <= self.layer1_visit_distance + 0.28
                    and (
                        not math.isfinite(hold_tof)
                        or guard_median < hold_tof
                    )
                ):
                    hold_tof = guard_median
                    hold_stable = True
                    hold_source = "front_guard_v21n5"

                safe = (
                    hold_stable
                    and self.tof_final_min_safe
                    <= hold_tof
                    <= self.layer1_visit_distance + 0.28
                    and self.pose.speed_xy <= 0.12
                    and yaw_error <= math.radians(5.0)
                )
                self._log_throttle(
                    "v21n_hold1",
                    0.40,
                    "info",
                    f"TOF_LAYER1_SAFETY_CHECK_V21N5 id={target.tree_id} "
                    f"tof={hold_tof:.2f} source={hold_source} "
                    f"mad={tof_mad:.2f} samples={tof_count} "
                    f"elapsed={elapsed:.2f}s safe={int(safe)}",
                )
                if safe and elapsed >= self.tof_layer_hold_time:
                    self._finish_tof_safe_visit(
                        target,
                        hold_tof,
                        tof_mad,
                        tof_count,
                    )
                    return
                if elapsed >= self.tof_layer_timeout:
                    self.get_logger().warning(
                        f"TOF_LAYER1_UNSAFE_TIMEOUT_V21N id={target.tree_id} "
                        f"tof={tof_median:.2f} mad={tof_mad:.2f} "
                        "action=hold_no_visit"
                    )
                    self.tof_stage_hold_started = time.monotonic()
                return

            else:
                self.get_logger().warning(
                    f"TOF_STAGE_RESET_V21N id={target.tree_id} "
                    f"unknown={stage} action=TO_3M"
                )
                self.tof_approach_stage = "TO_3M"
                self.motion_waypoint_xy = None
                self.motion_goal_xy = None
                return

            self._log_throttle(
                "approach_v21n",
                0.65,
                "info",
                f"TOF_APPROACH_V21N5 id={target.tree_id} "
                f"stage={self.tof_approach_stage} "
                f"tof={tof_for_log:.2f} "
                f"source={self.tof_selected_source_v21n2} "
                f"stable={int(stable_tof)} "
                f"mad={tof_mad:.2f} samples={tof_count} "
                f"target_dist={target_distance:.2f} "
                f"cloud_debug={cloud_front:.2f} "
                f"front_guard={guard_median:.2f} "
                f"stall={final_stall_elapsed_v21n12:.1f}s "
                f"frozen={int(final_lock_valid)}",
            )
            return

        if self.state == NavState.CLOSE_SETTLE:
            target = self._active_track()
            if target is None:
                self._set_state(NavState.SELECT_TARGET)
                return

            mini60 = (
                self.close_verify_purpose
                == "TOF3_MINISCAN60_V21N14"
            )
            if mini60:
                offset = self.tof3_sweep_offsets_v21n4[
                    self.tof3_sweep_index_v21n4
                ]
                desired_yaw = (
                    self.tof3_sweep_nominal_yaw_v21n4
                    + offset
                )
                settle_time = self.tof3_sweep_settle_time_v21n4
            else:
                desired_yaw = math.atan2(
                    target.y - self.pose.y_enu,
                    target.x - self.pose.x_enu,
                )
                settle_time = self.close_settle_time

            self.close_anchor_yaw = desired_yaw
            self._publish_position_enu(
                self.close_anchor_xy[0],
                self.close_anchor_xy[1],
                self.flight_altitude,
                desired_yaw,
            )

            yaw_error = abs(
                angle_diff(desired_yaw, self.pose.yaw_enu)
            )
            drift = math.hypot(
                self.pose.x_enu - self.close_anchor_xy[0],
                self.pose.y_enu - self.close_anchor_xy[1],
            )
            stable = (
                self.pose.speed_xy <= self.scan_max_speed
                and drift <= self.scan_max_drift
                and abs(
                    self.pose.altitude - self.close_anchor_altitude
                )
                <= self.scan_max_alt_error
                and yaw_error <= self.scan_yaw_tolerance
            )

            if mini60:
                self._log_throttle(
                    "v21n10_miniscan_settle",
                    0.55,
                    "info",
                    f"TOF3_MINISCAN_SETTLE_V21N14 "
                    f"id={target.tree_id} "
                    f"view={self.tof3_sweep_index_v21n4 + 1}/3 "
                    f"yaw_err={math.degrees(yaw_error):.1f}deg "
                    f"drift={drift:.2f} speed={self.pose.speed_xy:.2f}",
                )

            if stable and self._state_elapsed() >= settle_time:
                if mini60:
                    self.tof3_sweep_flush_start_seq_v21n10 = (
                        self.cloud_seq
                    )
                    self.tof3_sweep_flush_started_v21n10 = now
                    self._set_state(NavState.CLOSE_FLUSH)
                else:
                    self.close_anchor_altitude = self.pose.altitude
                    self.close_anchor_yaw = self.pose.yaw_enu
                    self.close_collect_frames = 0
                    self.close_matches = []
                    self._set_state(NavState.CLOSE_COLLECT)
            return

        if self.state == NavState.CLOSE_FLUSH:
            target = self._active_track()
            if target is None:
                self._set_state(NavState.SELECT_TARGET)
                return

            self._publish_position_enu(
                self.close_anchor_xy[0],
                self.close_anchor_xy[1],
                self.flight_altitude,
                self.close_anchor_yaw,
            )

            elapsed = (
                now - self.tof3_sweep_flush_started_v21n10
            )
            fresh = max(
                0,
                self.cloud_seq
                - self.tof3_sweep_flush_start_seq_v21n10,
            )
            enough = (
                elapsed >= self.tof3_sweep_flush_time_v21n10
                and fresh >= self.tof3_sweep_flush_fresh_v21n10
            )
            timeout = elapsed >= max(
                2.40,
                self.tof3_sweep_flush_time_v21n10 + 1.0,
            )

            self._log_throttle(
                "v21n10_miniscan_flush",
                0.55,
                "info",
                f"TOF3_MINISCAN_FLUSH_V21N14 "
                f"id={target.tree_id} "
                f"view={self.tof3_sweep_index_v21n4 + 1}/3 "
                f"fresh={fresh}/"
                f"{self.tof3_sweep_flush_fresh_v21n10} "
                f"elapsed={elapsed:.2f}/"
                f"{self.tof3_sweep_flush_time_v21n10:.2f}s",
            )

            if enough or timeout:
                self._freeze_close_anchor_v21n10()
                self.close_collect_frames = 0
                self._set_state(NavState.CLOSE_COLLECT)
            return

        if self.state == NavState.CLOSE_COLLECT:
            self._publish_position_enu(
                self.close_anchor_xy[0],
                self.close_anchor_xy[1],
                self.flight_altitude,
                self.close_anchor_yaw,
            )

            mini60 = (
                self.close_verify_purpose
                == "TOF3_MINISCAN60_V21N14"
            )
            if not mini60:
                if self._state_elapsed() >= self.close_collect_time:
                    self._finish_close_verify()
                else:
                    self._log_throttle(
                        "close_collect",
                        0.7,
                        "info",
                        f"CLOSE_VERIFY_COLLECT_V21H "
                        f"target={self.active_target_id} "
                        f"frames={self.close_collect_frames} "
                        f"matches={len(self.close_matches)}",
                    )
                return

            elapsed = self._state_elapsed()
            ready = (
                elapsed >= self.tof3_sweep_collect_time_v21n4
                and self.close_collect_frames
                >= self.tof3_sweep_collect_min_frames_v21n10
            )
            timeout = (
                elapsed
                >= self.tof3_sweep_collect_timeout_v21n10
            )

            current_view = self.tof3_sweep_index_v21n4
            view_hits = sum(
                1
                for _, _, frame_id, _ in self.close_matches
                if self.tof3_sweep_frame_view_v21n4.get(
                    frame_id,
                    -1,
                )
                == current_view
            )

            self._log_throttle(
                "v21n10_miniscan_collect",
                0.55,
                "info",
                f"TOF3_MINISCAN_COLLECT_V21N14 "
                f"id={self.active_target_id} "
                f"view={current_view + 1}/3 "
                f"offset={math.degrees(self.tof3_sweep_offsets_v21n4[current_view]):+.1f}deg "
                f"frames={self.close_collect_frames}/"
                f"{self.tof3_sweep_collect_min_frames_v21n10} "
                f"matches={view_hits}",
            )

            if not ready and not timeout:
                return

            self.get_logger().info(
                f"TOF3_MINISCAN_VIEW_DONE_V21N14 "
                f"id={self.active_target_id} "
                f"view={current_view + 1}/3 "
                f"offset={math.degrees(self.tof3_sweep_offsets_v21n4[current_view]):+.1f}deg "
                f"frames={self.close_collect_frames} "
                f"matches={view_hits} timeout={int(timeout)}"
            )

            next_view = current_view + 1
            if next_view < len(self.tof3_sweep_offsets_v21n4):
                self.tof3_sweep_index_v21n4 = next_view
                self.close_collect_frames = 0
                next_offset = self.tof3_sweep_offsets_v21n4[
                    next_view
                ]
                self.get_logger().info(
                    f"TOF3_MINISCAN_TURN_V21N14 "
                    f"id={self.active_target_id} "
                    f"view={next_view + 1}/3 "
                    f"offset={math.degrees(next_offset):+.1f}deg"
                )
                self._set_state(NavState.CLOSE_SETTLE)
                return

            self.get_logger().info(
                f"TOF3_MINISCAN_COMPLETE_V21N14 "
                f"id={self.active_target_id} arc=60.0deg "
                "action=fuse_and_sync_once"
            )
            self._finish_close_verify()
            return

        if self.state == NavState.EXPLORE_ALIGN:
            if self.explore_goal is None:
                self._choose_explore_goal()

            assert self.explore_goal is not None
            desired_yaw = math.atan2(
                self.explore_goal[1] - self.pose.y_enu,
                self.explore_goal[0] - self.pose.x_enu,
            )
            yaw_error = abs(angle_diff(desired_yaw, self.pose.yaw_enu))

            self._publish_position_enu(
                self.align_anchor_xy[0],
                self.align_anchor_xy[1],
                self.flight_altitude,
                desired_yaw,
            )

            if yaw_error <= self.target_align_tolerance:
                self._set_state(NavState.EXPLORE_MOVE)
            return

        if self.state == NavState.EXPLORE_MOVE:
            if self.explore_goal is None:
                self._set_state(NavState.SELECT_TARGET)
                return

            goal_distance = math.hypot(
                self.explore_goal[0] - self.pose.x_enu,
                self.explore_goal[1] - self.pose.y_enu,
            )
            desired_yaw = math.atan2(
                self.explore_goal[1] - self.pose.y_enu,
                self.explore_goal[0] - self.pose.x_enu,
            )
            front_range, front_source, front_tof, cloud_front = (
                self._combined_front_range()
            )

            if not self._horizontal_motion_ready(desired_yaw):
                return

            explore_limit = (
                self.moving_cloud_unmapped_stop
                if front_source == "pointcloud"
                else self.tof_unmapped_stop
            )
            if (
                math.isfinite(front_range)
                and front_range <= explore_limit
            ):
                self.get_logger().warning(
                    f"EXPLORE_FRONT_STOP_V21H source={front_source} "
                    f"range={front_range:.2f} tof={front_tof:.2f} "
                    f"cloud={cloud_front:.2f} action=brake_then_scan"
                )
                self.explore_goal = None
                self._enter_brake_hold(
                    f"explore_front_{front_source}",
                    next_action="scan",
                    scan_reason="explore_front_stop",
                )
                return

            if goal_distance <= self.explore_arrival_distance:
                self.explore_goal = None
                self._start_scan("random_explore_arrival")
                return

            self._command_toward(
                self.explore_goal[0],
                self.explore_goal[1],
                desired_yaw,
            )

            self._log_throttle(
                "explore_move",
                0.8,
                "info",
                f"EXPLORE_RANDOM_V21H goal=({self.explore_goal[0]:.2f},"
                f"{self.explore_goal[1]:.2f}) dist={goal_distance:.2f} "
                f"tof={front_tof if math.isfinite(front_tof) else float('inf'):.2f} "
                f"cloud={cloud_front if math.isfinite(cloud_front) else float('inf'):.2f}",
            )
            return

        if self.state == NavState.BRAKE_HOLD:
            self._publish_position_enu(
                self.brake_hold_xy[0],
                self.brake_hold_xy[1],
                self.flight_altitude,
                self.brake_hold_yaw,
            )
            stable = (
                self.pose.speed_xy <= 0.18
                and abs(self.pose.vz_up) <= 0.25
            )
            if (
                self._state_elapsed() >= self.brake_hold_time
                and stable
            ):
                action = self.brake_next_action
                scan_reason = self.brake_scan_reason
                self.get_logger().info(
                    f"BRAKE_HOLD_DONE_V21H reason={self.brake_reason} "
                    f"action={action}"
                )
                if action == "scan":
                    self._start_scan(scan_reason or "brake_hold_scan")
                else:
                    self._set_state(NavState.SELECT_TARGET)
            else:
                self._log_throttle(
                    "brake_hold",
                    0.6,
                    "warning",
                    f"BRAKE_HOLD_V21H reason={self.brake_reason} "
                    f"elapsed={self._state_elapsed():.2f}/"
                    f"{self.brake_hold_time:.2f}s "
                    f"speed={self.pose.speed_xy:.2f} vz={self.pose.vz_up:+.2f}",
                )
            return

        if self.state == NavState.HOLD:
            self._publish_position_enu(
                self.hold_xy[0],
                self.hold_xy[1],
                self.flight_altitude,
                self.hold_yaw,
            )
            return

        if self.state == NavState.COMPLETE:
            self._hold_current(self.pose.yaw_enu)
            self._log_throttle(
                "complete",
                2.0,
                "info",
                f"MISSION_COMPLETE_V21H visited={self._visited_count()}/"
                f"{self.target_tree_count}",
            )
            return

    # =========================================================================
    # Setpoint PX4
    # =========================================================================

    def _timestamp_us(self) -> int:
        return int(self.get_clock().now().nanoseconds // 1000)

    def _publish_offboard_mode(self) -> None:
        msg = OffboardControlMode()
        msg.timestamp = self._timestamp_us()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_pub.publish(msg)

    def _publish_position_enu(
        self,
        x_enu: float,
        y_enu: float,
        altitude: float,
        yaw_enu: float,
    ) -> None:
        if not finite(x_enu, y_enu, altitude, yaw_enu):
            self.get_logger().error(
                "INVALID_SETPOINT_BLOCK_V21H non-finite setpoint"
            )
            return

        # V21H final safety net: during translation, a commanded XY point may
        # never jump far away from the current vehicle pose. This is separate
        # from the normal latched micro-waypoint logic and only catches a
        # regression / bad coordinate conversion before it reaches PX4.
        if (
            self.pose is not None
            and self.state in (
                NavState.APPROACH,
                NavState.EXPLORE_MOVE,
                NavState.AVOID_OBSTACLE,
                NavState.RETRY_VERIFY,
                NavState.REACQUIRE_FINAL,
                NavState.RETREAT_VISITED,
            )
        ):
            dx_cmd = float(x_enu) - self.pose.x_enu
            dy_cmd = float(y_enu) - self.pose.y_enu
            cmd_distance = math.hypot(dx_cmd, dy_cmd)
            cmd_limit = max(0.18, self.max_command_ahead + 0.08)
            if cmd_distance > cmd_limit:
                scale = cmd_limit / max(cmd_distance, 1e-6)
                requested_x = float(x_enu)
                requested_y = float(y_enu)
                x_enu = self.pose.x_enu + dx_cmd * scale
                y_enu = self.pose.y_enu + dy_cmd * scale
                self._log_throttle(
                    "setpoint_distance_guard",
                    0.75,
                    "warning",
                    f"SETPOINT_DISTANCE_GUARD_V21H "
                    f"requested=({requested_x:.2f},{requested_y:.2f}) "
                    f"limited=({x_enu:.2f},{y_enu:.2f}) "
                    f"distance={cmd_distance:.2f} limit={cmd_limit:.2f}",
                )

        # Clamp and slew altitude so setpoint Z cannot suddenly jump.
        if self.state == NavState.PRESTREAM:
            requested_altitude = clamp(
                altitude,
                0.05,
                self.flight_altitude,
            )
        else:
            requested_altitude = clamp(
                altitude,
                max(0.55, self.flight_altitude - 0.12),
                self.flight_altitude + 0.10,
            )

        now = time.monotonic()
        dt = clamp(
            now - self.last_altitude_command_mono,
            0.0,
            0.20,
        )
        max_step = max(0.01, self.max_altitude_command_rate * dt)
        if requested_altitude > self.commanded_altitude:
            self.commanded_altitude = min(
                requested_altitude,
                self.commanded_altitude + max_step,
            )
        else:
            self.commanded_altitude = max(
                requested_altitude,
                self.commanded_altitude - max_step,
            )
        self.last_altitude_command_mono = now
        altitude = self.commanded_altitude

        # Slew-limit yaw. Direct 45-degree or target-yaw jumps were causing
        # aggressive roll/yaw coupling while the vehicle was translating.
        yaw_dt = clamp(
            now - self.last_yaw_command_mono,
            0.0,
            0.20,
        )
        max_yaw_step = max(
            math.radians(0.5),
            self.yaw_slew_rate * yaw_dt,
        )
        yaw_delta = angle_diff(yaw_enu, self.commanded_yaw_enu)
        yaw_delta = clamp(yaw_delta, -max_yaw_step, max_yaw_step)
        self.commanded_yaw_enu = wrap_pi(
            self.commanded_yaw_enu + yaw_delta
        )
        self.last_yaw_command_mono = now
        yaw_enu = self.commanded_yaw_enu

        # ENU relatif home -> NED absolut
        x_ned = self.home_ned_x + y_enu
        y_ned = self.home_ned_y + x_enu
        z_ned = self.home_ned_z - altitude
        yaw_ned = wrap_pi((math.pi / 2.0) - yaw_enu)

        if not finite(x_ned, y_ned, z_ned, yaw_ned):
            self.get_logger().error(
                "INVALID_SETPOINT_BLOCK_V21H transformed setpoint non-finite"
            )
            return

        msg = TrajectorySetpoint()
        msg.timestamp = self._timestamp_us()
        msg.position = [float(x_ned), float(y_ned), float(z_ned)]
        msg.velocity = [math.nan, math.nan, math.nan]
        msg.acceleration = [math.nan, math.nan, math.nan]
        msg.jerk = [math.nan, math.nan, math.nan]
        msg.yaw = float(yaw_ned)
        msg.yawspeed = math.nan
        self.trajectory_pub.publish(msg)

    def _command_toward(
        self,
        goal_x: float,
        goal_y: float,
        yaw_enu: float,
    ) -> None:
        """Move through a latched micro-waypoint with a speed governor.

        V21F updated a point ahead on every 20 Hz tick. That acts like
        a moving carrot: PX4 keeps accelerating and the vehicle reached more
        than 4 m/s. V21H holds one small waypoint until it is reached, then
        advances to the next node.
        """
        if self.pose is None:
            return

        now = time.monotonic()
        goal = self._clamp_goal_to_orchard(goal_x, goal_y)
        goal_changed = (
            self.motion_goal_xy is None
            or math.hypot(
                goal[0] - self.motion_goal_xy[0],
                goal[1] - self.motion_goal_xy[1],
            ) > 0.25
        )
        if goal_changed:
            self.motion_goal_xy = goal
            self.motion_waypoint_xy = None
            self.motion_brake_anchor_xy = None

        # Hard speed governor. The hold anchor is fixed; it never follows the
        # drifting current pose, so PX4 receives a real braking target.
        if self.pose.speed_xy >= self.motion_brake_speed:
            if self.motion_brake_anchor_xy is None:
                self.motion_brake_anchor_xy = (
                    self.pose.x_enu,
                    self.pose.y_enu,
                )
                self.get_logger().warning(
                    f"XY_SPEED_BRAKE_V21H speed={self.pose.speed_xy:.2f} "
                    f"anchor=({self.motion_brake_anchor_xy[0]:.2f},"
                    f"{self.motion_brake_anchor_xy[1]:.2f})"
                )
            self._publish_position_enu(
                self.motion_brake_anchor_xy[0],
                self.motion_brake_anchor_xy[1],
                self.flight_altitude,
                yaw_enu,
            )
            return

        if self.motion_brake_anchor_xy is not None:
            if self.pose.speed_xy > self.motion_release_speed:
                self._publish_position_enu(
                    self.motion_brake_anchor_xy[0],
                    self.motion_brake_anchor_xy[1],
                    self.flight_altitude,
                    yaw_enu,
                )
                return
            self.get_logger().info(
                f"XY_SPEED_RELEASE_V21H speed={self.pose.speed_xy:.2f}"
            )
            self.motion_brake_anchor_xy = None
            self.motion_waypoint_xy = None

        dx = goal[0] - self.pose.x_enu
        dy = goal[1] - self.pose.y_enu
        distance = math.hypot(dx, dy)

        step = min(self.motion_step_distance, self.max_command_ahead)
        front_range, front_source, _front_tof, _cloud_front = (
            self._combined_front_range()
        )
        hard_limit = (
            self.moving_cloud_hard_stop
            if front_source == "pointcloud"
            else self.tof_hard_stop
        )
        if (
            math.isfinite(front_range)
            and front_range < self.tof_slow_distance
        ):
            usable = max(
                0.0,
                (front_range - hard_limit)
                / max(self.tof_slow_distance - hard_limit, 1e-3),
            )
            step = self.min_motion_lookahead + usable * (
                step - self.min_motion_lookahead
            )
        step = clamp(
            step,
            self.min_motion_lookahead,
            min(self.motion_step_distance, self.max_command_ahead),
        )

        need_waypoint = self.motion_waypoint_xy is None
        if self.motion_waypoint_xy is not None:
            wp_distance = math.hypot(
                self.motion_waypoint_xy[0] - self.pose.x_enu,
                self.motion_waypoint_xy[1] - self.pose.y_enu,
            )
            reached = (
                wp_distance <= self.motion_waypoint_reach
                and self.pose.speed_xy <= self.motion_advance_speed
            )
            timed = (
                now - self.motion_waypoint_since
                >= self.motion_waypoint_timeout
                and self.pose.speed_xy <= self.motion_release_speed
            )
            escaped = wp_distance > max(1.0, 3.0 * step)
            need_waypoint = reached or timed or escaped

        if need_waypoint:
            if distance <= step:
                waypoint = goal
            else:
                scale = step / max(distance, 1e-6)
                waypoint = (
                    self.pose.x_enu + dx * scale,
                    self.pose.y_enu + dy * scale,
                )
            # V21H: jangan clamp micro-waypoint ke batas kebun.
            # Posisi awal drone berada di luar orchard_min_x. Clamp di sini
            # pernah mengubah node pertama dari sekitar 0.32 m menjadi x=9.50 m,
            # sehingga PX4 mendapat loncatan setpoint besar dan drone menyentak.
            # Final goal tetap sudah dibatasi oleh _clamp_goal_to_orchard().
            self.motion_waypoint_xy = (
                float(waypoint[0]),
                float(waypoint[1]),
            )
            self.motion_waypoint_since = now
            self._log_throttle(
                "motion_node",
                0.35,
                "info",
                f"MOTION_NODE_V21H waypoint="
                f"({self.motion_waypoint_xy[0]:.2f},"
                f"{self.motion_waypoint_xy[1]:.2f}) "
                f"goal=({goal[0]:.2f},{goal[1]:.2f}) "
                f"speed={self.pose.speed_xy:.2f} step={step:.2f}",
            )

        assert self.motion_waypoint_xy is not None
        self._publish_position_enu(
            self.motion_waypoint_xy[0],
            self.motion_waypoint_xy[1],
            self.flight_altitude,
            yaw_enu,
        )

    def _hold_current(self, yaw_enu: float) -> None:
        if self.pose is None:
            return
        self._publish_position_enu(
            self.pose.x_enu,
            self.pose.y_enu,
            self.flight_altitude,
            yaw_enu,
        )

    def _send_vehicle_command(
        self,
        command: int,
        **params: float,
    ) -> None:
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
        self.command_pub.publish(msg)

    def _send_offboard_and_arm(self) -> None:
        self._send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=6.0,
        )
        self._send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0,
        )
        self.get_logger().info("ARM_OFFBOARD_SENT_V21H")

    # =========================================================================
    # Eksplorasi
    # =========================================================================

    def _choose_explore_goal(self) -> None:
        min_x = self.orchard_min_x + self.explore_margin
        max_x = self.orchard_max_x - self.explore_margin
        min_y = self.orchard_min_y + self.explore_margin
        max_y = self.orchard_max_y - self.explore_margin

        for _ in range(30):
            x = self.random.uniform(min_x, max_x)
            y = self.random.uniform(min_y, max_y)

            too_close_to_visited = any(
                track.state == TrackState.VISITED
                and math.hypot(x - track.x, y - track.y) < 3.0
                for track in self.tracks.values()
            )
            if not too_close_to_visited:
                self.explore_goal = (x, y)
                break
        else:
            self.explore_goal = (
                self.random.uniform(min_x, max_x),
                self.random.uniform(min_y, max_y),
            )

        self.explore_count += 1
        self.get_logger().info(
            f"NEW_RANDOM_EXPLORE_GOAL_V21H count={self.explore_count} "
            f"goal=({self.explore_goal[0]:.2f},{self.explore_goal[1]:.2f})"
        )

    def _clamp_goal_to_orchard(
        self,
        x: float,
        y: float,
    ) -> Tuple[float, float]:
        return (
            clamp(x, self.orchard_min_x, self.orchard_max_x),
            clamp(y, self.orchard_min_y, self.orchard_max_y),
        )

    # =========================================================================
    # Marker dan debug
    # =========================================================================

    def _local_to_visual(self, x_local: float, y_local: float) -> Tuple[float, float]:
        """ENU relatif home -> Gazebo world ENU, khusus visualisasi."""
        return (
            self.visual_spawn_x + float(x_local),
            self.visual_spawn_y + float(y_local),
        )

    def _points_local_to_visual(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        result = np.asarray(points, dtype=np.float32).copy()
        result[:, 0] += float(self.visual_spawn_x)
        result[:, 1] += float(self.visual_spawn_y)
        return result

    def _actual_visual_positions(self) -> List[Tuple[float, float, float]]:
        if not self.publish_actual_visual or get_actual_tree_positions_gazebo is None:
            return []
        try:
            return [
                (float(x), float(y), float(z))
                for x, y, z in get_actual_tree_positions_gazebo(0.0)
            ]
        except Exception as exc:
            self._log_throttle(
                "actual_visual_error",
                2.0,
                "warning",
                f"ACTUAL_VISUAL_UNAVAILABLE_V21H error={exc}",
            )
            return []

    @staticmethod
    def _clear_marker_array(namespace: str = "") -> MarkerArray:
        array = MarkerArray()
        clear = Marker()
        clear.header.frame_id = "map"
        clear.action = Marker.DELETEALL
        clear.ns = namespace
        array.markers.append(clear)
        return array

    def _publish_markers(self) -> None:
        if self.pose is None:
            return

        stamp = self.get_clock().now().to_msg()
        actual_positions = self._actual_visual_positions()

        # ------------------------------------------------------------------
        # 1. Detected tree markers: /sawit/tree_markers
        # ------------------------------------------------------------------
        trees = self._clear_marker_array("tree_clear")
        for track in sorted(self.tracks.values(), key=lambda item: item.tree_id):
            vx, vy = self._local_to_visual(track.x, track.y)

            sphere = Marker()
            sphere.header.frame_id = "map"
            sphere.header.stamp = stamp
            sphere.ns = "detected_tree_centers"
            sphere.id = 1000 + track.tree_id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = vx
            sphere.pose.position.y = vy
            sphere.pose.position.z = 0.55
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.62
            sphere.scale.y = 0.62
            sphere.scale.z = 0.62

            visited_proven = (
                track.state == TrackState.VISITED
                and track.tree_id in self.visited_proof_ids_v21n8
            )
            if visited_proven:
                sphere.color.r, sphere.color.g, sphere.color.b = 0.05, 1.00, 0.10
            elif track.state == TrackState.VISITED:
                # Visited state without a 1 m proof must never look green.
                sphere.color.r, sphere.color.g, sphere.color.b = 0.65, 0.15, 0.95
            elif track.state == TrackState.CONFIRMED:
                sphere.color.r, sphere.color.g, sphere.color.b = 1.00, 0.82, 0.00
            elif track.state == TrackState.REJECTED:
                sphere.color.r, sphere.color.g, sphere.color.b = 1.00, 0.05, 0.05
            else:
                sphere.color.r, sphere.color.g, sphere.color.b = 1.00, 0.45, 0.00
            sphere.color.a = 1.0
            trees.markers.append(sphere)

            label = Marker()
            label.header.frame_id = "map"
            label.header.stamp = stamp
            label.ns = "detected_tree_labels"
            label.id = 2000 + track.tree_id
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = vx
            label.pose.position.y = vy
            label.pose.position.z = 1.15
            label.scale.z = 0.32
            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0
            label.color.a = 0.95
            if visited_proven:
                state_text = "visited_1m"
            elif track.state == TrackState.VISITED:
                state_text = "visited_unproven"
            else:
                state_text = track.state.value

            label.text = (
                f"D{track.tree_id} {state_text} "
                f"h={track.hits} views={len(track.sectors)}"
            )
            trees.markers.append(label)
        self.marker_pub.publish(trees)

        # ------------------------------------------------------------------
        # 2. Actual/SDF markers: /sawit/actual_tree_markers
        #    Pembanding visual saja, tidak pernah masuk selector/gate.
        # ------------------------------------------------------------------
        actual_array = self._clear_marker_array("actual_clear")
        for actual_id, (ax, ay, _az) in enumerate(actual_positions):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = stamp
            marker.ns = "actual_trees_sdf_visual_only"
            marker.id = actual_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = ax
            marker.pose.position.y = ay
            marker.pose.position.z = 1.20
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.34
            marker.scale.y = 0.34
            marker.scale.z = 2.40
            marker.color.r = 0.00
            marker.color.g = 0.15
            marker.color.b = 1.00
            marker.color.a = 1.00
            actual_array.markers.append(marker)

            label = Marker()
            label.header.frame_id = "map"
            label.header.stamp = stamp
            label.ns = "actual_tree_labels"
            label.id = 1000 + actual_id
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = ax
            label.pose.position.y = ay
            label.pose.position.z = 2.65
            label.scale.z = 0.30
            label.color.r = 0.40
            label.color.g = 0.65
            label.color.b = 1.00
            label.color.a = 1.00
            label.text = f"A{actual_id}"
            actual_array.markers.append(label)
        self.actual_pub.publish(actual_array)

        # ------------------------------------------------------------------
        # 3. Navigation: drone, orchard boundary, active target, goal
        # ------------------------------------------------------------------
        nav = self._clear_marker_array("nav_clear")

        grid = Marker()
        grid.header.frame_id = "map"
        grid.header.stamp = stamp
        grid.ns = "orchard_work_grid"
        grid.id = 1
        grid.type = Marker.LINE_LIST
        grid.action = Marker.ADD
        grid.scale.x = 0.035
        grid.color.r = 0.55
        grid.color.g = 0.55
        grid.color.b = 0.55
        grid.color.a = 0.90

        cols = 8
        rows = 8
        for i in range(cols + 1):
            ratio = i / float(cols)
            lx = self.orchard_min_x + ratio * (
                self.orchard_max_x - self.orchard_min_x
            )
            x1, y1 = self._local_to_visual(lx, self.orchard_min_y)
            x2, y2 = self._local_to_visual(lx, self.orchard_max_y)
            grid.points.extend(
                [self._point(x1, y1, 0.02), self._point(x2, y2, 0.02)]
            )
        for j in range(rows + 1):
            ratio = j / float(rows)
            ly = self.orchard_min_y + ratio * (
                self.orchard_max_y - self.orchard_min_y
            )
            x1, y1 = self._local_to_visual(self.orchard_min_x, ly)
            x2, y2 = self._local_to_visual(self.orchard_max_x, ly)
            grid.points.extend(
                [self._point(x1, y1, 0.02), self._point(x2, y2, 0.02)]
            )
        nav.markers.append(grid)

        drone_x, drone_y = self._local_to_visual(
            self.pose.x_enu,
            self.pose.y_enu,
        )
        drone = Marker()
        drone.header.frame_id = "map"
        drone.header.stamp = stamp
        drone.ns = "drone"
        drone.id = 10
        drone.type = Marker.ARROW
        drone.action = Marker.ADD
        drone.pose.position.x = drone_x
        drone.pose.position.y = drone_y
        drone.pose.position.z = float(max(0.15, self.pose.altitude))
        drone.pose.orientation.z = math.sin(self.pose.yaw_enu / 2.0)
        drone.pose.orientation.w = math.cos(self.pose.yaw_enu / 2.0)
        drone.scale.x = 1.35
        drone.scale.y = 0.46
        drone.scale.z = 0.46
        drone.color.r = 1.0
        drone.color.g = 0.02
        drone.color.b = 0.02
        drone.color.a = 1.0
        nav.markers.append(drone)

        # Beacon merah di proyeksi tanah supaya posisi drone tetap tampak
        # pada kamera Orbit/top-down walaupun arrow tertutup cloud atau jauh.
        drone_beacon = Marker()
        drone_beacon.header.frame_id = "map"
        drone_beacon.header.stamp = stamp
        drone_beacon.ns = "drone_ground_beacon"
        drone_beacon.id = 12
        drone_beacon.type = Marker.SPHERE
        drone_beacon.action = Marker.ADD
        drone_beacon.pose.position.x = drone_x
        drone_beacon.pose.position.y = drone_y
        drone_beacon.pose.position.z = 0.34
        drone_beacon.pose.orientation.w = 1.0
        drone_beacon.scale.x = 0.82
        drone_beacon.scale.y = 0.82
        drone_beacon.scale.z = 0.82
        drone_beacon.color.r = 1.0
        drone_beacon.color.g = 0.0
        drone_beacon.color.b = 0.0
        drone_beacon.color.a = 1.0
        nav.markers.append(drone_beacon)

        drone_mast = Marker()
        drone_mast.header.frame_id = "map"
        drone_mast.header.stamp = stamp
        drone_mast.ns = "drone_vertical_mast"
        drone_mast.id = 13
        drone_mast.type = Marker.LINE_LIST
        drone_mast.action = Marker.ADD
        drone_mast.scale.x = 0.10
        drone_mast.color.r = 1.0
        drone_mast.color.g = 0.0
        drone_mast.color.b = 0.0
        drone_mast.color.a = 0.95
        drone_mast.points = [
            self._point(drone_x, drone_y, 0.34),
            self._point(
                drone_x,
                drone_y,
                max(0.34, self.pose.altitude),
            ),
        ]
        nav.markers.append(drone_mast)

        drone_label = Marker()
        drone_label.header.frame_id = "map"
        drone_label.header.stamp = stamp
        drone_label.ns = "drone_label"
        drone_label.id = 11
        drone_label.type = Marker.TEXT_VIEW_FACING
        drone_label.action = Marker.ADD
        drone_label.pose.position.x = drone_x
        drone_label.pose.position.y = drone_y
        drone_label.pose.position.z = float(self.pose.altitude + 0.55)
        drone_label.scale.z = 0.42
        drone_label.color.r = 1.0
        drone_label.color.g = 1.0
        drone_label.color.b = 1.0
        drone_label.color.a = 1.0
        drone_label.text = (
            f"DRONE {self.state.value} z={self.pose.altitude:.2f}m"
        )
        nav.markers.append(drone_label)

        target = self._active_track()
        if target is not None:
            tx, ty = self._local_to_visual(target.x, target.y)
            active = Marker()
            active.header.frame_id = "map"
            active.header.stamp = stamp
            active.ns = "active_target"
            active.id = 20
            active.type = Marker.CYLINDER
            active.action = Marker.ADD
            active.pose.position.x = tx
            active.pose.position.y = ty
            active.pose.position.z = 0.25
            active.pose.orientation.w = 1.0
            active.scale.x = 0.90
            active.scale.y = 0.90
            active.scale.z = 0.16
            active.color.r = 0.80
            active.color.g = 0.05
            active.color.b = 1.00
            active.color.a = 0.95
            nav.markers.append(active)

            # V21N2 visual: ToF beam gates originate at the drone and follow
            # the sensor heading. They are not radius circles around the node.
            beam_yaw = self.pose.yaw_enu
            beam_dx = math.cos(beam_yaw)
            beam_dy = math.sin(beam_yaw)

            beam = Marker()
            beam.header.frame_id = "map"
            beam.header.stamp = stamp
            beam.ns = "tof_front_beam_v21n2"
            beam.id = 100
            beam.type = Marker.LINE_STRIP
            beam.action = Marker.ADD
            beam.scale.x = 0.10
            beam.color.r = 0.20
            beam.color.g = 0.85
            beam.color.b = 1.00
            beam.color.a = 0.95
            beam.points = [
                self._point(drone_x, drone_y, 0.26),
                self._point(
                    drone_x + 3.35 * beam_dx,
                    drone_y + 3.35 * beam_dy,
                    0.26,
                ),
            ]
            nav.markers.append(beam)

            gate_specs = (
                (
                    float(self.layer1_visit_distance),
                    "TOF 1m SAFE VISITED",
                    (0.05, 1.00, 0.10, 1.00),
                    101,
                ),
                (
                    float(self.layer2_stop_distance),
                    "TOF 2m SAFETY HOLD",
                    (1.00, 0.35, 0.00, 1.00),
                    102,
                ),
                (
                    float(self.layer3_object_distance),
                    "TOF 3m CHECK NEW/OLD",
                    (1.00, 0.90, 0.00, 1.00),
                    103,
                ),
            )

            for distance_m, gate_text, rgba, marker_id in gate_specs:
                gx = drone_x + distance_m * beam_dx
                gy = drone_y + distance_m * beam_dy
                px = -beam_dy
                py = beam_dx

                gate = Marker()
                gate.header.frame_id = "map"
                gate.header.stamp = stamp
                gate.ns = "tof_gate_ticks_v21n2"
                gate.id = marker_id
                gate.type = Marker.LINE_STRIP
                gate.action = Marker.ADD
                gate.scale.x = 0.13
                gate.color.r = rgba[0]
                gate.color.g = rgba[1]
                gate.color.b = rgba[2]
                gate.color.a = rgba[3]
                gate.points = [
                    self._point(gx - 0.45 * px, gy - 0.45 * py, 0.28),
                    self._point(gx + 0.45 * px, gy + 0.45 * py, 0.28),
                ]
                nav.markers.append(gate)

                gate_label = Marker()
                gate_label.header.frame_id = "map"
                gate_label.header.stamp = stamp
                gate_label.ns = "tof_gate_labels_v21n2"
                gate_label.id = 200 + marker_id
                gate_label.type = Marker.TEXT_VIEW_FACING
                gate_label.action = Marker.ADD
                gate_label.pose.position.x = gx + 0.55 * px
                gate_label.pose.position.y = gy + 0.55 * py
                gate_label.pose.position.z = 0.55
                gate_label.pose.orientation.w = 1.0
                gate_label.scale.z = 0.30
                gate_label.color.r = rgba[0]
                gate_label.color.g = rgba[1]
                gate_label.color.b = rgba[2]
                gate_label.color.a = 1.0
                gate_label.text = gate_text
                nav.markers.append(gate_label)

            _stable_gate, accepted_tof, _gate_mad, _gate_count = (
                self._tof_layer_stable()
            )
            if math.isfinite(accepted_tof):
                endpoint_distance = min(accepted_tof, 20.0)
                endpoint = Marker()
                endpoint.header.frame_id = "map"
                endpoint.header.stamp = stamp
                endpoint.ns = "tof_accepted_endpoint_v21n2"
                endpoint.id = 110
                endpoint.type = Marker.SPHERE
                endpoint.action = Marker.ADD
                endpoint.pose.position.x = drone_x + endpoint_distance * beam_dx
                endpoint.pose.position.y = drone_y + endpoint_distance * beam_dy
                endpoint.pose.position.z = 0.33
                endpoint.pose.orientation.w = 1.0
                endpoint.scale.x = 0.34
                endpoint.scale.y = 0.34
                endpoint.scale.z = 0.34
                endpoint.color.r = 0.00
                endpoint.color.g = 1.00
                endpoint.color.b = 1.00
                endpoint.color.a = 1.00
                nav.markers.append(endpoint)

            # Garis ukur drone -> pusat node dan teks status layer saat ini.
            target_distance_visual = math.hypot(
                target.x - self.pose.x_enu,
                target.y - self.pose.y_enu,
            )
            _stable_vis, tof_visual, tof_mad_visual, tof_count_visual = (
                self._tof_layer_stable()
            )
            front_range_visual = tof_visual
            front_source_visual = self.tof_selected_source_v21n2
            stage_visual = self.tof_approach_stage
            if stage_visual == "HOLD_1M":
                current_layer = "L1 SAFE VISITED CHECK"
                status_rgba = (0.05, 1.00, 0.10, 1.00)
            elif stage_visual in ("TO_1M", "HOLD_2M"):
                current_layer = "L2 SAFE -> TO 1M"
                status_rgba = (1.00, 0.35, 0.00, 1.00)
            elif stage_visual in ("TO_2M", "CHECK_3M"):
                current_layer = "L3 NEW -> TO 2M"
                status_rgba = (1.00, 0.90, 0.00, 1.00)
            else:
                current_layer = "TOF APPROACH TO 3M"
                status_rgba = (0.20, 0.85, 1.00, 1.00)

            measure_line = Marker()
            measure_line.header.frame_id = "map"
            measure_line.header.stamp = stamp
            measure_line.ns = "active_target_distance_v21n"
            measure_line.id = 150
            measure_line.type = Marker.LINE_STRIP
            measure_line.action = Marker.ADD
            measure_line.scale.x = 0.075
            measure_line.color.r = status_rgba[0]
            measure_line.color.g = status_rgba[1]
            measure_line.color.b = status_rgba[2]
            measure_line.color.a = 0.95
            measure_line.points = [
                self._point(drone_x, drone_y, 0.30),
                self._point(tx, ty, 0.30),
            ]
            nav.markers.append(measure_line)

            layer_status = Marker()
            layer_status.header.frame_id = "map"
            layer_status.header.stamp = stamp
            layer_status.ns = "active_target_layer_status_v21n"
            layer_status.id = 151
            layer_status.type = Marker.TEXT_VIEW_FACING
            layer_status.action = Marker.ADD
            layer_status.pose.position.x = (drone_x + tx) * 0.5
            layer_status.pose.position.y = (drone_y + ty) * 0.5
            layer_status.pose.position.z = 1.05
            layer_status.pose.orientation.w = 1.0
            layer_status.scale.z = 0.38
            layer_status.color.r = status_rgba[0]
            layer_status.color.g = status_rgba[1]
            layer_status.color.b = status_rgba[2]
            layer_status.color.a = 1.0
            front_text = (
                f"{front_range_visual:.2f}m/{front_source_visual}"
                if math.isfinite(front_range_visual)
                else "inf/no-front"
            )
            layer_status.text = (
                f"D{target.tree_id} {current_layer} "
                f"stage={self.tof_approach_stage} "
                f"map_target_dist={target_distance_visual:.2f}m "
                f"accepted_tof={front_text} mad={tof_mad_visual:.2f} "
                f"n={tof_count_visual}"
            )
            nav.markers.append(layer_status)

        if self.active_standoff_goal is not None:
            gx, gy = self._local_to_visual(*self.active_standoff_goal)
            goal = Marker()
            goal.header.frame_id = "map"
            goal.header.stamp = stamp
            goal.ns = "standoff_goal"
            goal.id = 21
            goal.type = Marker.SPHERE
            goal.action = Marker.ADD
            goal.pose.position.x = gx
            goal.pose.position.y = gy
            goal.pose.position.z = 0.22
            goal.pose.orientation.w = 1.0
            goal.scale.x = goal.scale.y = goal.scale.z = 0.35
            goal.color.r = 0.90
            goal.color.g = 0.00
            goal.color.b = 1.00
            goal.color.a = 1.0
            nav.markers.append(goal)

        self.nav_pub.publish(nav)

        # Topic khusus agar marker drone dapat ditambahkan sebagai display
        # Marker terpisah dan tidak tergantung pengaturan MarkerArray.
        dedicated_drone = Marker()
        dedicated_drone.header.frame_id = "map"
        dedicated_drone.header.stamp = stamp
        dedicated_drone.ns = "drone_dedicated"
        dedicated_drone.id = 0
        dedicated_drone.type = Marker.ARROW
        dedicated_drone.action = Marker.ADD
        dedicated_drone.pose.position.x = drone_x
        dedicated_drone.pose.position.y = drone_y
        dedicated_drone.pose.position.z = float(
            max(0.15, self.pose.altitude)
        )
        dedicated_drone.pose.orientation.z = math.sin(
            self.pose.yaw_enu / 2.0
        )
        dedicated_drone.pose.orientation.w = math.cos(
            self.pose.yaw_enu / 2.0
        )
        dedicated_drone.scale.x = 1.35
        dedicated_drone.scale.y = 0.46
        dedicated_drone.scale.z = 0.46
        dedicated_drone.color.r = 1.0
        dedicated_drone.color.g = 0.0
        dedicated_drone.color.b = 0.0
        dedicated_drone.color.a = 1.0
        self.drone_marker_pub.publish(dedicated_drone)

        # ------------------------------------------------------------------
        # 4. Route: /sawit/route_marker
        # ------------------------------------------------------------------
        route = Marker()
        route.header.frame_id = "map"
        route.header.stamp = stamp
        route.ns = "route"
        route.id = 0
        route.type = Marker.LINE_STRIP
        route.action = Marker.ADD
        route.scale.x = 0.055
        route.color.r = 1.0
        route.color.g = 1.0
        route.color.b = 1.0
        route.color.a = 1.0
        for x_local, y_local in self.path_points:
            vx, vy = self._local_to_visual(x_local, y_local)
            route.points.append(
                self._point(vx, vy, max(0.05, self.pose.altitude))
            )
        self.route_pub.publish(route)

        # ------------------------------------------------------------------
        # 5. Trunk models: /sawit/trunk_models
        # ------------------------------------------------------------------
        models = self._clear_marker_array("trunk_models_clear")
        for track in self.tracks.values():
            if track.state == TrackState.REJECTED:
                continue
            vx, vy = self._local_to_visual(track.x, track.y)
            model = Marker()
            model.header.frame_id = "map"
            model.header.stamp = stamp
            model.ns = "detected_trunk_models"
            model.id = track.tree_id
            model.type = Marker.CYLINDER
            model.action = Marker.ADD
            model.pose.position.x = vx
            model.pose.position.y = vy
            model.pose.position.z = 1.05
            model.pose.orientation.w = 1.0
            model.scale.x = 0.26
            model.scale.y = 0.26
            model.scale.z = 2.10
            if track.state == TrackState.VISITED:
                model.color.r, model.color.g, model.color.b = 0.05, 1.0, 0.10
            else:
                model.color.r, model.color.g, model.color.b = 1.0, 0.82, 0.0
            model.color.a = 0.82
            models.markers.append(model)
        self.trunk_models_pub.publish(models)

        # ------------------------------------------------------------------
        # 6. Visual comparison lines: /sawit/comparison_markers
        # ------------------------------------------------------------------
        comparison = self._clear_marker_array("comparison_clear")
        if actual_positions:
            comparison_id = 1
            errors: List[float] = []
            for track in self.tracks.values():
                if track.state == TrackState.REJECTED:
                    continue
                vx, vy = self._local_to_visual(track.x, track.y)
                nearest_id, nearest_pos, nearest_error = min(
                    (
                        (
                            actual_id,
                            (ax, ay),
                            math.hypot(vx - ax, vy - ay),
                        )
                        for actual_id, (ax, ay, _az) in enumerate(actual_positions)
                    ),
                    key=lambda item: item[2],
                )
                errors.append(nearest_error)

                line = Marker()
                line.header.frame_id = "map"
                line.header.stamp = stamp
                line.ns = "detected_to_actual_error"
                line.id = comparison_id
                comparison_id += 1
                line.type = Marker.LINE_LIST
                line.action = Marker.ADD
                line.scale.x = 0.045
                line.color.r = 0.00
                line.color.g = 1.00
                line.color.b = 1.00
                line.color.a = 0.90
                line.points = [
                    self._point(vx, vy, 0.72),
                    self._point(nearest_pos[0], nearest_pos[1], 0.72),
                ]
                comparison.markers.append(line)

                text = Marker()
                text.header.frame_id = "map"
                text.header.stamp = stamp
                text.ns = "comparison_error_labels"
                text.id = comparison_id
                comparison_id += 1
                text.type = Marker.TEXT_VIEW_FACING
                text.action = Marker.ADD
                text.pose.position.x = 0.5 * (vx + nearest_pos[0])
                text.pose.position.y = 0.5 * (vy + nearest_pos[1])
                text.pose.position.z = 1.45
                text.scale.z = 0.27
                text.color.r = 0.20
                text.color.g = 1.00
                text.color.b = 1.00
                text.color.a = 1.00
                text.text = (
                    f"D{track.tree_id}->A{nearest_id} "
                    f"e={nearest_error:.2f}m"
                )
                comparison.markers.append(text)

            if errors:
                self._log_throttle(
                    "visual_compare",
                    2.0,
                    "info",
                    f"VISUAL_COMPARE_V21H detected={len(errors)} "
                    f"mean={float(np.mean(errors)):.2f}m "
                    f"median={float(np.median(errors)):.2f}m "
                    "actual_used=visual_only",
                )
        self.comparison_pub.publish(comparison)

    def _point(self, x: float, y: float, z: float) -> Point:
        point = Point()
        point.x = float(x)
        point.y = float(y)
        point.z = float(z)
        return point

    def _publish_debug_clouds(self) -> None:
        stamp = self.get_clock().now().to_msg()

        if self.last_debug_stationary_points is not None:
            visual_stationary = self._points_local_to_visual(
                self.last_debug_stationary_points
            )
            header = Header()
            header.frame_id = "map"
            header.stamp = stamp
            msg = point_cloud2.create_cloud_xyz32(
                header,
                visual_stationary.tolist(),
            )
            self.debug_stationary_pub.publish(msg)
            # Alias lama untuk konfigurasi RViz sebelumnya.
            self.debug_roi_pub.publish(msg)

        if self.last_debug_candidate_points is not None:
            visual_candidates = self._points_local_to_visual(
                self.last_debug_candidate_points
            )
            header = Header()
            header.frame_id = "map"
            header.stamp = stamp
            msg = point_cloud2.create_cloud_xyz32(
                header,
                visual_candidates.tolist(),
            )
            self.debug_accepted_pub.publish(msg)

        landmark_points = []
        rejected_points = []
        for track in self.tracks.values():
            vx, vy = self._local_to_visual(track.x, track.y)
            point = (vx, vy, 0.55)
            if track.state == TrackState.REJECTED:
                rejected_points.append(point)
            else:
                landmark_points.append(point)

        header = Header()
        header.frame_id = "map"
        header.stamp = stamp

        landmark_msg = point_cloud2.create_cloud_xyz32(
            header,
            landmark_points,
        )
        self.debug_landmark_pub.publish(landmark_msg)

        rejected_msg = point_cloud2.create_cloud_xyz32(
            header,
            rejected_points,
        )
        self.debug_rejected_pub.publish(rejected_msg)


    # =========================================================================
    # Memory
    # =========================================================================

    def _load_or_reset_memory(self) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)

        if self.reset_memory_on_start:
            if self.memory_path.exists():
                backup = self.memory_path.with_suffix(
                    f".before_v21_{int(time.time())}.json"
                )
                try:
                    backup.write_text(self.memory_path.read_text())
                except Exception:
                    pass
            self.tracks = {}
            self.next_tree_id = 0
            self._save_memory()
            return

        if not self.memory_path.exists():
            return

        try:
            data = json.loads(self.memory_path.read_text())
            for item in data.get("tracks", []):
                track = TreeTrack(
                    tree_id=int(item["tree_id"]),
                    x=float(item["x"]),
                    y=float(item["y"]),
                    state=TrackState(str(item["state"])),
                    hits=int(item.get("hits", 0)),
                    strong_hits=int(item.get("strong_hits", 0)),
                    tof_hits=int(item.get("tof_hits", 0)),
                    sectors=set(int(v) for v in item.get("sectors", [])),
                    verify_failures=int(item.get("verify_failures", 0)),
                    last_score=float(item.get("last_score", 0.0)),
                    position_mad=float(item.get("position_mad", math.inf)),
                )
                self.tracks[track.tree_id] = track
            self.next_tree_id = (
                max(self.tracks.keys()) + 1 if self.tracks else 0
            )
        except Exception as exc:
            self.get_logger().error(
                f"MEMORY_LOAD_FAIL_V21H error={exc}; starting empty"
            )
            self.tracks = {}
            self.next_tree_id = 0

    def _save_memory(self) -> None:
        try:
            payload = {
                "version": "V21H",
                "frame": "ENU_relative_home",
                "updated_unix": time.time(),
                "tracks": [
                    {
                        "tree_id": track.tree_id,
                        "x": track.x,
                        "y": track.y,
                        "state": track.state.value,
                        "hits": track.hits,
                        "strong_hits": track.strong_hits,
                        "tof_hits": track.tof_hits,
                        "sectors": sorted(track.sectors),
                        "verify_failures": track.verify_failures,
                        "last_score": track.last_score,
                        "position_mad": track.position_mad,
                    }
                    for track in sorted(
                        self.tracks.values(),
                        key=lambda t: t.tree_id,
                    )
                ],
            }
            temp = self.memory_path.with_suffix(".tmp")
            temp.write_text(json.dumps(payload, indent=2))
            temp.replace(self.memory_path)
        except Exception as exc:
            self._log_throttle(
                "memory_save",
                2.0,
                "error",
                f"MEMORY_SAVE_FAIL_V21H error={exc}",
            )

    # =========================================================================
    # Helper
    # =========================================================================

    def _visited_count(self) -> int:
        return sum(
            track.state == TrackState.VISITED
            for track in self.tracks.values()
        )

    def _set_state(self, state: NavState) -> None:
        old = self.state
        self.state = state
        self.state_enter_mono = time.monotonic()

        if self.pose is not None and state in (
            NavState.ALIGN_TARGET,
            NavState.EXPLORE_ALIGN,
        ):
            self.align_anchor_xy = (
                self.pose.x_enu,
                self.pose.y_enu,
            )
            self.motion_waypoint_xy = None
            self.motion_brake_anchor_xy = None

        if state in (
            NavState.APPROACH,
            NavState.EXPLORE_MOVE,
        ) and old != state:
            self.motion_waypoint_xy = None
            self.motion_goal_xy = None
            self.motion_brake_anchor_xy = None

        if state not in (
            NavState.APPROACH,
            NavState.EXPLORE_MOVE,
            NavState.ALIGN_TARGET,
            NavState.EXPLORE_ALIGN,
        ):
            self.motion_waypoint_xy = None
            self.motion_goal_xy = None
            self.motion_brake_anchor_xy = None

        if old != state:
            self.get_logger().info(
                f"STATE_V21H {old.value} -> {state.value}"
            )

    def _state_elapsed(self) -> float:
        return time.monotonic() - self.state_enter_mono

    def _log_throttle(
        self,
        key: str,
        period: float,
        level: str,
        message: str,
    ) -> None:
        now = time.monotonic()
        previous = self.last_log_mono.get(key, -math.inf)
        if now - previous < period:
            return
        self.last_log_mono[key] = now

        logger = self.get_logger()
        if level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)
        else:
            logger.info(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SawitNavigatorV21H()

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
