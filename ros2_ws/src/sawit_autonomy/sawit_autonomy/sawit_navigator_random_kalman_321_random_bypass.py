#!/usr/bin/env python3
"""
Navigator random target + Kalman + verifikasi 3-2-1
dengan penghindaran obstacle virtual kiri/kanan.

Konsep penghindaran:
1. ToF mendeteksi obstacle yang berada sebelum target aktif.
2. Titik pusat obstacle diproyeksikan dari pose drone dan jarak ToF.
3. Obstacle diberi batas virtual 1 m ke kiri dan 1 m ke kanan.
4. Sistem memilih LEFT atau RIGHT secara acak satu kali.
5. Pilihan sisi dikunci selama target aktif yang sama.
6. Drone bergerak:
       HOLD -> BACKUP -> SIDE_ENTRY -> PASS_FORWARD
       -> VERIFY_CLEAR -> ALIGN_TARGET
7. Bila satu sisi gagal penuh, sisi dibalik satu kali.
8. Target aktif dan fixed random queue tidak dihapus atau dikonsumsi.

Ground truth Gazebo tidak digunakan untuk navigasi.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import rclpy
from visualization_msgs.msg import Marker, MarkerArray

from sawit_autonomy.sawit_navigator_random_kalman_321_v22 import (
    NavState,
    SawitRandomKalman321V22,
    TrackState,
)


Point2D = Tuple[float, float]


@dataclass
class RandomBypassPlan:
    target_id: int
    target_xy: Point2D
    obstacle_xy: Point2D
    forward_unit: Point2D
    left_unit: Point2D
    side: int  # +1 = LEFT, -1 = RIGHT

    backup_goal: Point2D
    side_entry_goal: Point2D
    pass_goal: Point2D

    phase: str
    phase_started: float
    hold_until: float

    last_remaining: float = math.inf
    last_progress_time: float = 0.0
    forward_extensions: int = 0
    emergency_backup_count: int = 0
    recovery_attempt: int = 0


class SawitRandomKalman321RandomBypass(
    SawitRandomKalman321V22
):
    """Penghindaran sederhana sesuai batas obstacle ±1 m."""

    def __init__(self) -> None:
        super().__init__()

        # --------------------------------------------------------------
        # Parameter obstacle virtual
        # --------------------------------------------------------------
        self.declare_parameter("random_bypass_enabled", True)

        # Bentuk pohon/obstacle menurut arahan:
        # 1 m ke kiri dan 1 m ke kanan dari pusat ToF.
        self.declare_parameter("obstacle_half_width", 1.00)

        # Tambahan agar badan/baling-baling tidak tepat di batas 1 m.
        self.declare_parameter("drone_safety_margin", 0.45)

        # Ambil alih hanya bila obstacle jelas berada sebelum target jauh.
        self.declare_parameter("bypass_target_min_distance", 5.00)
        self.declare_parameter("bypass_min_target_front_gap", 2.50)

        # Gerakan.
        self.declare_parameter("bypass_hold_time", 0.80)
        self.declare_parameter("bypass_backup_distance", 0.65)
        self.declare_parameter("bypass_entry_behind_obstacle", 0.70)
        self.declare_parameter("bypass_pass_ahead_obstacle", 1.80)

        # Bila garis menuju target masih memotong obstacle,
        # maju lagi di sisi yang sama.
        self.declare_parameter("bypass_forward_extension", 0.90)
        self.declare_parameter("bypass_max_extensions", 2)

        # Setpoint incremental agar gerakan tidak meloncat.
        self.declare_parameter("bypass_position_step", 0.12)
        self.declare_parameter("bypass_waypoint_tolerance", 0.25)

        # Anti-stuck.
        self.declare_parameter("bypass_progress_epsilon", 0.05)
        self.declare_parameter("bypass_progress_timeout", 12.0)
        self.declare_parameter("bypass_phase_timeout", 40.0)
        self.declare_parameter("bypass_verify_time", 1.00)

        # Hard safety.
        self.declare_parameter("bypass_emergency_front", 1.20)
        self.declare_parameter("bypass_emergency_before_flip", 2)
        self.declare_parameter("bypass_clearance_margin", 0.10)

        # Recovery setelah kedua sisi normal gagal.
        # Drone mundur lebih jauh lalu membangun jalur baru yang lebih lebar.
        self.declare_parameter("bypass_wide_recovery_enabled", True)
        self.declare_parameter("bypass_wide_recovery_max", 2)
        self.declare_parameter("bypass_deep_retreat_distance", 1.60)
        self.declare_parameter("bypass_deep_retreat_increment", 0.60)
        self.declare_parameter("bypass_wide_lateral_extra", 0.75)
        self.declare_parameter("bypass_wide_entry_extra", 0.35)
        self.declare_parameter("bypass_wide_pass_extra", 0.70)

        # Pengaman visited:
        # ToF dekat saja tidak cukup. Drone juga harus dekat terhadap
        # posisi target hasil deteksi pada peta lokal.
        self.declare_parameter("visit_map_gate_max", 2.50)

        # Satu pohon tidak boleh menghasilkan dua landmark VISITED.
        self.declare_parameter("visit_duplicate_radius", 2.50)

        # Obstacle dekat ketika target peta masih jauh harus masuk bypass,
        # bukan diperlakukan sebagai target 3-2-1.
        self.declare_parameter("far_obstacle_front_max", 3.40)
        self.declare_parameter("far_obstacle_confirmations", 2)
        self.declare_parameter("far_obstacle_repeat_interval", 1.00)
        self.declare_parameter("far_obstacle_distance_tolerance", 0.80)

        # Setelah COMPLETE, tampilkan hanya landmark VISITED yang sah.
        self.declare_parameter("completion_hide_nonvisited", True)

        # Final clean RViz bersifat visualisasi saja.
        # Kandidat confirmed/tentative tidak ditampilkan sebagai kuning/oranye.
        # Kontrol, filtering, Kalman, queue, avoidance, dan 3-2-1 tidak berubah.
        self.declare_parameter("clean_rviz_enabled", True)
        self.declare_parameter("clean_rviz_show_active_target", True)
        self.declare_parameter("clean_rviz_dedupe_radius", 2.50)

        # Recovery target hilang:
        # koordinat target hasil mapping sudah tercapai, tetapi sensor depan
        # tidak melihat batang. ID lama ditolak lalu scan 360 dilakukan.
        self.declare_parameter("target_lost_rescan_enabled", True)
        self.declare_parameter("target_lost_map_near_max", 1.20)
        self.declare_parameter("target_lost_tof_far_min", 4.50)
        self.declare_parameter("target_lost_confirm_time", 3.00)
        self.declare_parameter("target_lost_min_track_age", 2.00)
        # Latch recovery tidak boleh dibatalkan oleh satu pembacaan ToF
        # sekitar 3 m dari objek lain. Hanya return dekat yang stabil yang
        # boleh membatalkan relokalisasi.
        self.declare_parameter("target_lost_valid_near_max", 2.20)
        self.declare_parameter("target_lost_valid_near_count", 3)

        self.random_bypass_enabled = bool(
            self.get_parameter("random_bypass_enabled").value
        )
        self.obstacle_half_width = float(
            self.get_parameter("obstacle_half_width").value
        )
        self.drone_safety_margin = float(
            self.get_parameter("drone_safety_margin").value
        )
        self.bypass_lateral_offset = (
            self.obstacle_half_width
            + self.drone_safety_margin
        )

        self.bypass_target_min_distance = float(
            self.get_parameter(
                "bypass_target_min_distance"
            ).value
        )
        self.bypass_min_target_front_gap = float(
            self.get_parameter(
                "bypass_min_target_front_gap"
            ).value
        )
        self.bypass_hold_time = float(
            self.get_parameter("bypass_hold_time").value
        )
        self.bypass_backup_distance = float(
            self.get_parameter(
                "bypass_backup_distance"
            ).value
        )
        self.bypass_entry_behind_obstacle = float(
            self.get_parameter(
                "bypass_entry_behind_obstacle"
            ).value
        )
        self.bypass_pass_ahead_obstacle = float(
            self.get_parameter(
                "bypass_pass_ahead_obstacle"
            ).value
        )
        self.bypass_forward_extension = float(
            self.get_parameter(
                "bypass_forward_extension"
            ).value
        )
        self.bypass_max_extensions = max(
            0,
            int(
                self.get_parameter(
                    "bypass_max_extensions"
                ).value
            ),
        )
        self.bypass_position_step = float(
            self.get_parameter(
                "bypass_position_step"
            ).value
        )
        self.bypass_waypoint_tolerance = float(
            self.get_parameter(
                "bypass_waypoint_tolerance"
            ).value
        )
        self.bypass_progress_epsilon = float(
            self.get_parameter(
                "bypass_progress_epsilon"
            ).value
        )
        self.bypass_progress_timeout = float(
            self.get_parameter(
                "bypass_progress_timeout"
            ).value
        )
        self.bypass_phase_timeout = float(
            self.get_parameter(
                "bypass_phase_timeout"
            ).value
        )
        self.bypass_verify_time = float(
            self.get_parameter(
                "bypass_verify_time"
            ).value
        )
        self.bypass_emergency_front = float(
            self.get_parameter(
                "bypass_emergency_front"
            ).value
        )
        self.bypass_emergency_before_flip = max(
            1,
            int(
                self.get_parameter(
                    "bypass_emergency_before_flip"
                ).value
            ),
        )
        self.bypass_clearance_margin = float(
            self.get_parameter(
                "bypass_clearance_margin"
            ).value
        )
        self.bypass_wide_recovery_enabled = bool(
            self.get_parameter(
                "bypass_wide_recovery_enabled"
            ).value
        )
        self.bypass_wide_recovery_max = max(
            0,
            int(
                self.get_parameter(
                    "bypass_wide_recovery_max"
                ).value
            ),
        )
        self.bypass_deep_retreat_distance = float(
            self.get_parameter(
                "bypass_deep_retreat_distance"
            ).value
        )
        self.bypass_deep_retreat_increment = float(
            self.get_parameter(
                "bypass_deep_retreat_increment"
            ).value
        )
        self.bypass_wide_lateral_extra = float(
            self.get_parameter(
                "bypass_wide_lateral_extra"
            ).value
        )
        self.bypass_wide_entry_extra = float(
            self.get_parameter(
                "bypass_wide_entry_extra"
            ).value
        )
        self.bypass_wide_pass_extra = float(
            self.get_parameter(
                "bypass_wide_pass_extra"
            ).value
        )

        self.visit_map_gate_max = float(
            self.get_parameter("visit_map_gate_max").value
        )
        self.visit_duplicate_radius = float(
            self.get_parameter("visit_duplicate_radius").value
        )
        self.far_obstacle_front_max = float(
            self.get_parameter("far_obstacle_front_max").value
        )
        self.far_obstacle_confirmations = max(
            1,
            int(
                self.get_parameter(
                    "far_obstacle_confirmations"
                ).value
            ),
        )
        self.far_obstacle_repeat_interval = float(
            self.get_parameter(
                "far_obstacle_repeat_interval"
            ).value
        )
        self.far_obstacle_distance_tolerance = float(
            self.get_parameter(
                "far_obstacle_distance_tolerance"
            ).value
        )
        self.completion_hide_nonvisited = bool(
            self.get_parameter(
                "completion_hide_nonvisited"
            ).value
        )
        self.clean_rviz_enabled = bool(
            self.get_parameter("clean_rviz_enabled").value
        )
        self.clean_rviz_show_active_target = bool(
            self.get_parameter(
                "clean_rviz_show_active_target"
            ).value
        )
        self.clean_rviz_dedupe_radius = float(
            self.get_parameter(
                "clean_rviz_dedupe_radius"
            ).value
        )
        self.target_lost_rescan_enabled = bool(
            self.get_parameter(
                "target_lost_rescan_enabled"
            ).value
        )
        self.target_lost_map_near_max = float(
            self.get_parameter(
                "target_lost_map_near_max"
            ).value
        )
        self.target_lost_tof_far_min = float(
            self.get_parameter(
                "target_lost_tof_far_min"
            ).value
        )
        self.target_lost_confirm_time = float(
            self.get_parameter(
                "target_lost_confirm_time"
            ).value
        )
        self.target_lost_min_track_age = float(
            self.get_parameter(
                "target_lost_min_track_age"
            ).value
        )
        self.target_lost_valid_near_max = float(
            self.get_parameter(
                "target_lost_valid_near_max"
            ).value
        )
        self.target_lost_valid_near_count = max(
            1,
            int(
                self.get_parameter(
                    "target_lost_valid_near_count"
                ).value
            ),
        )

        # Random dibuat reproducible dari seed run.
        seed = int(
            getattr(
                self,
                "normal_random_seed",
                self.get_parameter(
                    "normal_random_seed"
                ).value,
            )
        )
        self._bypass_rng = random.Random(seed + 314159)

        self._bypass_plan: Optional[RandomBypassPlan] = None

        # Sisi dipilih random satu kali untuk target aktif.
        self._side_lock_by_target: Dict[int, int] = {}

        # Maksimal satu flip pada setiap tingkat recovery.
        self._side_flip_used_by_target: Dict[int, bool] = {}

        # Tingkat recovery per target: 0=normal, 1/2=jalur makin lebar.
        self._recovery_attempt_by_target: Dict[int, int] = {}

        self._last_bypass_log = 0.0
        self._last_target_warning = 0.0

        self._far_guard_target_id: Optional[int] = None
        self._far_guard_count = 0
        self._far_guard_last_front = math.inf
        self._far_guard_last_time = 0.0

        self._completion_cleanup_done = False

        self._target_lost_guard_id: Optional[int] = None
        self._target_lost_guard_since = 0.0
        self._target_lost_guard_last_log = 0.0
        self._target_lost_guard_signature: Optional[Tuple[float, float, int, float]] = None
        self._target_lost_near_evidence_count = 0

        self.get_logger().info(
            "START TARGET_LOST_RESCAN "
            f"enabled={int(self.target_lost_rescan_enabled)} "
            f"map_near_max={self.target_lost_map_near_max:.2f}m "
            f"tof_far_min={self.target_lost_tof_far_min:.2f}m "
            f"confirm_time={self.target_lost_confirm_time:.1f}s "
            f"min_track_age={self.target_lost_min_track_age:.1f}s "
            f"valid_near_max={self.target_lost_valid_near_max:.2f}m "
            f"valid_near_count={self.target_lost_valid_near_count} "
            "latched=1 actual_used_for_control=0"
        )
        self.get_logger().info(
            "START FINAL_CLEAN_RVIZ "
            f"enabled={int(self.clean_rviz_enabled)} "
            f"show_active={int(self.clean_rviz_show_active_target)} "
            f"visual_dedupe_radius="
            f"{self.clean_rviz_dedupe_radius:.2f}m "
            "hide_yellow_orange=1 "
            "control_logic_changed=0"
        )
        self.get_logger().info(
            "START VISIT_GUARD_CLEAN "
            f"map_gate={self.visit_map_gate_max:.2f}m "
            f"duplicate_radius={self.visit_duplicate_radius:.2f}m "
            f"far_obstacle_front_max="
            f"{self.far_obstacle_front_max:.2f}m "
            "actual_used_for_control=0"
        )
        self.get_logger().info(
            "START RANDOM_BYPASS_RECOVERY "
            f"enabled={int(self.bypass_wide_recovery_enabled)} "
            f"max={self.bypass_wide_recovery_max} "
            f"deep_retreat={self.bypass_deep_retreat_distance:.2f}m "
            f"wide_extra={self.bypass_wide_lateral_extra:.2f}m "
            "permanent_hold=0"
        )
        self.get_logger().info(
            "START RANDOM_BYPASS_SIDEFIX "
            "change=side_yaw_plus_emergency_flip "
            f"emergency_before_flip="
            f"{self.bypass_emergency_before_flip}"
        )
        self.get_logger().info(
            "START RANDOM_OBSTACLE_BYPASS "
            f"virtual_left={self.obstacle_half_width:.2f}m "
            f"virtual_right={self.obstacle_half_width:.2f}m "
            f"safety_margin={self.drone_safety_margin:.2f}m "
            f"lateral_waypoint={self.bypass_lateral_offset:.2f}m "
            "side_policy=random_once_then_lock"
        )

    # ==============================================================
    # Utility
    # ==============================================================

    @staticmethod
    def _distance(a: Point2D, b: Point2D) -> float:
        return math.hypot(
            a[0] - b[0],
            a[1] - b[1],
        )

    @staticmethod
    def _point_segment_distance(
        point: Point2D,
        start: Point2D,
        end: Point2D,
    ) -> float:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        denominator = dx * dx + dy * dy

        if denominator <= 1.0e-9:
            return math.hypot(
                point[0] - start[0],
                point[1] - start[1],
            )

        ratio = (
            (point[0] - start[0]) * dx
            + (point[1] - start[1]) * dy
        ) / denominator
        ratio = max(0.0, min(1.0, ratio))

        nearest = (
            start[0] + ratio * dx,
            start[1] + ratio * dy,
        )
        return math.hypot(
            point[0] - nearest[0],
            point[1] - nearest[1],
        )

    @staticmethod
    def _state_name(state: object) -> str:
        return str(
            getattr(
                state,
                "name",
                getattr(state, "value", state),
            )
        )

    def _set_nav_state(self, new_state: object) -> None:
        old_state = getattr(self, "state", None)

        # Gunakan helper basis bila tersedia.
        for function_name in (
            "_set_state",
            "set_state",
            "_transition_state",
            "_transition_to",
        ):
            function = getattr(
                self,
                function_name,
                None,
            )
            if not callable(function):
                continue

            try:
                function(new_state)
                return
            except TypeError:
                continue
            except Exception as error:
                self.get_logger().warning(
                    "RANDOM_BYPASS_STATE_HELPER_FAIL "
                    f"function={function_name} "
                    f"error={error}"
                )

        self.state = new_state
        self.get_logger().info(
            "RANDOM_BYPASS_STATE "
            f"{self._state_name(old_state)} "
            f"-> {self._state_name(new_state)}"
        )

    def _clear_old_avoidance_motion(self) -> None:
        for attribute in (
            "motion_waypoint_xy",
            "motion_goal_xy",
            "motion_brake_anchor_xy",
            "avoid_goal_xy",
            "avoid_waypoint_xy",
            "early_avoid_goal_xy",
            "early_obstacle_goal_xy",
        ):
            if hasattr(self, attribute):
                try:
                    setattr(self, attribute, None)
                except Exception:
                    pass

    # ==============================================================
    # Resolve target aktif
    # ==============================================================

    def _track_from_value(
        self,
        value: object,
    ) -> Optional[Tuple[int, object]]:
        tracks = getattr(self, "tracks", {})
        if not isinstance(tracks, dict):
            return None

        if isinstance(value, bool):
            return None

        if isinstance(value, int):
            if value in tracks:
                return int(value), tracks[value]
            return None

        if hasattr(value, "tree_id"):
            try:
                tree_id = int(value.tree_id)
            except (TypeError, ValueError):
                return None

            if tree_id in tracks:
                return tree_id, tracks[tree_id]

        return None

    def _resolve_active_target(
        self,
    ) -> Optional[Tuple[int, object]]:
        tracks = getattr(self, "tracks", {})
        if not isinstance(tracks, dict) or not tracks:
            return None

        likely_attributes = (
            "current_target_id",
            "active_target_id",
            "selected_target_id",
            "navigation_target_id",
            "nav_target_id",
            "current_tree_id",
            "target_track_id",
            "current_target",
            "active_target",
            "selected_target",
            "target_track",
        )

        for attribute in likely_attributes:
            resolved = self._track_from_value(
                getattr(self, attribute, None)
            )
            if resolved is not None:
                return resolved

        # Cari atribut internal lain yang mengandung "target".
        for attribute, value in vars(self).items():
            name = attribute.lower()
            if "target" not in name:
                continue
            if any(
                blocked in name
                for blocked in (
                    "distance",
                    "timeout",
                    "tolerance",
                    "bearing",
                    "position",
                    "visual",
                    "actual",
                    "candidate",
                    "count",
                    "minimum",
                    "maximum",
                    "radius",
                )
            ):
                continue

            resolved = self._track_from_value(value)
            if resolved is not None:
                return resolved

        # Fallback: elemen pertama queue yang belum selesai.
        possible_queues = (
            "target_queue",
            "random_queue",
            "fixed_random_queue",
            "random_batch_queue",
            "random_batch_snapshot_v21n6",
        )

        for attribute in possible_queues:
            value = getattr(self, attribute, None)
            if value is None or isinstance(
                value,
                (str, bytes, dict),
            ):
                continue

            try:
                items = list(value)
            except TypeError:
                continue

            for item in items:
                resolved = self._track_from_value(item)
                if resolved is None:
                    continue

                tree_id, track = resolved
                if getattr(track, "state", None) not in (
                    TrackState.VISITED,
                    TrackState.REJECTED,
                ):
                    return tree_id, track

        now = time.monotonic()
        if now - self._last_target_warning >= 3.0:
            self._last_target_warning = now
            target_attributes = sorted(
                attribute
                for attribute in vars(self)
                if "target" in attribute.lower()
            )
            self.get_logger().warning(
                "RANDOM_BYPASS_TARGET_NOT_FOUND "
                f"candidate_attributes="
                f"{target_attributes[:20]}"
            )

        return None

    def _current_geometry(
        self,
    ) -> Optional[
        Tuple[
            int,
            object,
            Point2D,
            Point2D,
            float,
            float,
        ]
    ]:
        pose = getattr(self, "pose", None)
        resolved = self._resolve_active_target()

        if pose is None or resolved is None:
            return None

        target_id, track = resolved

        try:
            drone_xy = (
                float(pose.x_enu),
                float(pose.y_enu),
            )
            target_xy = (
                float(track.x),
                float(track.y),
            )
        except (TypeError, ValueError, AttributeError):
            return None

        target_distance = self._distance(
            drone_xy,
            target_xy,
        )
        front_distance = float(
            self._front_cluster_distance_v22()
        )

        if not (
            math.isfinite(target_distance)
            and math.isfinite(front_distance)
        ):
            return None

        return (
            target_id,
            track,
            drone_xy,
            target_xy,
            target_distance,
            front_distance,
        )

    # ==============================================================
    # Rencana obstacle virtual ±1 m
    # ==============================================================

    def _choose_or_keep_side(
        self,
        target_id: int,
    ) -> int:
        existing = self._side_lock_by_target.get(
            target_id
        )
        if existing in (-1, 1):
            return existing

        side = self._bypass_rng.choice((1, -1))
        self._side_lock_by_target[target_id] = side
        self._side_flip_used_by_target[target_id] = False

        self.get_logger().info(
            "RANDOM_BYPASS_SIDE_RANDOM "
            f"id={target_id} "
            f"side={'LEFT' if side > 0 else 'RIGHT'} "
            "locked=1"
        )
        return side

    def _build_plan(self) -> bool:
        geometry = self._current_geometry()
        if geometry is None:
            return False

        (
            target_id,
            _track,
            drone_xy,
            target_xy,
            target_distance,
            front_distance,
        ) = geometry

        delta_x = target_xy[0] - drone_xy[0]
        delta_y = target_xy[1] - drone_xy[1]
        norm = math.hypot(delta_x, delta_y)

        if norm <= 1.0e-6:
            return False

        # Arah target menjadi sumbu forward obstacle virtual.
        forward = (
            delta_x / norm,
            delta_y / norm,
        )
        left = (
            -forward[1],
            forward[0],
        )

        # Pusat obstacle diproyeksikan dari jarak ToF.
        obstacle_xy = (
            drone_xy[0]
            + forward[0] * front_distance,
            drone_xy[1]
            + forward[1] * front_distance,
        )

        side = self._choose_or_keep_side(target_id)

        recovery_attempt = int(
            self._recovery_attempt_by_target.get(target_id, 0)
        )
        lateral_offset = (
            self.bypass_lateral_offset
            + recovery_attempt * self.bypass_wide_lateral_extra
        )
        backup_distance = (
            self.bypass_backup_distance
            + recovery_attempt * 0.25
        )
        entry_behind = (
            self.bypass_entry_behind_obstacle
            + recovery_attempt * self.bypass_wide_entry_extra
        )
        pass_ahead = (
            self.bypass_pass_ahead_obstacle
            + recovery_attempt * self.bypass_wide_pass_extra
        )

        # Mundur dari pose sekarang.
        backup_goal = (
            drone_xy[0] - forward[0] * backup_distance,
            drone_xy[1] - forward[1] * backup_distance,
        )

        # Masuk ke sisi obstacle sedikit sebelum pusat obstacle.
        side_entry_goal = (
            obstacle_xy[0]
            + side * left[0] * lateral_offset
            - forward[0] * entry_behind,
            obstacle_xy[1]
            + side * left[1] * lateral_offset
            - forward[1] * entry_behind,
        )

        # Lewati obstacle sambil tetap di luar batas lateral.
        pass_goal = (
            obstacle_xy[0]
            + side * left[0] * lateral_offset
            + forward[0] * pass_ahead,
            obstacle_xy[1]
            + side * left[1] * lateral_offset
            + forward[1] * pass_ahead,
        )

        now = time.monotonic()
        self._bypass_plan = RandomBypassPlan(
            target_id=target_id,
            target_xy=target_xy,
            obstacle_xy=obstacle_xy,
            forward_unit=forward,
            left_unit=left,
            side=side,
            backup_goal=backup_goal,
            side_entry_goal=side_entry_goal,
            pass_goal=pass_goal,
            phase="HOLD",
            phase_started=now,
            hold_until=now + self.bypass_hold_time,
            last_progress_time=now,
            recovery_attempt=recovery_attempt,
        )

        self._clear_old_avoidance_motion()

        left_boundary = (
            obstacle_xy[0]
            + left[0] * self.obstacle_half_width,
            obstacle_xy[1]
            + left[1] * self.obstacle_half_width,
        )
        right_boundary = (
            obstacle_xy[0]
            - left[0] * self.obstacle_half_width,
            obstacle_xy[1]
            - left[1] * self.obstacle_half_width,
        )

        self.get_logger().warning(
            "RANDOM_BYPASS_START "
            f"id={target_id} "
            f"target_dist={target_distance:.2f} "
            f"tof_front={front_distance:.2f} "
            f"side={'LEFT' if side > 0 else 'RIGHT'} "
            f"obstacle=({obstacle_xy[0]:.2f},"
            f"{obstacle_xy[1]:.2f}) "
            f"left_limit=({left_boundary[0]:.2f},"
            f"{left_boundary[1]:.2f}) "
            f"right_limit=({right_boundary[0]:.2f},"
            f"{right_boundary[1]:.2f}) "
            f"safe_offset={lateral_offset:.2f} "
            f"recovery={recovery_attempt}/"
            f"{self.bypass_wide_recovery_max} "
            "queue_consumed=0"
        )
        return True

    def _should_take_over(self) -> bool:
        if not self.random_bypass_enabled:
            return False

        avoid_state = getattr(
            NavState,
            "AVOID_OBSTACLE",
            None,
        )
        if avoid_state is None:
            return False
        if getattr(self, "state", None) != avoid_state:
            return False

        geometry = self._current_geometry()
        if geometry is None:
            return False

        (
            _target_id,
            _track,
            _drone_xy,
            _target_xy,
            target_distance,
            front_distance,
        ) = geometry

        target_front_gap = (
            target_distance - front_distance
        )

        # Obstacle harus benar-benar berada sebelum target,
        # bukan target final pada tahap 3-2-1.
        return (
            target_distance
            >= self.bypass_target_min_distance
            and target_front_gap
            >= self.bypass_min_target_front_gap
        )

    # ==============================================================
    # Offboard movement
    # ==============================================================

    def _publish_position(
        self,
        x: float,
        y: float,
        yaw: float,
    ) -> None:
        publish_offboard = getattr(
            self,
            "_publish_offboard_mode",
            None,
        )
        if callable(publish_offboard):
            publish_offboard()

        self._publish_position_enu(
            float(x),
            float(y),
            float(self.flight_altitude),
            float(yaw),
        )

    def _hold_position(self, yaw: float) -> None:
        pose = getattr(self, "pose", None)
        if pose is None:
            return

        self._publish_position(
            float(pose.x_enu),
            float(pose.y_enu),
            yaw,
        )

    def _move_to_goal(
        self,
        goal: Point2D,
        yaw: float,
    ) -> float:
        pose = getattr(self, "pose", None)
        if pose is None:
            return math.inf

        current = (
            float(pose.x_enu),
            float(pose.y_enu),
        )

        delta_x = goal[0] - current[0]
        delta_y = goal[1] - current[1]
        remaining = math.hypot(
            delta_x,
            delta_y,
        )

        if remaining <= self.bypass_waypoint_tolerance:
            self._publish_position(
                goal[0],
                goal[1],
                yaw,
            )
            return remaining

        command_step = min(
            max(0.03, self.bypass_position_step),
            remaining,
        )

        command_x = (
            current[0]
            + delta_x / remaining * command_step
        )
        command_y = (
            current[1]
            + delta_y / remaining * command_step
        )

        self._publish_position(
            command_x,
            command_y,
            yaw,
        )
        return remaining

    # ==============================================================
    # Phase execution
    # ==============================================================

    def _yaw_to_target(
        self,
        plan: RandomBypassPlan,
    ) -> float:
        pose = getattr(self, "pose", None)
        if pose is None:
            return 0.0

        return math.atan2(
            plan.target_xy[1] - float(pose.y_enu),
            plan.target_xy[0] - float(pose.x_enu),
        )

    def _forward_yaw(
        self,
        plan: RandomBypassPlan,
    ) -> float:
        return math.atan2(
            plan.forward_unit[1],
            plan.forward_unit[0],
        )

    def _side_yaw(
        self,
        plan: RandomBypassPlan,
    ) -> float:
        """Yaw diarahkan ke sisi gerak agar ToF melihat jalur sidestep."""
        side_x = plan.side * plan.left_unit[0]
        side_y = plan.side * plan.left_unit[1]
        return math.atan2(side_y, side_x)

    def _set_phase(
        self,
        plan: RandomBypassPlan,
        phase: str,
    ) -> None:
        now = time.monotonic()
        plan.phase = phase
        plan.phase_started = now
        plan.last_remaining = math.inf
        plan.last_progress_time = now

        self.get_logger().info(
            "RANDOM_BYPASS_PHASE "
            f"id={plan.target_id} "
            f"phase={phase} "
            f"side={'LEFT' if plan.side > 0 else 'RIGHT'}"
        )

    def _progress_is_valid(
        self,
        plan: RandomBypassPlan,
        remaining: float,
    ) -> bool:
        now = time.monotonic()

        if (
            not math.isfinite(plan.last_remaining)
            or remaining
            <= plan.last_remaining
            - self.bypass_progress_epsilon
        ):
            plan.last_remaining = remaining
            plan.last_progress_time = now
            return True

        return (
            now - plan.last_progress_time
            <= self.bypass_progress_timeout
        )

    def _corridor_clearance(
        self,
        plan: RandomBypassPlan,
    ) -> float:
        pose = getattr(self, "pose", None)
        if pose is None:
            return 0.0

        current = (
            float(pose.x_enu),
            float(pose.y_enu),
        )

        return self._point_segment_distance(
            plan.obstacle_xy,
            current,
            plan.target_xy,
        )

    def _finish_success(
        self,
        plan: RandomBypassPlan,
        clearance: float,
    ) -> None:
        self.get_logger().info(
            "RANDOM_BYPASS_DONE "
            f"id={plan.target_id} "
            f"side={'LEFT' if plan.side > 0 else 'RIGHT'} "
            f"corridor_clearance={clearance:.2f} "
            "action=return_same_target "
            "queue_consumed=0"
        )

        self._recovery_attempt_by_target.pop(
            plan.target_id,
            None,
        )
        self._side_flip_used_by_target[
            plan.target_id
        ] = False

        self._bypass_plan = None
        self._clear_old_avoidance_motion()

        align_state = getattr(
            NavState,
            "ALIGN_TARGET",
            None,
        )
        if align_state is not None:
            self._set_nav_state(align_state)

    def _request_rescan_after_bypass_failure(
        self,
        target_id: int,
        reason: str,
    ) -> None:
        """Keluar dari avoidance tanpa HOLD permanen."""
        self.get_logger().error(
            "RANDOM_BYPASS_RECOVERY_RESCAN "
            f"id={target_id} "
            f"reason={reason} "
            "action=clear_plan_and_rescan "
            "queue_consumed=0"
        )

        self._bypass_plan = None
        self._clear_old_avoidance_motion()
        self._side_lock_by_target.pop(target_id, None)
        self._side_flip_used_by_target.pop(target_id, None)
        self._recovery_attempt_by_target.pop(target_id, None)

        for method_name in (
            "_start_rescan",
            "_start_scan360",
            "_begin_rescan",
            "_begin_scan",
        ):
            method = getattr(self, method_name, None)
            if not callable(method):
                continue
            try:
                method()
                return
            except TypeError:
                continue
            except Exception:
                continue

        rotate_scan = getattr(NavState, "ROTATE_SCAN", None)
        if rotate_scan is not None:
            self._set_nav_state(rotate_scan)
            return

        select_target = getattr(NavState, "SELECT_TARGET", None)
        if select_target is not None:
            self._set_nav_state(select_target)
            return

        align_target = getattr(NavState, "ALIGN_TARGET", None)
        if align_target is not None:
            self._set_nav_state(align_target)

    def _start_wide_recovery(
        self,
        plan: RandomBypassPlan,
        reason: str,
    ) -> None:
        target_id = int(plan.target_id)
        next_attempt = int(
            self._recovery_attempt_by_target.get(target_id, 0)
        ) + 1

        if (
            not self.bypass_wide_recovery_enabled
            or next_attempt > self.bypass_wide_recovery_max
        ):
            self._request_rescan_after_bypass_failure(
                target_id,
                reason=(
                    f"recovery_exhausted_{reason}"
                ),
            )
            return

        pose = getattr(self, "pose", None)
        if pose is None:
            self._request_rescan_after_bypass_failure(
                target_id,
                reason="no_pose_for_deep_retreat",
            )
            return

        self._recovery_attempt_by_target[target_id] = next_attempt
        retreat_distance = (
            self.bypass_deep_retreat_distance
            + (next_attempt - 1)
            * self.bypass_deep_retreat_increment
        )

        current = (
            float(pose.x_enu),
            float(pose.y_enu),
        )
        plan.backup_goal = (
            current[0]
            - plan.forward_unit[0] * retreat_distance,
            current[1]
            - plan.forward_unit[1] * retreat_distance,
        )
        plan.recovery_attempt = next_attempt
        plan.emergency_backup_count = 0

        self.get_logger().warning(
            "RANDOM_BYPASS_WIDE_RECOVERY "
            f"id={target_id} "
            f"reason={reason} "
            f"attempt={next_attempt}/"
            f"{self.bypass_wide_recovery_max} "
            f"deep_retreat={retreat_distance:.2f}m "
            f"next_safe_offset="
            f"{self.bypass_lateral_offset + next_attempt * self.bypass_wide_lateral_extra:.2f}m "
            "action=retreat_then_replan "
            "queue_consumed=0"
        )
        self._set_phase(plan, "DEEP_RETREAT")

    def _restart_on_opposite_side(
        self,
        plan: RandomBypassPlan,
        reason: str,
    ) -> None:
        already_flipped = self._side_flip_used_by_target.get(
            plan.target_id,
            False,
        )

        if not already_flipped:
            new_side = -plan.side
            self._side_lock_by_target[
                plan.target_id
            ] = new_side
            self._side_flip_used_by_target[
                plan.target_id
            ] = True

            self.get_logger().warning(
                "RANDOM_BYPASS_SIDE_RETRY "
                f"id={plan.target_id} "
                f"reason={reason} "
                f"old={'LEFT' if plan.side > 0 else 'RIGHT'} "
                f"new={'LEFT' if new_side > 0 else 'RIGHT'} "
                "action=retry_once_queue_unchanged"
            )

            self._bypass_plan = None
            self._clear_old_avoidance_motion()

            avoid_state = getattr(
                NavState,
                "AVOID_OBSTACLE",
                None,
            )
            if avoid_state is not None:
                self._set_nav_state(avoid_state)

            # Bangun rencana baru dari pose terbaru.
            self._build_plan()
            return

        # Kedua sisi normal gagal. Jangan HOLD permanen:
        # mundur lebih jauh lalu replanning dengan koridor lebih lebar.
        self.get_logger().error(
            "RANDOM_BYPASS_BOTH_SIDES_FAILED "
            f"id={plan.target_id} "
            f"reason={reason} "
            "action=wide_recovery_not_permanent_hold "
            "queue_consumed=0"
        )
        self._start_wide_recovery(plan, reason)

    def _log_motion(
        self,
        plan: RandomBypassPlan,
        remaining: float,
        front_distance: float,
    ) -> None:
        now = time.monotonic()
        if now - self._last_bypass_log < 0.50:
            return

        self._last_bypass_log = now
        self.get_logger().info(
            "RANDOM_BYPASS_MOVE "
            f"id={plan.target_id} "
            f"phase={plan.phase} "
            f"side={'LEFT' if plan.side > 0 else 'RIGHT'} "
            f"remaining={remaining:.2f} "
            f"tof_front={front_distance:.2f} "
            "queue_consumed=0"
        )

    def _run_bypass(self) -> None:
        plan = self._bypass_plan
        if plan is None:
            return

        now = time.monotonic()
        yaw_to_target = self._yaw_to_target(plan)
        front_distance = float(
            self._front_cluster_distance_v22()
        )

        # Tidak boleh memakai fase selamanya.
        if (
            now - plan.phase_started
            > self.bypass_phase_timeout
            and plan.phase != "FAILSAFE_HOLD"
        ):
            if plan.phase == "DEEP_RETREAT":
                self._request_rescan_after_bypass_failure(
                    plan.target_id,
                    reason="timeout_DEEP_RETREAT",
                )
            else:
                self._restart_on_opposite_side(
                    plan,
                    reason=f"timeout_{plan.phase}",
                )
            return

        # Bila obstacle menjadi sangat dekat saat sidestep/pass,
        # kembali ke backup terlebih dahulu.
        if (
            plan.phase
            not in (
                "HOLD",
                "BACKUP",
                "DEEP_RETREAT",
                "FAILSAFE_HOLD",
            )
            and math.isfinite(front_distance)
            and front_distance
            < self.bypass_emergency_front
        ):
            pose = getattr(self, "pose", None)
            if pose is not None:
                current = (
                    float(pose.x_enu),
                    float(pose.y_enu),
                )
                plan.backup_goal = (
                    current[0]
                    - plan.forward_unit[0]
                    * self.bypass_backup_distance,
                    current[1]
                    - plan.forward_unit[1]
                    * self.bypass_backup_distance,
                )

                plan.emergency_backup_count += 1

                self.get_logger().warning(
                    "RANDOM_BYPASS_EMERGENCY_BACKUP "
                    f"id={plan.target_id} "
                    f"tof_front={front_distance:.2f} "
                    f"count={plan.emergency_backup_count}/"
                    f"{self.bypass_emergency_before_flip}"
                )

                if (
                    plan.emergency_backup_count
                    >= self.bypass_emergency_before_flip
                ):
                    self._restart_on_opposite_side(
                        plan,
                        reason=(
                            "repeated_emergency_front_"
                            f"{front_distance:.2f}"
                        ),
                    )
                    return

                self._set_phase(plan, "BACKUP")

        if plan.phase == "DEEP_RETREAT":
            remaining = self._move_to_goal(
                plan.backup_goal,
                yaw_to_target,
            )

            if remaining <= self.bypass_waypoint_tolerance:
                target_id = int(plan.target_id)
                attempt = int(plan.recovery_attempt)

                # Setelah jauh dari obstacle, pilih sisi baru secara
                # reproducible dan izinkan satu flip lagi.
                new_side = self._bypass_rng.choice((1, -1))
                self._side_lock_by_target[target_id] = new_side
                self._side_flip_used_by_target[target_id] = False

                self.get_logger().warning(
                    "RANDOM_BYPASS_RECOVERY_REPLAN "
                    f"id={target_id} "
                    f"attempt={attempt}/"
                    f"{self.bypass_wide_recovery_max} "
                    f"side={'LEFT' if new_side > 0 else 'RIGHT'} "
                    "action=build_wider_plan"
                )

                self._bypass_plan = None
                self._clear_old_avoidance_motion()
                avoid_state = getattr(
                    NavState,
                    "AVOID_OBSTACLE",
                    None,
                )
                if avoid_state is not None:
                    self._set_nav_state(avoid_state)

                if not self._build_plan():
                    self._request_rescan_after_bypass_failure(
                        target_id,
                        reason="wide_replan_build_failed",
                    )
                return

            if not self._progress_is_valid(plan, remaining):
                self._request_rescan_after_bypass_failure(
                    plan.target_id,
                    reason="deep_retreat_no_progress",
                )
                return

            self._log_motion(
                plan,
                remaining,
                front_distance,
            )
            return

        if plan.phase == "HOLD":
            self._hold_position(yaw_to_target)

            if now >= plan.hold_until:
                self._set_phase(
                    plan,
                    "BACKUP",
                )
            return

        if plan.phase == "BACKUP":
            remaining = self._move_to_goal(
                plan.backup_goal,
                yaw_to_target,
            )

            if remaining <= self.bypass_waypoint_tolerance:
                self._set_phase(
                    plan,
                    "SIDE_ENTRY",
                )
            elif not self._progress_is_valid(
                plan,
                remaining,
            ):
                self._restart_on_opposite_side(
                    plan,
                    reason="backup_no_progress",
                )

            self._log_motion(
                plan,
                remaining,
                front_distance,
            )
            return

        if plan.phase == "SIDE_ENTRY":
            # Hadapkan sensor ke arah gerak lateral. Dengan cara ini ToF
            # memeriksa jalur sidestep, bukan terus menatap batang yang
            # memang sedang dihindari.
            remaining = self._move_to_goal(
                plan.side_entry_goal,
                self._side_yaw(plan),
            )

            if remaining <= self.bypass_waypoint_tolerance:
                self._set_phase(
                    plan,
                    "PASS_FORWARD",
                )
            elif not self._progress_is_valid(
                plan,
                remaining,
            ):
                self._restart_on_opposite_side(
                    plan,
                    reason="side_entry_no_progress",
                )

            self._log_motion(
                plan,
                remaining,
                front_distance,
            )
            return

        if plan.phase == "PASS_FORWARD":
            remaining = self._move_to_goal(
                plan.pass_goal,
                self._forward_yaw(plan),
            )

            if remaining <= self.bypass_waypoint_tolerance:
                self._set_phase(
                    plan,
                    "VERIFY_CLEAR",
                )
            elif not self._progress_is_valid(
                plan,
                remaining,
            ):
                self._restart_on_opposite_side(
                    plan,
                    reason="pass_forward_no_progress",
                )

            self._log_motion(
                plan,
                remaining,
                front_distance,
            )
            return

        if plan.phase == "VERIFY_CLEAR":
            self._hold_position(yaw_to_target)

            if (
                now - plan.phase_started
                < self.bypass_verify_time
            ):
                return

            clearance = self._corridor_clearance(plan)
            required_clearance = (
                self.obstacle_half_width
                + self.drone_safety_margin
                + self.bypass_clearance_margin
            )

            if clearance >= required_clearance:
                self._finish_success(
                    plan,
                    clearance,
                )
                return

            if (
                plan.forward_extensions
                < self.bypass_max_extensions
            ):
                plan.forward_extensions += 1
                plan.pass_goal = (
                    plan.pass_goal[0]
                    + plan.forward_unit[0]
                    * self.bypass_forward_extension,
                    plan.pass_goal[1]
                    + plan.forward_unit[1]
                    * self.bypass_forward_extension,
                )

                self.get_logger().warning(
                    "RANDOM_BYPASS_EXTEND_FORWARD "
                    f"id={plan.target_id} "
                    f"extension="
                    f"{plan.forward_extensions}/"
                    f"{self.bypass_max_extensions} "
                    f"clearance={clearance:.2f} "
                    f"required={required_clearance:.2f}"
                )
                self._set_phase(
                    plan,
                    "PASS_FORWARD",
                )
                return

            self._restart_on_opposite_side(
                plan,
                reason=(
                    "corridor_not_clear_"
                    f"{clearance:.2f}"
                ),
            )
            return

        if plan.phase == "FAILSAFE_HOLD":
            self._hold_position(yaw_to_target)
            self._request_rescan_after_bypass_failure(
                plan.target_id,
                reason="legacy_failsafe_hold_auto_release",
            )
            return

        self._restart_on_opposite_side(
            plan,
            reason=f"unknown_phase_{plan.phase}",
        )

    # ==============================================================
    # Recovery target peta tercapai tetapi batang tidak ada di sensor
    # ==============================================================

    def _reset_target_lost_guard(self) -> None:
        self._target_lost_guard_id = None
        self._target_lost_guard_since = 0.0
        self._target_lost_guard_last_log = 0.0
        self._target_lost_guard_signature = None
        self._target_lost_near_evidence_count = 0

    @staticmethod
    def _track_age_for_target_lost(
        track: object,
        now: float,
    ) -> float:
        for name in ("updated_mono", "last_seen", "created_mono"):
            value = getattr(track, name, None)
            try:
                stamp = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(stamp) and stamp > 0.0:
                return max(0.0, now - stamp)
        return math.inf

    @staticmethod
    def _target_lost_track_signature(
        track: object,
    ) -> Tuple[float, float, int, float]:
        try:
            x = float(getattr(track, "x"))
            y = float(getattr(track, "y"))
        except (TypeError, ValueError, AttributeError):
            x = math.nan
            y = math.nan

        try:
            obs = int(getattr(track, "obs_count", 0))
        except (TypeError, ValueError):
            obs = 0

        last_seen = 0.0
        for name in ("updated_mono", "last_seen", "created_mono"):
            try:
                candidate = float(getattr(track, name))
            except (TypeError, ValueError, AttributeError):
                continue
            if math.isfinite(candidate):
                last_seen = candidate
                break

        return x, y, obs, last_seen

    @staticmethod
    def _target_lost_signature_changed(
        before: Optional[Tuple[float, float, int, float]],
        after: Tuple[float, float, int, float],
    ) -> bool:
        if before is None:
            return False

        bx, by, bobs, bseen = before
        ax, ay, aobs, aseen = after

        moved = False
        if all(math.isfinite(v) for v in (bx, by, ax, ay)):
            moved = math.hypot(ax - bx, ay - by) >= 0.15

        return moved or aobs > bobs or aseen > bseen + 0.05

    def _clear_active_target_after_lost(
        self,
        target_id: int,
        track: object,
    ) -> None:
        for name in (
            "current_target_id",
            "active_target_id",
            "selected_target_id",
            "navigation_target_id",
            "nav_target_id",
            "current_tree_id",
            "target_track_id",
        ):
            if not hasattr(self, name):
                continue
            value = getattr(self, name, None)
            try:
                same = int(value) == int(target_id)
            except (TypeError, ValueError):
                same = False
            if same:
                try:
                    setattr(self, name, None)
                except Exception:
                    pass

        for name in (
            "current_target",
            "active_target",
            "selected_target",
            "target_track",
        ):
            if hasattr(self, name) and getattr(self, name, None) is track:
                try:
                    setattr(self, name, None)
                except Exception:
                    pass

    def _start_target_lost_rescan(
        self,
        target_id: int,
    ) -> None:
        reason = f"target_lost_map_relocalize_id_{target_id}"

        method = getattr(self, "start_rotate_scan", None)
        if callable(method):
            try:
                method(reason)
                self.get_logger().warning(
                    "TARGET_LOST_RESCAN_STARTED "
                    f"id={target_id} method=start_rotate_scan "
                    "queue_consumed=0 actual_used_for_control=0"
                )
                return
            except Exception as exc:
                self.get_logger().error(
                    "TARGET_LOST_RESCAN_START_ERROR "
                    f"id={target_id} method=start_rotate_scan "
                    f"error={exc!r}"
                )

        for method_name in (
            "_start_rescan",
            "_start_scan360",
            "_begin_rescan",
            "_begin_scan",
        ):
            method = getattr(self, method_name, None)
            if not callable(method):
                continue
            try:
                method()
                self.get_logger().warning(
                    "TARGET_LOST_RESCAN_STARTED "
                    f"id={target_id} method={method_name} "
                    "queue_consumed=0 actual_used_for_control=0"
                )
                return
            except Exception:
                continue

        rotate_scan = getattr(NavState, "ROTATE_SCAN", None)
        if rotate_scan is not None:
            self._set_nav_state(rotate_scan)
            self.get_logger().warning(
                "TARGET_LOST_RESCAN_STARTED "
                f"id={target_id} method=state_ROTATE_SCAN "
                "queue_consumed=0 actual_used_for_control=0"
            )
            return

        select_target = getattr(NavState, "SELECT_TARGET", None)
        if select_target is not None:
            self._set_nav_state(select_target)
            self.get_logger().warning(
                "TARGET_LOST_RESCAN_FALLBACK "
                f"id={target_id} method=state_SELECT_TARGET"
            )

    def _target_lost_map_recovery(self) -> bool:
        """
        Recovery dilatch setelah mismatch pertama.

        Bug sebelumnya: TARGET_LOST_MAP_RESCAN_PENDING dapat hilang pada
        tick berikutnya ketika ToF sesaat membaca sekitar 3 m. Parent lalu
        menjalankan mini-scan 3 m, mempertahankan koordinat lama, dan masuk
        TO_2M. Versi ini menahan state machine parent selama masa konfirmasi.
        """
        if not self.target_lost_rescan_enabled:
            self._reset_target_lost_guard()
            return False

        approach_state = getattr(NavState, "APPROACH", None)
        if (
            approach_state is None
            or getattr(self, "state", None) != approach_state
            or str(getattr(self, "tof_approach_stage", "")) != "TO_3M"
        ):
            self._reset_target_lost_guard()
            return False

        resolved = self._resolve_active_target()
        pose = getattr(self, "pose", None)
        if resolved is None or pose is None:
            self._reset_target_lost_guard()
            return False

        target_id, track = resolved
        try:
            target_distance = math.hypot(
                float(track.x) - float(pose.x_enu),
                float(track.y) - float(pose.y_enu),
            )
        except (TypeError, ValueError, AttributeError):
            self._reset_target_lost_guard()
            return False

        try:
            front_distance = float(self._front_cluster_distance_v22())
        except Exception:
            front_distance = math.inf

        now = time.monotonic()
        track_age = self._track_age_for_target_lost(track, now)
        current_signature = self._target_lost_track_signature(track)
        guard_active = self._target_lost_guard_id == int(target_id)

        mismatch_now = (
            target_distance <= self.target_lost_map_near_max
            and (
                not math.isfinite(front_distance)
                or front_distance >= self.target_lost_tof_far_min
            )
            and track_age >= self.target_lost_min_track_age
        )

        if not guard_active:
            if not mismatch_now:
                self._reset_target_lost_guard()
                return False

            self._target_lost_guard_id = int(target_id)
            self._target_lost_guard_since = now
            self._target_lost_guard_last_log = 0.0
            self._target_lost_guard_signature = current_signature
            self._target_lost_near_evidence_count = 0
            guard_active = True

            self.get_logger().warning(
                "TARGET_LOST_MAP_RESCAN_LATCHED "
                f"id={target_id} map_dist={target_distance:.2f} "
                f"tof={front_distance:.2f} track_age={track_age:.1f}s "
                "parent_tof3_gate_blocked=1"
            )

        # Sesudah latch aktif, pembacaan sementara sekitar 3 m tidak boleh
        # mengembalikan kontrol ke parent. Hanya bukti dekat yang stabil
        # (sesuai layer 2/1) atau update track baru yang boleh membatalkan.
        fresh_track_evidence = self._target_lost_signature_changed(
            self._target_lost_guard_signature,
            current_signature,
        )
        valid_near = (
            math.isfinite(front_distance)
            and front_distance <= self.target_lost_valid_near_max
        )

        if valid_near:
            self._target_lost_near_evidence_count += 1
        else:
            self._target_lost_near_evidence_count = 0

        if (
            self._target_lost_near_evidence_count
            >= self.target_lost_valid_near_count
            or (fresh_track_evidence and valid_near)
        ):
            self.get_logger().warning(
                "TARGET_LOST_MAP_RESCAN_CANCELLED "
                f"id={target_id} map_dist={target_distance:.2f} "
                f"tof={front_distance:.2f} "
                f"near_count={self._target_lost_near_evidence_count} "
                f"fresh_track={int(fresh_track_evidence)} "
                "reason=stable_near_target_evidence"
            )
            self._reset_target_lost_guard()
            return False

        elapsed = now - self._target_lost_guard_since

        if now - self._target_lost_guard_last_log >= 1.0:
            self._target_lost_guard_last_log = now
            self.get_logger().warning(
                "TARGET_LOST_MAP_RESCAN_PENDING "
                f"id={target_id} "
                f"elapsed={elapsed:.1f}/"
                f"{self.target_lost_confirm_time:.1f}s "
                f"map_dist={target_distance:.2f} "
                f"tof={front_distance:.2f} "
                f"track_age={track_age:.1f}s "
                f"near_count={self._target_lost_near_evidence_count}/"
                f"{self.target_lost_valid_near_count} "
                "latched=1 parent_tof3_gate_blocked=1"
            )

        self._hold_current_v22("target_lost_map_guard_latched")

        if elapsed < self.target_lost_confirm_time:
            return True

        try:
            track.state = TrackState.REJECTED
        except Exception:
            pass
        for name, value in (
            ("rejected", True),
            ("rejected_reason", "map_reached_sensor_target_absent"),
            ("ghost", True),
        ):
            try:
                setattr(track, name, value)
            except Exception:
                pass

        try:
            self._kalman_tracks.pop(int(target_id), None)
        except Exception:
            pass
        try:
            self._last_track_update.pop(int(target_id), None)
        except Exception:
            pass

        self._clear_active_target_after_lost(int(target_id), track)
        self._clear_tof_for_visit_guard()
        self._clear_old_avoidance_motion()
        self._side_lock_by_target.pop(int(target_id), None)
        self._side_flip_used_by_target.pop(int(target_id), None)
        self._recovery_attempt_by_target.pop(int(target_id), None)

        try:
            self.tof_approach_stage = "TO_3M"
        except Exception:
            pass

        self.get_logger().error(
            "TARGET_LOST_MAP_REJECT_RESCAN "
            f"id={target_id} "
            f"map_dist={target_distance:.2f} "
            f"tof={front_distance:.2f} "
            f"track_age={track_age:.1f}s "
            "old_track=REJECTED "
            "action=stationary_360_relocalize "
            "queue_consumed=0 "
            "actual_used_for_control=0"
        )

        self._reset_target_lost_guard()
        self._start_target_lost_rescan(int(target_id))
        return True

    # ==============================================================
    # Final clean RViz — visualisasi saja, tidak memengaruhi kontrol
    # ==============================================================

    def _iter_tracks_clean_rviz(self) -> List[object]:
        tracks = getattr(self, "tracks", {})
        if isinstance(tracks, dict):
            return list(tracks.values())
        try:
            return list(tracks)
        except TypeError:
            return []

    @staticmethod
    def _track_id_clean_rviz(track: object) -> Optional[int]:
        try:
            return int(getattr(track, "tree_id"))
        except (TypeError, ValueError, AttributeError):
            return None

    @staticmethod
    def _track_xy_map_clean_rviz(
        track: object,
    ) -> Optional[Tuple[float, float]]:
        for x_name, y_name in (
            ("map_x", "map_y"),
            ("x", "y"),
        ):
            try:
                x = float(getattr(track, x_name))
                y = float(getattr(track, y_name))
            except (TypeError, ValueError, AttributeError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                return x, y
        return None

    @staticmethod
    def _track_is_visited_clean_rviz(track: object) -> bool:
        if bool(getattr(track, "visited", False)):
            return True
        return getattr(track, "state", None) == TrackState.VISITED

    @staticmethod
    def _track_is_rejected_clean_rviz(track: object) -> bool:
        if bool(getattr(track, "rejected", False)):
            return True
        return getattr(track, "state", None) == TrackState.REJECTED

    @staticmethod
    def _track_visual_strength_clean_rviz(track: object) -> float:
        total = 0.0
        for name, weight in (
            ("views", 30.0),
            ("view_count", 30.0),
            ("obs_count", 5.0),
            ("observations", 5.0),
            ("hits", 4.0),
            ("hit_count", 4.0),
            ("strong_hits", 10.0),
            ("score", 0.01),
        ):
            try:
                value = float(getattr(track, name))
            except (TypeError, ValueError, AttributeError):
                continue
            if math.isfinite(value):
                total += weight * value
        return total

    def _visible_visited_tracks_clean_rviz(
        self,
    ) -> List[object]:
        """
        NMS hanya untuk tampilan RViz.
        Track internal dan hasil CSV tidak diubah.
        """
        candidates = [
            track
            for track in self._iter_tracks_clean_rviz()
            if (
                self._track_is_visited_clean_rviz(track)
                and not self._track_is_rejected_clean_rviz(track)
                and self._track_xy_map_clean_rviz(track) is not None
            )
        ]

        candidates.sort(
            key=lambda track: (
                -self._track_visual_strength_clean_rviz(track),
                self._track_id_clean_rviz(track)
                if self._track_id_clean_rviz(track) is not None
                else 10**9,
            )
        )

        kept: List[object] = []
        for track in candidates:
            xy = self._track_xy_map_clean_rviz(track)
            if xy is None:
                continue

            if any(
                math.hypot(
                    xy[0] - kept_xy[0],
                    xy[1] - kept_xy[1],
                )
                <= self.clean_rviz_dedupe_radius
                for kept_track in kept
                for kept_xy in [
                    self._track_xy_map_clean_rviz(kept_track)
                ]
                if kept_xy is not None
            ):
                continue

            kept.append(track)

        return kept

    def _active_track_clean_rviz(self) -> Optional[object]:
        if not self.clean_rviz_show_active_target:
            return None

        active_id = getattr(self, "active_target_id", None)
        if active_id is None:
            return None

        try:
            active_id_int = int(active_id)
        except (TypeError, ValueError):
            return None

        for track in self._iter_tracks_clean_rviz():
            if self._track_id_clean_rviz(track) != active_id_int:
                continue
            if self._track_is_rejected_clean_rviz(track):
                return None
            if self._track_is_visited_clean_rviz(track):
                return None
            if self._track_xy_map_clean_rviz(track) is None:
                return None
            return track

        return None

    @staticmethod
    def _delete_all_marker_array(
        frame_id: str,
        namespace: str,
        stamp: object,
    ) -> MarkerArray:
        arr = MarkerArray()
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = 0
        marker.action = Marker.DELETEALL
        arr.markers.append(marker)
        return arr

    def _clear_debug_visual_topics_clean_rviz(self) -> None:
        stamp = self.get_clock().now().to_msg()

        debug_pub = getattr(self, "debug_pc_pub", None)
        if debug_pub is not None:
            try:
                debug_pub.publish(
                    self._delete_all_marker_array(
                        "map",
                        "debug_pc_pipeline",
                        stamp,
                    )
                )
            except Exception:
                pass

        # PointCloud2 debug lama juga dikosongkan agar RViz tidak
        # menyimpan accepted/rejected cloud dari scan sebelumnya.
        if hasattr(self, "make_xyz_cloud"):
            for pub_name in (
                "debug_roi_cloud_pub",
                "debug_accepted_cloud_pub",
                "debug_rejected_cloud_pub",
            ):
                pub = getattr(self, pub_name, None)
                if pub is None:
                    continue
                try:
                    pub.publish(self.make_xyz_cloud([]))
                except Exception:
                    pass

    def publish_trunk_models(self) -> None:
        """
        Bersihkan seluruh model batang tambahan.

        Dengan demikian satu hasil deteksi hanya tampil melalui
        /sawit/tree_markers, bukan sphere + cylinder sekaligus.
        """
        pub = getattr(self, "trunk_model_pub", None)
        if pub is None:
            return

        try:
            pub.publish(
                self._delete_all_marker_array(
                    "map",
                    "sawit_trunk_models",
                    self.get_clock().now().to_msg(),
                )
            )
        except Exception:
            pass

    def publish_markers(self) -> None:
        """
        Tampilan final:
        - hijau: landmark VISITED unik;
        - ungu: target aktif, opsional;
        - merah: drone;
        - biru actual tetap diterbitkan oleh publisher pembanding;
        - tidak ada marker kandidat kuning/oranye.

        Fungsi ini tidak mengubah track, queue, Kalman, ToF,
        PointCloud, status visited, atau CSV.
        """
        if not self.clean_rviz_enabled:
            return super().publish_markers()

        now = time.monotonic()
        if now - float(getattr(self, "last_marker_time", 0.0)) < 0.50:
            return
        self.last_marker_time = now

        stamp = self.get_clock().now().to_msg()
        arr = self._delete_all_marker_array(
            "map",
            "sawit_v17_trees",
            stamp,
        )

        visited_tracks = self._visible_visited_tracks_clean_rviz()
        visited_positions: List[Tuple[float, float]] = []

        for track in visited_tracks:
            tree_id = self._track_id_clean_rviz(track)
            xy = self._track_xy_map_clean_rviz(track)
            if tree_id is None or xy is None:
                continue

            visited_positions.append(xy)

            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = stamp
            marker.ns = "sawit_v17_trees"
            marker.id = tree_id + 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(xy[0])
            marker.pose.position.y = float(xy[1])
            marker.pose.position.z = 1.20
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.78
            marker.scale.y = 0.78
            marker.scale.z = 0.78
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.95
            arr.markers.append(marker)

        active_track = self._active_track_clean_rviz()
        if active_track is not None:
            active_id = self._track_id_clean_rviz(active_track)
            active_xy = self._track_xy_map_clean_rviz(active_track)

            not_near_visited = (
                active_xy is not None
                and not any(
                    math.hypot(
                        active_xy[0] - visited_xy[0],
                        active_xy[1] - visited_xy[1],
                    )
                    <= self.clean_rviz_dedupe_radius
                    for visited_xy in visited_positions
                )
            )

            if (
                active_id is not None
                and active_xy is not None
                and not_near_visited
            ):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = stamp
                marker.ns = "sawit_v17_trees"
                marker.id = active_id + 1
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = float(active_xy[0])
                marker.pose.position.y = float(active_xy[1])
                marker.pose.position.z = 1.20
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.68
                marker.scale.y = 0.68
                marker.scale.z = 0.68
                marker.color.r = 0.75
                marker.color.g = 0.0
                marker.color.b = 1.0
                marker.color.a = 0.95
                arr.markers.append(marker)

        # Drone marker merah.
        pose_x = getattr(self, "x", None)
        pose_y = getattr(self, "y", None)
        if pose_x is None or pose_y is None:
            pose = getattr(self, "pose", None)
            if pose is not None:
                pose_x = getattr(pose, "x_enu", None)
                pose_y = getattr(pose, "y_enu", None)

        try:
            drone_x = float(pose_x)
            drone_y = float(pose_y)
            if hasattr(self, "local_to_map"):
                drone_x, drone_y = self.local_to_map(
                    drone_x,
                    drone_y,
                )

            drone = Marker()
            drone.header.frame_id = "map"
            drone.header.stamp = stamp
            drone.ns = "drone"
            drone.id = 9000
            drone.type = Marker.CUBE
            drone.action = Marker.ADD
            drone.pose.position.x = float(drone_x)
            drone.pose.position.y = float(drone_y)
            drone.pose.position.z = 1.10
            drone.pose.orientation.w = 1.0
            drone.scale.x = 0.90
            drone.scale.y = 0.90
            drone.scale.z = 0.45
            drone.color.r = 1.0
            drone.color.g = 0.0
            drone.color.b = 0.0
            drone.color.a = 0.95
            arr.markers.append(drone)
        except (TypeError, ValueError):
            pass

        self.tree_pub.publish(arr)

        # Hapus sumber visual ganda dan debug lama.
        self.publish_trunk_models()
        self._clear_debug_visual_topics_clean_rviz()

        # Actual biru dan rute tetap untuk evaluasi visual.
        self.publish_actual_tree_markers()
        self.publish_route_marker()

    # ==============================================================
    # Visit guard dan completion cleanup
    # ==============================================================

    def _reset_far_obstacle_guard(self) -> None:
        self._far_guard_target_id = None
        self._far_guard_count = 0
        self._far_guard_last_front = math.inf
        self._far_guard_last_time = 0.0

    def _far_target_obstacle_takeover(self) -> bool:
        """
        Contoh bug yang dicegah:
        target peta masih 11 m, tetapi ToF melihat pohon lain 2 m.
        Objek 2 m harus dihindari, bukan membuat target jauh menjadi visited.
        """
        if not self.random_bypass_enabled:
            self._reset_far_obstacle_guard()
            return False
        if self._bypass_plan is not None:
            return False

        allowed_states = {
            getattr(NavState, "APPROACH", None),
            getattr(NavState, "CLOSE_SETTLE", None),
            getattr(NavState, "CLOSE_FLUSH", None),
            getattr(NavState, "CLOSE_COLLECT", None),
        }
        allowed_states.discard(None)

        if getattr(self, "state", None) not in allowed_states:
            self._reset_far_obstacle_guard()
            return False

        geometry = self._current_geometry()
        if geometry is None:
            self._reset_far_obstacle_guard()
            return False

        (
            target_id,
            _track,
            _drone_xy,
            _target_xy,
            target_distance,
            front_distance,
        ) = geometry

        gap = target_distance - front_distance
        condition = (
            target_distance >= self.bypass_target_min_distance
            and front_distance >= 0.45
            and front_distance <= self.far_obstacle_front_max
            and gap >= self.bypass_min_target_front_gap
        )

        if not condition:
            self._reset_far_obstacle_guard()
            return False

        now = time.monotonic()
        same_reading = (
            self._far_guard_target_id == int(target_id)
            and now - self._far_guard_last_time
            <= self.far_obstacle_repeat_interval
            and abs(front_distance - self._far_guard_last_front)
            <= self.far_obstacle_distance_tolerance
        )

        if same_reading:
            self._far_guard_count += 1
        else:
            self._far_guard_target_id = int(target_id)
            self._far_guard_count = 1

        self._far_guard_last_front = front_distance
        self._far_guard_last_time = now

        if self._far_guard_count < self.far_obstacle_confirmations:
            self.get_logger().info(
                "FAR_TARGET_OBSTACLE_PENDING "
                f"id={target_id} "
                f"count={self._far_guard_count}/"
                f"{self.far_obstacle_confirmations} "
                f"map_dist={target_distance:.2f} "
                f"tof={front_distance:.2f}"
            )
            return False

        self.get_logger().warning(
            "FAR_TARGET_OBSTACLE_BYPASS "
            f"id={target_id} "
            f"map_dist={target_distance:.2f} "
            f"tof={front_distance:.2f} "
            f"gap={gap:.2f} "
            "action=bypass_not_target_verification"
        )
        self._reset_far_obstacle_guard()

        avoid_state = getattr(
            NavState,
            "AVOID_OBSTACLE",
            None,
        )
        if avoid_state is not None:
            self._set_nav_state(avoid_state)

        self._clear_old_avoidance_motion()
        return self._build_plan()

    def _map_distance_to_track(
        self,
        track: object,
    ) -> float:
        pose = getattr(self, "pose", None)
        if pose is None:
            return math.inf

        try:
            return math.hypot(
                float(track.x) - float(pose.x_enu),
                float(track.y) - float(pose.y_enu),
            )
        except (TypeError, ValueError, AttributeError):
            return math.inf

    @staticmethod
    def _track_observation_strength(track: object) -> float:
        total = 0.0
        for name, weight in (
            ("views", 20.0),
            ("view_count", 20.0),
            ("hits", 3.0),
            ("hit_count", 3.0),
            ("observations", 3.0),
            ("strong_hits", 8.0),
            ("score", 0.01),
        ):
            value = getattr(track, name, None)
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                total += weight * number
        return total

    def _nearest_other_visited(
        self,
        track: object,
    ) -> Tuple[Optional[object], float]:
        nearest = None
        nearest_distance = math.inf

        for other in getattr(self, "tracks", {}).values():
            if other is track:
                continue
            if getattr(other, "state", None) != TrackState.VISITED:
                continue

            try:
                distance = math.hypot(
                    float(track.x) - float(other.x),
                    float(track.y) - float(other.y),
                )
            except (TypeError, ValueError, AttributeError):
                continue

            if distance < nearest_distance:
                nearest = other
                nearest_distance = distance

        return nearest, nearest_distance

    def _clear_tof_for_visit_guard(self) -> None:
        for name in (
            "tof_history",
            "tof_samples",
            "tof_target_history",
            "tof_target_samples",
            "tof_wide_history",
            "tof_wide_samples",
            "tof_narrow_history",
            "tof_narrow_samples",
            "tof_layer_samples",
            "tof_recent_samples",
            "tof1_samples",
            "tof2_samples",
            "tof3_samples",
        ):
            value = getattr(self, name, None)
            if hasattr(value, "clear"):
                try:
                    value.clear()
                except Exception:
                    pass

    def _set_target_reference(
        self,
        tree_id: int,
        track: object,
    ) -> None:
        for name in (
            "current_target_id",
            "active_target_id",
            "selected_target_id",
            "navigation_target_id",
            "nav_target_id",
            "current_tree_id",
            "target_track_id",
        ):
            if hasattr(self, name):
                try:
                    setattr(self, name, int(tree_id))
                except Exception:
                    pass

        for name in (
            "current_target",
            "active_target",
            "selected_target",
            "target_track",
        ):
            if hasattr(self, name):
                try:
                    setattr(self, name, track)
                except Exception:
                    pass

    def _recover_invalid_visit(
        self,
        track: object,
        reason: str,
        duplicate: bool,
        map_distance: float,
        duplicate_distance: float = math.inf,
    ) -> None:
        tree_id = int(track.tree_id)

        # Jangan tulis baris CSV untuk visit yang dibatalkan.
        self._visited_logged.discard(tree_id)

        if duplicate:
            track.state = TrackState.REJECTED
            try:
                setattr(track, "rejected_reason", "duplicate_visited")
            except Exception:
                pass
        else:
            track.state = TrackState.CONFIRMED
            self._set_target_reference(tree_id, track)

        self._clear_tof_for_visit_guard()
        self._clear_old_avoidance_motion()

        try:
            self.tof_approach_stage = "TO_3M"
        except Exception:
            pass

        next_state = (
            getattr(NavState, "SELECT_TARGET", None)
            if duplicate
            else getattr(NavState, "ALIGN_TARGET", None)
        )
        if next_state is not None:
            self._set_nav_state(next_state)

        self.get_logger().error(
            "FALSE_VISIT_BLOCKED "
            f"id={tree_id} "
            f"reason={reason} "
            f"map_dist={map_distance:.2f} "
            f"duplicate_dist={duplicate_distance:.2f} "
            f"new_state="
            f"{'REJECTED' if duplicate else 'CONFIRMED'} "
            "actual_used_for_gate=0"
        )

    def _capture_new_visits_v22(self) -> None:
        """
        Override pencatatan V22.

        VISITED sah bila:
        1. jarak drone ke target hasil deteksi <= visit_map_gate_max;
        2. target tidak berimpit dengan landmark VISITED lain.
        """
        for track in list(
            getattr(self, "tracks", {}).values()
        ):
            tree_id = int(track.tree_id)
            if (
                track.state != TrackState.VISITED
                or tree_id in self._visited_logged
            ):
                continue

            map_distance = self._map_distance_to_track(track)
            if (
                not math.isfinite(map_distance)
                or map_distance > self.visit_map_gate_max
            ):
                self._recover_invalid_visit(
                    track,
                    reason="drone_not_near_target_map",
                    duplicate=False,
                    map_distance=map_distance,
                )
                continue

            other, duplicate_distance = (
                self._nearest_other_visited(track)
            )
            if (
                other is not None
                and duplicate_distance
                <= self.visit_duplicate_radius
            ):
                # Track yang sudah visited lebih dulu dipertahankan.
                self._recover_invalid_visit(
                    track,
                    reason=(
                        "near_existing_visited_"
                        f"{int(other.tree_id)}"
                    ),
                    duplicate=True,
                    map_distance=map_distance,
                    duplicate_distance=duplicate_distance,
                )
                continue

            self._visited_logged.add(tree_id)
            self._log_visit_v22(track)

    def _completion_cleanup(self) -> None:
        if self._completion_cleanup_done:
            return

        complete_state = getattr(
            NavState,
            "COMPLETE",
            None,
        )
        if (
            complete_state is None
            or getattr(self, "state", None) != complete_state
        ):
            return

        tracks = getattr(self, "tracks", {})
        if not isinstance(tracks, dict):
            return

        visited_items = [
            (key, track)
            for key, track in tracks.items()
            if getattr(track, "state", None)
            == TrackState.VISITED
        ]

        # Pemeriksaan terakhir: tidak boleh ada dua visited berimpit.
        for index, (first_key, first) in enumerate(visited_items):
            for second_key, second in visited_items[index + 1:]:
                try:
                    distance = math.hypot(
                        float(first.x) - float(second.x),
                        float(first.y) - float(second.y),
                    )
                except (TypeError, ValueError, AttributeError):
                    continue

                if distance > self.visit_duplicate_radius:
                    continue

                first_strength = self._track_observation_strength(first)
                second_strength = self._track_observation_strength(second)

                if first_strength >= second_strength:
                    keeper, drop = first, second
                else:
                    keeper, drop = second, first

                drop.state = TrackState.REJECTED
                self._visited_logged.discard(int(drop.tree_id))

                select_state = getattr(
                    NavState,
                    "SELECT_TARGET",
                    None,
                )
                if select_state is not None:
                    self._set_nav_state(select_state)

                self.get_logger().error(
                    "COMPLETE_DUPLICATE_REOPEN "
                    f"keep={int(keeper.tree_id)} "
                    f"drop={int(drop.tree_id)} "
                    f"distance={distance:.2f} "
                    "action=mission_not_complete_rescan_required"
                )
                return

        removed_ids: List[int] = []
        if self.completion_hide_nonvisited:
            for key, track in list(tracks.items()):
                if getattr(track, "state", None) == TrackState.VISITED:
                    continue
                try:
                    removed_ids.append(int(track.tree_id))
                except Exception:
                    pass
                try:
                    del tracks[key]
                except Exception:
                    pass

            # Hilangkan evidence sementara agar marker kuning/oranye
            # tidak muncul lagi sesudah COMPLETE.
            for name in (
                "accumulators",
                "scan_accumulators",
                "dropped_candidates",
                "tentative_candidates",
                "deferred_candidates",
            ):
                container = getattr(self, name, None)
                if hasattr(container, "clear"):
                    try:
                        container.clear()
                    except Exception:
                        pass

        self._completion_cleanup_done = True
        self.get_logger().info(
            "COMPLETE_MARKER_CLEANUP "
            f"visited_kept={len(visited_items)} "
            f"nonvisited_removed={len(removed_ids)} "
            f"removed_ids={removed_ids} "
            "actual_blue_markers_unchanged=1"
        )

    # ==============================================================
    # Main control loop
    # ==============================================================

    def _control_loop(self) -> None:
        # Selama custom bypass aktif, state machine avoidance lama
        # tidak boleh mengubah sisi atau menyelesaikan sidestep lebih awal.
        if self._bypass_plan is not None:
            self._apply_kalman_updates_v22()

            safety_reason = self._safety_hold_reason_v22()

            # Timeout sensor tetap menghentikan drone.
            # Obstacle dekat ditangani oleh emergency backup custom.
            if (
                safety_reason is not None
                and not safety_reason.startswith(
                    "unverified_object_too_close"
                )
            ):
                self._hold_current_v22(
                    safety_reason
                )
                return

            self._run_bypass()
            self._capture_new_visits_v22()
            return

        # Koordinat target sudah tercapai, tetapi batang tidak ada pada
        # ToF/PointCloud: relokalisasi, bukan HOLD TO_3M tanpa batas.
        if self._target_lost_map_recovery():
            return

        # Periksa obstacle dekat sebelum parent salah menganggapnya
        # sebagai target 3 m yang masih jauh pada peta.
        if self._far_target_obstacle_takeover():
            self._run_bypass()
            self._capture_new_visits_v22()
            return

        # Semua scan, random queue, Kalman, dan 3-2-1 tetap berjalan.
        super()._control_loop()

        # Bersihkan marker duplikat/nonvisited hanya setelah misi
        # benar-benar COMPLETE.
        self._completion_cleanup()

        # Basis mendeteksi obstacle perantara dan masuk AVOID_OBSTACLE.
        # Ambil alih mulai tick berikutnya.
        if self._should_take_over():
            self._build_plan()


    # ==============================================================
    # VISIT_GUARD_RETREAT_FIX_V1
    # ==============================================================
    def _visit_guard_tof1_proven_v1(self, tree_id: int) -> bool:
        proof_ids = getattr(self, "visited_proof_ids_v21n8", set())
        try:
            return int(tree_id) in proof_ids
        except Exception:
            return False

    def _recover_invalid_visit(
        self,
        track: object,
        reason: str,
        duplicate: bool,
        map_distance: float,
        duplicate_distance: float = math.inf,
    ) -> None:
        # Batalkan visit tidak sah tanpa membuat drone terjebak pada
        # ALIGN_TARGET ketika ToF masih sekitar 1 meter.
        tree_id = int(track.tree_id)
        self._visited_logged.discard(tree_id)

        proof_ids = getattr(self, "visited_proof_ids_v21n8", None)
        if hasattr(proof_ids, "discard"):
            proof_ids.discard(tree_id)

        if duplicate:
            track.state = TrackState.REJECTED
            try:
                setattr(track, "rejected_reason", "duplicate_visited")
            except Exception:
                pass
        else:
            track.state = TrackState.CONFIRMED

        track.updated_mono = time.monotonic()

        retreat_goal = getattr(
            self,
            "post_visit_retreat_goal_v21n8",
            None,
        )
        retreat_id = getattr(
            self,
            "post_visit_retreat_id_v21n8",
            None,
        )
        retreat_ready = (
            retreat_goal is not None
            and retreat_id is not None
            and int(retreat_id) == tree_id
        )

        self._clear_tof_for_visit_guard()
        self._clear_old_avoidance_motion()

        if retreat_ready:
            for name in (
                "current_target_id",
                "active_target_id",
                "selected_target_id",
                "navigation_target_id",
                "nav_target_id",
                "current_tree_id",
                "target_track_id",
            ):
                if hasattr(self, name):
                    try:
                        setattr(self, name, None)
                    except Exception:
                        pass

            for name in (
                "current_target",
                "active_target",
                "selected_target",
                "target_track",
            ):
                if hasattr(self, name):
                    try:
                        setattr(self, name, None)
                    except Exception:
                        pass

            try:
                self.tof_approach_stage = "TO_1M"
            except Exception:
                pass

            retreat_state = getattr(
                NavState,
                "RETREAT_VISITED",
                None,
            )
            if retreat_state is not None:
                self._set_nav_state(retreat_state)

            self._save_memory()
            self.get_logger().error(
                "FALSE_VISIT_BLOCKED_RETREAT_V1 "
                f"id={tree_id} reason={reason} "
                f"map_dist={map_distance:.2f} "
                f"duplicate_dist={duplicate_distance:.2f} "
                f"new_state="
                f"{'REJECTED' if duplicate else 'CONFIRMED'} "
                "action=retreat_then_360_rescan "
                "actual_used_for_gate=0"
            )
            return

        if not duplicate:
            self._set_target_reference(tree_id, track)

        try:
            self.tof_approach_stage = "TO_3M"
        except Exception:
            pass

        next_state = (
            getattr(NavState, "SELECT_TARGET", None)
            if duplicate
            else getattr(NavState, "ALIGN_TARGET", None)
        )
        if next_state is not None:
            self._set_nav_state(next_state)

        self._save_memory()
        self.get_logger().error(
            "FALSE_VISIT_BLOCKED_FALLBACK_V1 "
            f"id={tree_id} reason={reason} "
            f"map_dist={map_distance:.2f} "
            f"duplicate_dist={duplicate_distance:.2f} "
            f"new_state="
            f"{'REJECTED' if duplicate else 'CONFIRMED'} "
            "actual_used_for_gate=0"
        )

    def _capture_new_visits_v22(self) -> None:
        # Visit tetap harus lolos map gate. Hasil ToF-1m yang sudah
        # memiliki proof diberi toleransi kecil 0,35 m untuk error peta.
        tracks = getattr(self, "tracks", {})
        for track in list(tracks.values()):
            tree_id = int(track.tree_id)
            if (
                track.state != TrackState.VISITED
                or tree_id in self._visited_logged
            ):
                continue

            map_distance = self._map_distance_to_track(track)
            tof1_proven = self._visit_guard_tof1_proven_v1(tree_id)

            base_gate = float(self.visit_map_gate_max)
            proof_slack = 0.35 if tof1_proven else 0.0
            effective_gate = base_gate + proof_slack

            if (
                not math.isfinite(map_distance)
                or map_distance > effective_gate
            ):
                self._recover_invalid_visit(
                    track,
                    reason="drone_not_near_target_map",
                    duplicate=False,
                    map_distance=map_distance,
                )
                continue

            if tof1_proven and map_distance > base_gate:
                self.get_logger().warning(
                    "VISIT_GUARD_TOLERANCE_ACCEPT_V1 "
                    f"id={tree_id} map_dist={map_distance:.2f} "
                    f"base_gate={base_gate:.2f} "
                    f"effective_gate={effective_gate:.2f} "
                    "tof1_proof=1 actual_used_for_gate=0"
                )

            other, duplicate_distance = (
                self._nearest_other_visited(track)
            )
            if (
                other is not None
                and duplicate_distance
                <= self.visit_duplicate_radius
            ):
                self._recover_invalid_visit(
                    track,
                    reason=(
                        "near_existing_visited_"
                        f"{int(other.tree_id)}"
                    ),
                    duplicate=True,
                    map_distance=map_distance,
                    duplicate_distance=duplicate_distance,
                )
                continue

            self._visited_logged.add(tree_id)
            self._log_visit_v22(track)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SawitRandomKalman321RandomBypass()

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
