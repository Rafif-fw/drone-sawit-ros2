#!/usr/bin/env python3
"""
Final normal-run navigator:
- Kalman Filter posisi target tetap aktif.
- Verifikasi kunjungan 3 m -> 2 m -> 1 m tetap aktif.
- Obstacle pohon dimodelkan 1 m ke kiri dan 1 m ke kanan.
- Sisi bypass dipilih random satu kali, lalu dikunci selama satu percobaan.
- Obstacle perantara dicek sebelum salah dianggap sebagai target verifikasi 3 m.
- Jika kedua sisi gagal, drone HOLD singkat lalu scan ulang, bukan stuck selamanya.
- Anti-double target tanpa memakai ground truth Gazebo.

Ground truth tetap hanya untuk evaluasi, bukan navigasi.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import rclpy

from sawit_autonomy.sawit_navigator_random_kalman_321_random_bypass import (
    NavState,
    SawitRandomKalman321RandomBypass,
    TrackState,
)
from sawit_autonomy.sawit_navigator_random_kalman_321_v22 import (
    SawitRandomKalman321V22,
)


class SawitRandomKalman321Final(SawitRandomKalman321RandomBypass):
    """Random target, Kalman, 3-2-1, robust bypass, dan anti-double."""

    def __init__(self) -> None:
        super().__init__()

        # Gunakan nama parameter final agar tidak bertabrakan dengan versi lama.
        self._declare_if_missing("final_bypass_front_min", 0.70)
        self._declare_if_missing("final_bypass_front_max", 6.50)
        self._declare_if_missing("final_bypass_target_min_distance", 5.00)
        self._declare_if_missing("final_bypass_min_target_front_gap", 2.50)
        self._declare_if_missing("final_bypass_confirmations", 2)
        self._declare_if_missing("final_bypass_front_tolerance", 1.00)
        self._declare_if_missing("final_bypass_max_interval", 1.00)
        self._declare_if_missing("final_bypass_cooldown", 1.50)
        self._declare_if_missing("final_bypass_failsafe_hold", 2.00)

        self._declare_if_missing("final_dedupe_enabled", True)
        self._declare_if_missing("final_dedupe_period", 0.50)
        self._declare_if_missing("final_dedupe_nonvisited_radius", 2.10)
        self._declare_if_missing("final_dedupe_visited_radius", 2.60)
        self._declare_if_missing("final_dedupe_strong_separation", 1.45)
        self._declare_if_missing("final_dedupe_strong_views", 3)

        # Kecepatan sedikit dinaikkan, tetapi tahap 1 m tetap konservatif.
        self._declare_if_missing("final_speed_stage3_step", 0.20)
        self._declare_if_missing("final_speed_stage2_step", 0.14)
        self._declare_if_missing("final_speed_stage1_step", 0.075)
        self._declare_if_missing("final_speed_bypass_step", 0.16)

        # Jangan menolak target sebagai ghost ketika drone masih jauh.
        self._declare_if_missing("final_ghost_reject_near_limit", 4.50)

        # Rescue bila state TO_1M salah, tetapi jarak masih sekitar 3 m.
        self._declare_if_missing("final_stage1_rescue_time", 8.00)
        self._declare_if_missing("final_stage1_rescue_map_min", 2.60)
        self._declare_if_missing("final_stage1_rescue_tof_min", 2.40)
        self._declare_if_missing("final_stage1_rescue_max", 2)

        self.final_bypass_front_min = float(
            self.get_parameter("final_bypass_front_min").value
        )
        self.final_bypass_front_max = float(
            self.get_parameter("final_bypass_front_max").value
        )
        self.final_bypass_target_min_distance = float(
            self.get_parameter("final_bypass_target_min_distance").value
        )
        self.final_bypass_min_target_front_gap = float(
            self.get_parameter("final_bypass_min_target_front_gap").value
        )
        self.final_bypass_confirmations = max(
            1,
            int(self.get_parameter("final_bypass_confirmations").value),
        )
        self.final_bypass_front_tolerance = float(
            self.get_parameter("final_bypass_front_tolerance").value
        )
        self.final_bypass_max_interval = float(
            self.get_parameter("final_bypass_max_interval").value
        )
        self.final_bypass_cooldown = float(
            self.get_parameter("final_bypass_cooldown").value
        )
        self.final_bypass_failsafe_hold = float(
            self.get_parameter("final_bypass_failsafe_hold").value
        )

        self.final_dedupe_enabled = bool(
            self.get_parameter("final_dedupe_enabled").value
        )
        self.final_dedupe_period = float(
            self.get_parameter("final_dedupe_period").value
        )
        self.final_dedupe_nonvisited_radius = float(
            self.get_parameter("final_dedupe_nonvisited_radius").value
        )
        self.final_dedupe_visited_radius = float(
            self.get_parameter("final_dedupe_visited_radius").value
        )
        self.final_dedupe_strong_separation = float(
            self.get_parameter("final_dedupe_strong_separation").value
        )
        self.final_dedupe_strong_views = max(
            1,
            int(self.get_parameter("final_dedupe_strong_views").value),
        )

        self.final_speed_stage3_step = float(
            self.get_parameter("final_speed_stage3_step").value
        )
        self.final_speed_stage2_step = float(
            self.get_parameter("final_speed_stage2_step").value
        )
        self.final_speed_stage1_step = float(
            self.get_parameter("final_speed_stage1_step").value
        )
        self.final_speed_bypass_step = float(
            self.get_parameter("final_speed_bypass_step").value
        )
        self.final_ghost_reject_near_limit = float(
            self.get_parameter("final_ghost_reject_near_limit").value
        )
        self.final_stage1_rescue_time = float(
            self.get_parameter("final_stage1_rescue_time").value
        )
        self.final_stage1_rescue_map_min = float(
            self.get_parameter("final_stage1_rescue_map_min").value
        )
        self.final_stage1_rescue_tof_min = float(
            self.get_parameter("final_stage1_rescue_tof_min").value
        )
        self.final_stage1_rescue_max = max(
            1,
            int(self.get_parameter("final_stage1_rescue_max").value),
        )

        self._final_proactive_target: Optional[int] = None
        self._final_proactive_count = 0
        self._final_proactive_front = math.inf
        self._final_proactive_time = 0.0
        self._final_bypass_cooldown_until = 0.0

        self._final_dedupe_last_time = 0.0
        self._final_dedupe_events = 0

        self._final_stage1_stall_target: Optional[int] = None
        self._final_stage1_stall_since = 0.0
        self._final_stage1_rescue_count: Dict[int, int] = {}

        self._apply_final_speed_profile()

        self.get_logger().info(
            "START RANDOM_KALMAN_321_FINAL "
            f"obstacle_half_width={float(self.obstacle_half_width):.2f}m "
            f"lateral_offset={float(self.bypass_lateral_offset):.2f}m "
            f"proactive_front={self.final_bypass_front_min:.2f}-"
            f"{self.final_bypass_front_max:.2f}m "
            f"confirmations={self.final_bypass_confirmations}"
        )
        self.get_logger().info(
            "START FINAL_SPEED_PROFILE "
            f"step_3m={self.final_speed_stage3_step:.3f} "
            f"step_2m={self.final_speed_stage2_step:.3f} "
            f"step_1m={self.final_speed_stage1_step:.3f} "
            f"bypass_step={self.final_speed_bypass_step:.3f}"
        )
        self.get_logger().info(
            "START FINAL_TRACK_DEDUPE "
            f"nonvisited_radius={self.final_dedupe_nonvisited_radius:.2f}m "
            f"visited_radius={self.final_dedupe_visited_radius:.2f}m "
            "policy=visited_then_active_then_strongest"
        )

    # ------------------------------------------------------------------
    # Parameter helper
    # ------------------------------------------------------------------

    def _declare_if_missing(self, name: str, default: object) -> None:
        try:
            exists = self.has_parameter(name)
        except Exception:
            exists = False
        if not exists:
            self.declare_parameter(name, default)

    # ------------------------------------------------------------------
    # Speed profile, false-red protection, dan stage rescue
    # ------------------------------------------------------------------

    def _apply_final_speed_profile(self) -> None:
        """
        Naikkan gerak sekitar 20-30 persen.
        Tahap akhir 1 m tetap jauh lebih kecil daripada tahap 3 m.
        """
        candidates = {
            "tof_stage3_step_v21n3": self.final_speed_stage3_step,
            "tof_stage3_step": self.final_speed_stage3_step,
            "tof_stage2_step_v21n3": self.final_speed_stage2_step,
            "tof_stage2_step": self.final_speed_stage2_step,
            "tof_stage1_step_v21n3": self.final_speed_stage1_step,
            "tof_stage1_step": self.final_speed_stage1_step,
        }

        applied = []
        for name, value in candidates.items():
            if not hasattr(self, name):
                continue
            try:
                setattr(self, name, float(value))
                applied.append(f"{name}={value:.3f}")
            except Exception:
                pass

        if hasattr(self, "bypass_position_step"):
            try:
                self.bypass_position_step = max(
                    float(self.bypass_position_step),
                    self.final_speed_bypass_step,
                )
                applied.append(
                    f"bypass_position_step="
                    f"{self.bypass_position_step:.3f}"
                )
            except Exception:
                pass

        self.get_logger().info(
            "FINAL_SPEED_APPLIED "
            + (" ".join(applied) if applied else "no_known_attributes")
        )

    def _track_state_snapshot(self) -> Dict[int, object]:
        snapshot: Dict[int, object] = {}
        tracks = getattr(self, "tracks", {})
        if not isinstance(tracks, dict):
            return snapshot

        for raw_id, track in tracks.items():
            try:
                tree_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if "REJECTED" not in self._state_label(track):
                snapshot[tree_id] = getattr(track, "state", None)
        return snapshot

    def _remove_false_reject_flags(self, tree_id: int) -> None:
        """
        Hapus ID dari container reject/drop generik bila parent
        sudah sempat memasukkannya.
        """
        for name, container in vars(self).items():
            lower = name.lower()
            if "reject" not in lower and "drop" not in lower:
                continue
            if isinstance(container, dict):
                continue

            try:
                if isinstance(container, set):
                    container.discard(tree_id)
                elif isinstance(container, list):
                    container[:] = [
                        item
                        for item in container
                        if self._item_id(item) != tree_id
                    ]
            except Exception:
                pass

    def _restore_false_far_rejections(
        self,
        before: Dict[int, object],
    ) -> None:
        """
        Ghost hanya boleh ditolak setelah drone benar-benar dekat.

        Pada log sebelumnya id=6 ditolak ketika target_dist=19.93 m.
        Ketiadaan frame dari jarak tersebut bukan bukti bahwa target ghost.
        """
        pose = getattr(self, "pose", None)
        tracks = getattr(self, "tracks", {})
        if pose is None or not isinstance(tracks, dict):
            return

        drone_xy = (
            float(pose.x_enu),
            float(pose.y_enu),
        )

        for tree_id, old_state in before.items():
            track = tracks.get(tree_id)
            if track is None:
                continue
            if "REJECTED" not in self._state_label(track):
                continue

            # Rejection anti-double tetap sah.
            duplicate_of = getattr(track, "duplicate_of", None)
            reason = str(
                getattr(track, "rejected_reason", "")
            ).lower()
            if duplicate_of is not None or "duplicate" in reason:
                continue

            try:
                distance = math.hypot(
                    float(track.x) - drone_xy[0],
                    float(track.y) - drone_xy[1],
                )
            except (TypeError, ValueError, AttributeError):
                continue

            if distance <= self.final_ghost_reject_near_limit:
                continue

            try:
                track.state = old_state
                setattr(
                    track,
                    "rejected_reason",
                    "far_no_frame_restored",
                )
            except Exception:
                continue

            self._remove_false_reject_flags(tree_id)
            self.get_logger().warning(
                "FINAL_FALSE_RED_RESTORED "
                f"id={tree_id} "
                f"distance={distance:.2f}m "
                f"near_limit={self.final_ghost_reject_near_limit:.2f}m "
                "reason=no_frame_while_far_not_valid_ghost_evidence"
            )

    def _reset_stage1_monitor(self) -> None:
        self._final_stage1_stall_target = None
        self._final_stage1_stall_since = 0.0

    def _monitor_and_rescue_stage1(self) -> bool:
        """
        Bila status sudah TO_1M tetapi target dan ToF masih sekitar 3 m,
        turunkan kembali ke TO_2M agar drone tidak melakukan reacquire
        TO_1M berkali-kali dengan langkah yang terlalu kecil.
        """
        approach_state = getattr(NavState, "APPROACH", None)
        if (
            approach_state is None
            or getattr(self, "state", None) != approach_state
            or str(getattr(self, "tof_approach_stage", "")) != "TO_1M"
        ):
            self._reset_stage1_monitor()
            return False

        geometry = self._current_geometry()
        if geometry is None:
            self._reset_stage1_monitor()
            return False

        (
            target_id,
            _track,
            _drone_xy,
            _target_xy,
            target_distance,
            front_distance,
        ) = geometry

        far_for_stage1 = (
            target_distance >= self.final_stage1_rescue_map_min
            and (
                not math.isfinite(front_distance)
                or front_distance >= self.final_stage1_rescue_tof_min
            )
        )

        if not far_for_stage1:
            self._reset_stage1_monitor()
            return False

        now = time.monotonic()
        if self._final_stage1_stall_target != int(target_id):
            self._final_stage1_stall_target = int(target_id)
            self._final_stage1_stall_since = now
            return False

        elapsed = now - self._final_stage1_stall_since
        if elapsed < self.final_stage1_rescue_time:
            return False

        count = self._final_stage1_rescue_count.get(
            int(target_id),
            0,
        ) + 1
        self._final_stage1_rescue_count[int(target_id)] = count
        self._reset_stage1_monitor()

        if count > self.final_stage1_rescue_max:
            self.get_logger().error(
                "FINAL_STAGE1_RESCUE_LIMIT "
                f"id={target_id} count={count} "
                "action=full_rescan"
            )
            self._clear_stale_tof_after_bypass()
            self._request_final_rescan()
            return True

        # Kembali ke tahap 2 m, bukan langsung visited.
        self.tof_approach_stage = "TO_2M"
        self._clear_stale_tof_after_bypass()
        self._clear_old_avoidance_motion()

        align_state = getattr(NavState, "ALIGN_TARGET", None)
        if align_state is not None:
            self._set_nav_state(align_state)

        self.get_logger().warning(
            "FINAL_STAGE1_DOWNGRADE_TO_2M "
            f"id={target_id} "
            f"elapsed={elapsed:.1f}s "
            f"target_dist={target_distance:.2f} "
            f"tof={front_distance:.2f} "
            f"rescue={count}/{self.final_stage1_rescue_max} "
            "action=resume_with_stage2_speed_and_safety"
        )
        return True

    # ------------------------------------------------------------------
    # Proactive obstacle takeover
    # ------------------------------------------------------------------

    def _reset_final_proactive(self) -> None:
        self._final_proactive_target = None
        self._final_proactive_count = 0
        self._final_proactive_front = math.inf
        self._final_proactive_time = 0.0

    def _should_take_over_proactively_final(self) -> bool:
        if not bool(getattr(self, "random_bypass_enabled", True)):
            self._reset_final_proactive()
            return False
        if getattr(self, "_bypass_plan", None) is not None:
            return False
        if time.monotonic() < self._final_bypass_cooldown_until:
            self._reset_final_proactive()
            return False

        allowed_states = {
            getattr(NavState, "APPROACH", None),
            getattr(NavState, "CLOSE_SETTLE", None),
            getattr(NavState, "CLOSE_FLUSH", None),
            getattr(NavState, "CLOSE_COLLECT", None),
        }
        allowed_states.discard(None)
        if getattr(self, "state", None) not in allowed_states:
            self._reset_final_proactive()
            return False

        stage = str(getattr(self, "tof_approach_stage", "TO_3M"))
        if stage != "TO_3M":
            self._reset_final_proactive()
            return False

        geometry = self._current_geometry()
        if geometry is None:
            self._reset_final_proactive()
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
            target_distance >= self.final_bypass_target_min_distance
            and gap >= self.final_bypass_min_target_front_gap
            and front_distance >= self.final_bypass_front_min
            and front_distance <= self.final_bypass_front_max
        )
        if not condition:
            self._reset_final_proactive()
            return False

        now = time.monotonic()
        same_obstacle = (
            self._final_proactive_target == target_id
            and now - self._final_proactive_time
            <= self.final_bypass_max_interval
            and abs(front_distance - self._final_proactive_front)
            <= self.final_bypass_front_tolerance
        )
        if same_obstacle:
            self._final_proactive_count += 1
        else:
            self._final_proactive_target = int(target_id)
            self._final_proactive_count = 1

        self._final_proactive_front = float(front_distance)
        self._final_proactive_time = now

        if self._final_proactive_count < self.final_bypass_confirmations:
            self.get_logger().info(
                "FINAL_BYPASS_TRIGGER_PENDING "
                f"id={target_id} "
                f"count={self._final_proactive_count}/"
                f"{self.final_bypass_confirmations} "
                f"target_dist={target_distance:.2f} "
                f"front={front_distance:.2f}"
            )
            return False

        self.get_logger().warning(
            "FINAL_BYPASS_PROACTIVE_TRIGGER "
            f"id={target_id} "
            f"target_dist={target_distance:.2f} "
            f"front={front_distance:.2f} "
            f"gap={gap:.2f} "
            f"state={self._state_name(self.state)} "
            "action=random_locked_side_before_3m_verify"
        )
        self._reset_final_proactive()
        return True

    def _clear_stale_tof_after_bypass(self) -> None:
        names = (
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
        )
        for name in names:
            value = getattr(self, name, None)
            if hasattr(value, "clear"):
                try:
                    value.clear()
                except Exception:
                    pass

        for name in (
            "tof_last_valid_distance",
            "tof_target_last_good",
            "tof_last_target_distance",
        ):
            if hasattr(self, name):
                try:
                    setattr(self, name, math.inf)
                except Exception:
                    pass

    def _finish_success(self, plan: object, clearance: float) -> None:
        super()._finish_success(plan, clearance)
        self._clear_stale_tof_after_bypass()
        self._final_bypass_cooldown_until = (
            time.monotonic() + self.final_bypass_cooldown
        )

    def _restart_on_opposite_side(self, plan: object, reason: str) -> None:
        super()._restart_on_opposite_side(plan, reason)
        if getattr(plan, "phase", "") == "FAILSAFE_HOLD":
            plan.hold_until = (
                time.monotonic() + self.final_bypass_failsafe_hold
            )

    def _request_final_rescan(self) -> None:
        for function_name in (
            "_start_rescan",
            "_start_scan360",
            "_begin_rescan",
            "_begin_scan",
        ):
            function = getattr(self, function_name, None)
            if not callable(function):
                continue
            try:
                function()
                self.get_logger().warning(
                    "FINAL_BYPASS_RESCAN "
                    f"method={function_name}"
                )
                return
            except TypeError:
                continue
            except Exception:
                continue

        rotate_scan = getattr(NavState, "ROTATE_SCAN", None)
        if rotate_scan is not None:
            self._set_nav_state(rotate_scan)
            self.get_logger().warning(
                "FINAL_BYPASS_RESCAN method=ROTATE_SCAN"
            )
            return

        align = getattr(NavState, "ALIGN_TARGET", None)
        if align is not None:
            self._set_nav_state(align)
            self.get_logger().warning(
                "FINAL_BYPASS_RESCAN method=ALIGN_TARGET_fallback"
            )

    def _run_bypass(self) -> None:
        plan = getattr(self, "_bypass_plan", None)
        if plan is not None and getattr(plan, "phase", "") == "FAILSAFE_HOLD":
            self._hold_position(self._yaw_to_target(plan))
            hold_until = float(
                getattr(
                    plan,
                    "hold_until",
                    time.monotonic() + self.final_bypass_failsafe_hold,
                )
            )
            if time.monotonic() >= hold_until:
                self.get_logger().warning(
                    "FINAL_BYPASS_FAILSAFE_RELEASE "
                    f"id={plan.target_id} action=rescan_not_stuck"
                )
                self._bypass_plan = None
                self._clear_old_avoidance_motion()
                self._clear_stale_tof_after_bypass()
                self._request_final_rescan()
            return

        super()._run_bypass()

    # ------------------------------------------------------------------
    # Anti-double target
    # ------------------------------------------------------------------

    @staticmethod
    def _state_label(track: object) -> str:
        state = getattr(track, "state", None)
        return str(
            getattr(state, "name", getattr(state, "value", state))
        ).upper()

    @staticmethod
    def _numeric_attr(
        track: object,
        names: Sequence[str],
        default: float = 0.0,
    ) -> float:
        for name in names:
            value = getattr(track, name, None)
            if isinstance(value, bool):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return number
        return float(default)

    def _views(self, track: object) -> int:
        return int(
            max(
                0.0,
                self._numeric_attr(
                    track,
                    (
                        "views",
                        "view_count",
                        "distinct_views",
                        "angle_bins",
                        "view_bins",
                    ),
                    0.0,
                ),
            )
        )

    def _quality(self, track: object) -> float:
        state = self._state_label(track)
        if "REJECTED" in state:
            return -math.inf
        if "VISITED" in state:
            state_score = 100000.0
        elif "CONFIRMED" in state:
            state_score = 10000.0
        elif "TENTATIVE" in state:
            state_score = 1000.0
        else:
            state_score = 0.0

        hits = self._numeric_attr(
            track,
            (
                "total_hits",
                "hits",
                "hit_count",
                "observations",
                "observation_count",
                "obs_count",
            ),
        )
        strong = self._numeric_attr(
            track,
            ("strong_hits", "strong_count", "strong_observations"),
        )
        tof = self._numeric_attr(
            track,
            ("tof_hits", "tof_support", "tof_support_count"),
        )
        score = self._numeric_attr(
            track,
            ("score", "confidence", "quality", "best_score"),
        )
        return (
            state_score
            + 30.0 * float(self._views(track))
            + 8.0 * strong
            + 3.0 * hits
            + 2.0 * tof
            + 0.01 * score
        )

    @staticmethod
    def _item_id(item: object) -> Optional[int]:
        if isinstance(item, bool):
            return None
        if isinstance(item, int):
            return int(item)
        if hasattr(item, "tree_id"):
            try:
                return int(item.tree_id)
            except (TypeError, ValueError):
                return None
        return None

    def _remove_id_from_queues(self, duplicate_id: int) -> None:
        names = {
            "target_queue",
            "random_queue",
            "fixed_random_queue",
            "random_batch_queue",
            "random_batch_snapshot_v21n6",
            "random_batch_snapshot",
            "pending_targets",
            "deferred_targets",
        }
        for name in vars(self):
            lower = name.lower()
            if "queue" in lower or "batch" in lower:
                names.add(name)

        for name in names:
            container = getattr(self, name, None)
            if container is None or isinstance(
                container, (str, bytes, dict)
            ):
                continue
            try:
                items = list(container)
            except TypeError:
                continue
            filtered = [
                item for item in items
                if self._item_id(item) != duplicate_id
            ]
            if len(filtered) == len(items):
                continue
            try:
                if isinstance(container, list):
                    container[:] = filtered
                elif isinstance(container, tuple):
                    setattr(self, name, tuple(filtered))
                elif isinstance(container, set):
                    container.clear()
                    container.update(filtered)
                elif hasattr(container, "clear") and hasattr(
                    container, "extend"
                ):
                    container.clear()
                    container.extend(filtered)
                else:
                    setattr(self, name, filtered)
            except Exception as error:
                self.get_logger().warning(
                    "FINAL_DEDUPE_QUEUE_FAIL "
                    f"attribute={name} error={error}"
                )

    def _active_id(self) -> Optional[int]:
        resolved = self._resolve_active_target()
        return int(resolved[0]) if resolved is not None else None

    def _choose_keeper(
        self,
        first_id: int,
        first: object,
        second_id: int,
        second: object,
        active_id: Optional[int],
    ) -> Tuple[int, object, int, object]:
        first_visited = "VISITED" in self._state_label(first)
        second_visited = "VISITED" in self._state_label(second)

        # Landmark visited selalu menang.
        if first_visited != second_visited:
            return (
                (first_id, first, second_id, second)
                if first_visited
                else (second_id, second, first_id, first)
            )

        # Jika keduanya belum visited, target aktif dipertahankan.
        if active_id == first_id and active_id != second_id:
            return first_id, first, second_id, second
        if active_id == second_id and active_id != first_id:
            return second_id, second, first_id, first

        first_quality = self._quality(first)
        second_quality = self._quality(second)
        if first_quality > second_quality:
            return first_id, first, second_id, second
        if second_quality > first_quality:
            return second_id, second, first_id, first

        first_created = self._numeric_attr(
            first,
            ("created_mono", "created_time", "first_seen"),
            math.inf,
        )
        second_created = self._numeric_attr(
            second,
            ("created_mono", "created_time", "first_seen"),
            math.inf,
        )
        if first_created < second_created:
            return first_id, first, second_id, second
        if second_created < first_created:
            return second_id, second, first_id, first

        return (
            (first_id, first, second_id, second)
            if first_id < second_id
            else (second_id, second, first_id, first)
        )

    def _rewrite_target_reference(
        self,
        duplicate_id: int,
        keeper_id: int,
        keeper: object,
    ) -> None:
        keeper_visited = "VISITED" in self._state_label(keeper)
        names = (
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
        for name in names:
            if not hasattr(self, name):
                continue
            value = getattr(self, name, None)
            if self._item_id(value) != duplicate_id:
                continue
            replacement = None
            if not keeper_visited:
                replacement = keeper if hasattr(value, "tree_id") else keeper_id
            try:
                setattr(self, name, replacement)
            except Exception:
                pass

        plan = getattr(self, "_bypass_plan", None)
        if plan is not None and plan.target_id == duplicate_id:
            if keeper_visited:
                self._bypass_plan = None
                self._clear_old_avoidance_motion()
            else:
                plan.target_id = keeper_id
                plan.target_xy = (float(keeper.x), float(keeper.y))

    def _reject_duplicate(
        self,
        keeper_id: int,
        keeper: object,
        duplicate_id: int,
        duplicate: object,
        distance: float,
    ) -> bool:
        try:
            duplicate.state = TrackState.REJECTED
        except Exception as error:
            self.get_logger().warning(
                "FINAL_DEDUPE_REJECT_FAIL "
                f"drop={duplicate_id} error={error}"
            )
            return False

        for name, value in (
            ("duplicate_of", keeper_id),
            ("duplicate_distance", float(distance)),
            ("rejected_reason", "duplicate_track"),
        ):
            try:
                setattr(duplicate, name, value)
            except Exception:
                pass

        self._remove_id_from_queues(duplicate_id)
        self._rewrite_target_reference(
            duplicate_id, keeper_id, keeper
        )
        self._kalman_tracks.pop(duplicate_id, None)
        self._last_track_update.pop(duplicate_id, None)
        self._side_lock_by_target.pop(duplicate_id, None)
        self._side_flip_used_by_target.pop(duplicate_id, None)

        self._final_dedupe_events += 1
        self.get_logger().warning(
            "FINAL_TRACK_DUPLICATE_SUPPRESSED "
            f"keep={keeper_id} drop={duplicate_id} "
            f"distance={distance:.2f}m "
            f"keep_state={self._state_label(keeper)} "
            f"keep_views={self._views(keeper)} "
            f"drop_views={self._views(duplicate)} "
            f"events={self._final_dedupe_events}"
        )
        return True

    def _dedupe_if_due(self) -> None:
        if not self.final_dedupe_enabled:
            return
        now = time.monotonic()
        if now - self._final_dedupe_last_time < self.final_dedupe_period:
            return
        self._final_dedupe_last_time = now

        tracks = getattr(self, "tracks", {})
        if not isinstance(tracks, dict) or len(tracks) < 2:
            return

        active_id = self._active_id()
        valid: List[Tuple[int, object]] = []
        for raw_id, track in tracks.items():
            try:
                tree_id = int(raw_id)
                x = float(track.x)
                y = float(track.y)
            except (TypeError, ValueError, AttributeError):
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            if "REJECTED" in self._state_label(track):
                continue
            valid.append((tree_id, track))

        pairs: List[Tuple[float, int, object, int, object]] = []
        for index, (first_id, first) in enumerate(valid):
            for second_id, second in valid[index + 1:]:
                distance = math.hypot(
                    float(first.x) - float(second.x),
                    float(first.y) - float(second.y),
                )
                first_visited = "VISITED" in self._state_label(first)
                second_visited = "VISITED" in self._state_label(second)
                radius = (
                    self.final_dedupe_visited_radius
                    if first_visited or second_visited
                    else self.final_dedupe_nonvisited_radius
                )
                if distance > radius:
                    continue

                # Jangan gabungkan dua kandidat yang sama-sama punya banyak view
                # kecuali jaraknya benar-benar sangat dekat.
                if (
                    not first_visited
                    and not second_visited
                    and distance > self.final_dedupe_strong_separation
                    and self._views(first) >= self.final_dedupe_strong_views
                    and self._views(second) >= self.final_dedupe_strong_views
                ):
                    continue

                pairs.append(
                    (distance, first_id, first, second_id, second)
                )

        pairs.sort(key=lambda item: item[0])
        rejected = set()
        for distance, first_id, first, second_id, second in pairs:
            if first_id in rejected or second_id in rejected:
                continue
            if (
                "REJECTED" in self._state_label(first)
                or "REJECTED" in self._state_label(second)
            ):
                continue
            keeper_id, keeper, drop_id, drop = self._choose_keeper(
                first_id,
                first,
                second_id,
                second,
                active_id,
            )
            if self._reject_duplicate(
                keeper_id, keeper, drop_id, drop, distance
            ):
                rejected.add(drop_id)

    # ------------------------------------------------------------------
    # Final control loop
    # ------------------------------------------------------------------

    def _control_loop(self) -> None:
        self._dedupe_if_due()

        # Rescue state TO_1M yang sebenarnya masih jauh.
        if self._monitor_and_rescue_stage1():
            return

        # Custom bypass sedang aktif.
        if getattr(self, "_bypass_plan", None) is not None:
            self._apply_kalman_updates_v22()
            safety_reason = self._safety_hold_reason_v22()
            if (
                safety_reason is not None
                and not safety_reason.startswith(
                    "unverified_object_too_close"
                )
            ):
                self._hold_current_v22(safety_reason)
                return

            self._run_bypass()
            self._capture_new_visits_v22()
            return

        # Kalman tetap aktif sebelum keputusan obstacle/target.
        self._apply_kalman_updates_v22()

        # Ambil alih sebelum parent salah memulai verifikasi 3 m.
        if self._should_take_over_proactively_final():
            if self._build_plan():
                self._run_bypass()
                self._capture_new_visits_v22()
                return

        # Simpan status agar rejection ghost yang terlalu jauh bisa dipulihkan.
        state_before = self._track_state_snapshot()

        # Jalankan scan, queue random, Kalman, dan verifikasi 3-2-1 induk.
        SawitRandomKalman321V22._control_loop(self)

        self._restore_false_far_rejections(state_before)
        self._dedupe_if_due()

        # Fallback bila induk sudah masuk AVOID_OBSTACLE.
        if self._should_take_over():
            self._build_plan()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SawitRandomKalman321Final()
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
