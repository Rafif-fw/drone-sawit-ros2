#!/usr/bin/env python3
"""
V3 collision-safe wrapper for the two Depth-Kalman anti-stuck variants.

This file does NOT replace the old algorithms. It adds two new executables:
- depth + ToF Kalman with verification 3-2-1
- depth + ToF Kalman direct to approximately 1 m

Fixes:
1. A latched simulation collision always overrides the custom bypass.
2. A pose-fault hold also overrides the custom bypass.
3. V22 near-object safety is only released to perform a controlled backup.
4. Bypass lateral clearance is increased and movement step is reduced.
5. Excessive actual XY speed during bypass immediately causes HOLD.
6. Strong reverse progress during a bypass phase is rejected immediately.

Actual/Gazebo ground truth is never used for navigation or Kalman updates.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional

import rclpy

from sawit_autonomy.sawit_navigator_depth_camera_kalman_dual_antistuck import (
    SawitDepthKalman321AntiStuck,
    SawitDepthKalmanDirect1MAntiStuck,
)
from sawit_autonomy.sawit_navigator_random_kalman_321_v22 import (
    SawitRandomKalman321V22,
)


class CollisionSafeBypassMixin:
    """Safety wrapper placed before the existing V2 classes in the MRO."""

    def __init__(self) -> None:
        super().__init__()

        self.declare_parameter("safe_bypass_lateral_offset", 2.00)
        self.declare_parameter("safe_bypass_emergency_front", 1.75)
        self.declare_parameter("safe_bypass_position_step", 0.08)
        self.declare_parameter("safe_bypass_backup_distance", 1.00)
        self.declare_parameter("safe_bypass_motion_speed_limit", 1.20)
        self.declare_parameter("safe_bypass_reverse_progress_limit", 0.18)
        self.declare_parameter("safe_pose_max_xy_speed", 2.00)

        self.safe_bypass_lateral_offset = float(
            self.get_parameter("safe_bypass_lateral_offset").value
        )
        self.safe_bypass_emergency_front = float(
            self.get_parameter("safe_bypass_emergency_front").value
        )
        self.safe_bypass_position_step = float(
            self.get_parameter("safe_bypass_position_step").value
        )
        self.safe_bypass_backup_distance = float(
            self.get_parameter("safe_bypass_backup_distance").value
        )
        self.safe_bypass_motion_speed_limit = float(
            self.get_parameter("safe_bypass_motion_speed_limit").value
        )
        self.safe_bypass_reverse_progress_limit = float(
            self.get_parameter("safe_bypass_reverse_progress_limit").value
        )
        self.safe_pose_max_xy_speed = float(
            self.get_parameter("safe_pose_max_xy_speed").value
        )

        # Keep the old code and queue logic, but make bypass geometry safer.
        self.bypass_lateral_offset = max(
            float(getattr(self, "bypass_lateral_offset", 0.0)),
            self.safe_bypass_lateral_offset,
        )
        if hasattr(self, "obstacle_half_width"):
            self.drone_safety_margin = max(
                float(getattr(self, "drone_safety_margin", 0.0)),
                self.bypass_lateral_offset
                - float(getattr(self, "obstacle_half_width", 1.0)),
            )

        self.bypass_emergency_front = max(
            float(getattr(self, "bypass_emergency_front", 0.0)),
            self.safe_bypass_emergency_front,
        )
        self.bypass_position_step = min(
            float(getattr(self, "bypass_position_step", 0.12)),
            self.safe_bypass_position_step,
        )
        self.bypass_backup_distance = max(
            float(getattr(self, "bypass_backup_distance", 0.65)),
            self.safe_bypass_backup_distance,
        )

        # The previous 5 m/s pose threshold was too late for this vehicle.
        if hasattr(self, "pose_max_xy_speed"):
            self.pose_max_xy_speed = min(
                float(self.pose_max_xy_speed),
                self.safe_pose_max_xy_speed,
            )

        self._safe_collision_stop_logged = False
        self._safe_speed_hold_last_log = 0.0
        self._safe_backup_last_reason = ""

        self.get_logger().info(
            "START DEPTH_COLLISION_SAFE_V3 "
            f"lateral_offset={self.bypass_lateral_offset:.2f}m "
            f"emergency_front={self.bypass_emergency_front:.2f}m "
            f"step={self.bypass_position_step:.2f}m "
            f"backup={self.bypass_backup_distance:.2f}m "
            f"motion_speed_limit={self.safe_bypass_motion_speed_limit:.2f}mps "
            f"pose_reject_limit={getattr(self, 'pose_max_xy_speed', math.nan):.2f}mps "
            "collision_latch_overrides_bypass=1"
        )

    @staticmethod
    def _safe_pose_speed(pose: Any) -> float:
        speed = getattr(pose, "speed_xy", None)
        if speed is not None:
            try:
                value = float(speed)
                if math.isfinite(value):
                    return value
            except (TypeError, ValueError):
                pass

        vx = float(getattr(pose, "vx_enu", 0.0))
        vy = float(getattr(pose, "vy_enu", 0.0))
        if not (math.isfinite(vx) and math.isfinite(vy)):
            return math.inf
        return math.hypot(vx, vy)

    def _clear_bypass_motion_safe(self) -> None:
        self._bypass_plan = None

        clear_old = getattr(self, "_clear_old_avoidance_motion", None)
        if callable(clear_old):
            clear_old()

        for name in (
            "motion_waypoint_xy",
            "motion_goal_xy",
            "motion_brake_anchor_xy",
        ):
            if hasattr(self, name):
                setattr(self, name, None)

    def _force_safe_backup(
        self,
        plan: Any,
        reason: str,
    ) -> bool:
        pose = getattr(self, "pose", None)
        if pose is None or plan is None:
            return False

        try:
            current_x = float(pose.x_enu)
            current_y = float(pose.y_enu)
            forward_x = float(plan.forward_unit[0])
            forward_y = float(plan.forward_unit[1])
        except (TypeError, ValueError, AttributeError, IndexError):
            return False

        values = (current_x, current_y, forward_x, forward_y)
        if not all(math.isfinite(value) for value in values):
            return False

        backup_distance = max(
            self.safe_bypass_backup_distance,
            float(getattr(self, "bypass_backup_distance", 0.65)),
        )
        plan.backup_goal = (
            current_x - forward_x * backup_distance,
            current_y - forward_y * backup_distance,
        )

        if str(getattr(plan, "phase", "")) != "BACKUP":
            set_phase = getattr(self, "_set_phase", None)
            if callable(set_phase):
                set_phase(plan, "BACKUP")
            else:
                plan.phase = "BACKUP"
                plan.phase_started = time.monotonic()
                plan.last_remaining = math.inf
                plan.last_progress_time = time.monotonic()

        if reason != self._safe_backup_last_reason:
            self._safe_backup_last_reason = reason
            self.get_logger().warning(
                "DEPTH_COLLISION_SAFE_FORCE_BACKUP_V3 "
                f"id={getattr(plan, 'target_id', -1)} "
                f"reason={reason} "
                f"backup_goal=({plan.backup_goal[0]:.2f},"
                f"{plan.backup_goal[1]:.2f}) "
                "action=backup_only_no_forward_release"
            )
        return True

    def _safety_hold_reason_v22(self) -> Optional[str]:
        """
        Read the original V22 reason directly, bypassing the V2 method that
        released near-object safety for every custom-bypass phase.
        """
        reason = SawitRandomKalman321V22._safety_hold_reason_v22(self)
        if reason is None:
            self._safe_backup_last_reason = ""
            return None

        plan = getattr(self, "_bypass_plan", None)
        if plan is None:
            return reason

        near_reason = (
            reason.startswith("final_hard_stop")
            or reason.startswith("unverified_object_too_close")
        )
        if not near_reason:
            return reason

        # Near-object safety may only be released after the phase has been
        # converted to BACKUP. It is never released for SIDE_ENTRY or
        # PASS_FORWARD.
        if self._force_safe_backup(plan, reason):
            return None

        return reason

    def _progress_is_valid(
        self,
        plan: Any,
        remaining: float,
    ) -> bool:
        previous = float(getattr(plan, "last_remaining", math.inf))
        if (
            math.isfinite(previous)
            and math.isfinite(remaining)
            and remaining
            > previous + self.safe_bypass_reverse_progress_limit
        ):
            self.get_logger().warning(
                "DEPTH_COLLISION_SAFE_REVERSE_PROGRESS_V3 "
                f"id={getattr(plan, 'target_id', -1)} "
                f"phase={getattr(plan, 'phase', 'unknown')} "
                f"previous={previous:.2f} remaining={remaining:.2f} "
                "action=reject_phase"
            )
            return False

        return super()._progress_is_valid(plan, remaining)

    def _run_bypass(self) -> None:
        plan = getattr(self, "_bypass_plan", None)
        if plan is None:
            return

        if bool(getattr(self, "collision_abort_latched_v21n7", False)):
            self._clear_bypass_motion_safe()
            self._hold_current_v22(
                "collision_abort_latched_reset_gazebo_required"
            )
            return

        now = time.monotonic()
        pose_fault_until = float(
            getattr(self, "pose_fault_until", 0.0)
        )
        if now < pose_fault_until:
            self._hold_current_v22(
                "pose_fault_during_bypass"
            )
            return

        pose = getattr(self, "pose", None)
        if pose is None:
            return

        speed = self._safe_pose_speed(pose)
        if speed > self.safe_bypass_motion_speed_limit:
            self._hold_current_v22(
                f"bypass_speed_guard speed={speed:.2f}mps "
                f"limit={self.safe_bypass_motion_speed_limit:.2f}mps"
            )
            if now - self._safe_speed_hold_last_log >= 0.75:
                self._safe_speed_hold_last_log = now
                self.get_logger().error(
                    "DEPTH_COLLISION_SAFE_SPEED_HOLD_V3 "
                    f"speed={speed:.2f}mps "
                    f"limit={self.safe_bypass_motion_speed_limit:.2f}mps "
                    "action=hold_no_new_bypass_setpoint"
                )
            return

        front_distance = float(self._front_cluster_distance_v22())
        phase = str(getattr(plan, "phase", ""))

        if (
            phase not in ("HOLD", "BACKUP", "FAILSAFE_HOLD")
            and math.isfinite(front_distance)
            and front_distance < self.safe_bypass_emergency_front
        ):
            self._force_safe_backup(
                plan,
                (
                    f"front_guard={front_distance:.2f}m"
                    f"<{self.safe_bypass_emergency_front:.2f}m"
                ),
            )

        super()._run_bypass()

    def _control_loop(self) -> None:
        """
        This wrapper runs before the random-bypass control loop. Therefore a
        collision latch or pose fault cannot be bypassed by custom movement.
        """
        if bool(getattr(self, "collision_abort_latched_v21n7", False)):
            self._clear_bypass_motion_safe()
            self._hold_current_v22(
                "collision_abort_latched_reset_gazebo_required"
            )

            if not self._safe_collision_stop_logged:
                self._safe_collision_stop_logged = True
                self.get_logger().error(
                    "DEPTH_COLLISION_SAFE_ABORT_V3 "
                    "action=stop_all_bypass_and_navigation_"
                    "reset_gazebo_required"
                )
            return

        now = time.monotonic()
        if (
            getattr(self, "_bypass_plan", None) is not None
            and now < float(getattr(self, "pose_fault_until", 0.0))
        ):
            self._hold_current_v22(
                "pose_fault_hold_overrides_custom_bypass"
            )
            return

        super()._control_loop()


class SawitDepthKalman321CollisionSafeV3(
    CollisionSafeBypassMixin,
    SawitDepthKalman321AntiStuck,
):
    """Depth/ToF Kalman every valid message with 3-2-1 verification."""

    def __init__(self) -> None:
        super().__init__()
        self.get_logger().info(
            "START DEPTH_KALMAN_321_COLLISION_SAFE_V3 "
            f"run_id={getattr(self, 'normal_run_id', '')} "
            "verification_321=1 actual_used_for_control=0"
        )


class SawitDepthKalmanDirect1MCollisionSafeV3(
    CollisionSafeBypassMixin,
    SawitDepthKalmanDirect1MAntiStuck,
):
    """Depth/ToF Kalman every valid message, direct approximately 1 m."""

    def __init__(self) -> None:
        super().__init__()
        self.get_logger().info(
            "START DEPTH_KALMAN_DIRECT1M_COLLISION_SAFE_V3 "
            f"run_id={getattr(self, 'normal_run_id', '')} "
            "verification_321=0 visit=direct_1m "
            "actual_used_for_control=0"
        )


def _spin(node_class, args=None) -> None:
    rclpy.init(args=args)
    node = node_class()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main_321(args=None) -> None:
    _spin(SawitDepthKalman321CollisionSafeV3, args)


def main_direct1m(args=None) -> None:
    _spin(SawitDepthKalmanDirect1MCollisionSafeV3, args)


if __name__ == "__main__":
    main_321()
