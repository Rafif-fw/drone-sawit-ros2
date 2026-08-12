#!/usr/bin/env python3
"""
Varian komparasi tanpa verifikasi bertingkat 3-2-1.

Yang tetap dipakai:
- scan stasioner 360 derajat;
- filtering PointCloud dan track tentative/confirmed;
- target pertama terdekat, sisanya fixed random queue;
- Kalman PointCloud lama;
- update Kalman pada setiap pesan ToF valid;
- dedupe, target-lost recovery, visit guard, dan random obstacle bypass;
- actual/Gazebo hanya untuk visualisasi dan evaluasi.

Yang dihilangkan:
- gate 3 meter;
- mini-scan 3 meter;
- safety hold 2 meter;
- transisi 3 m -> 2 m -> 1 m.

Target langsung didekati sampai satu gate kunjungan sekitar 1 meter.
"""

from __future__ import annotations

import math
import time

import rclpy

from sawit_autonomy.sawit_navigator_random_kalman_321_random_bypass import (
    NavState,
)
from sawit_autonomy.sawit_navigator_random_kalman_321_tof_every_update import (
    SawitRandomKalman321TofEveryUpdate,
)


class SawitTofKalmanDirect1M(
    SawitRandomKalman321TofEveryUpdate
):
    """Kalman PointCloud + ToF per pesan, langsung visited sekitar 1 m."""

    def __init__(self) -> None:
        super().__init__()

        self.declare_parameter("direct_visit_distance", 1.00)
        self.declare_parameter("direct_visit_tolerance", 0.10)
        self.declare_parameter("direct_visit_min_safe", 0.82)
        self.declare_parameter("direct_visit_max_mad", 0.15)
        self.declare_parameter("direct_visit_min_samples", 5)
        self.declare_parameter("direct_visit_hold_time", 0.80)

        # Micro-waypoint dinamis: cepat saat jauh, pelan dekat batang.
        self.declare_parameter("direct_step_far", 0.28)
        self.declare_parameter("direct_step_mid", 0.18)
        self.declare_parameter("direct_step_near", 0.10)
        self.declare_parameter("direct_step_final", 0.045)

        self.direct_visit_distance = float(
            self.get_parameter("direct_visit_distance").value
        )
        self.direct_visit_tolerance = float(
            self.get_parameter("direct_visit_tolerance").value
        )
        self.direct_visit_min_safe = float(
            self.get_parameter("direct_visit_min_safe").value
        )
        self.direct_visit_max_mad = float(
            self.get_parameter("direct_visit_max_mad").value
        )
        self.direct_visit_min_samples = int(
            self.get_parameter("direct_visit_min_samples").value
        )
        self.direct_visit_hold_time = float(
            self.get_parameter("direct_visit_hold_time").value
        )

        self.direct_step_far = float(
            self.get_parameter("direct_step_far").value
        )
        self.direct_step_mid = float(
            self.get_parameter("direct_step_mid").value
        )
        self.direct_step_near = float(
            self.get_parameter("direct_step_near").value
        )
        self.direct_step_final = float(
            self.get_parameter("direct_step_final").value
        )

        self.direct_visit_max = (
            self.direct_visit_distance
            + self.direct_visit_tolerance
        )

        # Pakai hanya mekanisme final 1 m milik basis.
        # Nilai final diverifikasi ulang di _finish_tof_safe_visit agar
        # tidak lolos hanya karena toleransi internal basis lebih longgar.
        self.layer1_visit_distance = self.direct_visit_distance
        self.tof_layer_tolerance = self.direct_visit_tolerance
        self.tof_final_min_safe = self.direct_visit_min_safe
        self.tof_layer1_near_max_v21n12 = self.direct_visit_max
        self.tof_front_guard_layer1_v21n5 = self.direct_visit_max
        self.tof_near_gate_min_samples_v21n12 = max(
            self.tof_near_gate_min_samples_v21n12,
            self.direct_visit_min_samples,
        )
        self.tof_near_gate_max_mad_v21n12 = min(
            self.tof_near_gate_max_mad_v21n12,
            self.direct_visit_max_mad,
        )
        self.tof_layer_hold_time = self.direct_visit_hold_time

        # Basis V22 membatasi step tahap 1 m menjadi sangat kecil.
        # Varian ini mulai langsung dari tahap final, sehingga cap harus
        # dibuka dan kemudian diturunkan secara dinamis oleh fungsi di bawah.
        self.tof_stage1_step_v21n3 = max(
            self.tof_stage1_step_v21n3,
            self.direct_step_far,
        )

        self._direct_target_id = None
        self._direct_visit_reject_count = 0

        self.get_logger().info(
            "START TOF_KALMAN_DIRECT_1M_V1 "
            f"run_id={getattr(self, 'normal_run_id', '')} "
            f"seed={getattr(self, 'normal_random_seed', '')} "
            "pointcloud_kalman=kept "
            "tof_every_valid_message_kalman=1 "
            "gate3_disabled=1 miniscan3_disabled=1 "
            "gate2_disabled=1 "
            f"visit_nominal={self.direct_visit_distance:.2f}m "
            f"visit_band={self.direct_visit_min_safe:.2f}-"
            f"{self.direct_visit_max:.2f}m "
            f"mad_max={self.direct_visit_max_mad:.2f} "
            f"min_samples={self.direct_visit_min_samples} "
            "actual_used_for_control=0"
        )

    def _prepare_direct_1m_target(self, target) -> None:
        """Inisialisasi target langsung ke satu-satunya gate 1 m."""
        self.tof_approach_stage = "TO_1M"
        self.tof_stage_target_id = int(target.tree_id)
        self.tof_stage_hold_started = 0.0

        self.tof_recovery_history.clear()
        self.tof_dropout_started = 0.0
        self.tof_last_valid_distance = math.inf
        self.tof_last_valid_mono = 0.0
        self.tof_last_valid_target_distance = math.inf
        self.tof_selected_source_v21n2 = "none"
        self.tof_front_guard_history_v21n5.clear()

        # Tidak ada bearing beku hasil mini-scan 3 m.
        # Bearing selalu mengikuti posisi Kalman terbaru.
        self.tof_final_yaw_v21n3 = math.nan
        self.tof_final_target_id_v21n3 = None
        self.close_verify_purpose = ""

        self.final_progress_target_id_v21n12 = None
        self.final_progress_stage_v21n12 = ""
        self.final_progress_last_mono_v21n12 = 0.0

        self.motion_waypoint_xy = None
        self.motion_goal_xy = None
        self.motion_brake_anchor_xy = None

        self._direct_target_id = int(target.tree_id)

        self.get_logger().info(
            f"DIRECT_1M_TARGET_START_V1 id={target.tree_id} "
            f"target=({target.x:.2f},{target.y:.2f}) "
            "action=approach_direct_to_1m_no_3m_no_2m"
        )

    def _control_loop(self) -> None:
        # Pengaman: tidak boleh ada state mini-scan 3 m dari basis.
        if self.state in (
            NavState.CLOSE_SETTLE,
            NavState.CLOSE_FLUSH,
            NavState.CLOSE_COLLECT,
        ):
            if (
                getattr(self, "close_verify_purpose", "")
                == "TOF3_MINISCAN60_V21N14"
            ):
                self.close_verify_purpose = ""
                self.tof_approach_stage = "TO_1M"
                self.motion_waypoint_xy = None
                self.motion_goal_xy = None
                self.get_logger().warning(
                    "DIRECT_1M_BLOCK_MINISCAN3_V1 "
                    "action=return_to_approach_1m"
                )
                self._set_state(NavState.APPROACH)

        if self.state == NavState.APPROACH:
            target = self._active_track()
            if target is not None:
                wrong_target = (
                    self.tof_stage_target_id != target.tree_id
                )
                forbidden_stage = self.tof_approach_stage in (
                    "TO_3M",
                    "CHECK_3M",
                    "CHECK_3M_MINISCAN60",
                    "TO_2M",
                    "HOLD_2M",
                )
                unknown_stage = self.tof_approach_stage not in (
                    "TO_1M",
                    "HOLD_1M",
                )

                if wrong_target or forbidden_stage or unknown_stage:
                    self._prepare_direct_1m_target(target)

        # Random bypass, Kalman lama, Kalman ToF per pesan, dedupe,
        # visit guard, rescan, dan completion tetap dipanggil dari parent.
        super()._control_loop()

    def _final_step_cap_v21n12(
        self,
        stage: str,
        physical_distance: float,
    ) -> float:
        """Gerakan langsung 1 m: cepat saat jauh, konservatif dekat batang."""
        if stage != "TO_1M":
            return super()._final_step_cap_v21n12(
                stage,
                physical_distance,
            )

        if not math.isfinite(physical_distance):
            return 0.08
        if physical_distance > 6.0:
            return self.direct_step_far
        if physical_distance > 3.5:
            return min(self.direct_step_far, 0.24)
        if physical_distance > 2.0:
            return self.direct_step_mid
        if physical_distance > 1.35:
            return self.direct_step_near
        return self.direct_step_final

    def _finish_tof_safe_visit(
        self,
        target,
        tof_median: float,
        tof_mad: float,
        tof_count: int,
    ) -> None:
        """Final gate tunggal: visited hanya sekitar 1 m dan stabil."""
        valid = (
            math.isfinite(tof_median)
            and self.direct_visit_min_safe
            <= tof_median
            <= self.direct_visit_max
            and math.isfinite(tof_mad)
            and tof_mad <= self.direct_visit_max_mad
            and tof_count >= self.direct_visit_min_samples
        )

        if not valid:
            self._direct_visit_reject_count += 1
            self.tof_approach_stage = "TO_1M"
            self.tof_stage_hold_started = 0.0
            self.motion_waypoint_xy = None
            self.motion_goal_xy = None
            self.motion_brake_anchor_xy = None

            self.get_logger().warning(
                f"DIRECT_1M_VISIT_REJECT_V1 id={target.tree_id} "
                f"tof={tof_median:.2f} "
                f"allowed={self.direct_visit_min_safe:.2f}-"
                f"{self.direct_visit_max:.2f} "
                f"mad={tof_mad:.2f}/"
                f"{self.direct_visit_max_mad:.2f} "
                f"samples={tof_count}/"
                f"{self.direct_visit_min_samples} "
                f"reject_count={self._direct_visit_reject_count} "
                "action=continue_direct_approach"
            )
            return

        self.get_logger().info(
            f"DIRECT_1M_VISIT_ACCEPT_V1 id={target.tree_id} "
            f"tof={tof_median:.2f} mad={tof_mad:.2f} "
            f"samples={tof_count} "
            f"kalman=({target.x:.2f},{target.y:.2f}) "
            "verification=single_1m_gate no_3m=1 no_2m=1"
        )

        # Fungsi lama hanya dipakai untuk menetapkan VISITED, retreat,
        # rescan, penyimpanan memory, CSV, dan penyelesaian misi.
        super()._finish_tof_safe_visit(
            target,
            tof_median,
            tof_mad,
            tof_count,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SawitTofKalmanDirect1M()
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
