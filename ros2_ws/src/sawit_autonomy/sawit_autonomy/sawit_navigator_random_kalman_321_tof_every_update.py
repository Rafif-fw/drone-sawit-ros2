#!/usr/bin/env python3
"""
Varian komparasi:
- Seluruh algoritma lama random + Kalman PointCloud + 3-2-1 + bypass tetap dipakai.
- Setiap pesan ToF valid yang TERASOSIASI dengan target aktif juga menjadi measurement
  XY untuk Kalman yang sama.
- Ground truth Gazebo tetap visual/evaluasi saja.

Penting:
ToF tidak boleh dimasukkan secara buta. Pesan hanya diterima sebagai measurement target
jika target aktif ada, yaw menghadap target, range sesuai dengan jarak peta, dan state
bukan avoidance. Hal ini mencegah obstacle/daun/latar menggeser posisi pohon.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rclpy
from sensor_msgs.msg import LaserScan

from sawit_autonomy.sawit_navigator_random_kalman_321_v22 import (
    KalmanTrack2D,
)
from sawit_autonomy.sawit_navigator_random_kalman_321_random_bypass import (
    NavState,
    SawitRandomKalman321RandomBypass,
    TrackState,
)


def wrap_pi(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


class SawitRandomKalman321TofEveryUpdate(
    SawitRandomKalman321RandomBypass
):
    """Kalman lama + measurement tambahan dari setiap ToF valid."""

    def __init__(self) -> None:
        super().__init__()

        self.declare_parameter("tof_kalman_every_msg_enabled", True)
        self.declare_parameter("tof_kalman_center_offset", 0.18)
        self.declare_parameter("tof_kalman_measurement_std", 0.35)
        self.declare_parameter("tof_kalman_yaw_gate_deg", 12.0)
        self.declare_parameter("tof_kalman_range_gate", 2.20)
        self.declare_parameter("tof_kalman_xy_gate", 2.20)
        self.declare_parameter("tof_kalman_min_range", 0.72)
        self.declare_parameter("tof_kalman_max_range", 8.00)
        self.declare_parameter("tof_kalman_log_each_update", True)
        self.declare_parameter(
            "tof_kalman_csv_path",
            str(
                Path.home()
                / "ros2_ws/src/sawit_autonomy/data/"
                  "tof_every_message_kalman_updates.csv"
            ),
        )

        self.tof_kalman_every_msg_enabled = bool(
            self.get_parameter("tof_kalman_every_msg_enabled").value
        )
        self.tof_kalman_center_offset = float(
            self.get_parameter("tof_kalman_center_offset").value
        )
        self.tof_kalman_measurement_variance = float(
            self.get_parameter("tof_kalman_measurement_std").value
        ) ** 2
        self.tof_kalman_yaw_gate = math.radians(
            float(self.get_parameter("tof_kalman_yaw_gate_deg").value)
        )
        self.tof_kalman_range_gate = float(
            self.get_parameter("tof_kalman_range_gate").value
        )
        self.tof_kalman_xy_gate = float(
            self.get_parameter("tof_kalman_xy_gate").value
        )
        self.tof_kalman_min_range = float(
            self.get_parameter("tof_kalman_min_range").value
        )
        self.tof_kalman_max_range = float(
            self.get_parameter("tof_kalman_max_range").value
        )
        self.tof_kalman_log_each_update = bool(
            self.get_parameter("tof_kalman_log_each_update").value
        )
        self.tof_kalman_csv_path = Path(
            str(self.get_parameter("tof_kalman_csv_path").value)
        ).expanduser()

        self._tof_kalman_rx = 0
        self._tof_kalman_valid = 0
        self._tof_kalman_accepted = 0
        self._tof_kalman_rejected = 0
        self._tof_kalman_last_summary = 0.0
        self._tof_kalman_start_mono = time.monotonic()

        self._ensure_tof_kalman_csv()

        self.get_logger().info(
            "START TOF_EVERY_MESSAGE_KALMAN_V1 "
            "old_pointcloud_kalman=kept "
            "tof_additional_measurement=1 "
            f"center_offset={self.tof_kalman_center_offset:.2f}m "
            f"measurement_std="
            f"{math.sqrt(self.tof_kalman_measurement_variance):.2f}m "
            f"yaw_gate={math.degrees(self.tof_kalman_yaw_gate):.1f}deg "
            f"range_gate={self.tof_kalman_range_gate:.2f}m "
            f"xy_gate={self.tof_kalman_xy_gate:.2f}m "
            "actual_used_for_control=0"
        )

    @staticmethod
    def _tof_csv_fields() -> list[str]:
        return [
            "ros_time_sec",
            "run_id",
            "random_seed",
            "rx_index",
            "target_id",
            "nav_state",
            "stage",
            "tof_range",
            "map_distance_before",
            "range_residual",
            "yaw_error_deg",
            "prior_x",
            "prior_y",
            "measurement_x",
            "measurement_y",
            "post_x",
            "post_y",
            "kalman_gain_x",
            "kalman_gain_y",
            "innovation_xy",
            "accepted",
            "reason",
        ]

    def _ensure_tof_kalman_csv(self) -> None:
        self.tof_kalman_csv_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        if (
            not self.tof_kalman_csv_path.exists()
            or self.tof_kalman_csv_path.stat().st_size == 0
        ):
            with self.tof_kalman_csv_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                csv.DictWriter(
                    handle,
                    fieldnames=self._tof_csv_fields(),
                ).writeheader()

    def _write_tof_kalman_csv(
        self,
        *,
        target_id: int,
        tof_range: float,
        map_distance: float,
        range_residual: float,
        yaw_error: float,
        prior_xy: Tuple[float, float],
        measurement_xy: Tuple[float, float],
        post_xy: Tuple[float, float],
        gain_xy: Tuple[float, float],
        innovation_xy: float,
        accepted: bool,
        reason: str,
    ) -> None:
        ros_time = self.get_clock().now().nanoseconds / 1.0e9
        state = getattr(getattr(self, "state", None), "value", "")
        stage = str(getattr(self, "tof_approach_stage", ""))

        row = {
            "ros_time_sec": f"{ros_time:.6f}",
            "run_id": str(getattr(self, "normal_run_id", "")),
            "random_seed": int(getattr(self, "normal_random_seed", 0)),
            "rx_index": self._tof_kalman_rx,
            "target_id": target_id,
            "nav_state": state,
            "stage": stage,
            "tof_range": (
                f"{tof_range:.6f}" if math.isfinite(tof_range) else ""
            ),
            "map_distance_before": (
                f"{map_distance:.6f}"
                if math.isfinite(map_distance)
                else ""
            ),
            "range_residual": (
                f"{range_residual:.6f}"
                if math.isfinite(range_residual)
                else ""
            ),
            "yaw_error_deg": (
                f"{math.degrees(yaw_error):.6f}"
                if math.isfinite(yaw_error)
                else ""
            ),
            "prior_x": f"{prior_xy[0]:.6f}",
            "prior_y": f"{prior_xy[1]:.6f}",
            "measurement_x": f"{measurement_xy[0]:.6f}",
            "measurement_y": f"{measurement_xy[1]:.6f}",
            "post_x": f"{post_xy[0]:.6f}",
            "post_y": f"{post_xy[1]:.6f}",
            "kalman_gain_x": f"{gain_xy[0]:.6f}",
            "kalman_gain_y": f"{gain_xy[1]:.6f}",
            "innovation_xy": (
                f"{innovation_xy:.6f}"
                if math.isfinite(innovation_xy)
                else ""
            ),
            "accepted": int(accepted),
            "reason": reason,
        }

        with self.tof_kalman_csv_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as handle:
            csv.DictWriter(
                handle,
                fieldnames=self._tof_csv_fields(),
            ).writerow(row)

    def _current_tof_measurement(self) -> float:
        """Gunakan frame ToF yang baru saja diterima, bukan nilai lama."""
        try:
            value = float(self._tof_at_bearing(0.0))
        except Exception:
            value = math.inf
        if not math.isfinite(value):
            return math.inf
        if value < self.tof_kalman_min_range:
            return math.inf
        if value > self.tof_kalman_max_range:
            return math.inf
        return value

    def _tof_target_context(
        self,
        tof_range: float,
    ) -> Tuple[
        Optional[object],
        str,
        float,
        float,
        Tuple[float, float],
    ]:
        pose = getattr(self, "pose", None)
        if pose is None:
            return None, "no_pose", math.inf, math.inf, (0.0, 0.0)

        active_id = getattr(self, "active_target_id", None)
        if active_id is None:
            return None, "no_active_target", math.inf, math.inf, (
                float(pose.x_enu),
                float(pose.y_enu),
            )

        track = getattr(self, "tracks", {}).get(active_id)
        if track is None:
            return None, "active_track_missing", math.inf, math.inf, (
                float(pose.x_enu),
                float(pose.y_enu),
            )

        if track.state in (TrackState.VISITED, TrackState.REJECTED):
            return None, f"track_state_{track.state.value}", math.inf, math.inf, (
                float(pose.x_enu),
                float(pose.y_enu),
            )

        forbidden = {
            getattr(NavState, "AVOID_OBSTACLE", None),
            getattr(NavState, "SCAN_TURN", None),
            getattr(NavState, "SCAN_SETTLE", None),
            getattr(NavState, "SCAN_FLUSH", None),
            getattr(NavState, "SCAN_COLLECT", None),
            getattr(NavState, "RETREAT_VISITED", None),
            getattr(NavState, "COMPLETE", None),
        }
        forbidden.discard(None)
        if getattr(self, "state", None) in forbidden:
            return None, "state_not_target_measurement", math.inf, math.inf, (
                float(pose.x_enu),
                float(pose.y_enu),
            )

        dx = float(track.x) - float(pose.x_enu)
        dy = float(track.y) - float(pose.y_enu)
        map_distance = math.hypot(dx, dy)
        target_bearing = math.atan2(dy, dx)
        yaw_error = wrap_pi(target_bearing - float(pose.yaw_enu))

        if abs(yaw_error) > self.tof_kalman_yaw_gate:
            return track, "yaw_gate", map_distance, yaw_error, (
                float(pose.x_enu),
                float(pose.y_enu),
            )

        expected_surface = max(
            self.tof_kalman_min_range,
            map_distance - self.tof_kalman_center_offset,
        )
        residual = abs(tof_range - expected_surface)
        if residual > self.tof_kalman_range_gate:
            return track, "range_gate", map_distance, yaw_error, (
                float(pose.x_enu),
                float(pose.y_enu),
            )

        return track, "accepted", map_distance, yaw_error, (
            float(pose.x_enu),
            float(pose.y_enu),
        )

    def _update_target_from_tof(
        self,
        tof_range: float,
    ) -> None:
        track, reason, map_distance, yaw_error, pose_xy = (
            self._tof_target_context(tof_range)
        )

        if track is None or reason != "accepted":
            self._tof_kalman_rejected += 1
            self._log_tof_kalman_summary()
            return

        self._tof_kalman_valid += 1

        # Proses dulu measurement PointCloud yang mungkin baru masuk,
        # kemudian ToF menjadi measurement tambahan pada filter yang sama.
        self._apply_kalman_updates_v22()

        tree_id = int(track.tree_id)
        filt = self._kalman_tracks.get(tree_id)
        now = time.monotonic()

        if filt is None:
            filt = KalmanTrack2D(
                x=float(track.x),
                y=float(track.y),
                variance_x=float(self.kalman_initial_variance),
                variance_y=float(self.kalman_initial_variance),
                last_time=now,
            )
            self._kalman_tracks[tree_id] = filt
            update_stamp = float(
                getattr(track, "updated_mono", 0.0)
            )
            if update_stamp > 0.0:
                self._last_track_update[tree_id] = update_stamp

        pose = getattr(self, "pose", None)
        if pose is None:
            self._tof_kalman_rejected += 1
            return

        measurement_distance = (
            tof_range + self.tof_kalman_center_offset
        )
        measurement_x = (
            float(pose.x_enu)
            + math.cos(float(pose.yaw_enu)) * measurement_distance
        )
        measurement_y = (
            float(pose.y_enu)
            + math.sin(float(pose.yaw_enu)) * measurement_distance
        )

        prior_x = float(filt.x)
        prior_y = float(filt.y)
        innovation = math.hypot(
            measurement_x - prior_x,
            measurement_y - prior_y,
        )
        range_residual = abs(
            tof_range
            - max(
                self.tof_kalman_min_range,
                map_distance - self.tof_kalman_center_offset,
            )
        )

        if innovation > self.tof_kalman_xy_gate:
            self._tof_kalman_rejected += 1
            self._write_tof_kalman_csv(
                target_id=tree_id,
                tof_range=tof_range,
                map_distance=map_distance,
                range_residual=range_residual,
                yaw_error=yaw_error,
                prior_xy=(prior_x, prior_y),
                measurement_xy=(measurement_x, measurement_y),
                post_xy=(prior_x, prior_y),
                gain_xy=(float(filt.gain_x), float(filt.gain_y)),
                innovation_xy=innovation,
                accepted=False,
                reason="xy_innovation_gate",
            )
            self.get_logger().warning(
                f"TOF_KALMAN_GATE_REJECT_V1 id={tree_id} "
                f"tof={tof_range:.2f} "
                f"innovation={innovation:.2f}m "
                f"limit={self.tof_kalman_xy_gate:.2f}m"
            )
            self._log_tof_kalman_summary()
            return

        filt.predict(
            now=now,
            process_variance_per_sec=self.kalman_process_variance,
        )
        post_x, post_y = filt.update(
            measurement_x=measurement_x,
            measurement_y=measurement_y,
            measurement_variance=self.tof_kalman_measurement_variance,
        )

        track.x = float(post_x)
        track.y = float(post_y)
        self._tof_kalman_accepted += 1

        self._write_tof_kalman_csv(
            target_id=tree_id,
            tof_range=tof_range,
            map_distance=map_distance,
            range_residual=range_residual,
            yaw_error=yaw_error,
            prior_xy=(prior_x, prior_y),
            measurement_xy=(measurement_x, measurement_y),
            post_xy=(post_x, post_y),
            gain_xy=(float(filt.gain_x), float(filt.gain_y)),
            innovation_xy=innovation,
            accepted=True,
            reason="accepted",
        )

        if self.tof_kalman_log_each_update:
            self.get_logger().info(
                f"TOF_KALMAN_UPDATE_EVERY_MSG_V1 "
                f"rx={self._tof_kalman_rx} id={tree_id} "
                f"state={getattr(self.state, 'value', self.state)} "
                f"stage={getattr(self, 'tof_approach_stage', '')} "
                f"tof={tof_range:.2f} "
                f"map_before={map_distance:.2f} "
                f"yaw_err={math.degrees(yaw_error):+.1f}deg "
                f"z=({measurement_x:.2f},{measurement_y:.2f}) "
                f"prior=({prior_x:.2f},{prior_y:.2f}) "
                f"K=({filt.gain_x:.3f},{filt.gain_y:.3f}) "
                f"post=({post_x:.2f},{post_y:.2f}) "
                f"updates={filt.updates}"
            )

        self._log_tof_kalman_summary()

    def _log_tof_kalman_summary(self) -> None:
        now = time.monotonic()
        if now - self._tof_kalman_last_summary < 1.0:
            return
        self._tof_kalman_last_summary = now
        self.get_logger().info(
            f"TOF_KALMAN_MONITOR_V1 "
            f"elapsed={now - self._tof_kalman_start_mono:.1f}s "
            f"rx={self._tof_kalman_rx} "
            f"valid_target={self._tof_kalman_valid} "
            f"accepted={self._tof_kalman_accepted} "
            f"rejected_or_unassociated={self._tof_kalman_rejected}"
        )

    def _on_tof(self, msg: LaserScan) -> None:
        # Semua perilaku ToF lama tetap dijalankan lebih dahulu.
        super()._on_tof(msg)

        self._tof_kalman_rx += 1
        if not self.tof_kalman_every_msg_enabled:
            return

        tof_range = self._current_tof_measurement()
        if not math.isfinite(tof_range):
            self._tof_kalman_rejected += 1
            self._log_tof_kalman_summary()
            return

        self._update_target_from_tof(tof_range)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SawitRandomKalman321TofEveryUpdate()
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
