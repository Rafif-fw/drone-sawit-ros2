#!/usr/bin/env python3
"""
Dua algoritma pembanding:

A. Depth-camera Kalman + ToF Kalman dengan 3-2-1.
B. Depth-camera Kalman + ToF Kalman tanpa 3-2-1,
   langsung kunjungan sekitar 1 meter.

Pada setiap pesan PointCloud2:
1. PointCloud difilter kembali.
2. Kandidat batang dihitung kembali.
3. Kandidat diasosiasikan dengan track pohon.
4. Centroid valid menjadi measurement Kalman.
5. Posisi track diperbarui dari hasil Kalman.

Actual/ground truth Gazebo tidak digunakan dalam navigasi.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import List, Optional, Tuple

import rclpy
from sensor_msgs.msg import PointCloud2

from sawit_autonomy.sawit_navigator_random_kalman_321_v22 import (
    KalmanTrack2D,
)
from sawit_autonomy.sawit_navigator_random_kalman_321_random_bypass import (
    NavState,
    TrackState,
)
from sawit_autonomy.sawit_navigator_random_kalman_321_tof_every_update import (
    SawitRandomKalman321TofEveryUpdate,
)
from sawit_autonomy.sawit_navigator_tof_kalman_direct_1m import (
    SawitTofKalmanDirect1M,
)


def wrap_pi(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


class DepthCameraEveryMessageKalmanMixin:
    """
    Mixin untuk menghitung ulang centroid dan memperbarui Kalman
    pada setiap pesan depth-camera yang valid.
    """

    def __init__(self) -> None:
        super().__init__()

        self.declare_parameter(
            "depth_every_message_kalman_enabled",
            True,
        )
        self.declare_parameter(
            "depth_kalman_association_gate",
            1.35,
        )
        self.declare_parameter(
            "depth_kalman_bearing_gate_deg",
            20.0,
        )
        self.declare_parameter(
            "depth_kalman_range_gate",
            2.00,
        )
        self.declare_parameter(
            "depth_kalman_measurement_std",
            0.28,
        )
        self.declare_parameter(
            "depth_kalman_max_speed",
            0.80,
        )
        self.declare_parameter(
            "depth_kalman_min_score",
            150.0,
        )
        self.declare_parameter(
            "depth_kalman_log_each_update",
            False,
        )
        self.declare_parameter(
            "depth_kalman_csv_path",
            str(
                Path.home()
                / "ros2_ws/src/sawit_autonomy/data/"
                  "depth_camera_kalman_updates.csv"
            ),
        )

        self.depth_kalman_enabled = bool(
            self.get_parameter(
                "depth_every_message_kalman_enabled"
            ).value
        )
        self.depth_assoc_gate = float(
            self.get_parameter(
                "depth_kalman_association_gate"
            ).value
        )
        self.depth_bearing_gate = math.radians(
            float(
                self.get_parameter(
                    "depth_kalman_bearing_gate_deg"
                ).value
            )
        )
        self.depth_range_gate = float(
            self.get_parameter(
                "depth_kalman_range_gate"
            ).value
        )
        self.depth_measurement_std = float(
            self.get_parameter(
                "depth_kalman_measurement_std"
            ).value
        )
        self.depth_max_speed = float(
            self.get_parameter(
                "depth_kalman_max_speed"
            ).value
        )
        self.depth_min_score = float(
            self.get_parameter(
                "depth_kalman_min_score"
            ).value
        )
        self.depth_log_each_update = bool(
            self.get_parameter(
                "depth_kalman_log_each_update"
            ).value
        )
        self.depth_csv_path = Path(
            str(
                self.get_parameter(
                    "depth_kalman_csv_path"
                ).value
            )
        ).expanduser()

        self.depth_rx = 0
        self.depth_parsed = 0
        self.depth_update_frames = 0
        self.depth_accepted = 0
        self.depth_rejected = 0

        self.depth_started_mono = time.monotonic()
        self.depth_last_summary = 0.0

        self._prepare_depth_csv()

    @staticmethod
    def _depth_fields() -> List[str]:
        return [
            "ros_time_sec",
            "run_id",
            "seed",
            "depth_rx",
            "cloud_seq",
            "target_id",
            "state",
            "candidate_range",
            "candidate_score",
            "candidate_strong",
            "association_distance",
            "bearing_error_deg",
            "range_residual",
            "prior_x",
            "prior_y",
            "measurement_x",
            "measurement_y",
            "post_x",
            "post_y",
            "kalman_gain_x",
            "kalman_gain_y",
            "measurement_std",
        ]

    def _prepare_depth_csv(self) -> None:
        self.depth_csv_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if (
            not self.depth_csv_path.exists()
            or self.depth_csv_path.stat().st_size == 0
        ):
            with self.depth_csv_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                csv.DictWriter(
                    handle,
                    fieldnames=self._depth_fields(),
                ).writeheader()

    def _write_depth_csv(
        self,
        *,
        track,
        candidate,
        association_distance: float,
        bearing_error: float,
        range_residual: float,
        prior_x: float,
        prior_y: float,
        measurement_x: float,
        measurement_y: float,
        post_x: float,
        post_y: float,
        gain_x: float,
        gain_y: float,
        measurement_std: float,
    ) -> None:
        state_name = getattr(
            self.state,
            "value",
            str(self.state),
        )

        row = {
            "ros_time_sec": (
                f"{self.get_clock().now().nanoseconds / 1e9:.6f}"
            ),
            "run_id": str(
                getattr(self, "normal_run_id", "")
            ),
            "seed": int(
                getattr(self, "normal_random_seed", 0)
            ),
            "depth_rx": self.depth_rx,
            "cloud_seq": int(self.cloud_seq),
            "target_id": int(track.tree_id),
            "state": state_name,
            "candidate_range": (
                f"{candidate.range_m:.6f}"
            ),
            "candidate_score": (
                f"{candidate.score:.6f}"
            ),
            "candidate_strong": int(candidate.strong),
            "association_distance": (
                f"{association_distance:.6f}"
            ),
            "bearing_error_deg": (
                f"{math.degrees(bearing_error):.6f}"
            ),
            "range_residual": (
                f"{range_residual:.6f}"
            ),
            "prior_x": f"{prior_x:.6f}",
            "prior_y": f"{prior_y:.6f}",
            "measurement_x": f"{measurement_x:.6f}",
            "measurement_y": f"{measurement_y:.6f}",
            "post_x": f"{post_x:.6f}",
            "post_y": f"{post_y:.6f}",
            "kalman_gain_x": f"{gain_x:.6f}",
            "kalman_gain_y": f"{gain_y:.6f}",
            "measurement_std": (
                f"{measurement_std:.6f}"
            ),
        }

        with self.depth_csv_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as handle:
            csv.DictWriter(
                handle,
                fieldnames=self._depth_fields(),
            ).writerow(row)

    def _on_cloud(self, msg: PointCloud2) -> None:
        self.depth_rx += 1

        # Seluruh fungsi PointCloud lama tetap dijalankan:
        # scan, safety cloud, accumulation, debug, dan mini-scan.
        super()._on_cloud(msg)

        if not self.depth_kalman_enabled:
            return

        self._update_kalman_from_depth_message(msg)

    def _eligible_tracks_and_anchor(
        self,
    ) -> Tuple[
        List[object],
        Optional[Tuple[float, float]],
        Optional[float],
        str,
    ]:
        pose = getattr(self, "pose", None)

        if pose is None:
            return [], None, None, "no_pose"

        if getattr(self, "_bypass_plan", None) is not None:
            return [], None, None, "bypass_active"

        if self.state == NavState.AVOID_OBSTACLE:
            return [], None, None, "avoidance_active"

        # Scan 360 derajat: gunakan anchor yang dibekukan.
        if self.state == NavState.SCAN_COLLECT:
            if not self._scan_capture_stable():
                return [], None, None, "scan_not_stable"

            tracks = [
                track
                for track in self.tracks.values()
                if track.state in (
                    TrackState.TENTATIVE,
                    TrackState.CONFIRMED,
                )
            ]

            return (
                tracks,
                self.sector_anchor_xy,
                self.sector_anchor_yaw,
                "scan_collect",
            )

        # Mini-scan 3 m untuk algoritma 3-2-1.
        if self.state == NavState.CLOSE_COLLECT:
            target = self._active_track()

            if target is None:
                return [], None, None, "no_close_target"

            return (
                [target],
                self.close_anchor_xy,
                self.close_anchor_yaw,
                "close_collect",
            )

        target_states = {
            NavState.ALIGN_TARGET,
            NavState.APPROACH,
            NavState.RETRY_VERIFY,
            NavState.REACQUIRE_FINAL,
        }

        optional_names = (
            "BRAKE_HOLD",
            "HOLD",
        )
        for name in optional_names:
            value = getattr(NavState, name, None)
            if value is not None:
                target_states.add(value)

        if self.state in target_states:
            if pose.speed_xy > self.depth_max_speed:
                return [], None, None, "moving_too_fast"

            target = self._active_track()

            if target is None:
                return [], None, None, "no_active_target"

            if target.state in (
                TrackState.VISITED,
                TrackState.REJECTED,
            ):
                return [], None, None, "inactive_track"

            return (
                [target],
                (pose.x_enu, pose.y_enu),
                pose.yaw_enu,
                "active_target",
            )

        # Saat eksplorasi, track lama yang confirmed masih dapat
        # diperbarui bila kandidatnya benar-benar berasosiasi.
        explore_states = set()

        for name in (
            "EXPLORE_ALIGN",
            "EXPLORE_MOVE",
        ):
            value = getattr(NavState, name, None)
            if value is not None:
                explore_states.add(value)

        if self.state in explore_states:
            if pose.speed_xy > self.depth_max_speed:
                return [], None, None, "explore_too_fast"

            tracks = [
                track
                for track in self.tracks.values()
                if track.state == TrackState.CONFIRMED
            ]

            return (
                tracks,
                (pose.x_enu, pose.y_enu),
                pose.yaw_enu,
                "explore",
            )

        return [], None, None, "state_not_used"

    def _ensure_depth_filter(
        self,
        track,
    ) -> KalmanTrack2D:
        tree_id = int(track.tree_id)

        filt = self._kalman_tracks.get(tree_id)

        if filt is None:
            filt = KalmanTrack2D(
                x=float(track.x),
                y=float(track.y),
                variance_x=float(
                    self.kalman_initial_variance
                ),
                variance_y=float(
                    self.kalman_initial_variance
                ),
                last_time=time.monotonic(),
            )

            self._kalman_tracks[tree_id] = filt

            self.get_logger().info(
                f"DEPTH_KALMAN_INIT_EVERY_CAMERA_V1 "
                f"id={tree_id} "
                f"x={track.x:.2f} y={track.y:.2f}"
            )

        return filt

    def _update_kalman_from_depth_message(
        self,
        msg: PointCloud2,
    ) -> None:
        (
            tracks,
            anchor_xy,
            anchor_yaw,
            context,
        ) = self._eligible_tracks_and_anchor()

        if (
            not tracks
            or anchor_xy is None
            or anchor_yaw is None
        ):
            self.depth_rejected += 1
            self._depth_summary(context)
            return

        try:
            raw_points = self._pointcloud_to_xyz(msg)
            self.depth_parsed += 1

            if raw_points.size == 0:
                self.depth_rejected += 1
                self._depth_summary("empty_cloud")
                return

            candidates, _ = self._extract_trunk_candidates(
                raw_points
            )

        except Exception as exc:
            self.depth_rejected += 1

            self.get_logger().warning(
                "DEPTH_KALMAN_PROCESS_REJECT_V1 "
                f"type={type(exc).__name__} "
                f"error={exc}"
            )

            self._depth_summary("parse_error")
            return

        if not candidates:
            self.depth_rejected += 1
            self._depth_summary("no_candidate")
            return

        possible_pairs = []

        for candidate_index, candidate in enumerate(
            candidates
        ):
            if candidate.score < self.depth_min_score:
                continue

            measurement_x, measurement_y = (
                self._body_to_map(
                    candidate.forward,
                    candidate.left,
                    anchor_xy,
                    anchor_yaw,
                )
            )

            for track in tracks:
                dx = float(track.x) - anchor_xy[0]
                dy = float(track.y) - anchor_xy[1]

                expected_range = math.hypot(dx, dy)

                expected_bearing = wrap_pi(
                    math.atan2(dy, dx) - anchor_yaw
                )

                association_distance = math.hypot(
                    measurement_x - float(track.x),
                    measurement_y - float(track.y),
                )

                bearing_error = abs(
                    wrap_pi(
                        candidate.bearing
                        - expected_bearing
                    )
                )

                range_residual = abs(
                    candidate.range_m
                    - expected_range
                )

                if (
                    association_distance
                    > self.depth_assoc_gate
                ):
                    continue

                if bearing_error > self.depth_bearing_gate:
                    continue

                if range_residual > self.depth_range_gate:
                    continue

                cost = (
                    association_distance
                    + 0.20 * range_residual
                    + 0.30 * bearing_error
                    - (
                        0.08
                        if candidate.strong
                        else 0.0
                    )
                )

                possible_pairs.append(
                    (
                        cost,
                        candidate_index,
                        int(track.tree_id),
                        candidate,
                        track,
                        measurement_x,
                        measurement_y,
                        association_distance,
                        bearing_error,
                        range_residual,
                    )
                )

        if not possible_pairs:
            self.depth_rejected += 1
            self._depth_summary("association_reject")
            return

        possible_pairs.sort(key=lambda row: row[0])

        # Measurement PointCloud basis yang belum diproses
        # diselesaikan sebelum frame depth baru dimasukkan.
        self._apply_kalman_updates_v22()

        used_candidates = set()
        used_tracks = set()
        accepted_this_frame = 0

        for (
            _cost,
            candidate_index,
            tree_id,
            candidate,
            track,
            measurement_x,
            measurement_y,
            association_distance,
            bearing_error,
            range_residual,
        ) in possible_pairs:

            if candidate_index in used_candidates:
                continue

            if tree_id in used_tracks:
                continue

            filt = self._ensure_depth_filter(track)

            prior_x = float(filt.x)
            prior_y = float(filt.y)

            innovation = math.hypot(
                measurement_x - prior_x,
                measurement_y - prior_y,
            )

            if innovation > self.depth_assoc_gate:
                continue

            # Frame jauh atau kandidat yang kurang kuat diberi
            # measurement noise lebih besar.
            measurement_std = (
                self.depth_measurement_std
                + min(
                    0.18,
                    0.012 * candidate.range_m,
                )
                + (
                    0.08
                    if not candidate.strong
                    else 0.0
                )
            )

            filt.predict(
                now=time.monotonic(),
                process_variance_per_sec=(
                    self.kalman_process_variance
                ),
            )

            post_x, post_y = filt.update(
                measurement_x=measurement_x,
                measurement_y=measurement_y,
                measurement_variance=(
                    measurement_std ** 2
                ),
            )

            track.x = float(post_x)
            track.y = float(post_y)

            used_candidates.add(candidate_index)
            used_tracks.add(tree_id)

            accepted_this_frame += 1
            self.depth_accepted += 1

            self._write_depth_csv(
                track=track,
                candidate=candidate,
                association_distance=(
                    association_distance
                ),
                bearing_error=bearing_error,
                range_residual=range_residual,
                prior_x=prior_x,
                prior_y=prior_y,
                measurement_x=measurement_x,
                measurement_y=measurement_y,
                post_x=post_x,
                post_y=post_y,
                gain_x=float(filt.gain_x),
                gain_y=float(filt.gain_y),
                measurement_std=measurement_std,
            )

            if self.depth_log_each_update:
                self.get_logger().info(
                    "DEPTH_KALMAN_UPDATE_EVERY_CAMERA_V1 "
                    f"rx={self.depth_rx} "
                    f"seq={self.cloud_seq} "
                    f"id={tree_id} "
                    f"context={context} "
                    f"range={candidate.range_m:.2f} "
                    f"score={candidate.score:.1f} "
                    f"assoc={association_distance:.2f} "
                    f"z=({measurement_x:.2f},"
                    f"{measurement_y:.2f}) "
                    f"prior=({prior_x:.2f},"
                    f"{prior_y:.2f}) "
                    f"K=({filt.gain_x:.3f},"
                    f"{filt.gain_y:.3f}) "
                    f"post=({post_x:.2f},"
                    f"{post_y:.2f})"
                )

        if accepted_this_frame > 0:
            self.depth_update_frames += 1
            reason = "accepted"
        else:
            self.depth_rejected += 1
            reason = "innovation_reject"

        self._depth_summary(reason)

    def _depth_summary(
        self,
        reason: str,
    ) -> None:
        now = time.monotonic()

        if now - self.depth_last_summary < 1.0:
            return

        self.depth_last_summary = now

        self.get_logger().info(
            "DEPTH_KALMAN_MONITOR_EVERY_CAMERA_V1 "
            f"elapsed="
            f"{now - self.depth_started_mono:.1f}s "
            f"received={self.depth_rx} "
            f"parsed={self.depth_parsed} "
            f"update_frames={self.depth_update_frames} "
            f"accepted_measurements="
            f"{self.depth_accepted} "
            f"rejected_or_unassociated="
            f"{self.depth_rejected} "
            f"last_reason={reason} "
            f"tof_received="
            f"{getattr(self, '_tof_kalman_rx', 0)} "
            f"tof_accepted="
            f"{getattr(self, '_tof_kalman_accepted', 0)}"
        )


class SawitDepthCameraKalman321(
    DepthCameraEveryMessageKalmanMixin,
    SawitRandomKalman321TofEveryUpdate,
):
    """
    Depth setiap frame + ToF setiap pesan + Kalman,
    tetap menggunakan verifikasi 3-2-1.
    """

    def __init__(self) -> None:
        super().__init__()

        self.get_logger().info(
            "START DEPTH_CAMERA_KALMAN_321_V1 "
            f"run_id={self.normal_run_id} "
            f"seed={self.normal_random_seed} "
            "depth_every_valid_camera_message=1 "
            "tof_every_valid_message=1 "
            "verification_321=1"
        )


class SawitDepthCameraKalmanDirect1M(
    DepthCameraEveryMessageKalmanMixin,
    SawitTofKalmanDirect1M,
):
    """
    Depth setiap frame + ToF setiap pesan + Kalman,
    tanpa 3-2-1 dan langsung kunjungan sekitar 1 meter.
    """

    def __init__(self) -> None:
        super().__init__()

        self.get_logger().info(
            "START DEPTH_CAMERA_KALMAN_DIRECT_1M_V1 "
            f"run_id={self.normal_run_id} "
            f"seed={self.normal_random_seed} "
            "depth_every_valid_camera_message=1 "
            "tof_every_valid_message=1 "
            "verification_321=0 "
            "visit=direct_1m"
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
    _spin(SawitDepthCameraKalman321, args)


def main_direct_1m(args=None) -> None:
    _spin(SawitDepthCameraKalmanDirect1M, args)


if __name__ == "__main__":
    main_321()
