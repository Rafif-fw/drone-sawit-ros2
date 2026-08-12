#!/usr/bin/env python3
"""
Dua algoritma pembanding dengan pembaruan Kalman dari setiap topic
depth camera / PointCloud2 yang VALID:

A. Depth + ToF Kalman dengan verifikasi bertingkat 3-2-1.
B. Depth + ToF Kalman tanpa 3-2-1, langsung kunjungan sekitar 1 meter.

Perbaikan anti-mandek:
- Saat custom RANDOM_BYPASS aktif, V22 final_hard_stop tidak boleh
  menahan drone terus-menerus.
- Obstacle sangat dekat saat bypass ditangani oleh emergency BACKUP
  milik random bypass, seperti versi yang sebelumnya berhasil.
- Watchdog sensor timeout, pose fault, dan collision guard tetap aktif.

Actual/Gazebo ground truth tidak pernah digunakan untuk navigasi,
association, atau update Kalman.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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


class DepthCameraEveryMessageKalmanAntiStuckMixin:
    """
    Menghitung kandidat batang dan meng-update Kalman pada setiap
    pesan PointCloud2 yang valid.

    Catatan:
    - Initial scan tetap membuat centroid/track lewat detector basis.
    - Setelah track tersedia, setiap frame valid yang berasosiasi
      memperbarui filter Kalman track tersebut.
    - Saat bypass/avoidance, depth tetap diterima tetapi tidak dipakai
      untuk menggeser centroid target.
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
            18.0,
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
            "depth_kalman_require_strong_while_moving",
            True,
        )
        self.declare_parameter(
            "depth_kalman_log_each_update",
            False,
        )
        self.declare_parameter(
            "depth_kalman_summary_period",
            1.00,
        )
        self.declare_parameter(
            "depth_kalman_csv_path",
            str(
                Path.home()
                / "ros2_ws/src/sawit_autonomy/data/"
                  "depth_camera_kalman_updates_antistuck.csv"
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
        self.depth_require_strong_moving = bool(
            self.get_parameter(
                "depth_kalman_require_strong_while_moving"
            ).value
        )
        self.depth_log_each_update = bool(
            self.get_parameter(
                "depth_kalman_log_each_update"
            ).value
        )
        self.depth_summary_period = max(
            0.20,
            float(
                self.get_parameter(
                    "depth_kalman_summary_period"
                ).value
            ),
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
        self.depth_candidate_frames = 0
        self.depth_update_frames = 0
        self.depth_accepted = 0
        self.depth_rejected = 0
        self.depth_initial_scan_frames = 0

        self.depth_started_mono = time.monotonic()
        self.depth_last_summary = 0.0

        # Cache satu pesan supaya callback basis dan mixin tidak
        # membaca buffer PointCloud2 dua kali.
        self._depth_cache_key: Optional[
            Tuple[int, int, int, int, int]
        ] = None
        self._depth_cache_xyz = None

        self._depth_bypass_release_logged = False
        self._prepare_depth_csv()

    # ==========================================================
    # Anti-mandek custom bypass
    # ==========================================================

    def _safety_hold_reason_v22(self) -> Optional[str]:
        """
        Random bypass harus tetap boleh melakukan BACKUP ketika objek
        sangat dekat. Pada versi sebelumnya, direct-1m memakai stage
        TO_1M sehingga V22 mengeluarkan final_hard_stop dan menghentikan
        _run_bypass selamanya.

        Hanya hold obstacle-dekat yang dilepas saat custom plan aktif.
        Timeout sensor, pose fault, dan watchdog lain tetap berlaku.
        """
        reason = super()._safety_hold_reason_v22()

        bypass_active = (
            getattr(self, "_bypass_plan", None) is not None
        )
        if not bypass_active or reason is None:
            self._depth_bypass_release_logged = False
            return reason

        bypass_handled_reasons = (
            "final_hard_stop",
            "unverified_object_too_close",
        )

        if reason.startswith(bypass_handled_reasons):
            if not self._depth_bypass_release_logged:
                self.get_logger().warning(
                    "DEPTH_DUAL_BYPASS_SAFETY_RELEASE_V2 "
                    f"reason={reason} "
                    "action=allow_custom_emergency_backup "
                    "watchdog_timeout_still_enabled=1"
                )
                self._depth_bypass_release_logged = True
            return None

        return reason

    # ==========================================================
    # CSV
    # ==========================================================

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
            "context",
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
            "innovation_xy",
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
        track: Any,
        candidate: Any,
        context: str,
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
        innovation: float,
        measurement_std: float,
    ) -> None:
        row: Dict[str, Any] = {
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
            "cloud_seq": int(
                getattr(self, "cloud_seq", -1)
            ),
            "target_id": int(track.tree_id),
            "state": str(
                getattr(
                    getattr(self, "state", None),
                    "value",
                    getattr(self, "state", ""),
                )
            ),
            "context": context,
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
            "innovation_xy": f"{innovation:.6f}",
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

    # ==========================================================
    # PointCloud cache
    # ==========================================================

    @staticmethod
    def _depth_message_key(
        msg: PointCloud2,
    ) -> Tuple[int, int, int, int, int]:
        return (
            int(msg.header.stamp.sec),
            int(msg.header.stamp.nanosec),
            int(msg.width),
            int(msg.height),
            int(msg.point_step),
        )

    def _pointcloud_to_xyz(self, msg: PointCloud2):
        key = self._depth_message_key(msg)

        if (
            self._depth_cache_key == key
            and self._depth_cache_xyz is not None
        ):
            return self._depth_cache_xyz

        xyz = super()._pointcloud_to_xyz(msg)
        self._depth_cache_key = key
        self._depth_cache_xyz = xyz
        return xyz

    # ==========================================================
    # Callback camera depth
    # ==========================================================

    def _on_cloud(self, msg: PointCloud2) -> None:
        self.depth_rx += 1

        # Detector, scan, mini-scan, moving safety, debug, dan
        # memory basis tetap berjalan lebih dahulu.
        super()._on_cloud(msg)

        if not self.depth_kalman_enabled:
            return

        self._update_kalman_from_depth_message(msg)

    def _state_set(self, names: Tuple[str, ...]) -> Set[Any]:
        result: Set[Any] = set()
        for name in names:
            value = getattr(NavState, name, None)
            if value is not None:
                result.add(value)
        return result

    def _depth_context_and_anchor(
        self,
    ) -> Tuple[
        List[Any],
        Optional[Tuple[float, float]],
        Optional[float],
        str,
        bool,
    ]:
        """
        Return:
        tracks, anchor_xy, anchor_yaw, context, moving_context
        """
        pose = getattr(self, "pose", None)
        if pose is None:
            return [], None, None, "no_pose", False

        if getattr(self, "_bypass_plan", None) is not None:
            return [], None, None, "bypass_active", True

        avoid_state = getattr(
            NavState,
            "AVOID_OBSTACLE",
            None,
        )
        if (
            avoid_state is not None
            and getattr(self, "state", None) == avoid_state
        ):
            return [], None, None, "avoidance_active", True

        scan_collect = getattr(
            NavState,
            "SCAN_COLLECT",
            None,
        )
        if (
            scan_collect is not None
            and self.state == scan_collect
        ):
            stable = bool(self._scan_capture_stable())
            if not stable:
                return (
                    [],
                    self.sector_anchor_xy,
                    self.sector_anchor_yaw,
                    "scan_not_stable",
                    False,
                )

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
                False,
            )

        close_collect = getattr(
            NavState,
            "CLOSE_COLLECT",
            None,
        )
        if (
            close_collect is not None
            and self.state == close_collect
        ):
            target = self._active_track()
            tracks = [target] if target is not None else []
            return (
                tracks,
                self.close_anchor_xy,
                self.close_anchor_yaw,
                "close_collect",
                False,
            )

        target_states = self._state_set(
            (
                "ALIGN_TARGET",
                "APPROACH",
                "RETRY_VERIFY",
                "REACQUIRE_FINAL",
                "BRAKE_HOLD",
                "HOLD",
            )
        )
        if self.state in target_states:
            if pose.speed_xy > self.depth_max_speed:
                return (
                    [],
                    (pose.x_enu, pose.y_enu),
                    pose.yaw_enu,
                    "moving_too_fast",
                    True,
                )

            target = self._active_track()
            tracks = []
            if (
                target is not None
                and target.state not in (
                    TrackState.VISITED,
                    TrackState.REJECTED,
                )
            ):
                tracks = [target]

            return (
                tracks,
                (pose.x_enu, pose.y_enu),
                pose.yaw_enu,
                "active_target",
                True,
            )

        explore_states = self._state_set(
            (
                "EXPLORE_ALIGN",
                "EXPLORE_MOVE",
            )
        )
        if self.state in explore_states:
            if pose.speed_xy > self.depth_max_speed:
                return (
                    [],
                    (pose.x_enu, pose.y_enu),
                    pose.yaw_enu,
                    "explore_too_fast",
                    True,
                )

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
                True,
            )

        return [], None, None, "state_not_used", False

    def _ensure_depth_filter(
        self,
        track: Any,
    ) -> KalmanTrack2D:
        tree_id = int(track.tree_id)
        filt = self._kalman_tracks.get(tree_id)

        if filt is None:
            now = time.monotonic()
            filt = KalmanTrack2D(
                x=float(track.x),
                y=float(track.y),
                variance_x=float(
                    self.kalman_initial_variance
                ),
                variance_y=float(
                    self.kalman_initial_variance
                ),
                last_time=now,
            )
            self._kalman_tracks[tree_id] = filt

            update_stamp = float(
                getattr(track, "updated_mono", 0.0)
            )
            if update_stamp > 0.0:
                self._last_track_update[tree_id] = (
                    update_stamp
                )

            self.get_logger().info(
                "DEPTH_KALMAN_INIT_EVERY_CAMERA_V2 "
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
            moving_context,
        ) = self._depth_context_and_anchor()

        # Camera tetap masuk setiap saat, tetapi hanya state yang
        # memiliki anchor transform aman yang boleh diparsing.
        if anchor_xy is None or anchor_yaw is None:
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
                "DEPTH_KALMAN_PARSE_REJECT_V2 "
                f"type={type(exc).__name__} "
                f"error={exc}"
            )
            self._depth_summary("parse_error")
            return

        if context == "scan_collect":
            self.depth_initial_scan_frames += 1

        if not candidates:
            self.depth_rejected += 1
            self._depth_summary("no_candidate")
            return

        self.depth_candidate_frames += 1

        # Initial scan basis tetap mengumpulkan centroid meskipun
        # belum ada track yang dapat diberi Kalman.
        if not tracks:
            self.depth_rejected += 1
            self._depth_summary(
                "no_existing_track_initial_detection"
                if context == "scan_collect"
                else "no_eligible_track"
            )
            return

        possible_pairs = []

        for candidate_index, candidate in enumerate(
            candidates
        ):
            if candidate.score < self.depth_min_score:
                continue

            if (
                moving_context
                and self.depth_require_strong_moving
                and not candidate.strong
            ):
                continue

            measurement_x, measurement_y = (
                self._body_to_map(
                    float(candidate.forward),
                    float(candidate.left),
                    anchor_xy,
                    float(anchor_yaw),
                )
            )

            for track in tracks:
                dx = float(track.x) - anchor_xy[0]
                dy = float(track.y) - anchor_xy[1]

                expected_range = math.hypot(dx, dy)
                expected_bearing = wrap_pi(
                    math.atan2(dy, dx)
                    - float(anchor_yaw)
                )

                association_distance = math.hypot(
                    measurement_x - float(track.x),
                    measurement_y - float(track.y),
                )
                bearing_error = abs(
                    wrap_pi(
                        float(candidate.bearing)
                        - expected_bearing
                    )
                )
                range_residual = abs(
                    float(candidate.range_m)
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

        # Measurement basis yang tertunda diproses dahulu.
        self._apply_kalman_updates_v22()

        used_candidates: Set[int] = set()
        used_tracks: Set[int] = set()
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

            measurement_std = (
                self.depth_measurement_std
                + min(
                    0.18,
                    0.012 * float(candidate.range_m),
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
                measurement_x=float(measurement_x),
                measurement_y=float(measurement_y),
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
                context=context,
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
                innovation=innovation,
                measurement_std=measurement_std,
            )

            if self.depth_log_each_update:
                self.get_logger().info(
                    "DEPTH_KALMAN_UPDATE_EVERY_CAMERA_V2 "
                    f"rx={self.depth_rx} "
                    f"seq={getattr(self, 'cloud_seq', -1)} "
                    f"id={tree_id} "
                    f"context={context} "
                    f"range={candidate.range_m:.2f} "
                    f"score={candidate.score:.1f} "
                    f"strong={int(candidate.strong)} "
                    f"assoc={association_distance:.2f} "
                    f"bearing_err="
                    f"{math.degrees(bearing_error):.1f}deg "
                    f"z=({measurement_x:.2f},"
                    f"{measurement_y:.2f}) "
                    f"prior=({prior_x:.2f},"
                    f"{prior_y:.2f}) "
                    f"K=({filt.gain_x:.3f},"
                    f"{filt.gain_y:.3f}) "
                    f"post=({post_x:.2f},"
                    f"{post_y:.2f}) "
                    f"updates={filt.updates}"
                )

        if accepted_this_frame > 0:
            self.depth_update_frames += 1
            reason = "accepted"
        else:
            self.depth_rejected += 1
            reason = "innovation_reject"

        self._depth_summary(reason)

    def _depth_summary(self, reason: str) -> None:
        now = time.monotonic()
        if (
            now - self.depth_last_summary
            < self.depth_summary_period
        ):
            return

        self.depth_last_summary = now

        self.get_logger().info(
            "DEPTH_KALMAN_MONITOR_EVERY_CAMERA_V2 "
            f"elapsed="
            f"{now - self.depth_started_mono:.1f}s "
            f"received={self.depth_rx} "
            f"parsed={self.depth_parsed} "
            f"candidate_frames={self.depth_candidate_frames} "
            f"initial_scan_frames="
            f"{self.depth_initial_scan_frames} "
            f"update_frames={self.depth_update_frames} "
            f"accepted_measurements="
            f"{self.depth_accepted} "
            f"rejected_or_unassociated="
            f"{self.depth_rejected} "
            f"last_reason={reason} "
            f"tof_received="
            f"{getattr(self, '_tof_kalman_rx', 0)} "
            f"tof_accepted="
            f"{getattr(self, '_tof_kalman_accepted', 0)} "
            f"bypass_active="
            f"{int(getattr(self, '_bypass_plan', None) is not None)}"
        )


class SawitDepthKalman321AntiStuck(
    DepthCameraEveryMessageKalmanAntiStuckMixin,
    SawitRandomKalman321TofEveryUpdate,
):
    """Depth per frame + ToF per pesan + 3-2-1."""

    def __init__(self) -> None:
        super().__init__()

        self.get_logger().info(
            "START DEPTH_KALMAN_321_ANTISTUCK_V2 "
            f"run_id={getattr(self, 'normal_run_id', '')} "
            f"seed={getattr(self, 'normal_random_seed', '')} "
            "depth_every_valid_camera_message=1 "
            "tof_every_valid_message=1 "
            "verification_321=1 "
            "bypass_final_hard_stop_release=1 "
            "actual_used_for_control=0"
        )


class SawitDepthKalmanDirect1MAntiStuck(
    DepthCameraEveryMessageKalmanAntiStuckMixin,
    SawitTofKalmanDirect1M,
):
    """Depth per frame + ToF per pesan + direct sekitar 1 m."""

    def __init__(self) -> None:
        super().__init__()

        self.get_logger().info(
            "START DEPTH_KALMAN_DIRECT1M_ANTISTUCK_V2 "
            f"run_id={getattr(self, 'normal_run_id', '')} "
            f"seed={getattr(self, 'normal_random_seed', '')} "
            "depth_every_valid_camera_message=1 "
            "tof_every_valid_message=1 "
            "verification_321=0 "
            "visit=direct_1m "
            "bypass_final_hard_stop_release=1 "
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
    _spin(SawitDepthKalman321AntiStuck, args)


def main_direct1m(args=None) -> None:
    _spin(SawitDepthKalmanDirect1MAntiStuck, args)


if __name__ == "__main__":
    main_321()
