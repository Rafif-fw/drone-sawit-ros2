#!/usr/bin/env python3
"""
Sawit Random Kalman 3-2-1 V22
=============================

Add-on untuk basis sawit_navigator_fast V21N13.

Kebijakan yang dipertahankan dari basis:
- Scan stasioner 360 derajat / 8 sektor.
- Target pertama = CONFIRMED terdekat.
- Target berikutnya diacak satu kali menjadi fixed random queue.
- Setelah visited, rescan 360 derajat dan queue lama dipertahankan.
- Kandidat baru hasil rescan ditunda ke batch berikutnya.
- Ghost target, deduplikasi, dan recovery tetap ditangani basis V21N13.

Tambahan V22:
- Kalman Filter 2D constant-position untuk posisi target X-Y.
- 3 m: berhenti dan verifikasi ulang PointCloud.
- 2 m: safety hold; hanya kondisi aman yang boleh lanjut.
- 1 m: status VISITED.
- Objek belum terverifikasi yang masuk <1.5 m memicu HOLD.
- Watchdog ToF pada kondisi normal; tidak ada fault injection.
- CSV hasil setiap target yang berubah menjadi VISITED.

Catatan:
- Koordinat actual/Gazebo hanya untuk evaluasi CSV, bukan target navigasi.
- Versi ini adalah eksperimen kondisi normal. Delay/drop sengaja belum
  disuntikkan agar baseline normal dapat dikumpulkan terlebih dahulu.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rclpy

from sawit_autonomy import sawit_navigator_fast as base


# ---------------------------------------------------------------------------
# Cari kelas basis aktif dari keluarga V21.
# V21N13 saat ini tetap memakai nama kelas SawitNavigatorV21H.
# ---------------------------------------------------------------------------
_BASE_CLASS = None
for _class_name in (
    "SawitNavigatorV21H1",
    "SawitNavigatorV21N13",
    "SawitNavigatorV21N",
    "SawitNavigatorV21J",
    "SawitNavigatorV21I",
    "SawitNavigatorV21H",
    "SawitNavigatorV21G",
    "SawitNavigatorV21F",
    "SawitNavigatorV21E",
    "SawitNavigatorV21C",
    "SawitNavigatorV21B",
):
    _candidate = getattr(base, _class_name, None)
    if _candidate is not None:
        _BASE_CLASS = _candidate
        break

if _BASE_CLASS is None:
    raise ImportError(
        "Kelas basis V21 tidak ditemukan. Pasang "
        "sawit_navigator_fast V21N13 terlebih dahulu."
    )

NavState = base.NavState
TrackState = base.TrackState


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


@dataclass
class KalmanTrack2D:
    """Kalman Filter 2D untuk target statis dengan state [x, y]."""

    x: float
    y: float
    variance_x: float
    variance_y: float
    last_time: float
    gain_x: float = 1.0
    gain_y: float = 1.0
    updates: int = 1

    def predict(self, now: float, process_variance_per_sec: float) -> None:
        dt = max(0.01, min(5.0, float(now) - float(self.last_time)))
        added = max(0.0, float(process_variance_per_sec)) * dt
        self.variance_x += added
        self.variance_y += added
        self.last_time = float(now)

    def update(
        self,
        measurement_x: float,
        measurement_y: float,
        measurement_variance: float,
    ) -> Tuple[float, float]:
        r = max(1.0e-6, float(measurement_variance))

        self.gain_x = self.variance_x / (self.variance_x + r)
        self.gain_y = self.variance_y / (self.variance_y + r)

        self.x += self.gain_x * (float(measurement_x) - self.x)
        self.y += self.gain_y * (float(measurement_y) - self.y)

        self.variance_x = max(
            1.0e-9,
            (1.0 - self.gain_x) * self.variance_x,
        )
        self.variance_y = max(
            1.0e-9,
            (1.0 - self.gain_y) * self.variance_y,
        )
        self.updates += 1
        return self.x, self.y


class SawitRandomKalman321V22(_BASE_CLASS):
    """Fixed-random target visit + Kalman XY + layered 3-2-1 safety."""

    def __init__(self) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Parameter eksperimen kondisi normal
        # ------------------------------------------------------------------
        self.declare_parameter("normal_run_id", "normal_01")
        self.declare_parameter("normal_random_seed", 101)
        self.declare_parameter("normal_csv_path", str(
            Path.home()
            / "ros2_ws/src/sawit_autonomy/data/"
              "normal_kalman_321_v22.csv"
        ))

        # Kalman: target pohon diasumsikan statis.
        self.declare_parameter("kalman_enabled", True)
        self.declare_parameter("kalman_initial_std", 0.60)
        self.declare_parameter("kalman_process_std_per_sqrt_s", 0.10)
        self.declare_parameter("kalman_measurement_std", 0.45)
        self.declare_parameter("kalman_measurement_gate", 2.20)

        # Aturan 3-2-1 dan keselamatan.
        self.declare_parameter("v22_layer3_verify_distance", 3.00)
        self.declare_parameter("v22_layer2_safety_distance", 2.00)
        self.declare_parameter("v22_layer1_visit_distance", 1.00)
        self.declare_parameter("v22_unverified_collision_guard", 1.50)
        self.declare_parameter("v22_final_hard_stop", 0.72)

        # Watchdog baseline normal. Tidak menyuntikkan delay/drop.
        self.declare_parameter("v22_tof_timeout", 1.00)
        self.declare_parameter("v22_cloud_timeout", 1.50)
        self.declare_parameter("v22_watchdog_enabled", True)

        self.normal_run_id = str(
            self.get_parameter("normal_run_id").value
        )
        self.normal_random_seed = int(
            self.get_parameter("normal_random_seed").value
        )
        self.normal_csv_path = Path(
            str(self.get_parameter("normal_csv_path").value)
        ).expanduser()

        self.kalman_enabled = bool(
            self.get_parameter("kalman_enabled").value
        )
        self.kalman_initial_variance = float(
            self.get_parameter("kalman_initial_std").value
        ) ** 2
        self.kalman_process_variance = float(
            self.get_parameter(
                "kalman_process_std_per_sqrt_s"
            ).value
        ) ** 2
        self.kalman_measurement_variance = float(
            self.get_parameter("kalman_measurement_std").value
        ) ** 2
        self.kalman_measurement_gate = float(
            self.get_parameter("kalman_measurement_gate").value
        )

        self.v22_layer3 = float(
            self.get_parameter("v22_layer3_verify_distance").value
        )
        self.v22_layer2 = float(
            self.get_parameter("v22_layer2_safety_distance").value
        )
        self.v22_layer1 = float(
            self.get_parameter("v22_layer1_visit_distance").value
        )
        self.v22_collision_guard = float(
            self.get_parameter("v22_unverified_collision_guard").value
        )
        self.v22_final_hard_stop = float(
            self.get_parameter("v22_final_hard_stop").value
        )
        self.v22_tof_timeout = float(
            self.get_parameter("v22_tof_timeout").value
        )
        self.v22_cloud_timeout = float(
            self.get_parameter("v22_cloud_timeout").value
        )
        self.v22_watchdog_enabled = bool(
            self.get_parameter("v22_watchdog_enabled").value
        )

        # ------------------------------------------------------------------
        # Terapkan parameter 3-2-1 pada mesin V21N13.
        # ------------------------------------------------------------------
        self.layer3_object_distance = self.v22_layer3
        self.layer2_stop_distance = self.v22_layer2
        self.layer1_visit_distance = self.v22_layer1

        # Ketelitian/stabilitas final.
        self.tof_layer_tolerance = 0.10
        self.tof_layer_max_mad = min(
            float(getattr(self, "tof_layer_max_mad", 0.25)),
            0.25,
        )
        self.tof_layer_hold_time = max(
            float(getattr(self, "tof_layer_hold_time", 0.80)),
            0.80,
        )
        self.tof_final_min_safe = max(
            float(getattr(self, "tof_final_min_safe", 0.65)),
            0.65,
        )

        # Dekat pohon dibuat konservatif.
        if hasattr(self, "tof_stage2_step_v21n3"):
            self.tof_stage2_step_v21n3 = min(
                float(self.tof_stage2_step_v21n3),
                0.12,
            )
        if hasattr(self, "tof_stage1_step_v21n3"):
            self.tof_stage1_step_v21n3 = min(
                float(self.tof_stage1_step_v21n3),
                0.055,
            )
        if hasattr(self, "tof_layer2_near_max_v21n12"):
            self.tof_layer2_near_max_v21n12 = 2.30
        if hasattr(self, "tof_layer1_near_max_v21n12"):
            self.tof_layer1_near_max_v21n12 = 1.22

        # Seed dibuat eksplisit agar tiga run dapat direplikasi.
        random_engine = getattr(self, "random", None)
        if random_engine is not None and hasattr(random_engine, "seed"):
            random_engine.seed(self.normal_random_seed)

        self._kalman_tracks: Dict[int, KalmanTrack2D] = {}
        self._last_track_update: Dict[int, float] = {}
        self._visited_logged: set[int] = {
            int(track.tree_id)
            for track in getattr(self, "tracks", {}).values()
            if track.state == TrackState.VISITED
        }
        self._last_watchdog_reason = ""
        self._actual_visual_cache: Optional[
            List[Tuple[float, float]]
        ] = None

        self._ensure_csv_header()

        self.get_logger().info(
            "START V22 NORMAL RANDOM KALMAN 3-2-1 "
            f"run_id={self.normal_run_id} "
            f"seed={self.normal_random_seed} "
            f"layer3={self.v22_layer3:.2f}m "
            f"layer2={self.v22_layer2:.2f}m "
            f"layer1={self.v22_layer1:.2f}m "
            f"unverified_guard={self.v22_collision_guard:.2f}m "
            "fault_injection=0"
        )
        self.get_logger().info(
            "V22 POLICY first=nearest remaining=fixed_random "
            "3m=pointcloud_reverify 2m=safety_gate "
            "safe_then_1m=visited unsafe=hold"
        )
        self.get_logger().info(
            "V22 KALMAN model=constant_position state=[x,y] "
            f"initial_std={math.sqrt(self.kalman_initial_variance):.2f} "
            f"measurement_std="
            f"{math.sqrt(self.kalman_measurement_variance):.2f} "
            f"gate={self.kalman_measurement_gate:.2f}m"
        )
        self.get_logger().info(
            f"V22 CSV path={self.normal_csv_path}"
        )

    # ======================================================================
    # Kalman Filter
    # ======================================================================

    def _apply_kalman_updates_v22(self) -> None:
        if not self.kalman_enabled:
            return

        now = time.monotonic()
        tracks = getattr(self, "tracks", {})

        # Hapus filter untuk ID yang sudah tidak ada.
        valid_ids = {int(tree_id) for tree_id in tracks.keys()}
        for tree_id in list(self._kalman_tracks):
            if tree_id not in valid_ids:
                self._kalman_tracks.pop(tree_id, None)
                self._last_track_update.pop(tree_id, None)

        for tree_id_raw, track in tracks.items():
            tree_id = int(tree_id_raw)

            # VISITED immutable; REJECTED tidak perlu difilter.
            if track.state in (TrackState.VISITED, TrackState.REJECTED):
                continue

            measurement_x = float(track.x)
            measurement_y = float(track.y)
            if not (_finite(measurement_x) and _finite(measurement_y)):
                continue

            update_stamp = float(
                getattr(track, "updated_mono", 0.0)
            )
            if update_stamp <= 0.0:
                # Fallback untuk versi basis yang tidak mengisi updated_mono.
                update_stamp = float(
                    getattr(track, "created_mono", now)
                )

            previous_stamp = self._last_track_update.get(
                tree_id,
                -math.inf,
            )
            if update_stamp <= previous_stamp + 1.0e-9:
                continue
            self._last_track_update[tree_id] = update_stamp

            filt = self._kalman_tracks.get(tree_id)
            if filt is None:
                self._kalman_tracks[tree_id] = KalmanTrack2D(
                    x=measurement_x,
                    y=measurement_y,
                    variance_x=self.kalman_initial_variance,
                    variance_y=self.kalman_initial_variance,
                    last_time=now,
                )
                self.get_logger().info(
                    f"KALMAN_INIT_V22 id={tree_id} "
                    f"xy=({measurement_x:.2f},{measurement_y:.2f})"
                )
                continue

            prior_x = float(filt.x)
            prior_y = float(filt.y)
            innovation_distance = math.hypot(
                measurement_x - prior_x,
                measurement_y - prior_y,
            )

            # Measurement gate mencegah satu centroid salah menyeret target.
            if innovation_distance > self.kalman_measurement_gate:
                track.x = prior_x
                track.y = prior_y
                self.get_logger().warning(
                    f"KALMAN_GATE_REJECT_V22 id={tree_id} "
                    f"measurement=({measurement_x:.2f},"
                    f"{measurement_y:.2f}) "
                    f"prior=({prior_x:.2f},{prior_y:.2f}) "
                    f"innovation={innovation_distance:.2f}m "
                    f"limit={self.kalman_measurement_gate:.2f}m"
                )
                continue

            filt.predict(
                now=now,
                process_variance_per_sec=self.kalman_process_variance,
            )
            filtered_x, filtered_y = filt.update(
                measurement_x=measurement_x,
                measurement_y=measurement_y,
                measurement_variance=self.kalman_measurement_variance,
            )

            track.x = float(filtered_x)
            track.y = float(filtered_y)

            self.get_logger().info(
                f"KALMAN_UPDATE_V22 id={tree_id} "
                f"z=({measurement_x:.2f},{measurement_y:.2f}) "
                f"prior=({prior_x:.2f},{prior_y:.2f}) "
                f"K=({filt.gain_x:.3f},{filt.gain_y:.3f}) "
                f"post=({filtered_x:.2f},{filtered_y:.2f}) "
                f"updates={filt.updates}"
            )

    # ======================================================================
    # ToF robust front guard
    # ======================================================================

    def _front_cluster_distance_v22(self) -> float:
        ranges = getattr(self, "tof_ranges", None)
        if ranges is None:
            return math.inf

        arr = np.asarray(ranges, dtype=np.float64)
        if arr.size == 0:
            return math.inf

        angle_min = float(getattr(self, "tof_angle_min", 0.0))
        angle_inc = float(
            getattr(self, "tof_angle_increment", 0.0)
        )
        range_min = float(getattr(self, "tof_range_min", 0.0))
        range_max = float(getattr(self, "tof_range_max", math.inf))
        half_window = math.radians(30.0)

        selected: List[Tuple[int, float]] = []
        for index, distance in enumerate(arr):
            if not math.isfinite(float(distance)):
                continue
            angle = angle_min + float(index) * angle_inc
            if abs(angle) > half_window:
                continue
            if float(distance) < max(0.05, range_min):
                continue
            if math.isfinite(range_max) and float(distance) > range_max:
                continue
            selected.append((index, float(distance)))

        if not selected:
            return math.inf

        # Cluster berdasarkan ray berurutan dan kemiripan jarak.
        clusters: List[List[float]] = []
        current: List[float] = []
        previous_index: Optional[int] = None
        previous_distance: Optional[float] = None

        for index, distance in selected:
            continuous = (
                previous_index is not None
                and index == previous_index + 1
                and previous_distance is not None
                and abs(distance - previous_distance) <= 0.40
            )
            if not continuous and current:
                clusters.append(current)
                current = []
            current.append(distance)
            previous_index = index
            previous_distance = distance

        if current:
            clusters.append(current)

        robust = [
            float(np.median(cluster))
            for cluster in clusters
            if len(cluster) >= 3
        ]
        if robust:
            return min(robust)

        # Fallback bila resolusi ToF sangat kecil.
        values = np.asarray(
            [distance for _, distance in selected],
            dtype=np.float64,
        )
        if values.size >= 3:
            values.sort()
            return float(np.median(values[: min(5, values.size)]))
        return float(np.min(values))

    def _movement_state_v22(self) -> bool:
        moving_names = (
            "ALIGN_TARGET",
            "APPROACH",
            "AVOID_OBSTACLE",
            "RETRY_VERIFY",
            "REACQUIRE_FINAL",
            "EXPLORE_ALIGN",
            "EXPLORE_MOVE",
            # Fase ini tidak bergerak, tetapi tetap membutuhkan data sensor
            # segar agar verifikasi tidak memakai frame lama.
            "CLOSE_SETTLE",
            "CLOSE_FLUSH",
            "CLOSE_COLLECT",
        )
        moving_states = {
            getattr(NavState, name, None)
            for name in moving_names
        }
        moving_states.discard(None)
        return getattr(self, "state", None) in moving_states

    def _hold_current_v22(self, reason: str) -> None:
        pose = getattr(self, "pose", None)
        if pose is None:
            return

        publish_offboard = getattr(self, "_publish_offboard_mode", None)
        if callable(publish_offboard):
            publish_offboard()

        self._publish_position_enu(
            float(pose.x_enu),
            float(pose.y_enu),
            float(self.flight_altitude),
            float(pose.yaw_enu),
        )
        self.motion_waypoint_xy = None
        self.motion_goal_xy = None
        self.motion_brake_anchor_xy = None

        if reason != self._last_watchdog_reason:
            self.get_logger().warning(
                f"V22_SAFETY_HOLD reason={reason} "
                f"state={getattr(self.state, 'value', self.state)} "
                f"stage={getattr(self, 'tof_approach_stage', 'n/a')}"
            )
            self._last_watchdog_reason = reason

    def _safety_hold_reason_v22(self) -> Optional[str]:
        if not self._movement_state_v22():
            self._last_watchdog_reason = ""
            return None

        now = time.monotonic()
        stage = str(
            getattr(self, "tof_approach_stage", "TO_3M")
        )

        if self.v22_watchdog_enabled:
            tof_receipt = float(
                getattr(self, "tof_receipt_mono", 0.0)
            )
            tof_age = (
                now - tof_receipt
                if tof_receipt > 0.0
                else math.inf
            )
            if tof_age > self.v22_tof_timeout:
                return (
                    f"tof_timeout age={tof_age:.2f}s "
                    f"limit={self.v22_tof_timeout:.2f}s"
                )

            # PointCloud wajib segar pada fase verifikasi dekat.
            close_states = {
                getattr(NavState, "CLOSE_SETTLE", None),
                getattr(NavState, "CLOSE_FLUSH", None),
                getattr(NavState, "CLOSE_COLLECT", None),
            }
            close_states.discard(None)
            if getattr(self, "state", None) in close_states:
                cloud_receipt = float(
                    getattr(self, "last_cloud_receipt_mono", 0.0)
                )
                cloud_age = (
                    now - cloud_receipt
                    if cloud_receipt > 0.0
                    else math.inf
                )
                if cloud_age > self.v22_cloud_timeout:
                    return (
                        f"cloud_timeout age={cloud_age:.2f}s "
                        f"limit={self.v22_cloud_timeout:.2f}s"
                    )

        front_distance = self._front_cluster_distance_v22()

        # Sebelum lolos safety gate 2 m, objek <1.5 m dianggap tidak aman.
        if (
            stage in ("TO_3M", "TO_2M")
            and math.isfinite(front_distance)
            and front_distance < self.v22_collision_guard
        ):
            return (
                f"unverified_object_too_close "
                f"front={front_distance:.2f}m "
                f"guard={self.v22_collision_guard:.2f}m"
            )

        # Setelah safety gate berhasil, basis boleh menyelesaikan visited 1 m.
        # Hanya jarak yang benar-benar terlalu dekat yang dipaksa HOLD.
        if (
            stage in ("TO_1M", "HOLD_1M")
            and math.isfinite(front_distance)
            and front_distance < self.v22_final_hard_stop
        ):
            return (
                f"final_hard_stop front={front_distance:.2f}m "
                f"limit={self.v22_final_hard_stop:.2f}m"
            )

        self._last_watchdog_reason = ""
        return None

    # ======================================================================
    # CSV hasil kunjungan
    # ======================================================================

    @staticmethod
    def _csv_fields() -> List[str]:
        return [
            "timestamp_ros_sec",
            "run_id",
            "random_seed",
            "scan_generation",
            "queue_position",
            "target_id",
            "target_state",
            "verification_stage",
            "drone_x_enu",
            "drone_y_enu",
            "drone_altitude",
            "estimate_x_enu",
            "estimate_y_enu",
            "estimate_x_visual",
            "estimate_y_visual",
            "actual_x_visual",
            "actual_y_visual",
            "tof_distance",
            "actual_visit_distance",
            "position_error",
            "visited_correct",
            "kalman_gain_x",
            "kalman_gain_y",
            "kalman_updates",
        ]

    def _ensure_csv_header(self) -> None:
        self.normal_csv_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        if (
            not self.normal_csv_path.exists()
            or self.normal_csv_path.stat().st_size == 0
        ):
            with self.normal_csv_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=self._csv_fields(),
                )
                writer.writeheader()

    @staticmethod
    def _parse_actual_points(
        raw: object,
    ) -> List[Tuple[float, float]]:
        points: List[Tuple[float, float]] = []

        if raw is None:
            return points

        if isinstance(raw, dict):
            items: Iterable[object] = raw.values()
        elif isinstance(raw, (list, tuple)):
            items = raw
        else:
            return points

        for item in items:
            x: Optional[float] = None
            y: Optional[float] = None

            if isinstance(item, dict):
                for x_key in ("x", "world_x", "map_x"):
                    if x_key in item and _finite(item[x_key]):
                        x = float(item[x_key])
                        break
                for y_key in ("y", "world_y", "map_y"):
                    if y_key in item and _finite(item[y_key]):
                        y = float(item[y_key])
                        break
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                if _finite(item[0]) and _finite(item[1]):
                    x = float(item[0])
                    y = float(item[1])
            else:
                if hasattr(item, "x") and hasattr(item, "y"):
                    if _finite(item.x) and _finite(item.y):
                        x = float(item.x)
                        y = float(item.y)

            if x is not None and y is not None:
                points.append((x, y))

        return points

    def _actual_visual_points_v22(
        self,
    ) -> List[Tuple[float, float]]:
        if self._actual_visual_cache is not None:
            return self._actual_visual_cache

        raw = None
        for attribute_name in (
            "actual_tree_positions",
            "actual_positions",
            "actual_trees",
        ):
            candidate = getattr(self, attribute_name, None)
            if candidate:
                raw = candidate
                break

        if raw is None:
            function = getattr(
                base,
                "get_actual_tree_positions_gazebo",
                None,
            )
            if callable(function):
                try:
                    raw = function()
                except Exception as exc:
                    self.get_logger().warning(
                        f"ACTUAL_LOAD_FAIL_V22 error={exc}"
                    )

        self._actual_visual_cache = self._parse_actual_points(raw)
        return self._actual_visual_cache

    def _to_visual_v22(
        self,
        x_enu: float,
        y_enu: float,
    ) -> Tuple[float, float]:
        converter = getattr(self, "_local_to_visual", None)
        if callable(converter):
            try:
                vx, vy = converter(float(x_enu), float(y_enu))
                return float(vx), float(vy)
            except Exception:
                pass
        return float(x_enu), float(y_enu)

    def _queue_position_v22(self, tree_id: int) -> int:
        snapshot: Sequence[int] = getattr(
            self,
            "random_batch_snapshot_v21n6",
            [],
        )
        try:
            return list(snapshot).index(int(tree_id)) + 1
        except ValueError:
            return 0

    def _latest_tof_v22(self) -> float:
        value = float(
            getattr(self, "tof_last_valid_distance", math.inf)
        )
        if math.isfinite(value):
            return value
        return self._front_cluster_distance_v22()

    def _log_visit_v22(self, track: object) -> None:
        pose = getattr(self, "pose", None)
        if pose is None:
            return

        tree_id = int(track.tree_id)
        estimate_x = float(track.x)
        estimate_y = float(track.y)
        estimate_visual = self._to_visual_v22(
            estimate_x,
            estimate_y,
        )
        drone_visual = self._to_visual_v22(
            float(pose.x_enu),
            float(pose.y_enu),
        )

        actual_x = math.nan
        actual_y = math.nan
        position_error = math.nan
        actual_visit_distance = math.nan

        actual_points = self._actual_visual_points_v22()
        if actual_points:
            actual_x, actual_y = min(
                actual_points,
                key=lambda point: math.hypot(
                    point[0] - estimate_visual[0],
                    point[1] - estimate_visual[1],
                ),
            )
            position_error = math.hypot(
                estimate_visual[0] - actual_x,
                estimate_visual[1] - actual_y,
            )
            actual_visit_distance = math.hypot(
                drone_visual[0] - actual_x,
                drone_visual[1] - actual_y,
            )

        filt = self._kalman_tracks.get(tree_id)
        gain_x = filt.gain_x if filt is not None else math.nan
        gain_y = filt.gain_y if filt is not None else math.nan
        updates = filt.updates if filt is not None else 0

        tof_distance = self._latest_tof_v22()
        visited_correct = (
            int(
                math.isfinite(actual_visit_distance)
                and actual_visit_distance <= 1.50
            )
            if math.isfinite(actual_visit_distance)
            else ""
        )

        ros_seconds = (
            self.get_clock().now().nanoseconds / 1.0e9
        )

        row = {
            "timestamp_ros_sec": f"{ros_seconds:.6f}",
            "run_id": self.normal_run_id,
            "random_seed": self.normal_random_seed,
            "scan_generation": int(
                getattr(self, "scan_generation", 0)
            ),
            "queue_position": self._queue_position_v22(tree_id),
            "target_id": tree_id,
            "target_state": str(track.state.value),
            "verification_stage": str(
                getattr(self, "tof_approach_stage", "")
            ),
            "drone_x_enu": f"{float(pose.x_enu):.6f}",
            "drone_y_enu": f"{float(pose.y_enu):.6f}",
            "drone_altitude": f"{float(pose.altitude):.6f}",
            "estimate_x_enu": f"{estimate_x:.6f}",
            "estimate_y_enu": f"{estimate_y:.6f}",
            "estimate_x_visual": f"{estimate_visual[0]:.6f}",
            "estimate_y_visual": f"{estimate_visual[1]:.6f}",
            "actual_x_visual": (
                f"{actual_x:.6f}"
                if math.isfinite(actual_x)
                else ""
            ),
            "actual_y_visual": (
                f"{actual_y:.6f}"
                if math.isfinite(actual_y)
                else ""
            ),
            "tof_distance": (
                f"{tof_distance:.6f}"
                if math.isfinite(tof_distance)
                else ""
            ),
            "actual_visit_distance": (
                f"{actual_visit_distance:.6f}"
                if math.isfinite(actual_visit_distance)
                else ""
            ),
            "position_error": (
                f"{position_error:.6f}"
                if math.isfinite(position_error)
                else ""
            ),
            "visited_correct": visited_correct,
            "kalman_gain_x": (
                f"{gain_x:.6f}"
                if math.isfinite(gain_x)
                else ""
            ),
            "kalman_gain_y": (
                f"{gain_y:.6f}"
                if math.isfinite(gain_y)
                else ""
            ),
            "kalman_updates": updates,
        }

        with self.normal_csv_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=self._csv_fields(),
            )
            writer.writerow(row)

        self.get_logger().info(
            f"NORMAL_VISIT_RECORDED_V22 "
            f"run={self.normal_run_id} id={tree_id} "
            f"queue={row['queue_position']} "
            f"tof={tof_distance:.2f} "
            f"actual_visit={actual_visit_distance:.2f} "
            f"position_error={position_error:.2f} "
            f"csv={self.normal_csv_path}"
        )

    def _capture_new_visits_v22(self) -> None:
        for track in getattr(self, "tracks", {}).values():
            tree_id = int(track.tree_id)
            if (
                track.state == TrackState.VISITED
                and tree_id not in self._visited_logged
            ):
                self._visited_logged.add(tree_id)
                self._log_visit_v22(track)

    # ======================================================================
    # Control loop
    # ======================================================================

    def _control_loop(self) -> None:
        # Filter hanya bekerja ketika basis menghasilkan measurement baru.
        self._apply_kalman_updates_v22()

        reason = self._safety_hold_reason_v22()
        if reason is not None:
            self._hold_current_v22(reason)
            return

        # Seluruh scan, queue random, 3m/2m/1m, rescan, dedupe,
        # ghost recovery, dan PX4 control tetap milik basis V21N13.
        super()._control_loop()

        # Deteksi transisi target menjadi VISITED untuk pelaporan.
        self._capture_new_visits_v22()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SawitRandomKalman321V22()
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
