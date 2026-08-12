#!/usr/bin/env python3
"""Monitor waktu, kecepatan, dan penyelesaian misi drone sawit.

Jalankan sebelum navigator. Hasil direkam tepat sekali saat rosout memuat
MISSION_COMPLETE... visited=16/16.
"""

from __future__ import annotations

import csv
import math
import re
import time
from pathlib import Path
from typing import Optional

import rclpy
from px4_msgs.msg import VehicleLocalPosition
from rcl_interfaces.msg import Log
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)


class SawitFlightTimeSpeedMonitor(Node):
    START_PATTERNS = (
        (
            "START DEPTH_KALMAN_321_ANTISTUCK_V2",
            "depth_every_message_kalman_321_antistuck",
            30,
        ),
        (
            "START DEPTH_KALMAN_DIRECT1M_ANTISTUCK_V2",
            "depth_every_message_kalman_direct1m_antistuck",
            31,
        ),
        (
            "START TOF_KALMAN_DIRECT_1M_V1",
            "tof_every_message_kalman_direct1m",
            20,
        ),
        (
            "START TOF_EVERY_MESSAGE_KALMAN_V1",
            "tof_every_message_kalman_321",
            19,
        ),
        ("START V21H", "baseline_v21h", 10),
    )

    COMPLETE_RE = re.compile(
        r"MISSION_COMPLETE(?:_[A-Za-z0-9]+)?\s+visited=(\d+)\s*/\s*(\d+)",
        re.IGNORECASE,
    )
    VISITED_RE = re.compile(
        r"\bvisited=(\d+)(?:\s*/\s*(\d+))?",
        re.IGNORECASE,
    )
    RUN_RE = re.compile(r"\brun_id=([^\s]+)")
    SEED_RE = re.compile(r"\bseed=([^\s]+)")

    def __init__(self) -> None:
        super().__init__("sawit_flight_time_speed_monitor")

        self.declare_parameter("label", "sawit_run")
        self.declare_parameter("target_count", 16)
        self.declare_parameter("airborne_height", 0.35)
        self.declare_parameter("moving_speed_threshold", 0.05)
        self.declare_parameter("speed_noise_floor", 0.02)
        self.declare_parameter("period_sec", 1.0)
        self.declare_parameter(
            "csv_path",
            str(
                Path.home()
                / "ros2_ws/src/sawit_autonomy/data/flight_time_speed_16.csv"
            ),
        )
        self.declare_parameter(
            "latest_result_path",
            str(
                Path.home()
                / "ros2_ws/src/sawit_autonomy/data/flight_time_speed_latest.txt"
            ),
        )

        self.label = str(self.get_parameter("label").value)
        self.target_count = int(self.get_parameter("target_count").value)
        self.airborne_height = float(
            self.get_parameter("airborne_height").value
        )
        self.moving_speed_threshold = float(
            self.get_parameter("moving_speed_threshold").value
        )
        self.speed_noise_floor = float(
            self.get_parameter("speed_noise_floor").value
        )
        self.period_sec = max(
            0.2,
            float(self.get_parameter("period_sec").value),
        )
        self.csv_path = Path(
            str(self.get_parameter("csv_path").value)
        ).expanduser()
        self.latest_result_path = Path(
            str(self.get_parameter("latest_result_path").value)
        ).expanduser()

        self.monitor_start = time.monotonic()
        self.algorithm_start: Optional[float] = None
        self.airborne_start: Optional[float] = None
        self.mission_end: Optional[float] = None
        self.last_velocity_time: Optional[float] = None
        self.home_z: Optional[float] = None

        self.algorithm_name = ""
        self.algorithm_rank = 0
        self.run_id = ""
        self.seed = ""
        self.finished = False
        self.visited_count = 0
        self.reported_target_count = self.target_count

        self.max_altitude = 0.0
        self.current_speed = 0.0
        self.max_speed = 0.0
        self.speed_time_integral = 0.0
        self.speed_integration_time = 0.0
        self.moving_speed_integral = 0.0
        self.moving_time = 0.0
        self.estimated_horizontal_distance = 0.0
        self.speed_samples = 0

        px4_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        rosout_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=500,
        )

        self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self._on_position,
            px4_qos,
        )
        self.create_subscription(
            Log,
            "/rosout",
            self._on_log,
            rosout_qos,
        )
        self.create_timer(self.period_sec, self._periodic)

        self._ensure_output_files()
        self.get_logger().info(
            "FLIGHT_SPEED_TIME_MONITOR_START "
            f"label={self.label} target={self.target_count} "
            f"airborne_height={self.airborne_height:.2f}m "
            f"csv={self.csv_path}"
        )

    def _ensure_output_files(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.latest_result_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with self.csv_path.open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                csv.writer(handle).writerow(
                    [
                        "wall_time",
                        "label",
                        "algorithm_name",
                        "run_id",
                        "seed",
                        "visited",
                        "target_count",
                        "algorithm_duration_sec",
                        "flight_duration_sec",
                        "current_speed_at_complete_mps",
                        "average_speed_all_flight_mps",
                        "average_speed_while_moving_mps",
                        "max_horizontal_speed_mps",
                        "estimated_horizontal_distance_m",
                        "max_altitude_m",
                        "speed_samples",
                        "status",
                        "completed",
                    ]
                )

    @staticmethod
    def _safe_float(value: object) -> Optional[float]:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    def _on_position(self, msg: VehicleLocalPosition) -> None:
        now = time.monotonic()

        z = self._safe_float(msg.z)
        if z is not None:
            if self.home_z is None:
                self.home_z = z
            altitude = float(self.home_z - z)
            self.max_altitude = max(self.max_altitude, altitude)
            if (
                self.airborne_start is None
                and altitude >= self.airborne_height
            ):
                self.airborne_start = now
                self.last_velocity_time = now
                self.get_logger().info(
                    "FLIGHT_TIMER_AIRBORNE_START_SPEED_MONITOR "
                    f"altitude={altitude:.3f}m"
                )

        vx = self._safe_float(msg.vx)
        vy = self._safe_float(msg.vy)
        if vx is None or vy is None:
            self.last_velocity_time = now
            return

        speed = math.hypot(vx, vy)
        if speed < self.speed_noise_floor:
            speed = 0.0

        self.current_speed = speed
        self.max_speed = max(self.max_speed, speed)
        self.speed_samples += 1

        if self.airborne_start is None or self.finished:
            self.last_velocity_time = now
            return

        if self.last_velocity_time is None:
            self.last_velocity_time = now
            return

        dt = now - self.last_velocity_time
        self.last_velocity_time = now
        if dt <= 0.0 or dt > 2.0:
            return

        self.speed_time_integral += speed * dt
        self.speed_integration_time += dt
        self.estimated_horizontal_distance += speed * dt
        if speed >= self.moving_speed_threshold:
            self.moving_speed_integral += speed * dt
            self.moving_time += dt

    def _register_algorithm(
        self, text: str, name: str, rank: int
    ) -> None:
        now = time.monotonic()
        if self.algorithm_start is None:
            self.algorithm_start = now
        if rank >= self.algorithm_rank:
            self.algorithm_name = name
            self.algorithm_rank = rank

        run_match = self.RUN_RE.search(text)
        seed_match = self.SEED_RE.search(text)
        if run_match:
            self.run_id = run_match.group(1)
        if seed_match:
            self.seed = seed_match.group(1)

        self.get_logger().info(
            "ALGORITHM_TIMER_IDENTIFIED_SPEED_MONITOR "
            f"algorithm={self.algorithm_name} "
            f"run_id={self.run_id or '-'} seed={self.seed or '-'}"
        )

    def _on_log(self, msg: Log) -> None:
        if self.finished:
            return
        text = str(msg.msg)

        for token, name, rank in self.START_PATTERNS:
            if token in text:
                self._register_algorithm(text, name, rank)

        visited_match = self.VISITED_RE.search(text)
        if visited_match:
            visited = int(visited_match.group(1))
            total = (
                int(visited_match.group(2))
                if visited_match.group(2)
                else self.target_count
            )
            if 0 <= visited <= max(total, self.target_count):
                self.visited_count = max(self.visited_count, visited)
                self.reported_target_count = total

        complete_match = self.COMPLETE_RE.search(text)
        if complete_match:
            visited = int(complete_match.group(1))
            total = int(complete_match.group(2))
            self.visited_count = visited
            self.reported_target_count = total
            if visited >= self.target_count and total >= self.target_count:
                self._finish("completed", 1)
                return

        if (
            "SIM_COLLISION_ABORT_LATCH_V21N7" in text
            or "MISSION_COLLISION_ABORT_V21N8" in text
        ):
            self._finish("collision_abort", 0)

    def _durations(self, now: float) -> tuple[float, float]:
        algorithm_base = (
            self.algorithm_start
            if self.algorithm_start is not None
            else self.monitor_start
        )
        flight_base = (
            self.airborne_start
            if self.airborne_start is not None
            else algorithm_base
        )
        return now - algorithm_base, now - flight_base

    def _average_speed_all(self) -> float:
        if self.speed_integration_time <= 0.0:
            return 0.0
        return self.speed_time_integral / self.speed_integration_time

    def _average_speed_moving(self) -> float:
        if self.moving_time <= 0.0:
            return 0.0
        return self.moving_speed_integral / self.moving_time

    @staticmethod
    def _hms(seconds: float) -> str:
        seconds = max(0.0, seconds)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    def _periodic(self) -> None:
        if self.finished:
            return
        now = time.monotonic()
        algorithm_elapsed, flight_elapsed = self._durations(now)
        self.get_logger().info(
            "FLIGHT_SPEED_TIME_MONITOR "
            f"label={self.label} "
            f"algorithm={self.algorithm_name or 'waiting'} "
            f"visited={self.visited_count}/{self.target_count} "
            f"current_speed={self.current_speed:.3f}mps "
            f"avg_speed={self._average_speed_all():.3f}mps "
            f"moving_avg={self._average_speed_moving():.3f}mps "
            f"max_speed={self.max_speed:.3f}mps "
            f"distance={self.estimated_horizontal_distance:.2f}m "
            f"algorithm_elapsed={algorithm_elapsed:.1f}s "
            f"flight_elapsed={flight_elapsed:.1f}s "
            f"max_altitude={self.max_altitude:.2f}m "
            "status=running"
        )

    def _finish(self, status: str, completed: int) -> None:
        if self.finished:
            return
        self.finished = True
        self.mission_end = time.monotonic()
        algorithm_duration, flight_duration = self._durations(
            self.mission_end
        )
        avg_speed_all = self._average_speed_all()
        avg_speed_moving = self._average_speed_moving()
        wall_time = time.strftime("%Y-%m-%d %H:%M:%S")

        with self.csv_path.open(
            "a", newline="", encoding="utf-8"
        ) as handle:
            csv.writer(handle).writerow(
                [
                    wall_time,
                    self.label,
                    self.algorithm_name,
                    self.run_id,
                    self.seed,
                    self.visited_count,
                    self.reported_target_count,
                    f"{algorithm_duration:.3f}",
                    f"{flight_duration:.3f}",
                    f"{self.current_speed:.4f}",
                    f"{avg_speed_all:.4f}",
                    f"{avg_speed_moving:.4f}",
                    f"{self.max_speed:.4f}",
                    f"{self.estimated_horizontal_distance:.3f}",
                    f"{self.max_altitude:.3f}",
                    self.speed_samples,
                    status,
                    completed,
                ]
            )

        latest_text = "\n".join(
            [
                "HASIL MONITOR MISI DRONE SAWIT",
                f"Waktu pencatatan       : {wall_time}",
                f"Label                  : {self.label}",
                f"Algoritma              : {self.algorithm_name}",
                f"Run ID                 : {self.run_id}",
                f"Seed                   : {self.seed}",
                f"Kunjungan              : {self.visited_count}/{self.reported_target_count}",
                f"Waktu algoritma        : {algorithm_duration:.3f} s ({self._hms(algorithm_duration)})",
                f"Waktu penerbangan      : {flight_duration:.3f} s ({self._hms(flight_duration)})",
                f"Kecepatan saat selesai : {self.current_speed:.4f} m/s",
                f"Kecepatan rata-rata    : {avg_speed_all:.4f} m/s (termasuk hold/scan)",
                f"Rata-rata saat bergerak: {avg_speed_moving:.4f} m/s",
                f"Kecepatan maksimum     : {self.max_speed:.4f} m/s",
                f"Estimasi jarak mendatar: {self.estimated_horizontal_distance:.3f} m",
                f"Ketinggian maksimum    : {self.max_altitude:.3f} m",
                f"Status                 : {status}",
                f"CSV                    : {self.csv_path}",
                "",
            ]
        )
        self.latest_result_path.write_text(
            latest_text, encoding="utf-8"
        )

        self.get_logger().info(
            "FLIGHT_SPEED_TIME_RESULT "
            f"label={self.label} "
            f"algorithm={self.algorithm_name} "
            f"run_id={self.run_id} seed={self.seed} "
            f"visited={self.visited_count}/{self.reported_target_count} "
            f"algorithm_duration={algorithm_duration:.3f}s "
            f"flight_duration={flight_duration:.3f}s "
            f"current_speed={self.current_speed:.4f}mps "
            f"avg_speed={avg_speed_all:.4f}mps "
            f"moving_avg={avg_speed_moving:.4f}mps "
            f"max_speed={self.max_speed:.4f}mps "
            f"distance={self.estimated_horizontal_distance:.3f}m "
            f"max_altitude={self.max_altitude:.3f}m "
            f"status={status} completed={completed} "
            f"csv={self.csv_path} latest={self.latest_result_path}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SawitFlightTimeSpeedMonitor()
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
