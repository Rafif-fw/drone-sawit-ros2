#!/usr/bin/env python3
"""Monitor waktu V2 untuk baseline, ToF-Kalman 3-2-1, dan direct 1 m."""

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


class SawitFlightTimeMonitorV2(Node):
    def __init__(self) -> None:
        super().__init__("sawit_flight_time_monitor_v2")

        self.declare_parameter("label", "comparison")
        self.declare_parameter("target_count", 16)
        self.declare_parameter("airborne_height", 0.35)
        self.declare_parameter(
            "csv_path",
            str(
                Path.home()
                / "ros2_ws/src/sawit_autonomy/data/"
                  "flight_time_comparison.csv"
            ),
        )

        self.label = str(self.get_parameter("label").value)
        self.target_count = int(
            self.get_parameter("target_count").value
        )
        self.airborne_height = float(
            self.get_parameter("airborne_height").value
        )
        self.csv_path = Path(
            str(self.get_parameter("csv_path").value)
        ).expanduser()

        self.monitor_start = time.monotonic()
        self.algorithm_start: Optional[float] = None
        self.airborne_start: Optional[float] = None
        self.mission_end: Optional[float] = None
        self.home_z: Optional[float] = None
        self.max_altitude = 0.0

        self.algorithm_name = ""
        self.algorithm_rank = 0
        self.run_id = ""
        self.seed = ""
        self.completed = False

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
            depth=200,
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
        self.create_timer(1.0, self._periodic)

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if (
            not self.csv_path.exists()
            or self.csv_path.stat().st_size == 0
        ):
            with self.csv_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                csv.writer(handle).writerow([
                    "wall_time",
                    "label",
                    "algorithm_name",
                    "run_id",
                    "seed",
                    "target_count",
                    "algorithm_duration_sec",
                    "flight_duration_sec",
                    "max_altitude_m",
                    "completed",
                ])

        self.get_logger().info(
            f"FLIGHT_TIME_MONITOR_V2_START label={self.label} "
            f"airborne_height={self.airborne_height:.2f}m "
            f"csv={self.csv_path}"
        )

    def _on_position(self, msg: VehicleLocalPosition) -> None:
        z = float(msg.z)
        if not math.isfinite(z):
            return

        if self.home_z is None:
            self.home_z = z

        altitude = float(self.home_z - z)
        self.max_altitude = max(self.max_altitude, altitude)

        if (
            self.airborne_start is None
            and altitude >= self.airborne_height
        ):
            self.airborne_start = time.monotonic()
            self.get_logger().info(
                "FLIGHT_TIMER_AIRBORNE_START "
                f"altitude={altitude:.2f}m"
            )

    def _register_algorithm(
        self,
        text: str,
        name: str,
        rank: int,
    ) -> None:
        now = time.monotonic()
        if self.algorithm_start is None:
            self.algorithm_start = now

        if rank >= self.algorithm_rank:
            self.algorithm_name = name
            self.algorithm_rank = rank

        run_match = re.search(r"run_id=([^\s]+)", text)
        seed_match = re.search(r"seed=([^\s]+)", text)
        if run_match:
            self.run_id = run_match.group(1)
        if seed_match:
            self.seed = seed_match.group(1)

        self.get_logger().info(
            f"ALGORITHM_TIMER_IDENTIFIED_V2 "
            f"algorithm={self.algorithm_name} "
            f"run_id={self.run_id} seed={self.seed}"
        )

    def _on_log(self, msg: Log) -> None:
        text = str(msg.msg)
        now = time.monotonic()

        if "START V22 NORMAL RANDOM KALMAN 3-2-1" in text:
            self._register_algorithm(
                text,
                "pointcloud_kalman_baseline_321",
                1,
            )

        if "START TOF_EVERY_MESSAGE_KALMAN_V1" in text:
            self._register_algorithm(
                text,
                "tof_every_message_kalman_321",
                2,
            )

        if "START TOF_KALMAN_DIRECT_1M_V1" in text:
            self._register_algorithm(
                text,
                "tof_every_message_kalman_direct_1m",
                3,
            )

        complete_token = (
            f"MISSION_COMPLETE_V21H visited="
            f"{self.target_count}/{self.target_count}"
        )
        if not self.completed and complete_token in text:
            self.completed = True
            self.mission_end = now
            self._finish()

    def _periodic(self) -> None:
        if self.completed:
            return

        now = time.monotonic()
        algorithm_elapsed = (
            now - self.algorithm_start
            if self.algorithm_start is not None
            else 0.0
        )
        flight_elapsed = (
            now - self.airborne_start
            if self.airborne_start is not None
            else 0.0
        )
        self.get_logger().info(
            f"FLIGHT_TIME_MONITOR_V2 "
            f"label={self.label} "
            f"algorithm={self.algorithm_name or 'waiting'} "
            f"algorithm_elapsed={algorithm_elapsed:.1f}s "
            f"flight_elapsed={flight_elapsed:.1f}s "
            f"max_altitude={self.max_altitude:.2f}m "
            "completed=0"
        )

    def _finish(self) -> None:
        assert self.mission_end is not None

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
        algorithm_duration = self.mission_end - algorithm_base
        flight_duration = self.mission_end - flight_base

        with self.csv_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as handle:
            csv.writer(handle).writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                self.label,
                self.algorithm_name,
                self.run_id,
                self.seed,
                self.target_count,
                f"{algorithm_duration:.3f}",
                f"{flight_duration:.3f}",
                f"{self.max_altitude:.3f}",
                1,
            ])

        self.get_logger().info(
            "FLIGHT_TIME_RESULT_V2 "
            f"label={self.label} "
            f"algorithm={self.algorithm_name} "
            f"run_id={self.run_id} seed={self.seed} "
            f"algorithm_duration={algorithm_duration:.3f}s "
            f"flight_duration={flight_duration:.3f}s "
            f"max_altitude={self.max_altitude:.3f}m "
            f"csv={self.csv_path}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SawitFlightTimeMonitorV2()
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
