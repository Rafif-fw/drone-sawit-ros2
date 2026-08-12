#!/usr/bin/env bash
set -eo pipefail

LABEL="${1:-depth_kalman_comparison_01}"

set +u
source /opt/ros/humble/setup.bash
source "$HOME/ros2_ws/install/setup.bash"
set -u

exec ros2 run sawit_autonomy \
  sawit_flight_time_monitor_depth_dual \
  --ros-args \
  -p label:="$LABEL" \
  -p target_count:=16 \
  -p airborne_height:=0.35 \
  -p csv_path:="$HOME/ros2_ws/src/sawit_autonomy/data/flight_time_depth_kalman_comparison.csv"
