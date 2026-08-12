# Panduan Menjalankan Simulasi Drone Sawit

Gunakan terminal Ubuntu/WSL yang berbeda untuk setiap proses. Jangan menjalankan dua navigator secara bersamaan.

## Terminal 1 — PX4 SITL dan Gazebo

```bash
pkill -f px4 2>/dev/null || true
pkill -f "gz sim" 2>/dev/null || true

cd ~/PX4-Autopilot

PX4_GZ_WORLD=kebun_sawit \
PX4_GZ_MODEL_POSE="-25,0,0.5,0,0,0" \
make px4_sitl gz_x500
```

Setelah PX4 shell muncul:

```text
param set COM_RC_IN_MODE 4
param set NAV_DLL_ACT 0
param set NAV_RCL_ACT 0
param set CBRK_SUPPLY_CHK 894281
param set COM_ARM_WO_GPS 1
```

## Terminal 2 — Micro XRCE-DDS Agent

```bash
pkill -f MicroXRCEAgent 2>/dev/null || true
MicroXRCEAgent udp4 -p 8888
```

## Terminal 3 — Bridge sensor Gazebo ke ROS 2

```bash
pkill -f parameter_bridge 2>/dev/null || true

source /opt/ros/humble/setup.bash

GZ_IP=127.0.0.1 ros2 run ros_gz_bridge parameter_bridge \
"/camera/image@sensor_msgs/msg/Image[gz.msgs.Image" \
"/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image" \
"/camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked" \
"/tof_front@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan" \
--ros-args -p lazy:=false
```

## Terminal 4 — Build package setelah kode berubah

```bash
source /opt/ros/humble/setup.bash
cd ~/ros2_ws

colcon build --packages-select sawit_autonomy
source ~/ros2_ws/install/setup.bash
```

Build tidak perlu diulang jika kode tidak berubah.

## Pilihan A — Menjalankan base code

Jalankan di terminal navigator:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 run sawit_autonomy sawit_navigator_fast --ros-args \
  -p target_tree_count:=16 \
  -p visual_spawn_x:=-25.0 \
  -p visual_spawn_y:=0.0 \
  -p reset_memory_on_start:=true
```

Monitor base pada terminal terpisah:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 topic echo /rosout \
| grep --line-buffered -E \
"START V21|STATE_V21H|SCAN|TARGET|TREE_VISITED|VISUAL_COMPARE|MISSION_COLLISION_ABORT|Traceback"
```

## Pilihan B — Percobaan tanpa 3–2–1

Terminal monitor:

```bash
~/monitor_sawit_depth_kalman_time.sh depth_direct_fix_01
```

Terminal navigator:

```bash
~/run_sawit_depth_kalman_direct1m_antistuck.sh depth_direct_fix_01
```

## Pilihan C — Percobaan dengan 3–2–1 antistuck

Terminal monitor:

```bash
~/monitor_sawit_depth_kalman_time.sh depth321_fix_01
```

Terminal navigator:

```bash
~/run_sawit_depth_kalman_321_antistuck.sh depth321_fix_01
```

## Menghentikan simulasi

Tekan `Ctrl+C` pada terminal navigator, monitor, bridge, agent, dan PX4. Setelah itu jalankan:

```bash
pkill -f MicroXRCEAgent 2>/dev/null || true
pkill -f parameter_bridge 2>/dev/null || true
pkill -f px4 2>/dev/null || true
pkill -f "gz sim" 2>/dev/null || true
```

