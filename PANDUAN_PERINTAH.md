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

## Percobaan tanpa 3–2–1

```bash
~/drone-sawit-ros2/scripts/monitor_sawit_depth_kalman_time.sh depth_direct_fix_01
```

```bash
~/drone-sawit-ros2/scripts/run_sawit_depth_kalman_direct1m_antistuck.sh depth_direct_fix_01
```

## Percobaan dengan 3–2–1

```bash
~/drone-sawit-ros2/scripts/monitor_sawit_depth_kalman_time.sh depth321_fix_01
```

```bash
~/drone-sawit-ros2/scripts/run_sawit_depth_kalman_321_antistuck.sh depth321_fix_01
```

## Monitor Topik

```bash
rviz2
```

## Menghentikan simulasi

Tekan `Ctrl+C` pada terminal navigator, monitor, bridge, agent, dan PX4. Setelah itu jalankan:

```bash
pkill -f MicroXRCEAgent 2>/dev/null || true
pkill -f parameter_bridge 2>/dev/null || true
pkill -f px4 2>/dev/null || true
pkill -f "gz sim" 2>/dev/null || true
```
