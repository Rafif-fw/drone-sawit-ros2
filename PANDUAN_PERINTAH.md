# Panduan Perintah Menjalankan Simulasi

Panduan ini memakai lima terminal Ubuntu/WSL. Jalankan setiap bagian pada terminal yang berbeda dan biarkan prosesnya tetap hidup.

## 0. Persiapan satu kali

Clone dan masuk ke repository:

```bash
cd ~
git clone https://github.com/Rafif-fw/drone-sawit-ros2.git
cd ~/drone-sawit-ros2
```

Salin model sawit dan world ke PX4:

```bash
mkdir -p ~/PX4-Autopilot/Tools/simulation/gz/models/sawit_sedeng

rsync -a \
  ~/drone-sawit-ros2/px4_custom/Tools/simulation/gz/models/sawit_sedeng/ \
  ~/PX4-Autopilot/Tools/simulation/gz/models/sawit_sedeng/

install -m 0644 \
  ~/drone-sawit-ros2/px4_custom/Tools/simulation/gz/worlds/kebun_sawit.sdf \
  ~/PX4-Autopilot/Tools/simulation/gz/worlds/kebun_sawit.sdf
```

Salin package ROS 2 dan build:

```bash
mkdir -p ~/ros2_ws/src

rsync -a \
  ~/drone-sawit-ros2/ros2_ws/src/sawit_autonomy/ \
  ~/ros2_ws/src/sawit_autonomy/

source /opt/ros/humble/setup.bash
cd ~/ros2_ws

rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select sawit_autonomy
source install/setup.bash
```

Pastikan `px4_msgs` tersedia pada workspace atau instalasi ROS 2 yang digunakan.

## 1. Membersihkan proses simulasi lama

Jalankan sebelum memulai ulang simulasi:

```bash
pkill -f MicroXRCEAgent 2>/dev/null || true
pkill -f parameter_bridge 2>/dev/null || true
pkill -f px4 2>/dev/null || true
pkill -f "gz sim" 2>/dev/null || true
```

## 2. Terminal 1 — PX4 SITL dan world kebun sawit

```bash
cd ~/PX4-Autopilot

PX4_GZ_WORLD=kebun_sawit \
PX4_GZ_MODEL_POSE="-25,0,0.5,0,0,0" \
make px4_sitl gz_x500
```

Setelah PX4 shell muncul, parameter berikut dapat diterapkan sesuai versi PX4:

```text
param set COM_RC_IN_MODE 4
param set NAV_DLL_ACT 0
param set NAV_RCL_ACT 0
param set CBRK_SUPPLY_CHK 894281
param set COM_ARM_WO_GPS 1
```

Jika suatu parameter tidak tersedia pada versi PX4 yang dipakai, periksa nama penggantinya dengan `param show NAMA_PARAMETER`.

## 3. Terminal 2 — Micro XRCE-DDS Agent

```bash
MicroXRCEAgent udp4 -p 8888
```

Biarkan terminal ini tetap berjalan.

## 4. Terminal 3 — Bridge sensor Gazebo ke ROS 2

```bash
source /opt/ros/humble/setup.bash

GZ_IP=127.0.0.1 ros2 run ros_gz_bridge parameter_bridge \
"/camera/image@sensor_msgs/msg/Image[gz.msgs.Image" \
"/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image" \
"/camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked" \
"/tof_front@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan" \
--ros-args -p lazy:=false
```

Nama topic harus sama dengan topic yang diterbitkan model sensor Gazebo. Periksa dengan:

```bash
gz topic -l | grep -E 'camera|points|tof|scan'
```

Jika simulator menggunakan awalan `/camera_front`, ganti nama topic bridge dan jalankan navigator dengan parameter `cloud_topic:=/camera_front/points`.

## 5. Memeriksa koneksi sebelum terbang

Buka terminal baru:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 topic list | grep -E 'fmu|camera|points|tof_front'
```

Periksa data PX4:

```bash
ros2 topic echo /fmu/out/vehicle_local_position_v1 --once
ros2 topic echo /fmu/out/vehicle_status_v4 --once
```

Periksa frekuensi sensor:

```bash
ros2 topic hz /camera/points
```

Hentikan pemeriksaan frekuensi dengan `Ctrl+C`, kemudian periksa ToF:

```bash
ros2 topic echo /tof_front --once
```

Jangan menjalankan navigator sebelum topic posisi PX4, status kendaraan, PointCloud, dan ToF tersedia.

## 6. Terminal 4 — Menjalankan navigator utama

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 run sawit_autonomy sawit_navigator_fast --ros-args \
-p target_tree_count:=16 \
-p visual_spawn_x:=-25.0 \
-p visual_spawn_y:=0.0 \
-p reset_memory_on_start:=true
```

Node menggunakan PointCloud `/camera/points` dan ToF `/tof_front` secara default.

Untuk topic PointCloud lain:

```bash
ros2 run sawit_autonomy sawit_navigator_fast --ros-args \
-p cloud_topic:=/camera_front/points \
-p tof_topic:=/tof_front
```

Mode `reset_memory_on_start:=true` memulai pemetaan baru. Gunakan `false` hanya ketika ingin memuat kembali memory JSON yang sudah ada.

## 7. Terminal 5 — Monitor log penting

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 topic echo /rosout \
| grep --line-buffered -E \
"START V21|STATE_V21H|SCAN|TARGET|TREE_VISITED|VISUAL_COMPARE|MISSION_COLLISION_ABORT|Traceback"
```

## 8. RViz

Jalankan:

```bash
rviz2
```

Atur **Fixed Frame** menjadi `map`. Tambahkan display untuk:

- `/sawit/tree_markers`
- `/sawit/actual_tree_markers`
- `/sawit/navigation_markers`
- `/sawit/route_marker`
- `/sawit/drone_marker`
- `/sawit/trunk_models`
- `/sawit/comparison_markers`
- `/sawit/debug_pc/stationary_raw`
- `/sawit/debug_pc/accepted`
- `/sawit/debug_pc/rejected`
- `/sawit/debug_pc/landmark_memory`

## 9. Menjalankan RTAB-Map (opsional)

Pastikan topic RGB, depth, dan camera info tersedia. Kemudian:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 launch sawit_autonomy sawit_slam.launch.py \
rgb_topic:=/camera_front/image \
depth_topic:=/camera_front/depth_image \
camera_info_topic:=/camera_front/camera_info
```

Sesuaikan ketiga topic tersebut dengan keluaran aktual dari `ros2 topic list`.

## 10. Menghentikan simulasi

Tekan `Ctrl+C` pada navigator, bridge, agent, dan PX4. Jika masih ada proses tertinggal:

```bash
pkill -f MicroXRCEAgent 2>/dev/null || true
pkill -f parameter_bridge 2>/dev/null || true
pkill -f px4 2>/dev/null || true
pkill -f "gz sim" 2>/dev/null || true
```

## 11. Troubleshooting cepat

### PX4 topic tidak muncul

- Pastikan Micro XRCE-DDS Agent berjalan pada UDP port 8888.
- Pastikan versi `px4_msgs` kompatibel dengan PX4.
- Jalankan `ros2 topic list | grep fmu`.

### PointCloud tidak muncul

- Jalankan `gz topic -l | grep -E 'camera|points'`.
- Pastikan model x500 yang dipakai memang memiliki depth camera.
- Samakan nama topic pada `parameter_bridge`.
- Periksa dengan `ros2 topic info /camera/points -v`.

### ToF tidak muncul

- Jalankan `gz topic -l | grep -E 'tof|scan'`.
- Pastikan sensor depan menerbitkan pesan LaserScan.
- Samakan topic bridge dengan parameter `tof_topic`.

### Drone tidak arm atau tidak masuk Offboard

- Pastikan topic posisi dan status PX4 sudah diterima.
- Pastikan bridge sensor tidak menyebabkan beban berlebihan.
- Periksa pesan pada PX4 shell dan `/rosout`.
- Jangan mengubah parameter keselamatan tanpa memahami dampaknya.

## Catatan konfigurasi sensor x500

Repository ini belum menyimpan model x500 khusus yang memasang kamera depth dan ToF. Pada komputer penelitian, jalankan dengan folder PX4-Autopilot yang selama ini sudah dimodifikasi. Untuk komputer baru, salin juga konfigurasi model sensor x500 sebelum menjalankan bridge.
