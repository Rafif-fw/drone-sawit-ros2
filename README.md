# Drone Sawit ROS 2

Simulasi navigasi drone otonom untuk pencarian, pemetaan, pemilihan target, dan kunjungan pohon kelapa sawit menggunakan **ROS 2 Humble**, **PX4 SITL**, dan **Gazebo (GZ)**.

Repository ini merupakan arsip kode dan aset simulasi penelitian:

> **Simulasi Akurasi Jarak dan Kunjungan Drone Pencarian Objek Pohon Kelapa Sawit dengan Metode Pilihan Target Random**

## Fitur utama

- World Gazebo kebun sawit berisi **16 pohon** dalam susunan 4 × 4.
- Model pohon 3D Collada beserta texture.
- Navigasi PX4 Offboard melalui ROS 2.
- Pemrosesan PointCloud untuk mencari kandidat batang.
- Scan stasioner 360° dalam 8 sektor.
- Penggabungan kandidat dan pencegahan ID pohon ganda.
- Pemilihan target menggunakan urutan random yang dikunci setelah scan.
- ToF untuk verifikasi dan keselamatan pendekatan 3 m, 2 m, dan 1 m.
- Penyimpanan posisi/status pohon ke JSON.
- Marker RViz untuk target, rute, drone, landmark, dan perbandingan posisi aktual.

## Struktur repository

```text
drone-sawit-ros2/
├── px4_custom/
│   └── Tools/simulation/gz/
│       ├── models/sawit_sedeng/      # mesh, model.sdf, dan texture sawit
│       └── worlds/kebun_sawit.sdf    # world 16 pohon
├── ros2_ws/
│   └── src/sawit_autonomy/
│       ├── launch/
│       ├── data/
│       ├── sawit_autonomy/           # node Python ROS 2
│       ├── package.xml
│       └── setup.py
├── scripts/                         # monitor dan runner percobaan
├── PANDUAN_PERINTAH.md
└── README.md
```

## Node utama

Node aktif utama:

```bash
ros2 run sawit_autonomy sawit_navigator_fast
```

Entry point tersebut menjalankan node `sawit_navigator_fast_v21h`. Repository menyertakan dua varian percobaan utama: tanpa verifikasi 3–2–1 dan dengan verifikasi 3–2–1.

## Topic utama

### Input

| Topic | Tipe | Fungsi |
|---|---|---|
| `/fmu/out/vehicle_local_position_v1` | `px4_msgs/msg/VehicleLocalPosition` | posisi dan kecepatan lokal PX4 |
| `/fmu/out/vehicle_status_v4` | `px4_msgs/msg/VehicleStatus` | status kendaraan |
| `/camera/points` | `sensor_msgs/msg/PointCloud2` | PointCloud kamera depth |
| `/tof_front` | `sensor_msgs/msg/LaserScan` | jarak ToF depan |

### Output utama

| Topic | Fungsi |
|---|---|
| `/fmu/in/offboard_control_mode` | mode kendali Offboard |
| `/fmu/in/trajectory_setpoint` | setpoint posisi/yaw |
| `/fmu/in/vehicle_command` | perintah arm dan mode |
| `/sawit/tree_markers` | marker pohon terdeteksi |
| `/sawit/navigation_markers` | target dan status navigasi |
| `/sawit/route_marker` | lintasan drone |
| `/sawit/drone_marker` | posisi drone |
| `/sawit/comparison_markers` | garis galat deteksi terhadap posisi aktual |
| `/sawit/debug_pc/*` | PointCloud hasil tahap filtering |

Nama topic dapat diganti melalui parameter ROS 2, misalnya:

```bash
ros2 run sawit_autonomy sawit_navigator_fast --ros-args -p cloud_topic:=/camera_front/points
```

## Aset Gazebo

Model pohon:

```text
px4_custom/Tools/simulation/gz/models/sawit_sedeng/
```

World kebun:

```text
px4_custom/Tools/simulation/gz/worlds/kebun_sawit.sdf
```

World tersebut menempatkan 16 model sawit pada koordinat -12, -4, 4, dan 12 meter pada sumbu X/Y.

## Instalasi singkat

### 1. Clone repository

```bash
git clone https://github.com/Rafif-fw/drone-sawit-ros2.git
cd drone-sawit-ros2
```

### 2. Salin aset ke PX4-Autopilot

```bash
mkdir -p ~/PX4-Autopilot/Tools/simulation/gz/models/sawit_sedeng
rsync -a px4_custom/Tools/simulation/gz/models/sawit_sedeng/ ~/PX4-Autopilot/Tools/simulation/gz/models/sawit_sedeng/

install -m 0644 px4_custom/Tools/simulation/gz/worlds/kebun_sawit.sdf ~/PX4-Autopilot/Tools/simulation/gz/worlds/kebun_sawit.sdf
```

### 3. Salin dan build package ROS 2

```bash
mkdir -p ~/ros2_ws/src
rsync -a ros2_ws/src/sawit_autonomy/ ~/ros2_ws/src/sawit_autonomy/

source /opt/ros/humble/setup.bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select sawit_autonomy
source install/setup.bash
```

Lihat [PANDUAN_PERINTAH.md](PANDUAN_PERINTAH.md) untuk urutan terminal PX4, Micro XRCE-DDS Agent, bridge Gazebo–ROS, navigator, pemeriksaan topic, dan RViz.

## Dependensi utama

- Ubuntu 22.04
- ROS 2 Humble
- PX4-Autopilot SITL
- Gazebo/GZ yang kompatibel dengan versi PX4
- Micro XRCE-DDS Agent
- `px4_msgs`
- `ros_gz_bridge`
- `sensor_msgs_py`
- NumPy
- RTAB-Map ROS (hanya untuk launch SLAM)

## Catatan reproduksi penting

Repository saat ini berisi **world kebun**, **model pohon 3D**, dan **package ROS 2**, tetapi belum berisi modifikasi model drone x500 yang memasang kamera depth/PointCloud serta sensor ToF. Perintah utama dapat langsung digunakan pada komputer penelitian yang sudah memiliki PX4-Autopilot dengan konfigurasi sensor tersebut.

Untuk pemasangan baru, model drone harus menerbitkan topic Gazebo yang kemudian dijembatani menjadi `/camera/points` dan `/tof_front`. Gunakan `gz topic -l` untuk melihat nama topic aktual.

Posisi aktual dari SDF hanya dipakai untuk visualisasi/perbandingan hasil dan tidak dipakai sebagai target atau gate navigasi.

## Data penelitian

Folder `ros2_ws/src/sawit_autonomy/data/` disediakan untuk keluaran CSV dan JSON saat program dijalankan.

## Status

Kode ini merupakan perangkat lunak penelitian/simulasi. Jalankan hanya di SITL sampai seluruh pemeriksaan keselamatan dan integrasi sensor selesai.
