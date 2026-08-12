#!/usr/bin/env bash
set -eo pipefail

RUN_ID="${1:-depth321_previous_01}"

set +u
source /opt/ros/humble/setup.bash
source "$HOME/ros2_ws/install/setup.bash"
set -u

# Kebun world tetap pada X [-15.5, 15.5] dan Y [-15, 15].
# Pose spawn dibaca otomatis sebelum navigator melakukan takeoff, sehingga
# perintah run tetap sama untuk tengah, kiri, dan kanan.
read -r SPAWN_X SPAWN_Y ORCHARD_MIN_X ORCHARD_MAX_X ORCHARD_MIN_Y ORCHARD_MAX_Y SPAWN_SOURCE < <(
python3 - <<'PY'
import os
import re
import subprocess
import sys

WORLD_MIN_X = -15.5
WORLD_MAX_X = 15.5
WORLD_MIN_Y = -15.0
WORLD_MAX_Y = 15.0

def parse_pose_env(value: str):
    parts = value.replace(',', ' ').split()
    if len(parts) >= 2:
        try:
            return float(parts[0]), float(parts[1]), 'PX4_GZ_MODEL_POSE'
        except ValueError:
            pass
    return None

def parse_gz_pose():
    try:
        topics = subprocess.run(
            ['gz', 'topic', '-l'],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=4,
            check=False,
        ).stdout.splitlines()
    except Exception:
        topics = []

    preferred = [t for t in topics if t == '/world/kebun_sawit/pose/info']
    candidates = preferred or [
        t for t in topics
        if t.startswith('/world/') and t.endswith('/pose/info')
    ]

    for topic in candidates:
        try:
            result = subprocess.run(
                ['timeout', '4s', 'gz', 'topic', '-e', '-t', topic],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=6,
                check=False,
            )
            text = result.stdout
        except Exception:
            continue

        match = re.search(
            r'name\s*:\s*"x500_0"(?:(?!name\s*:).){0,1800}?'
            r'position\s*\{\s*x\s*:\s*([-+0-9.eE]+)\s*'
            r'y\s*:\s*([-+0-9.eE]+)',
            text,
            flags=re.S,
        )
        if match:
            x = float(match.group(1))
            y = float(match.group(2))
            # Hilangkan noise fisika sangat kecil saat model masih diam.
            return round(x, 2), round(y, 2), f'gazebo:{topic}'
    return None

pose = parse_pose_env(os.environ.get('PX4_GZ_MODEL_POSE', ''))
if pose is None:
    pose = parse_gz_pose()

if pose is None:
    print(
        'GAGAL GAGAL GAGAL GAGAL GAGAL GAGAL unknown',
    )
    sys.exit(0)

sx, sy, source = pose
print(
    f'{sx:.3f} {sy:.3f} '
    f'{WORLD_MIN_X - sx:.3f} {WORLD_MAX_X - sx:.3f} '
    f'{WORLD_MIN_Y - sy:.3f} {WORLD_MAX_Y - sy:.3f} '
    f'{source}'
)
PY
)

if [[ "$SPAWN_X" == "GAGAL" ]]; then
    echo "GAGAL: pose x500_0 tidak dapat dibaca dari Gazebo." >&2
    echo "Pastikan Gazebo kebun_sawit sudah aktif sebelum navigator." >&2
    echo "Cek: gz topic -e -t /world/kebun_sawit/pose/info" >&2
    exit 1
fi

echo "Mode          : Depth setiap frame + ToF setiap pesan + Kalman 3-2-1"
echo "Visit         : verifikasi bertahap 3 m, 2 m, dan 1 m"
echo "Run ID        : $RUN_ID"
echo "Spawn source  : $SPAWN_SOURCE"
echo "Visual spawn  : ($SPAWN_X, $SPAWN_Y)"
echo "Orchard local : X[$ORCHARD_MIN_X,$ORCHARD_MAX_X] Y[$ORCHARD_MIN_Y,$ORCHARD_MAX_Y]"
echo "Actual/SDF    : visual dan evaluasi saja; bukan gate deteksi"

exec ros2 run sawit_autonomy \
  sawit_navigator_depth_kalman_321_antistuck \
  --ros-args \
  -p normal_run_id:="$RUN_ID" \
  -p visual_spawn_x:="$SPAWN_X" \
  -p visual_spawn_y:="$SPAWN_Y" \
  -p orchard_min_x:="$ORCHARD_MIN_X" \
  -p orchard_max_x:="$ORCHARD_MAX_X" \
  -p orchard_min_y:="$ORCHARD_MIN_Y" \
  -p orchard_max_y:="$ORCHARD_MAX_Y"
