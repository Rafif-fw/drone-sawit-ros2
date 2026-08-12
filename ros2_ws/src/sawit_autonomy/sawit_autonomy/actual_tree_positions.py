# actual_tree_positions.py
# Posisi pohon actual langsung dari kebun_sawit.sdf
# Format Gazebo/SDF world ENU:
#   x = East
#   y = North

ACTUAL_TREE_POSITIONS_GAZEBO_ENU = [
    (-12.0, -12.0), (-12.0, -4.0), (-12.0, 4.0), (-12.0, 12.0),
    (-4.0, -12.0),  (-4.0, -4.0),  (-4.0, 4.0),  (-4.0, 12.0),
    (4.0, -12.0),   (4.0, -4.0),   (4.0, 4.0),   (4.0, 12.0),
    (12.0, -12.0),  (12.0, -4.0),  (12.0, 4.0),  (12.0, 12.0),
]


def get_actual_tree_positions_gazebo(z: float = 0.0):
    return [
        (float(x), float(y), float(z))
        for x, y in ACTUAL_TREE_POSITIONS_GAZEBO_ENU
    ]
