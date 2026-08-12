from glob import glob
import os

from setuptools import find_packages, setup


package_name = "sawit_autonomy"


setup(
    name=package_name,
    version="0.0.1",

    packages=find_packages(
        exclude=[
            "test",
            "test.*",
        ]
    ),

    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [
                os.path.join(
                    "resource",
                    package_name,
                )
            ],
        ),
        (
            os.path.join(
                "share",
                package_name,
            ),
            [
                "package.xml",
            ],
        ),
        (
            os.path.join(
                "share",
                package_name,
                "launch",
            ),
            glob("launch/*.launch.py"),
        ),
    ],

    install_requires=[
        "setuptools",
    ],

    zip_safe=True,

    maintainer="Rafif Fernanda Wibowo",
    maintainer_email="rafifdarkblood4869@gmail.com",

    description=(
        "Oil palm RGB-D SLAM and autonomous "
        "navigation experiments using ROS 2 and PX4 SITL"
    ),

    license="Apache-2.0",

    tests_require=[
        "pytest",
    ],

    entry_points={
        "console_scripts": [
            'sawit_navigator_depth_kalman_direct1m_collision_safe = sawit_autonomy.sawit_navigator_depth_kalman_collision_safe_v3:main_direct1m',
            'sawit_navigator_depth_kalman_321_collision_safe = sawit_autonomy.sawit_navigator_depth_kalman_collision_safe_v3:main_321',
            'sawit_flight_time_speed_monitor = sawit_autonomy.sawit_flight_time_speed_monitor:main',
            'sawit_flight_time_monitor_depth_dual = sawit_autonomy.sawit_flight_time_monitor_depth_dual:main',
            'sawit_navigator_depth_kalman_direct1m_antistuck = sawit_autonomy.sawit_navigator_depth_camera_kalman_dual_antistuck:main_direct1m',
            'sawit_navigator_depth_kalman_321_antistuck = sawit_autonomy.sawit_navigator_depth_camera_kalman_dual_antistuck:main_321',
            'sawit_navigator_depth_camera_kalman_direct_1m = sawit_autonomy.sawit_navigator_depth_camera_kalman_dual:main_direct_1m',
            'sawit_navigator_depth_camera_kalman_321 = sawit_autonomy.sawit_navigator_depth_camera_kalman_dual:main_321',
            'sawit_navigator_random_kalman_321_final = sawit_autonomy.sawit_navigator_random_kalman_321_final:main',
            'sawit_navigator_random_kalman_321_random_bypass = sawit_autonomy.sawit_navigator_random_kalman_321_random_bypass:main',
            'sawit_flight_time_monitor = sawit_autonomy.sawit_flight_time_monitor:main',
            'sawit_navigator_random_kalman_321_tof_every_update = sawit_autonomy.sawit_navigator_random_kalman_321_tof_every_update:main',
            'sawit_flight_time_monitor_v2 = sawit_autonomy.sawit_flight_time_monitor_v2:main',
            'sawit_navigator_tof_kalman_direct_1m = sawit_autonomy.sawit_navigator_tof_kalman_direct_1m:main',
            'sawit_navigator_random_kalman_321 = sawit_autonomy.sawit_navigator_random_kalman_321_v22:main',
        "sawit_navigator = sawit_autonomy.sawit_navigator:main",
            "sawit_navigator_fast = sawit_autonomy.sawit_navigator_fast:main",
            'sawit_random_3layer_navigator = sawit_autonomy.sawit_random_3layer_navigator:main',
            "sawit_navigator_tof_camera_memory = sawit_autonomy.sawit_navigator_tof_camera_memory:main",
        (
            "sawit_random_pointcloud_navigator = "
            "sawit_autonomy.sawit_random_pointcloud_navigator:main"
        ),
        (
            "sawit_random_depth_navigator = "
            "sawit_autonomy.sawit_random_depth_navigator:main"
        ),
        (
            "sawit_slam_random_navigator = "
            "sawit_autonomy.sawit_slam_random_navigator:main"
        ),
                    'sawit_navigator_bfs = sawit_autonomy.sawit_navigator_bfs:main',
                    'sawit_tree_memory_mapper = sawit_autonomy.sawit_tree_memory_mapper:main',
            'sawit_navigator_dfs = sawit_autonomy.sawit_navigator_dfs:main',
],
    },
)
