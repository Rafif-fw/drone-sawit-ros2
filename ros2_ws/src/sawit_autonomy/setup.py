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
            # Navigator utama/base
            "sawit_navigator_fast = sawit_autonomy.sawit_navigator_fast:main",

            # Percobaan tanpa metode 3-2-1
            "sawit_navigator_depth_kalman_direct1m_antistuck = sawit_autonomy.sawit_navigator_depth_camera_kalman_dual_antistuck:main_direct1m",

            # Percobaan dengan metode 3-2-1
            "sawit_navigator_depth_kalman_321_antistuck = sawit_autonomy.sawit_navigator_depth_camera_kalman_dual_antistuck:main_321",
            "sawit_navigator_depth_kalman_321_collision_safe = sawit_autonomy.sawit_navigator_depth_kalman_collision_safe_v3:main_321",

            # Monitor percobaan
            "sawit_flight_time_speed_monitor = sawit_autonomy.sawit_flight_time_speed_monitor:main",
            "sawit_flight_time_monitor_depth_dual = sawit_autonomy.sawit_flight_time_monitor_depth_dual:main",
        ],
    },
)
