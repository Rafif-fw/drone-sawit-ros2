#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    rgb_topic = LaunchConfiguration("rgb_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    camera_frame = LaunchConfiguration("camera_frame")
    database_path = LaunchConfiguration("database_path")

    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                FindPackageShare("rtabmap_launch"),
                "/launch/rtabmap.launch.py",
            ]
        ),
        launch_arguments={
            # Semua node harus mengikuti waktu Gazebo.
            "use_sim_time": use_sim_time,

            # Sensor RGB-D depan.
            "rgb_topic": rgb_topic,
            "depth_topic": depth_topic,
            "camera_info_topic": camera_info_topic,

            # Untuk uji awal masih memakai frame depth.
            "frame_id": camera_frame,

            # RGB-D visual odometry.
            "visual_odometry": "true",
            "icp_odometry": "false",

            # Sinkronisasi Gazebo.
            "approx_sync": "true",
            "approx_sync_max_interval": "0.04",
            "topic_queue_size": "100",
            "sync_queue_size": "100",

            # ros_gz_bridge memakai SensorDataQoS / Best Effort.
            "qos": "2",
            "qos_image": "2",
            "qos_camera_info": "2",
            "qos_odom": "2",

            # Tidak memakai laser.
            "subscribe_scan": "false",
            "subscribe_scan_cloud": "false",
            "subscribe_imu": "false",
            "wait_imu_to_init": "false",

            # Mapping baru.
            "localization": "false",

            # Kurangi beban GUI dahulu.
            "rviz": "false",
            "rtabmap_viz": "true",

            "database_path": database_path,

            "rtabmap_args": (
                "--delete_db_on_start "
                "--Rtabmap/DetectionRate 2 "
                "--RGBD/LinearUpdate 0.10 "
                "--RGBD/AngularUpdate 0.05 "
                "--RGBD/ProximityBySpace false "
                "--Mem/IncrementalMemory true "
                "--Vis/MinInliers 12 "
                "--Vis/CorType 0 "
                "--Vis/FeatureType 6 "
                "--Kp/MaxFeatures 1200"
            ),
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
            ),
            DeclareLaunchArgument(
                "rgb_topic",
                default_value="/camera_front/image",
            ),
            DeclareLaunchArgument(
                "depth_topic",
                default_value="/camera",
            ),
            DeclareLaunchArgument(
                "camera_info_topic",
                default_value="/camera_front/camera_info",
            ),
            DeclareLaunchArgument(
                "camera_frame",
                default_value=(
                    "x500_0/front_sensor_link/"
                    "camera_front_depth"
                ),
            ),
            DeclareLaunchArgument(
                "database_path",
                default_value=(
                    "/home/rafif/.ros/"
                    "sawit_rtabmap.db"
                ),
            ),
            rtabmap_launch,
        ]
    )