#!/usr/bin/env python3

import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge


class CloudOverlayDebug(Node):
    def __init__(self):
        super().__init__("cloud_overlay_debug")

        self.bridge = CvBridge()
        self.last_img = None

        self.fx = 554.0
        self.fy = 554.0
        self.cx = 320.0
        self.cy = 240.0

        self.create_subscription(Image, "/camera/image", self.image_cb, 10)
        self.create_subscription(CameraInfo, "/camera/camera_info", self.info_cb, 10)
        self.create_subscription(PointCloud2, "/debug/cloud_filtered_trunk", self.cloud_cb, 10)

        self.pub = self.create_publisher(Image, "/debug/cloud_overlay_image", 10)

        self.get_logger().info("Cloud overlay debug ON: /camera/image + /debug/cloud_filtered_trunk -> /debug/cloud_overlay_image")

    def info_cb(self, msg):
        if len(msg.k) >= 9:
            self.fx = float(msg.k[0])
            self.fy = float(msg.k[4])
            self.cx = float(msg.k[2])
            self.cy = float(msg.k[5])

    def image_cb(self, msg):
        try:
            self.last_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"image convert failed: {e}")

    def cloud_cb(self, msg):
        if self.last_img is None:
            return

        img = self.last_img.copy()
        h, w = img.shape[:2]

        pts2d = []

        # PointCloud axis kamu: x=forward/depth, y=left, z=up
        for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            x = float(p[0])  # forward / depth
            y = float(p[1])  # left
            z = float(p[2])  # up

            if x <= 0.2 or x > 40.0:
                continue

            u = int(self.cx - (y * self.fx / x))
            v = int(self.cy - (z * self.fy / x))

            if 0 <= u < w and 0 <= v < h:
                pts2d.append((u, v))
                cv2.circle(img, (u, v), 2, (0, 255, 0), -1)

        if pts2d:
            arr = np.array(pts2d, dtype=np.int32)
            cu = int(np.mean(arr[:, 0]))
            cv = int(np.mean(arr[:, 1]))
            cv2.circle(img, (cu, cv), 8, (0, 0, 255), 2)
            cv2.putText(img, f"filtered cloud pts={len(pts2d)}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)
        else:
            cv2.putText(img, "no filtered cloud projected",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2)

        out = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        out.header = msg.header
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CloudOverlayDebug()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
