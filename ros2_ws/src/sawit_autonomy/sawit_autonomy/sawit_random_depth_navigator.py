#!/usr/bin/env python3
import math
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

from sawit_autonomy.navigation_common import ReactiveNavigatorBase


class RandomDepthNavigator(ReactiveNavigatorBase):
    """
    Non-SLAM random navigator using a 32FC1 depth image.

    This detects a compact near-depth region, not the semantic class "oil palm".
    To prove that an object is specifically an oil-palm tree, RGB/YOLO or a
    trained classifier is still required.
    """

    def __init__(self):
        super().__init__("sawit_random_depth_navigator")

        self.declare_parameter("depth_topic", "/camera")
        self.declare_parameter("camera_info_topic", "/camera_front/camera_info")
        self.declare_parameter("minimum_depth", 1.5)
        self.declare_parameter("maximum_depth", 14.0)
        self.declare_parameter("minimum_component_area", 250)
        self.declare_parameter("maximum_component_area", 90000)
        self.declare_parameter("center_vertical_start_ratio", 0.18)
        self.declare_parameter("center_vertical_end_ratio", 0.92)
        self.declare_parameter("horizontal_margin_ratio", 0.08)
        self.declare_parameter("process_hz", 5.0)

        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(
            self.get_parameter("camera_info_topic").value
        )
        self.minimum_depth = float(self.get_parameter("minimum_depth").value)
        self.maximum_depth = float(self.get_parameter("maximum_depth").value)
        self.minimum_component_area = int(
            self.get_parameter("minimum_component_area").value
        )
        self.maximum_component_area = int(
            self.get_parameter("maximum_component_area").value
        )
        self.roi_top_ratio = float(
            self.get_parameter("center_vertical_start_ratio").value
        )
        self.roi_bottom_ratio = float(
            self.get_parameter("center_vertical_end_ratio").value
        )
        self.margin_ratio = float(
            self.get_parameter("horizontal_margin_ratio").value
        )
        self.process_period = 1.0 / max(
            0.5, float(self.get_parameter("process_hz").value)
        )

        self.fx: Optional[float] = None
        self.cx: Optional[float] = None
        self.last_process = 0.0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Image,
            self.depth_topic,
            self._depth_callback,
            qos,
        )
        self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._camera_info_callback,
            qos,
        )
        self.get_logger().info(
            f"Depth input: {self.depth_topic}; info: {self.camera_info_topic}"
        )

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        if len(msg.k) >= 9 and msg.k[0] > 0.0:
            self.fx = float(msg.k[0])
            self.cx = float(msg.k[2])

    def _decode_depth(self, msg: Image) -> Optional[np.ndarray]:
        if msg.encoding == "32FC1":
            dtype = np.dtype(np.float32)
            if msg.is_bigendian:
                dtype = dtype.newbyteorder(">")
            array = np.frombuffer(msg.data, dtype=dtype)
            expected = msg.height * (msg.step // 4)
            if array.size < expected:
                return None
            array = array[:expected].reshape(msg.height, msg.step // 4)
            return array[:, : msg.width].astype(np.float32, copy=False)

        if msg.encoding == "16UC1":
            dtype = np.dtype(np.uint16)
            if msg.is_bigendian:
                dtype = dtype.newbyteorder(">")
            array = np.frombuffer(msg.data, dtype=dtype)
            expected = msg.height * (msg.step // 2)
            if array.size < expected:
                return None
            array = array[:expected].reshape(msg.height, msg.step // 2)
            return array[:, : msg.width].astype(np.float32) / 1000.0

        self.get_logger().error(
            f"Unsupported depth encoding: {msg.encoding}",
            throttle_duration_sec=3.0,
        )
        return None

    def _depth_callback(self, msg: Image) -> None:
        now = time.monotonic()
        if now - self.last_process < self.process_period:
            return
        self.last_process = now

        depth = self._decode_depth(msg)
        if depth is None or depth.size == 0:
            self.clear_detection()
            return

        height, width = depth.shape
        y0 = int(height * self.roi_top_ratio)
        y1 = int(height * self.roi_bottom_ratio)
        x0 = int(width * self.margin_ratio)
        x1 = int(width * (1.0 - self.margin_ratio))

        roi = depth[y0:y1, x0:x1]
        valid = np.isfinite(roi)
        valid &= roi >= self.minimum_depth
        valid &= roi <= self.maximum_depth

        # Divide depth into near-range bands so different trees do not merge.
        selected = None
        for near in np.arange(
            self.minimum_depth,
            self.maximum_depth,
            1.0,
            dtype=np.float32,
        ):
            far = min(float(near + 1.5), self.maximum_depth)
            mask = valid & (roi >= near) & (roi < far)
            mask_u8 = (mask.astype(np.uint8) * 255)
            kernel = np.ones((5, 5), dtype=np.uint8)
            mask_u8 = cv2.morphologyEx(
                mask_u8,
                cv2.MORPH_OPEN,
                kernel,
                iterations=1,
            )
            mask_u8 = cv2.morphologyEx(
                mask_u8,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=2,
            )

            count, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask_u8,
                connectivity=8,
            )

            candidates = []
            for label in range(1, count):
                area = int(stats[label, cv2.CC_STAT_AREA])
                if area < self.minimum_component_area:
                    continue
                if area > self.maximum_component_area:
                    continue

                component = labels == label
                component_depth = roi[component]
                component_depth = component_depth[np.isfinite(component_depth)]
                if component_depth.size == 0:
                    continue

                median_depth = float(np.median(component_depth))
                u_roi = float(centroids[label][0])
                v_roi = float(centroids[label][1])
                u = u_roi + x0
                v = v_roi + y0

                # Prefer nearer, tall-ish components near the image center.
                bbox_h = int(stats[label, cv2.CC_STAT_HEIGHT])
                bbox_w = int(stats[label, cv2.CC_STAT_WIDTH])
                center_cost = abs(u - width / 2.0) / max(1.0, width)
                shape_bonus = min(1.0, bbox_h / max(1.0, bbox_w))
                score = median_depth + 2.0 * center_cost - 0.5 * shape_bonus
                candidates.append((score, median_depth, u, v, area))

            if candidates:
                candidates.sort(key=lambda item: item[0])
                selected = candidates[0]
                break

        if selected is None:
            self.clear_detection()
            self.get_logger().info(
                "DEPTH: no compact target",
                throttle_duration_sec=1.5,
            )
            return

        _, forward, u, _, area = selected

        fx = self.fx if self.fx and self.fx > 0.0 else width * 0.80
        cx = self.cx if self.cx is not None else width / 2.0

        horizontal_angle = math.atan2(u - cx, fx)
        # Image u increases right; camera "left" must therefore be negative.
        left = -forward * math.tan(horizontal_angle)

        self.update_detection(forward, left)
        self.get_logger().info(
            f"DEPTH_TARGET: distance={math.hypot(forward, left):.2f}, "
            f"forward={forward:.2f}, left={left:.2f}, area={area}",
            throttle_duration_sec=0.8,
        )


def main(args=None):
    rclpy.init(args=args)
    node = RandomDepthNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
