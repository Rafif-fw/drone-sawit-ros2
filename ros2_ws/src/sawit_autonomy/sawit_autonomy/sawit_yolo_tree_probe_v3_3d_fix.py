#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone ROS 2 test for a custom Ultralytics YOLO model (batang.pt).

Purpose:
- Do NOT move the drone.
- Verify whether the model actually detects palm trunks on /camera/image.
- Estimate each detection's 3-D position using the organized /camera/points cloud.
- Fall back to /camera/depth_image when the cloud is not organized/aligned.
- Publish an annotated image and RViz markers on separate test topics.

Default model path under WSL:
  /mnt/c/Users/rafif/Downloads/batang.pt
"""

from __future__ import annotations

import math
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

from sensor_msgs.msg import Image, LaserScan, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from px4_msgs.msg import VehicleLocalPosition

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("OpenCV belum terpasang. Install: python3 -m pip install opencv-python") from exc

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Ultralytics belum terpasang. Install: python3 -m pip install --user ultralytics"
    ) from exc


@dataclass
class Detection3D:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    forward: float
    left: float
    up: float
    distance: float
    source: str
    map_x: float
    map_y: float


@dataclass
class StableTrack:
    track_id: int
    map_x: float
    map_y: float
    forward: float
    left: float
    confidence: float
    class_name: str
    observations: int = 1
    first_seen: float = 0.0
    last_seen: float = 0.0
    samples: List[Tuple[float, float]] = field(default_factory=list)


class YoloBatangTest(Node):
    def __init__(self) -> None:
        super().__init__("sawit_yolo_batang_test")

        self.declare_parameter("model_path", "/mnt/c/Users/rafif/Downloads/batang.pt")
        self.declare_parameter("confidence", 0.05)
        self.declare_parameter("iou", 0.50)
        self.declare_parameter("inference_hz", 1.5)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("image_size", 1280)
        self.declare_parameter("camera_hfov", 1.047)
        self.declare_parameter("image_topic", "/camera/image")
        self.declare_parameter("use_tiling", True)
        self.declare_parameter("probe_confidence", 0.005)
        self.declare_parameter("spawn_map_x", -25.0)
        self.declare_parameter("spawn_map_y", 0.0)
        self.declare_parameter("stable_observations", 2)
        self.declare_parameter("track_merge_radius", 1.8)
        self.declare_parameter("track_ttl", 4.0)
        self.declare_parameter("max_sensor_age", 6.0)

        self.model_path = os.path.expanduser(str(self.get_parameter("model_path").value))
        self.conf_threshold = float(self.get_parameter("confidence").value)
        self.iou_threshold = float(self.get_parameter("iou").value)
        self.inference_hz = max(0.5, float(self.get_parameter("inference_hz").value))
        self.device = str(self.get_parameter("device").value)
        self.image_size = int(self.get_parameter("image_size").value)
        self.camera_hfov = float(self.get_parameter("camera_hfov").value)
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.use_tiling = bool(self.get_parameter("use_tiling").value)
        self.probe_confidence = float(self.get_parameter("probe_confidence").value)
        self.spawn_map_x = float(self.get_parameter("spawn_map_x").value)
        self.spawn_map_y = float(self.get_parameter("spawn_map_y").value)
        self.stable_observations = max(1, int(self.get_parameter("stable_observations").value))
        self.track_merge_radius = float(self.get_parameter("track_merge_radius").value)
        self.track_ttl = float(self.get_parameter("track_ttl").value)
        self.max_sensor_age = max(0.5, float(self.get_parameter("max_sensor_age").value))

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"Model tidak ditemukan: {self.model_path}. "
                "Di WSL path Windows C:\\Users\\rafif\\Downloads\\batang.pt harus menjadi "
                "/mnt/c/Users/rafif/Downloads/batang.pt"
            )

        self.get_logger().info(f"Loading YOLO model: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.names = self._normalise_names(getattr(self.model, "names", {}))
        self.get_logger().info(f"YOLO classes: {self.names}")

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2,
        )
        px4_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(Image, self.image_topic, self.on_image, sensor_qos)
        self.create_subscription(Image, "/camera/depth_image", self.on_depth, sensor_qos)
        self.create_subscription(PointCloud2, "/camera/points", self.on_cloud, sensor_qos)
        self.create_subscription(LaserScan, "/tof_front", self.on_tof, sensor_qos)
        self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self.on_local_position,
            px4_qos,
        )

        self.debug_pub = self.create_publisher(Image, "/sawit/yolo_batang/debug_image", 2)
        self.marker_pub = self.create_publisher(MarkerArray, "/sawit/yolo_batang/markers", 10)

        self.latest_image: Optional[Image] = None
        self.latest_depth: Optional[Image] = None
        self.latest_cloud: Optional[PointCloud2] = None
        self.latest_image_time = 0.0
        self.latest_cloud_time = 0.0
        self.latest_depth_time = 0.0

        self.local_x = 0.0
        self.local_y = 0.0
        self.heading = 0.0
        self.have_pose = False
        self.tof_distance = float("inf")

        self.tracks: List[StableTrack] = []
        self.next_track_id = 0
        self.busy = False
        self.last_summary_time = 0.0
        self.logged_image_info = False
        self.logged_cloud_info = False

        self.timer = self.create_timer(1.0 / self.inference_hz, self.process_once)

        self.get_logger().info("START YOLO-TREE PROBE V3 3D-FIX (2D + synchronized snapshot cloud/depth, drone tidak digerakkan)")
        self.get_logger().info(f"Image topic: {self.image_topic}; imgsz={self.image_size}; conf={self.conf_threshold}; tiling={self.use_tiling}")
        self.get_logger().info("Debug image: /sawit/yolo_batang/debug_image")
        self.get_logger().info("RViz markers: /sawit/yolo_batang/markers")

    @staticmethod
    def _normalise_names(names) -> Dict[int, str]:
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, (list, tuple)):
            return {i: str(v) for i, v in enumerate(names)}
        return {}

    def on_image(self, msg: Image) -> None:
        self.latest_image = msg
        self.latest_image_time = time.monotonic()

    def on_depth(self, msg: Image) -> None:
        self.latest_depth = msg
        self.latest_depth_time = time.monotonic()

    def on_cloud(self, msg: PointCloud2) -> None:
        self.latest_cloud = msg
        self.latest_cloud_time = time.monotonic()

    def on_local_position(self, msg: VehicleLocalPosition) -> None:
        if math.isfinite(float(msg.x)) and math.isfinite(float(msg.y)):
            self.local_x = float(msg.x)
            self.local_y = float(msg.y)
            self.have_pose = True
        if math.isfinite(float(msg.heading)):
            self.heading = float(msg.heading)

    def on_tof(self, msg: LaserScan) -> None:
        vals: List[float] = []
        for i, r in enumerate(msg.ranges):
            if not math.isfinite(float(r)):
                continue
            angle = float(msg.angle_min) + i * float(msg.angle_increment)
            if abs(angle) <= math.radians(8.0) and 0.1 < float(r) < 30.0:
                vals.append(float(r))
        if vals:
            self.tof_distance = float(np.median(np.asarray(vals, dtype=np.float32)))

    @staticmethod
    def decode_image(msg: Image) -> np.ndarray:
        h = int(msg.height)
        w = int(msg.width)
        enc = msg.encoding.lower()
        if enc in ("rgb8", "bgr8"):
            channels = 3
            row_pixels = int(msg.step) // channels if int(msg.step) else w
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, row_pixels, channels)[:, :w, :]
            if enc == "rgb8":
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return np.ascontiguousarray(arr)
        if enc in ("rgba8", "bgra8"):
            channels = 4
            row_pixels = int(msg.step) // channels if int(msg.step) else w
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, row_pixels, channels)[:, :w, :]
            code = cv2.COLOR_RGBA2BGR if enc == "rgba8" else cv2.COLOR_BGRA2BGR
            return cv2.cvtColor(arr, code)
        if enc in ("mono8", "8uc1"):
            row_pixels = int(msg.step) if int(msg.step) else w
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, row_pixels)[:, :w]
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        raise ValueError(f"Encoding image belum didukung: {msg.encoding}")

    @staticmethod
    def decode_depth(msg: Image) -> Optional[np.ndarray]:
        try:
            h = int(msg.height)
            w = int(msg.width)
            enc = msg.encoding.lower()
            if enc in ("32fc1", "32fc"):
                return np.frombuffer(msg.data, dtype=np.float32).reshape(h, w).copy()
            if enc in ("16uc1", "mono16"):
                return (
                    np.frombuffer(msg.data, dtype=np.uint16)
                    .reshape(h, w)
                    .astype(np.float32)
                    * 0.001
                )
        except Exception:
            return None
        return None

    @staticmethod
    def _field_offsets(cloud: PointCloud2) -> Optional[Tuple[int, int, int]]:
        offsets = {str(f.name): int(f.offset) for f in cloud.fields}
        if not all(k in offsets for k in ("x", "y", "z")):
            return None
        return offsets["x"], offsets["y"], offsets["z"]

    def xyz_from_cloud_pixels(
        self,
        cloud: PointCloud2,
        pixel_mask: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> Optional[Tuple[float, float, float, int]]:
        image_h, image_w = image_shape
        cloud_h = int(cloud.height)
        cloud_w = int(cloud.width)
        if cloud_h <= 1 or cloud_w <= 1:
            return None

        # GZ image and depth point cloud may use different organized resolutions.
        # Resize the YOLO pixel mask into cloud coordinates instead of rejecting it.
        if cloud_h != image_h or cloud_w != image_w:
            pixel_mask = cv2.resize(
                pixel_mask.astype(np.uint8),
                (cloud_w, cloud_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        offsets = self._field_offsets(cloud)
        if offsets is None:
            return None
        ox, oy, oz = offsets
        endian = ">" if bool(cloud.is_bigendian) else "<"
        fmt = endian + "f"
        row_step = int(cloud.row_step)
        point_step = int(cloud.point_step)
        raw = cloud.data

        ys, xs = np.nonzero(pixel_mask)
        if xs.size == 0:
            return None

        # Bound CPU cost while keeping points spread over the full YOLO region.
        sample_count = min(1400, xs.size)
        if xs.size > sample_count:
            idx = np.linspace(0, xs.size - 1, sample_count, dtype=np.int64)
            xs = xs[idx]
            ys = ys[idx]

        points: List[Tuple[float, float, float]] = []
        for u, v in zip(xs.tolist(), ys.tolist()):
            base = int(v) * row_step + int(u) * point_step
            try:
                x = float(struct.unpack_from(fmt, raw, base + ox)[0])
                y = float(struct.unpack_from(fmt, raw, base + oy)[0])
                z = float(struct.unpack_from(fmt, raw, base + oz)[0])
            except (struct.error, IndexError):
                continue
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            # Confirmed sensor axes: x forward, y lateral/left, z up.
            if not (0.35 <= x <= 22.0):
                continue
            if abs(y) > 9.0 or not (-2.0 <= z <= 3.0):
                continue
            points.append((x, y, z))

        if len(points) < 12:
            return None

        arr = np.asarray(points, dtype=np.float32)
        # First robust center, then remove background/ground outliers.
        med = np.median(arr, axis=0)
        keep = (
            (np.abs(arr[:, 0] - med[0]) <= 1.0)
            & (np.abs(arr[:, 1] - med[1]) <= 0.9)
            & (np.abs(arr[:, 2] - med[2]) <= 1.2)
        )
        core = arr[keep]
        if core.shape[0] < 10:
            core = arr

        # 35th percentile forward is more resistant to background behind a thin trunk.
        forward = float(np.percentile(core[:, 0], 35.0))
        near = core[np.abs(core[:, 0] - forward) <= 0.8]
        if near.shape[0] >= 8:
            core = near
        left = float(np.median(core[:, 1]))
        up = float(np.median(core[:, 2]))
        return forward, left, up, int(core.shape[0])

    def xyz_from_depth(
        self,
        depth_msg: Optional[Image],
        bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, int],
    ) -> Optional[Tuple[float, float, float, int]]:
        if depth_msg is None:
            return None
        depth = self.decode_depth(depth_msg)
        if depth is None:
            return None
        h, w = image_shape
        if depth.shape[0] != h or depth.shape[1] != w:
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        x0, y0, x1, y1 = bbox
        bw = max(1, x1 - x0)
        bh = max(1, y1 - y0)
        # Central strip avoids background around the trunk bbox.
        rx0 = max(0, int(x0 + 0.32 * bw))
        rx1 = min(w, int(x0 + 0.68 * bw))
        ry0 = max(0, int(y0 + 0.18 * bh))
        ry1 = min(h, int(y0 + 0.92 * bh))
        vals = depth[ry0:ry1, rx0:rx1].reshape(-1)
        vals = vals[np.isfinite(vals)]
        vals = vals[(vals >= 0.35) & (vals <= 22.0)]
        if vals.size < 8:
            return None
        forward = float(np.median(vals))
        center_u = 0.5 * (x0 + x1)
        norm_x = (center_u / max(w - 1, 1) - 0.5) * 2.0
        angle = norm_x * self.camera_hfov * 0.5
        left = -forward * math.tan(angle)
        return forward, left, 0.0, int(vals.size)

    def make_detection_mask(
        self,
        bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, int],
        segmentation_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        h, w = image_shape
        x0, y0, x1, y1 = bbox
        mask = np.zeros((h, w), dtype=np.uint8)
        if segmentation_mask is not None:
            seg = segmentation_mask
            if seg.shape != (h, w):
                seg = cv2.resize(seg.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            mask[seg > 0.5] = 1
            # Keep the central vertical part to reduce leaves/background.
            bw = max(1, x1 - x0)
            central = np.zeros_like(mask)
            cx0 = max(0, int(x0 + 0.18 * bw))
            cx1 = min(w, int(x0 + 0.82 * bw))
            central[max(0, y0):min(h, y1), cx0:cx1] = 1
            mask &= central
        else:
            bw = max(1, x1 - x0)
            bh = max(1, y1 - y0)
            rx0 = max(0, int(x0 + 0.28 * bw))
            rx1 = min(w, int(x0 + 0.72 * bw))
            ry0 = max(0, int(y0 + 0.12 * bh))
            ry1 = min(h, int(y0 + 0.95 * bh))
            mask[ry0:ry1, rx0:rx1] = 1
        return mask.astype(bool)

    def body_to_local(self, forward: float, left: float) -> Tuple[float, float]:
        # PX4 local: x=N, y=E. Body forward and left transformed using heading.
        lx = self.local_x + forward * math.cos(self.heading) - left * math.sin(self.heading)
        ly = self.local_y + forward * math.sin(self.heading) + left * math.cos(self.heading)
        return lx, ly

    def local_to_map(self, local_x: float, local_y: float) -> Tuple[float, float]:
        # Same visual transform used by the current sawit navigator.
        return self.spawn_map_x + local_y, self.spawn_map_y + local_x

    @staticmethod
    def _nms_boxes(
        boxes: List[Tuple[int, int, int, int]],
        scores: List[float],
        score_threshold: float,
        nms_threshold: float,
    ) -> List[int]:
        if not boxes:
            return []
        xywh = [[x0, y0, max(1, x1 - x0), max(1, y1 - y0)] for x0, y0, x1, y1 in boxes]
        indices = cv2.dnn.NMSBoxes(xywh, scores, score_threshold, nms_threshold)
        if indices is None or len(indices) == 0:
            return []
        return [int(i) for i in np.asarray(indices).reshape(-1).tolist()]

    def infer_boxes_multiscale(
        self, frame: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float], List[int], float]:
        """Run a low-threshold probe and return filtered boxes in original image coordinates."""
        h, w = frame.shape[:2]
        raw_boxes: List[Tuple[int, int, int, int]] = []
        raw_scores: List[float] = []
        raw_classes: List[int] = []
        max_raw_conf = 0.0

        regions: List[Tuple[int, int, int, int]] = [(0, 0, w, h)]
        if self.use_tiling:
            tile_w = max(32, int(round(w * 0.62)))
            tile_h = max(32, int(round(h * 0.72)))
            x_starts = sorted(set([0, max(0, w - tile_w)]))
            y_starts = sorted(set([0, max(0, h - tile_h)]))
            regions.extend((x0, y0, min(w, x0 + tile_w), min(h, y0 + tile_h))
                           for y0 in y_starts for x0 in x_starts)

        for rx0, ry0, rx1, ry1 in regions:
            crop = frame[ry0:ry1, rx0:rx1]
            if crop.size == 0:
                continue
            results = self.model.predict(
                source=crop,
                conf=self.probe_confidence,
                iou=self.iou_threshold,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )
            if not results:
                continue
            boxes_obj = getattr(results[0], "boxes", None)
            if boxes_obj is None or len(boxes_obj) == 0:
                continue
            xyxy = boxes_obj.xyxy.cpu().numpy()
            confs = boxes_obj.conf.cpu().numpy()
            classes = boxes_obj.cls.cpu().numpy().astype(int)
            for box, conf, cls_id in zip(xyxy, confs, classes):
                conf = float(conf)
                max_raw_conf = max(max_raw_conf, conf)
                if conf < self.conf_threshold:
                    continue
                x0 = max(0, min(w - 1, int(round(float(box[0]) + rx0))))
                y0 = max(0, min(h - 1, int(round(float(box[1]) + ry0))))
                x1 = max(x0 + 1, min(w, int(round(float(box[2]) + rx0))))
                y1 = max(y0 + 1, min(h, int(round(float(box[3]) + ry0))))
                raw_boxes.append((x0, y0, x1, y1))
                raw_scores.append(conf)
                raw_classes.append(int(cls_id))

        keep = self._nms_boxes(raw_boxes, raw_scores, self.conf_threshold, self.iou_threshold)
        return (
            [raw_boxes[i] for i in keep],
            [raw_scores[i] for i in keep],
            [raw_classes[i] for i in keep],
            max_raw_conf,
        )

    def process_once(self) -> None:
        if self.busy or self.latest_image is None:
            return
        if time.monotonic() - self.latest_image_time > 1.0:
            return

        self.busy = True
        try:
            # Snapshot mutually corresponding sensor messages BEFORE slow YOLO inference.
            # The old V2 checked sensor age after 3-4 s inference, so every valid box
            # became "no-3D" even though cloud/depth were present.
            image_msg = self.latest_image
            cloud_msg = self.latest_cloud
            depth_msg = self.latest_depth
            now_at_start = time.monotonic()
            cloud_age_at_start = (
                now_at_start - self.latest_cloud_time if cloud_msg is not None else float("inf")
            )
            depth_age_at_start = (
                now_at_start - self.latest_depth_time if depth_msg is not None else float("inf")
            )

            frame = self.decode_image(image_msg)
            h, w = frame.shape[:2]

            if not self.logged_image_info:
                self.logged_image_info = True
                self.get_logger().info(
                    f"INPUT_IMAGE topic={self.image_topic} encoding={image_msg.encoding} "
                    f"size={w}x{h} min={int(frame.min())} max={int(frame.max())} "
                    f"mean={float(frame.mean()):.1f} std={float(frame.std()):.1f}"
                )

            if not self.logged_cloud_info and cloud_msg is not None:
                self.logged_cloud_info = True
                fields = {str(f.name): int(f.offset) for f in cloud_msg.fields}
                self.get_logger().info(
                    f"INPUT_CLOUD size={int(cloud_msg.width)}x{int(cloud_msg.height)} "
                    f"organized={int(cloud_msg.height) > 1} point_step={int(cloud_msg.point_step)} "
                    f"fields={fields} age_at_start={cloud_age_at_start:.3f}s; "
                    f"depth_age_at_start={depth_age_at_start:.3f}s max_sensor_age={self.max_sensor_age:.1f}s"
                )

            box_list, conf_list, class_list, max_raw_conf = self.infer_boxes_multiscale(frame)
            detections: List[Detection3D] = []
            for bbox, conf, cls_id in zip(box_list, conf_list, class_list):
                x0, y0, x1, y1 = bbox
                pixel_mask = self.make_detection_mask(bbox, (h, w), None)

                xyz = None
                source = "none"
                if cloud_msg is not None and cloud_age_at_start <= self.max_sensor_age:
                    xyz = self.xyz_from_cloud_pixels(cloud_msg, pixel_mask, (h, w))
                    if xyz is not None:
                        source = "organized_pointcloud"
                if xyz is None and depth_msg is not None and depth_age_at_start <= self.max_sensor_age:
                    xyz = self.xyz_from_depth(depth_msg, bbox, (h, w))
                    if xyz is not None:
                        source = "depth_fallback"
                if xyz is None:
                    self.draw_box(
                        frame,
                        bbox,
                        f"{self.names.get(int(cls_id), str(cls_id))} {float(conf):.2f} no-3D",
                        (0, 0, 255),
                    )
                    continue

                forward, left, up, point_count = xyz
                distance = math.hypot(forward, left)
                local_x, local_y = self.body_to_local(forward, left)
                map_x, map_y = self.local_to_map(local_x, local_y)
                class_name = self.names.get(int(cls_id), str(cls_id))

                det = Detection3D(
                    class_id=int(cls_id),
                    class_name=class_name,
                    confidence=float(conf),
                    bbox=bbox,
                    forward=forward,
                    left=left,
                    up=up,
                    distance=distance,
                    source=source,
                    map_x=map_x,
                    map_y=map_y,
                )
                detections.append(det)

                tof_text = "inf" if not math.isfinite(self.tof_distance) else f"{self.tof_distance:.2f}"
                label = (
                    f"{class_name} {float(conf):.2f} "
                    f"PC={distance:.2f}m ToF={tof_text} n={point_count}"
                )
                self.draw_box(frame, bbox, label, (0, 255, 255))

            stable = self.update_tracks(detections)
            self.publish_markers(stable)
            self.publish_debug(frame, image_msg.header)

            now = time.monotonic()
            if now - self.last_summary_time >= 1.0:
                self.last_summary_time = now
                boxes_2d = len(box_list)
                no_3d = max(0, boxes_2d - len(detections))
                if detections:
                    details = "; ".join(
                        f"{d.class_name} conf={d.confidence:.2f} dist={d.distance:.2f}m "
                        f"f={d.forward:.2f} l={d.left:.2f} src={d.source}"
                        for d in detections
                    )
                    self.get_logger().info(
                        f"YOLO_TREE boxes2d={boxes_2d} detections3d={len(detections)} "
                        f"no3d={no_3d} stable={len(stable)} max_raw_conf={max_raw_conf:.4f} | {details}"
                    )
                else:
                    self.get_logger().info(
                        f"YOLO_TREE boxes2d={boxes_2d} detections3d=0 no3d={no_3d} "
                        f"stable=0 max_raw_conf={max_raw_conf:.4f} "
                        f"cloud_age_start={cloud_age_at_start:.2f}s depth_age_start={depth_age_at_start:.2f}s"
                    )
        except Exception as exc:
            self.get_logger().error(f"YOLO inference gagal: {type(exc).__name__}: {exc}")
        finally:
            self.busy = False

    @staticmethod
    def draw_box(
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        label: str,
        color: Tuple[int, int, int],
    ) -> None:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
        baseline_y = max(18, y0 - 7)
        cv2.putText(
            image,
            label,
            (x0, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    def update_tracks(self, detections: Sequence[Detection3D]) -> List[StableTrack]:
        now = time.monotonic()
        for det in detections:
            nearest: Optional[StableTrack] = None
            nearest_d = float("inf")
            for track in self.tracks:
                d = math.hypot(det.map_x - track.map_x, det.map_y - track.map_y)
                if d < nearest_d and d <= self.track_merge_radius:
                    nearest = track
                    nearest_d = d
            if nearest is None:
                track = StableTrack(
                    track_id=self.next_track_id,
                    map_x=det.map_x,
                    map_y=det.map_y,
                    forward=det.forward,
                    left=det.left,
                    confidence=det.confidence,
                    class_name=det.class_name,
                    observations=1,
                    first_seen=now,
                    last_seen=now,
                    samples=[(det.map_x, det.map_y)],
                )
                self.next_track_id += 1
                self.tracks.append(track)
            else:
                nearest.observations += 1
                nearest.last_seen = now
                nearest.confidence = max(nearest.confidence, det.confidence)
                nearest.forward = 0.65 * nearest.forward + 0.35 * det.forward
                nearest.left = 0.65 * nearest.left + 0.35 * det.left
                nearest.samples.append((det.map_x, det.map_y))
                nearest.samples = nearest.samples[-12:]
                arr = np.asarray(nearest.samples, dtype=np.float32)
                nearest.map_x = float(np.median(arr[:, 0]))
                nearest.map_y = float(np.median(arr[:, 1]))

        self.tracks = [t for t in self.tracks if now - t.last_seen <= self.track_ttl]
        return [t for t in self.tracks if t.observations >= self.stable_observations]

    def publish_debug(self, bgr: np.ndarray, header) -> None:
        msg = Image()
        msg.header = header
        msg.height = int(bgr.shape[0])
        msg.width = int(bgr.shape[1])
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = int(bgr.shape[1] * 3)
        msg.data = np.ascontiguousarray(bgr).tobytes()
        self.debug_pub.publish(msg)

    def publish_markers(self, tracks: Sequence[StableTrack]) -> None:
        arr = MarkerArray()

        delete_all = Marker()
        delete_all.header.frame_id = "map"
        delete_all.header.stamp = self.get_clock().now().to_msg()
        delete_all.action = Marker.DELETEALL
        arr.markers.append(delete_all)

        for track in tracks:
            sphere = Marker()
            sphere.header.frame_id = "map"
            sphere.header.stamp = self.get_clock().now().to_msg()
            sphere.ns = "yolo_batang"
            sphere.id = int(track.track_id * 2)
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(track.map_x)
            sphere.pose.position.y = float(track.map_y)
            sphere.pose.position.z = 0.65
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.55
            sphere.scale.y = 0.55
            sphere.scale.z = 0.55
            sphere.color.r = 1.0
            sphere.color.g = 0.85
            sphere.color.b = 0.0
            sphere.color.a = 1.0
            arr.markers.append(sphere)

            text = Marker()
            text.header = sphere.header
            text.ns = "yolo_batang_text"
            text.id = int(track.track_id * 2 + 1)
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(track.map_x)
            text.pose.position.y = float(track.map_y)
            text.pose.position.z = 1.25
            text.pose.orientation.w = 1.0
            text.scale.z = 0.38
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = (
                f"YOLO {track.class_name} id={track.track_id} "
                f"obs={track.observations} conf={track.confidence:.2f}"
            )
            arr.markers.append(text)

        self.marker_pub.publish(arr)


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[YoloBatangTest] = None
    try:
        node = YoloBatangTest()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
