#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO tree probe V4 for ROS 2 Humble.

Tujuan:
- Tidak menggerakkan drone.
- YOLO mendeteksi pohon sawit pada /camera/image.
- Jarak dan posisi 3D diambil dari organized /camera/points.
- Inferensi dilakukan di worker thread agar callback image/cloud/depth tidak berhenti.
- ROI 3D difokuskan ke batang bagian tengah-bawah bounding box.
- Depth fallback dimatikan secara default karena kurang presisi untuk mapping.

Default model:
  /mnt/c/Users/rafif/Downloads/tree.pt
"""

from __future__ import annotations

import math
import os
import struct
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
)

from sensor_msgs.msg import Image, LaserScan, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from px4_msgs.msg import VehicleLocalPosition

try:
    import cv2
except ImportError as exc:
    raise RuntimeError(
        "OpenCV gagal dimuat. Pastikan NumPy 1.26.4 dan opencv-python 4.10.0.84 terpasang."
    ) from exc

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise RuntimeError(
        "Ultralytics belum terpasang. Jalankan: python3 -m pip install --user ultralytics"
    ) from exc


@dataclass
class TimedMessage:
    stamp: float
    received: float
    msg: object


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
    point_count: int
    forward_spread: float
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


@dataclass
class InferenceSnapshot:
    image_msg: Image
    cloud_msg: Optional[PointCloud2]
    depth_msg: Optional[Image]
    image_received: float
    cloud_received: float
    depth_received: float
    cloud_stamp_delta: float
    depth_stamp_delta: float
    pose_x: float
    pose_y: float
    heading: float
    tof_distance: float


@dataclass
class InferenceResult:
    frame: np.ndarray
    image_header: object
    boxes2d: int
    detections: List[Detection3D]
    no3d: int
    max_raw_conf: float
    cloud_rx_age: float
    depth_rx_age: float
    cloud_stamp_delta: float
    depth_stamp_delta: float


class YoloTreeProbeV4(Node):
    def __init__(self) -> None:
        super().__init__("sawit_yolo_tree_probe_v4")

        self.declare_parameter("model_path", "/mnt/c/Users/rafif/Downloads/tree.pt")
        self.declare_parameter("image_topic", "/camera/image")
        self.declare_parameter("cloud_topic", "/camera/points")
        self.declare_parameter("depth_topic", "/camera/depth_image")
        self.declare_parameter("confidence", 0.10)
        self.declare_parameter("probe_confidence", 0.001)
        self.declare_parameter("iou", 0.35)
        self.declare_parameter("image_size", 1280)
        self.declare_parameter("use_tiling", True)
        self.declare_parameter("inference_hz", 0.5)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("camera_hfov", 1.047)
        self.declare_parameter("allow_depth_fallback", False)
        self.declare_parameter("max_sensor_rx_age", 2.0)
        self.declare_parameter("max_stamp_delta", 0.75)
        self.declare_parameter("spawn_map_x", -25.0)
        self.declare_parameter("spawn_map_y", 0.0)
        self.declare_parameter("stable_observations", 2)
        self.declare_parameter("track_merge_radius", 2.2)
        self.declare_parameter("track_ttl", 12.0)
        self.declare_parameter("trunk_roi_x_min", 0.36)
        self.declare_parameter("trunk_roi_x_max", 0.64)
        self.declare_parameter("trunk_roi_y_min", 0.38)
        self.declare_parameter("trunk_roi_y_max", 0.92)

        self.model_path = os.path.expanduser(str(self.get_parameter("model_path").value))
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.cloud_topic = str(self.get_parameter("cloud_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.conf_threshold = float(self.get_parameter("confidence").value)
        self.probe_confidence = float(self.get_parameter("probe_confidence").value)
        self.iou_threshold = float(self.get_parameter("iou").value)
        self.image_size = int(self.get_parameter("image_size").value)
        self.use_tiling = bool(self.get_parameter("use_tiling").value)
        self.inference_hz = max(0.1, float(self.get_parameter("inference_hz").value))
        self.device = str(self.get_parameter("device").value)
        self.camera_hfov = float(self.get_parameter("camera_hfov").value)
        self.allow_depth_fallback = bool(self.get_parameter("allow_depth_fallback").value)
        self.max_sensor_rx_age = max(0.2, float(self.get_parameter("max_sensor_rx_age").value))
        self.max_stamp_delta = max(0.0, float(self.get_parameter("max_stamp_delta").value))
        self.spawn_map_x = float(self.get_parameter("spawn_map_x").value)
        self.spawn_map_y = float(self.get_parameter("spawn_map_y").value)
        self.stable_observations = max(1, int(self.get_parameter("stable_observations").value))
        self.track_merge_radius = max(0.2, float(self.get_parameter("track_merge_radius").value))
        self.track_ttl = max(2.0, float(self.get_parameter("track_ttl").value))
        self.roi_x_min = float(self.get_parameter("trunk_roi_x_min").value)
        self.roi_x_max = float(self.get_parameter("trunk_roi_x_max").value)
        self.roi_y_min = float(self.get_parameter("trunk_roi_y_min").value)
        self.roi_y_max = float(self.get_parameter("trunk_roi_y_max").value)

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model tidak ditemukan: {self.model_path}")

        self.get_logger().info(f"Loading YOLO model: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.names = self._normalise_names(getattr(self.model, "names", {}))
        self.get_logger().info(f"YOLO classes: {self.names}")

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        px4_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(Image, self.image_topic, self.on_image, sensor_qos)
        self.create_subscription(PointCloud2, self.cloud_topic, self.on_cloud, sensor_qos)
        self.create_subscription(Image, self.depth_topic, self.on_depth, sensor_qos)
        self.create_subscription(LaserScan, "/tof_front", self.on_tof, sensor_qos)
        self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self.on_local_position,
            px4_qos,
        )

        self.debug_pub = self.create_publisher(Image, "/sawit/yolo_tree_v4/debug_image", 2)
        self.marker_pub = self.create_publisher(MarkerArray, "/sawit/yolo_tree_v4/markers", 10)

        self.data_lock = threading.Lock()
        self.latest_image: Optional[TimedMessage] = None
        self.cloud_buffer: Deque[TimedMessage] = deque(maxlen=12)
        self.depth_buffer: Deque[TimedMessage] = deque(maxlen=12)

        self.local_x = 0.0
        self.local_y = 0.0
        self.heading = 0.0
        self.have_pose = False
        self.tof_distance = float("inf")

        self.tracks: List[StableTrack] = []
        self.next_track_id = 0
        self.last_summary_time = 0.0
        self.last_submit_time = 0.0
        self.logged_input = False

        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo_tree_v4")
        self.future: Optional[Future] = None

        self.create_timer(0.05, self.schedule_inference)
        self.create_timer(0.05, self.poll_inference)

        self.get_logger().info(
            "START YOLO-TREE PROBE V4 ASYNC-CLOUD "
            "(inferensi worker, callback sensor tetap realtime, drone tidak digerakkan)"
        )
        self.get_logger().info(
            f"image={self.image_topic} cloud={self.cloud_topic} depth={self.depth_topic} "
            f"imgsz={self.image_size} conf={self.conf_threshold} tiling={self.use_tiling}"
        )
        self.get_logger().info("Debug Image: /sawit/yolo_tree_v4/debug_image")
        self.get_logger().info("MarkerArray: /sawit/yolo_tree_v4/markers")

    @staticmethod
    def _normalise_names(names) -> Dict[int, str]:
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, (list, tuple)):
            return {i: str(v) for i, v in enumerate(names)}
        return {}

    @staticmethod
    def stamp_seconds(msg) -> float:
        try:
            stamp = msg.header.stamp
            return float(stamp.sec) + float(stamp.nanosec) * 1e-9
        except Exception:
            return 0.0

    def on_image(self, msg: Image) -> None:
        item = TimedMessage(self.stamp_seconds(msg), time.monotonic(), msg)
        with self.data_lock:
            self.latest_image = item

    def on_cloud(self, msg: PointCloud2) -> None:
        item = TimedMessage(self.stamp_seconds(msg), time.monotonic(), msg)
        with self.data_lock:
            self.cloud_buffer.append(item)

    def on_depth(self, msg: Image) -> None:
        item = TimedMessage(self.stamp_seconds(msg), time.monotonic(), msg)
        with self.data_lock:
            self.depth_buffer.append(item)

    def on_local_position(self, msg: VehicleLocalPosition) -> None:
        with self.data_lock:
            if math.isfinite(float(msg.x)) and math.isfinite(float(msg.y)):
                self.local_x = float(msg.x)
                self.local_y = float(msg.y)
                self.have_pose = True
            if math.isfinite(float(msg.heading)):
                self.heading = float(msg.heading)

    def on_tof(self, msg: LaserScan) -> None:
        vals: List[float] = []
        for i, r in enumerate(msg.ranges):
            rr = float(r)
            if not math.isfinite(rr):
                continue
            angle = float(msg.angle_min) + i * float(msg.angle_increment)
            if abs(angle) <= math.radians(8.0) and 0.1 < rr < 30.0:
                vals.append(rr)
        if vals:
            with self.data_lock:
                self.tof_distance = float(np.median(np.asarray(vals, dtype=np.float32)))

    @staticmethod
    def closest_message(buffer: Sequence[TimedMessage], target_stamp: float) -> Tuple[Optional[TimedMessage], float]:
        if not buffer:
            return None, float("inf")
        if target_stamp <= 0.0:
            item = buffer[-1]
            return item, float("inf")
        item = min(buffer, key=lambda x: abs(x.stamp - target_stamp) if x.stamp > 0.0 else float("inf"))
        delta = abs(item.stamp - target_stamp) if item.stamp > 0.0 else float("inf")
        return item, delta

    def build_snapshot(self) -> Optional[InferenceSnapshot]:
        now = time.monotonic()
        with self.data_lock:
            image = self.latest_image
            if image is None or now - image.received > 1.0:
                return None
            cloud, cloud_delta = self.closest_message(list(self.cloud_buffer), image.stamp)
            depth, depth_delta = self.closest_message(list(self.depth_buffer), image.stamp)
            pose_x = self.local_x
            pose_y = self.local_y
            heading = self.heading
            tof = self.tof_distance

        return InferenceSnapshot(
            image_msg=image.msg,
            cloud_msg=cloud.msg if cloud is not None else None,
            depth_msg=depth.msg if depth is not None else None,
            image_received=image.received,
            cloud_received=cloud.received if cloud is not None else 0.0,
            depth_received=depth.received if depth is not None else 0.0,
            cloud_stamp_delta=cloud_delta,
            depth_stamp_delta=depth_delta,
            pose_x=pose_x,
            pose_y=pose_y,
            heading=heading,
            tof_distance=tof,
        )

    def schedule_inference(self) -> None:
        if self.future is not None:
            return
        now = time.monotonic()
        if now - self.last_submit_time < 1.0 / self.inference_hz:
            return
        snapshot = self.build_snapshot()
        if snapshot is None:
            return
        self.last_submit_time = now
        self.future = self.worker.submit(self.run_inference, snapshot)

    def poll_inference(self) -> None:
        future = self.future
        if future is None or not future.done():
            return
        self.future = None
        try:
            result: InferenceResult = future.result()
        except Exception as exc:
            self.get_logger().error(f"YOLO worker gagal: {type(exc).__name__}: {exc}")
            return

        stable = self.update_tracks(result.detections)
        self.publish_markers(stable)
        self.publish_debug(result.frame, result.image_header)

        now = time.monotonic()
        if now - self.last_summary_time >= 0.8:
            self.last_summary_time = now
            if result.detections:
                details = "; ".join(
                    f"{d.class_name} conf={d.confidence:.2f} PC={d.distance:.2f}m "
                    f"f={d.forward:.2f} l={d.left:.2f} n={d.point_count} "
                    f"spread={d.forward_spread:.2f}"
                    for d in result.detections
                )
                self.get_logger().info(
                    f"YOLO_TREE_V4 boxes2d={result.boxes2d} detections3d={len(result.detections)} "
                    f"no3d={result.no3d} stable={len(stable)} max_conf={result.max_raw_conf:.4f} "
                    f"cloud_rx_age={result.cloud_rx_age:.2f}s stamp_dt={result.cloud_stamp_delta:.3f}s | {details}"
                )
            else:
                self.get_logger().info(
                    f"YOLO_TREE_V4 boxes2d={result.boxes2d} detections3d=0 no3d={result.no3d} "
                    f"stable={len(stable)} max_conf={result.max_raw_conf:.4f} "
                    f"cloud_rx_age={result.cloud_rx_age:.2f}s stamp_dt={result.cloud_stamp_delta:.3f}s"
                )

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
                return np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w).astype(np.float32) * 0.001
        except Exception:
            return None
        return None

    @staticmethod
    def _field_offsets(cloud: PointCloud2) -> Optional[Tuple[int, int, int]]:
        offsets = {str(f.name): int(f.offset) for f in cloud.fields}
        if not all(k in offsets for k in ("x", "y", "z")):
            return None
        return offsets["x"], offsets["y"], offsets["z"]

    def make_trunk_mask(self, bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int]) -> np.ndarray:
        h, w = image_shape
        x0, y0, x1, y1 = bbox
        bw = max(1, x1 - x0)
        bh = max(1, y1 - y0)
        rx0 = max(0, int(round(x0 + self.roi_x_min * bw)))
        rx1 = min(w, int(round(x0 + self.roi_x_max * bw)))
        ry0 = max(0, int(round(y0 + self.roi_y_min * bh)))
        ry1 = min(h, int(round(y0 + self.roi_y_max * bh)))
        mask = np.zeros((h, w), dtype=np.uint8)
        if rx1 > rx0 and ry1 > ry0:
            mask[ry0:ry1, rx0:rx1] = 1
        return mask.astype(bool)

    def xyz_from_cloud_pixels(
        self,
        cloud: PointCloud2,
        pixel_mask: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> Optional[Tuple[float, float, float, int, float]]:
        image_h, image_w = image_shape
        cloud_h = int(cloud.height)
        cloud_w = int(cloud.width)
        if cloud_h <= 1 or cloud_w <= 1:
            return None

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
        sample_count = min(2200, xs.size)
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
            if not (0.4 <= x <= 25.0):
                continue
            if abs(y) > 12.0 or not (-2.0 <= z <= 3.0):
                continue
            points.append((x, y, z))

        if len(points) < 10:
            return None
        arr = np.asarray(points, dtype=np.float32)

        # Cari permukaan objek terdekat yang membentuk kelompok kedalaman padat.
        bin_width = 0.25
        min_x = float(np.min(arr[:, 0]))
        max_x = float(np.max(arr[:, 0]))
        if max_x - min_x < bin_width:
            core = arr
        else:
            edges = np.arange(min_x, max_x + bin_width, bin_width, dtype=np.float32)
            hist, _ = np.histogram(arr[:, 0], bins=edges)
            max_count = int(hist.max()) if hist.size else 0
            min_dense = max(6, int(round(0.10 * max_count)))
            candidate_bins = [i for i, count in enumerate(hist.tolist()) if count >= min_dense]
            if not candidate_bins:
                return None
            chosen = candidate_bins[0]  # nearest sufficiently dense depth layer
            center = 0.5 * float(edges[chosen] + edges[chosen + 1])
            core = arr[np.abs(arr[:, 0] - center) <= 0.42]
            if core.shape[0] < 8:
                return None

        forward = float(np.median(core[:, 0]))
        left = float(np.median(core[:, 1]))
        up = float(np.median(core[:, 2]))
        spread = float(np.percentile(core[:, 0], 90.0) - np.percentile(core[:, 0], 10.0))
        if spread > 1.20:
            return None
        return forward, left, up, int(core.shape[0]), spread

    def xyz_from_depth(
        self,
        depth_msg: Optional[Image],
        bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, int],
    ) -> Optional[Tuple[float, float, float, int, float]]:
        if depth_msg is None:
            return None
        depth = self.decode_depth(depth_msg)
        if depth is None:
            return None
        h, w = image_shape
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = self.make_trunk_mask(bbox, image_shape)
        vals = depth[mask]
        vals = vals[np.isfinite(vals)]
        vals = vals[(vals >= 0.4) & (vals <= 25.0)]
        if vals.size < 8:
            return None
        forward = float(np.percentile(vals, 30.0))
        x0, _, x1, _ = bbox
        center_u = 0.5 * (x0 + x1)
        norm_x = (center_u / max(w - 1, 1) - 0.5) * 2.0
        angle = norm_x * self.camera_hfov * 0.5
        left = -forward * math.tan(angle)
        spread = float(np.percentile(vals, 90.0) - np.percentile(vals, 10.0))
        return forward, left, 0.0, int(vals.size), spread

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
            regions.extend(
                (x0, y0, min(w, x0 + tile_w), min(h, y0 + tile_h))
                for y0 in y_starts
                for x0 in x_starts
            )

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
                confidence = float(conf)
                max_raw_conf = max(max_raw_conf, confidence)
                if confidence < self.conf_threshold:
                    continue
                x0 = max(0, min(w - 1, int(round(float(box[0]) + rx0))))
                y0 = max(0, min(h - 1, int(round(float(box[1]) + ry0))))
                x1 = max(x0 + 1, min(w, int(round(float(box[2]) + rx0))))
                y1 = max(y0 + 1, min(h, int(round(float(box[3]) + ry0))))
                # Box sangat kecil biasanya hasil tile/noise.
                if (x1 - x0) < 10 or (y1 - y0) < 22:
                    continue
                raw_boxes.append((x0, y0, x1, y1))
                raw_scores.append(confidence)
                raw_classes.append(int(cls_id))

        keep = self._nms_boxes(raw_boxes, raw_scores, self.conf_threshold, self.iou_threshold)
        return (
            [raw_boxes[i] for i in keep],
            [raw_scores[i] for i in keep],
            [raw_classes[i] for i in keep],
            max_raw_conf,
        )

    @staticmethod
    def draw_box(
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        label: str,
        color: Tuple[int, int, int],
    ) -> None:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            image,
            label,
            (x0, max(18, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            1,
            cv2.LINE_AA,
        )

    def body_to_map(
        self,
        forward: float,
        left: float,
        pose_x: float,
        pose_y: float,
        heading: float,
    ) -> Tuple[float, float]:
        local_x = pose_x + forward * math.cos(heading) - left * math.sin(heading)
        local_y = pose_y + forward * math.sin(heading) + left * math.cos(heading)
        return self.spawn_map_x + local_y, self.spawn_map_y + local_x

    def run_inference(self, snapshot: InferenceSnapshot) -> InferenceResult:
        start = time.monotonic()
        frame = self.decode_image(snapshot.image_msg)
        h, w = frame.shape[:2]
        boxes, scores, classes, max_raw_conf = self.infer_boxes_multiscale(frame)

        cloud_age = start - snapshot.cloud_received if snapshot.cloud_msg is not None else float("inf")
        depth_age = start - snapshot.depth_received if snapshot.depth_msg is not None else float("inf")
        cloud_ok = (
            snapshot.cloud_msg is not None
            and cloud_age <= self.max_sensor_rx_age
            and (
                not math.isfinite(snapshot.cloud_stamp_delta)
                or snapshot.cloud_stamp_delta <= self.max_stamp_delta
            )
        )
        depth_ok = (
            self.allow_depth_fallback
            and snapshot.depth_msg is not None
            and depth_age <= self.max_sensor_rx_age
            and (
                not math.isfinite(snapshot.depth_stamp_delta)
                or snapshot.depth_stamp_delta <= self.max_stamp_delta
            )
        )

        detections: List[Detection3D] = []
        for bbox, conf, cls_id in zip(boxes, scores, classes):
            mask = self.make_trunk_mask(bbox, (h, w))
            xyz = None
            source = "none"
            if cloud_ok:
                xyz = self.xyz_from_cloud_pixels(snapshot.cloud_msg, mask, (h, w))
                if xyz is not None:
                    source = "organized_pointcloud"
            if xyz is None and depth_ok:
                xyz = self.xyz_from_depth(snapshot.depth_msg, bbox, (h, w))
                if xyz is not None:
                    source = "depth_fallback"

            class_name = self.names.get(int(cls_id), str(cls_id))
            if xyz is None:
                self.draw_box(frame, bbox, f"{class_name} {conf:.2f} no-PC", (0, 0, 255))
                continue

            forward, left, up, point_count, spread = xyz
            distance = math.hypot(forward, left)
            map_x, map_y = self.body_to_map(
                forward,
                left,
                snapshot.pose_x,
                snapshot.pose_y,
                snapshot.heading,
            )
            det = Detection3D(
                class_id=int(cls_id),
                class_name=class_name,
                confidence=float(conf),
                bbox=bbox,
                forward=forward,
                left=left,
                up=up,
                distance=distance,
                point_count=point_count,
                forward_spread=spread,
                source=source,
                map_x=map_x,
                map_y=map_y,
            )
            detections.append(det)
            tof_text = "inf" if not math.isfinite(snapshot.tof_distance) else f"{snapshot.tof_distance:.2f}"
            self.draw_box(
                frame,
                bbox,
                f"{class_name} {conf:.2f} PC={distance:.2f}m ToF={tof_text} n={point_count}",
                (0, 255, 255),
            )

        return InferenceResult(
            frame=frame,
            image_header=snapshot.image_msg.header,
            boxes2d=len(boxes),
            detections=detections,
            no3d=max(0, len(boxes) - len(detections)),
            max_raw_conf=max_raw_conf,
            cloud_rx_age=cloud_age,
            depth_rx_age=depth_age,
            cloud_stamp_delta=snapshot.cloud_stamp_delta,
            depth_stamp_delta=snapshot.depth_stamp_delta,
        )

    def update_tracks(self, detections: Sequence[Detection3D]) -> List[StableTrack]:
        now = time.monotonic()
        for det in detections:
            nearest: Optional[StableTrack] = None
            nearest_distance = float("inf")
            for track in self.tracks:
                distance = math.hypot(det.map_x - track.map_x, det.map_y - track.map_y)
                if distance <= self.track_merge_radius and distance < nearest_distance:
                    nearest = track
                    nearest_distance = distance
            if nearest is None:
                self.tracks.append(
                    StableTrack(
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
                )
                self.next_track_id += 1
            else:
                nearest.observations += 1
                nearest.last_seen = now
                nearest.confidence = max(nearest.confidence, det.confidence)
                nearest.forward = 0.7 * nearest.forward + 0.3 * det.forward
                nearest.left = 0.7 * nearest.left + 0.3 * det.left
                nearest.samples.append((det.map_x, det.map_y))
                nearest.samples = nearest.samples[-15:]
                samples = np.asarray(nearest.samples, dtype=np.float32)
                nearest.map_x = float(np.median(samples[:, 0]))
                nearest.map_y = float(np.median(samples[:, 1]))

        self.tracks = [track for track in self.tracks if now - track.last_seen <= self.track_ttl]
        return [track for track in self.tracks if track.observations >= self.stable_observations]

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
        marker_array = MarkerArray()
        delete_all = Marker()
        delete_all.header.frame_id = "map"
        delete_all.header.stamp = self.get_clock().now().to_msg()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        for track in tracks:
            sphere = Marker()
            sphere.header.frame_id = "map"
            sphere.header.stamp = self.get_clock().now().to_msg()
            sphere.ns = "yolo_tree_v4"
            sphere.id = track.track_id * 2
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
            marker_array.markers.append(sphere)

            text = Marker()
            text.header = sphere.header
            text.ns = "yolo_tree_v4_text"
            text.id = track.track_id * 2 + 1
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
                f"YOLO tree id={track.track_id} obs={track.observations} "
                f"conf={track.confidence:.2f}"
            )
            marker_array.markers.append(text)

        self.marker_pub.publish(marker_array)

    def destroy_node(self):
        self.worker.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloTreeProbeV4()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
