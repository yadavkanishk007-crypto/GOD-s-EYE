"""
GOD's EYE — Object Detector
YOLOv8n wrapper for lightweight CPU-based object detection.
Detects people, vehicles, and uses HSV heuristic for fire/smoke.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Detection:
    """A single detected object."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float
    center: Tuple[int, int] = field(default=(0, 0))

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class FireRegion:
    """A detected fire/smoke region from HSV analysis."""
    bbox: Tuple[int, int, int, int]
    area: float
    confidence: float


class ObjectDetector:
    """
    YOLOv8n-based object detector with HSV fire detection.
    Runs on CPU for edge deployment.
    """

    # COCO class name mapping for relevant classes
    CLASS_NAMES = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
        5: "bus", 7: "truck"
    }

    def __init__(self, model_path: str = "yolov8n.pt",
                 confidence: float = 0.4,
                 iou_threshold: float = 0.45,
                 device: str = "cpu",
                 target_classes: List[int] = None,
                 fire_config: dict = None):
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device
        self.target_classes = target_classes or [0, 1, 2, 3, 5, 7]

        # Fire detection HSV config
        self.fire_enabled = True
        self.fire_hsv_lower = np.array([0, 120, 200])
        self.fire_hsv_upper = np.array([25, 255, 255])
        self.fire_min_area = 3000
        if fire_config:
            self.fire_enabled = fire_config.get("enabled", True)
            self.fire_hsv_lower = np.array(fire_config.get("hsv_lower", [0, 120, 200]))
            self.fire_hsv_upper = np.array(fire_config.get("hsv_upper", [25, 255, 255]))
            self.fire_min_area = fire_config.get("min_area", 3000)

        self._model = None

    def load_model(self):
        """Load YOLO model (lazy loading — downloads on first run)."""
        from ultralytics import YOLO
        self._model = YOLO(self.model_path)
        # Warm up with a dummy inference
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model(dummy, verbose=False, device=self.device)

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], List[FireRegion]]:
        """
        Run detection on a frame.
        Returns (detections, fire_regions).
        """
        if self._model is None:
            self.load_model()

        detections = []
        fire_regions = []

        # --- YOLO Detection ---
        results = self._model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            classes=self.target_classes,
            verbose=False
        )

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls_id = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    cls_name = self.CLASS_NAMES.get(cls_id, f"class_{cls_id}")

                    detections.append(Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf
                    ))

        # --- HSV Fire Detection ---
        if self.fire_enabled:
            fire_regions = self._detect_fire(frame)

        return detections, fire_regions

    def _detect_fire(self, frame: np.ndarray) -> List[FireRegion]:
        """Detect fire/smoke using HSV color space heuristic."""
        regions = []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.fire_hsv_lower, self.fire_hsv_upper)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.fire_min_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Confidence based on area relative to threshold
                conf = min(1.0, area / (self.fire_min_area * 5))
                regions.append(FireRegion(
                    bbox=(x, y, x + w, y + h),
                    area=area,
                    confidence=conf
                ))

        return regions

    def get_annotated_frame(self, frame: np.ndarray,
                            detections: List[Detection],
                            fire_regions: List[FireRegion],
                            track_ids: dict = None) -> np.ndarray:
        """Draw detection overlays on frame."""
        annotated = frame.copy()

        # Color map for classes
        colors = {
            "person": (0, 255, 200),      # cyan-green
            "car": (255, 200, 0),          # amber
            "truck": (255, 200, 0),
            "bus": (255, 200, 0),
            "motorcycle": (255, 150, 0),
            "bicycle": (200, 255, 0),
        }

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = colors.get(det.class_name, (200, 200, 200))

            # Draw bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"{det.class_name} {det.confidence:.2f}"
            if track_ids and id(det) in track_ids:
                label = f"ID:{track_ids[id(det)]} {label}"

            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 6),
                          (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw fire regions
        for fire in fire_regions:
            x1, y1, x2, y2 = fire.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(annotated, f"FIRE {fire.confidence:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

        return annotated
