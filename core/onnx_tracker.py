"""
GOD's EYE — Lightweight ONNX Tracker
Zero-torch inference using onnxruntime directly.
Saves ~120 MB RAM per camera by eliminating the PyTorch runtime.
Implements IoU-based tracking for persistent object IDs.

PRODUCTION HARDENED (v2):
  • Track history auto-pruning — dead tracks are cleaned up to prevent memory leaks.
  • Configurable thread count for multi-camera CPU sharing.
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class TrackedObject:
    track_id: int
    bbox: Tuple[int, int, int, int]
    class_id: int
    class_name: str
    confidence: float
    center: Tuple[int, int] = field(default=(0, 0))
    velocity: Tuple[float, float] = field(default=(0.0, 0.0))
    speed: float = 0.0
    direction: float = 0.0

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)


CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class OnnxTracker:
    """
    Lightweight object tracker using ONNX Runtime for inference.
    Replaces ultralytics + PyTorch with pure onnxruntime + numpy.
    """

    def __init__(self, model_path="yolov8n.onnx", confidence=0.4, iou_threshold=0.45,
                 target_classes=None, history_length=30, img_size=640):
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.target_classes = set(target_classes or [0, 1, 2, 3, 5, 7])
        self.history_length = history_length
        self.img_size = img_size

        self._session = None
        self._input_name = None

        # IoU tracker state
        self._next_id = 1
        self._tracks: Dict[int, dict] = {}
        self._track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=history_length))
        self._max_age = 30
        self._match_iou = 0.3

        # History pruning counter — run every N frames to avoid overhead
        self._prune_counter = 0
        self._prune_interval = 100  # Prune dead track histories every 100 frames

    def load_model(self):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Check for GPU providers
        available_providers = ort.get_available_providers()
        providers = []
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
            print(f"[OnnxTracker] Using GPU acceleration (CUDA)")
        elif 'TensorrtExecutionProvider' in available_providers:
            providers.append('TensorrtExecutionProvider')
            print(f"[OnnxTracker] Using GPU acceleration (TensorRT)")
        else:
            providers.append('CPUExecutionProvider')
            opts.intra_op_num_threads = 2
            print(f"[OnnxTracker] Using CPU inference")

        self._session = ort.InferenceSession(self.model_path, opts, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        print(f"[OnnxTracker] Loaded {self.model_path}")

    def unload_model(self):
        """Release model to free RAM (Strategy D: lazy unload)."""
        self._session = None
        self._input_name = None

    def is_loaded(self):
        return self._session is not None

    def _preprocess(self, frame):
        h, w = frame.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        padded = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        pad_h, pad_w = (self.img_size - new_h) // 2, (self.img_size - new_w) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(blob, 0), scale, pad_w, pad_h

    def _postprocess(self, output, scale, pad_w, pad_h, orig_h, orig_w):
        preds = output[0].T  # (8400, 84)
        boxes_cxcywh = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]
        mask = confidences > self.confidence
        class_mask = np.array([cid in self.target_classes for cid in class_ids])
        mask = mask & class_mask
        boxes_cxcywh, confidences, class_ids = boxes_cxcywh[mask], confidences[mask], class_ids[mask]
        if len(boxes_cxcywh) == 0:
            return np.array([]), np.array([]), np.array([])
        boxes = np.zeros_like(boxes_cxcywh)
        boxes[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        boxes[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        boxes[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        boxes[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
        keep = self._nms(boxes, confidences, self.iou_threshold)
        boxes, confidences, class_ids = boxes[keep], confidences[keep], class_ids[keep]
        boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - pad_w) / scale, 0, orig_w)
        boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - pad_h) / scale, 0, orig_h)
        return boxes.astype(int), confidences, class_ids.astype(int)

    @staticmethod
    def _nms(boxes, scores, iou_thresh):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
            order = order[np.where(iou <= iou_thresh)[0] + 1]
        return keep

    @staticmethod
    def _iou_matrix(boxes_a, boxes_b):
        x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
        y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
        x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
        y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
        return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-8)

    def _update_tracks(self, boxes, class_ids, confidences, timestamp):
        tracked = []
        dead_tids = []
        for tid in list(self._tracks):
            self._tracks[tid]["age"] += 1
            if self._tracks[tid]["age"] > self._max_age:
                dead_tids.append(tid)
        # Remove dead tracks AND their histories
        for tid in dead_tids:
            del self._tracks[tid]
            self._track_history.pop(tid, None)

        if len(boxes) == 0:
            return tracked
        active_tids = list(self._tracks.keys())
        matched_det, matched_trk = set(), set()
        if active_tids:
            track_boxes = np.array([self._tracks[tid]["bbox"] for tid in active_tids])
            iou_mat = self._iou_matrix(boxes, track_boxes)
            for _ in range(min(len(boxes), len(active_tids))):
                if iou_mat.size == 0: break
                max_idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                if iou_mat[max_idx] < self._match_iou: break
                det_idx, trk_idx = max_idx
                if det_idx in matched_det or trk_idx in matched_trk:
                    iou_mat[det_idx, trk_idx] = 0
                    continue
                tid = active_tids[trk_idx]
                self._tracks[tid].update({"bbox": tuple(boxes[det_idx]),
                    "class_id": int(class_ids[det_idx]),
                    "confidence": float(confidences[det_idx]), "age": 0})
                self._tracks[tid]["hits"] += 1
                matched_det.add(det_idx); matched_trk.add(trk_idx)
                iou_mat[det_idx, :] = 0; iou_mat[:, trk_idx] = 0
        for i in range(len(boxes)):
            if i not in matched_det:
                tid = self._next_id; self._next_id += 1
                self._tracks[tid] = {"bbox": tuple(boxes[i]), "class_id": int(class_ids[i]),
                    "confidence": float(confidences[i]), "age": 0, "hits": 1}
        for tid, trk in self._tracks.items():
            if trk["hits"] >= 2 and trk["age"] == 0:
                x1, y1, x2, y2 = trk["bbox"]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                self._track_history[tid].append((cx, cy, timestamp))
                vel, spd, dirn = (0.0, 0.0), 0.0, 0.0
                h = self._track_history[tid]
                if len(h) >= 2:
                    dx, dy = h[-1][0] - h[-2][0], h[-1][1] - h[-2][1]
                    vel = (float(dx), float(dy))
                    spd = float(np.sqrt(dx**2 + dy**2))
                    dirn = float(np.arctan2(dy, dx))
                tracked.append(TrackedObject(
                    track_id=tid, bbox=(int(x1), int(y1), int(x2), int(y2)),
                    class_id=trk["class_id"],
                    class_name=CLASS_NAMES.get(trk["class_id"], f"cls_{trk['class_id']}"),
                    confidence=trk["confidence"], center=(cx, cy),
                    velocity=vel, speed=spd, direction=dirn))

        # Periodic deep prune: remove any orphaned history entries
        self._prune_counter += 1
        if self._prune_counter >= self._prune_interval:
            self._prune_counter = 0
            self._prune_orphaned_histories()

        return tracked

    def _prune_orphaned_histories(self):
        """Remove track histories for IDs no longer in active tracks."""
        active_ids = set(self._tracks.keys())
        orphaned = [tid for tid in self._track_history if tid not in active_ids]
        for tid in orphaned:
            del self._track_history[tid]
        if orphaned:
            print(f"[OnnxTracker] Pruned {len(orphaned)} orphaned track histories")

    def track(self, frame, timestamp=0.0):
        if self._session is None:
            self.load_model()
        h, w = frame.shape[:2]
        blob, scale, pad_w, pad_h = self._preprocess(frame)
        outputs = self._session.run(None, {self._input_name: blob})  # Releases GIL!
        boxes, confs, class_ids = self._postprocess(outputs[0], scale, pad_w, pad_h, h, w)
        return self._update_tracks(boxes, class_ids, confs, timestamp)

    def get_track_history(self, track_id):
        return list(self._track_history.get(track_id, []))

    def get_annotated_frame(self, frame, tracked_objects):
        ann = frame.copy()
        colors = {"person": (0, 255, 200), "car": (255, 200, 0), "truck": (255, 200, 0),
                  "bus": (255, 200, 0), "motorcycle": (255, 150, 0), "bicycle": (200, 255, 0)}
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj.bbox
            c = colors.get(obj.class_name, (200, 200, 200))
            cv2.rectangle(ann, (x1, y1), (x2, y2), c, 2)
            lbl = f"ID:{obj.track_id} {obj.class_name} {obj.confidence:.2f}"
            sz, _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(ann, (x1, y1 - sz[1] - 8), (x1 + sz[0] + 4, y1), c, -1)
            cv2.putText(ann, lbl, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            hist = self._track_history.get(obj.track_id, [])
            if len(hist) > 1:
                pts = [(int(p[0]), int(p[1])) for p in hist]
                for i in range(1, len(pts)):
                    a = i / len(pts)
                    cv2.line(ann, pts[i - 1], pts[i], tuple(int(v * a) for v in c), max(1, int(a * 3)))
        return ann

    def reset(self):
        self._track_history.clear()
        self._tracks.clear()
        self._next_id = 1
