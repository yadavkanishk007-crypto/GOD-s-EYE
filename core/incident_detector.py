"""
GOD's EYE — Advanced Incident Detector  v3
Multi-modal, CPU-only detection for Fire, Smoke, and Traffic Accidents.

FALSE-POSITIVE HARDENING (v3 fixes — on top of v2):
  • Frame-area ratio check — sunsets cover >15% of frame, real fires rarely do.
  • Sky-region bias — fire covering the top 40% of frame = sunset, not fire.
  • Accident: speed-before-stop requirement — vehicles must have been moving fast
    before stopping to trigger cluster_stop evidence.
  • Accident: cluster_stop alone cannot pair with flow_spike — requires vehicle_overlap.
  • Optical flow only runs every 4th call to halve CPU load.

All methods produce a confidence ∈ [0, 1] that is fused via a weighted sum.
"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class FireRegion:
    bbox: Tuple[int, int, int, int]
    area: float
    confidence: float


@dataclass
class SmokeRegion:
    bbox: Tuple[int, int, int, int]
    area: float
    confidence: float


@dataclass
class AccidentSignal:
    """Aggregated accident evidence for this frame."""
    confidence: float                    # 0-1
    bbox: Optional[Tuple[int, int, int, int]] = None  # approximate location
    description: str = ""


@dataclass
class IncidentResult:
    fire_regions: List[FireRegion] = field(default_factory=list)
    smoke_regions: List[SmokeRegion] = field(default_factory=list)
    accident: Optional[AccidentSignal] = None

    @property
    def fire_detected(self) -> bool:
        return len(self.fire_regions) > 0

    @property
    def smoke_detected(self) -> bool:
        return len(self.smoke_regions) > 0

    @property
    def accident_detected(self) -> bool:
        return self.accident is not None and self.accident.confidence >= 0.55

    @property
    def fire_confidence(self) -> float:
        return max((r.confidence for r in self.fire_regions), default=0.0)

    @property
    def smoke_confidence(self) -> float:
        return max((r.confidence for r in self.smoke_regions), default=0.0)


# ─────────────────────────────────────────────
# Main detector class
# ─────────────────────────────────────────────

class IncidentDetector:
    """
    Multi-modal incident detector with strong false-positive suppression.
    v3: Frame-area gating, sky-bias rejection, speed-before-stop for accidents.
    """

    # ── HSV bounds ──────────────────────────────────────────────────
    # Fire: narrow orange-yellow band (H 8-25).  H < 8 catches red signs/buses.
    FIRE_HSV_LOWER  = np.array([8,   150, 200], np.uint8)
    FIRE_HSV_UPPER  = np.array([25,  255, 255], np.uint8)
    # Deep-red wrap: H 170-179 at very high saturation + value only
    FIRE_HSV_LOWER2 = np.array([170, 180, 220], np.uint8)
    FIRE_HSV_UPPER2 = np.array([179, 255, 255], np.uint8)

    # Smoke: very low saturation, medium value (gray haze)
    SMOKE_HSV_LOWER = np.array([0,   0,  130], np.uint8)
    SMOKE_HSV_UPPER = np.array([179, 40, 200], np.uint8)

    FIRE_MIN_AREA   = 3000    # pixels² — raised from 1500 to filter small red objects
    SMOKE_MIN_AREA  = 8000    # raised from 5000

    # Temporal persistence thresholds
    FIRE_PERSIST_FRAMES  = 4   # fire must be seen in N of last M frames
    SMOKE_PERSIST_FRAMES = 5

    # v3: Frame-area ratio caps — fire/smoke covering > this fraction = false positive
    FIRE_MAX_FRAME_RATIO  = 0.15   # real fire rarely covers >15% of frame
    SMOKE_MAX_FRAME_RATIO = 0.40   # smoke can be larger but cap at 40%

    def __init__(self,
                 temporal_window: int = 12,
                 accident_flow_thresh: float = 22.0,
                 accident_overlap_thresh: float = 0.30):
        self._temporal_window   = temporal_window
        self._flow_thresh       = accident_flow_thresh
        self._overlap_thresh    = accident_overlap_thresh

        # Temporal buffers
        self._brightness_hist: deque  = deque(maxlen=temporal_window)
        self._fire_hit_hist: deque    = deque(maxlen=temporal_window)   # bool per frame
        self._smoke_hit_hist: deque   = deque(maxlen=temporal_window)
        self._flow_hist: deque        = deque(maxlen=temporal_window)
        self._vehicle_stop_hist: deque = deque(maxlen=temporal_window)
        # v3: Track vehicle speeds over time for speed-before-stop check
        self._vehicle_speed_hist: deque = deque(maxlen=temporal_window)

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_fire_mask: Optional[np.ndarray] = None

        # Morphological kernel
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Optical flow frame counter (run only every 4th call to save CPU)
        self._detect_call_count = 0

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def detect(self, frame: np.ndarray,
               tracked_objects=None) -> IncidentResult:
        self._detect_call_count += 1
        result = IncidentResult()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        objects = tracked_objects or []

        # Build YOLO occupancy mask — any pixel covered by a known object
        yolo_mask = self._build_yolo_mask(frame.shape[:2], objects)

        # 1. Fire (with YOLO suppression + temporal gating + area-ratio check)
        result.fire_regions = self._detect_fire(frame, hsv, gray, yolo_mask)

        # 2. Smoke (with position bias + temporal gating)
        result.smoke_regions = self._detect_smoke(frame, hsv, gray, yolo_mask)

        # 3. Accident (optical flow every 4th call only)
        run_flow = (self._detect_call_count % 4 == 0)
        result.accident = self._detect_accident(gray, objects, run_flow)

        self._prev_gray = gray
        return result

    # ──────────────────────────────────────────
    # YOLO occupancy mask
    # ──────────────────────────────────────────

    @staticmethod
    def _build_yolo_mask(shape: Tuple[int, int], objects) -> np.ndarray:
        """Binary mask where 255 = covered by a YOLO-detected object."""
        mask = np.zeros(shape, dtype=np.uint8)
        for obj in objects:
            x1, y1, x2, y2 = obj.bbox
            # Shrink bbox slightly (80%) to avoid edge noise
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            hw, hh = int((x2 - x1) * 0.4), int((y2 - y1) * 0.4)
            mask[max(0, cy - hh):cy + hh, max(0, cx - hw):cx + hw] = 255
        return mask

    # ──────────────────────────────────────────
    # Fire detection
    # ──────────────────────────────────────────

    def _detect_fire(self, frame, hsv, gray, yolo_mask) -> List[FireRegion]:
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w

        # HSV masks
        mask = cv2.inRange(hsv, self.FIRE_HSV_LOWER, self.FIRE_HSV_UPPER)
        mask2 = cv2.inRange(hsv, self.FIRE_HSV_LOWER2, self.FIRE_HSV_UPPER2)
        mask = cv2.bitwise_or(mask, mask2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel)

        # ── YOLO suppression: subtract pixels covered by known objects ──
        mask = cv2.subtract(mask, yolo_mask)

        # ── v3: Total fire-pixel area ratio check ──
        # If fire-colored pixels cover >15% of frame → sunset/billboard, not fire
        total_fire_pixels = np.count_nonzero(mask)
        if total_fire_pixels / frame_area > self.FIRE_MAX_FRAME_RATIO:
            self._fire_hit_hist.append(False)
            return []

        # ── Motion-in-region check ──
        motion_ratio = 0.0
        if self._prev_fire_mask is not None and self._prev_fire_mask.shape == mask.shape:
            diff = cv2.absdiff(mask, self._prev_fire_mask)
            fire_pixels = max(1, np.count_nonzero(mask))
            changed_pixels = np.count_nonzero(diff)
            motion_ratio = changed_pixels / fire_pixels
        self._prev_fire_mask = mask.copy()

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.FIRE_MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(c)

            # ── v3: Sky-region bias — fire in top 30% of frame = sunset ──
            center_y = y + h // 2
            if center_y < frame_h * 0.30 and w > frame_w * 0.3:
                # Wide region in the sky → sunset, not fire
                continue

            # ── Shape check: fire tends to be irregular ──
            perimeter = cv2.arcLength(c, True)
            circularity = (4 * np.pi * area) / (perimeter * perimeter + 1e-8)
            rect_area = w * h
            compactness = area / (rect_area + 1e-8)
            if compactness > 0.85:
                continue

            # ── Motion requirement: static red objects don't flicker ──
            if motion_ratio < 0.05:
                continue

            # Base confidence from area (capped lower than before)
            base_conf = min(0.50, area / 30000.0)

            # Flicker boost
            brightness = float(np.mean(gray[y:y+h, x:x+w]))
            self._brightness_hist.append(brightness)
            flicker = float(np.std(self._brightness_hist)) / 255.0 if len(self._brightness_hist) > 5 else 0.0
            flicker_boost = min(0.20, flicker * 0.3)

            # Saturation boost (fire = high saturation)
            roi_hsv = hsv[y:y+h, x:x+w]
            avg_sat = float(np.mean(roi_hsv[:, :, 1])) / 255.0
            sat_boost = max(0.0, (avg_sat - 0.5) * 0.2)

            conf = min(1.0, base_conf + flicker_boost + sat_boost)
            candidates.append(FireRegion(bbox=(x, y, x+w, y+h), area=area, confidence=conf))

        # ── Temporal gating: require persistence ──
        self._fire_hit_hist.append(len(candidates) > 0)
        recent_hits = sum(self._fire_hit_hist)
        if recent_hits < self.FIRE_PERSIST_FRAMES:
            return []

        return candidates

    # ──────────────────────────────────────────
    # Smoke detection
    # ──────────────────────────────────────────

    def _detect_smoke(self, frame, hsv, gray, yolo_mask) -> List[SmokeRegion]:
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w

        mask = cv2.inRange(hsv, self.SMOKE_HSV_LOWER, self.SMOKE_HSV_UPPER)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        # Subtract YOLO-known objects (gray buildings ≠ smoke)
        mask = cv2.subtract(mask, yolo_mask)

        # ── v3: Total smoke-pixel area ratio check ──
        total_smoke_pixels = np.count_nonzero(mask)
        if total_smoke_pixels / frame_area > self.SMOKE_MAX_FRAME_RATIO:
            self._smoke_hit_hist.append(False)
            return []

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.SMOKE_MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(c)

            # ── Position bias: smoke rises → should be in upper half of frame ──
            center_y = y + h // 2
            if center_y > frame_h * 0.7:
                continue

            # ── Shape check: smoke is amorphous, not rectangular ──
            rect_area = w * h
            compactness = area / (rect_area + 1e-8)
            if compactness > 0.80:
                continue

            # Texture: smoke is blurry/flat
            roi_gray = gray[y:y+h, x:x+w]
            laplacian_var = float(cv2.Laplacian(roi_gray, cv2.CV_64F).var())
            if laplacian_var > 300:
                continue
            texture_score = max(0.0, 1.0 - laplacian_var / 300.0)

            conf = min(1.0, (area / 50000.0) * 0.5 + texture_score * 0.5)
            if conf > 0.25:
                candidates.append(SmokeRegion(bbox=(x, y, x+w, y+h), area=area, confidence=conf))

        # ── Temporal gating ──
        self._smoke_hit_hist.append(len(candidates) > 0)
        recent_hits = sum(self._smoke_hit_hist)
        if recent_hits < self.SMOKE_PERSIST_FRAMES:
            return []

        return candidates

    # ──────────────────────────────────────────
    # Accident detection
    # ──────────────────────────────────────────

    def _detect_accident(self, gray, tracked_objects, run_flow: bool) -> Optional[AccidentSignal]:
        evidences = {}

        # ── A. Optical-flow magnitude spike (only every 4th frame) ───────
        if run_flow and self._prev_gray is not None:
            h, w = gray.shape
            # Resize to 1/4 resolution for speed
            small_prev = cv2.resize(self._prev_gray, (w // 4, h // 4))
            small_curr = cv2.resize(gray,            (w // 4, h // 4))
            flow = cv2.calcOpticalFlowFarneback(
                small_prev, small_curr, None,
                0.5, 2, 10, 3, 5, 1.2, 0   # fewer pyramid levels, smaller window
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            top_mag = float(np.percentile(mag, 97))
            self._flow_hist.append(top_mag)
            if len(self._flow_hist) > 6:
                baseline = float(np.mean(list(self._flow_hist)[:-3]))
                spike = top_mag / (baseline + 1e-3)
                if spike > 3.0 and top_mag > self._flow_thresh:
                    evidences["flow_spike"] = min(1.0, spike / 6.0)
        elif not run_flow:
            pass  # Keep existing flow_hist, just skip computation
        else:
            self._flow_hist.append(0.0)

        # ── B. Vehicle bounding-box overlap ──────────────────────────────
        vehicles = [o for o in tracked_objects
                    if o.class_name in ("car", "truck", "bus", "motorcycle")]
        overlap_conf = 0.0
        crash_bbox = None
        if len(vehicles) >= 2:
            best_iou = 0.0
            for i in range(len(vehicles)):
                for j in range(i + 1, len(vehicles)):
                    iou = self._bbox_iou(vehicles[i].bbox, vehicles[j].bbox)
                    if iou > best_iou:
                        best_iou = iou
                        x1 = min(vehicles[i].bbox[0], vehicles[j].bbox[0])
                        y1 = min(vehicles[i].bbox[1], vehicles[j].bbox[1])
                        x2 = max(vehicles[i].bbox[2], vehicles[j].bbox[2])
                        y2 = max(vehicles[i].bbox[3], vehicles[j].bbox[3])
                        crash_bbox = (x1, y1, x2, y2)
            if best_iou > self._overlap_thresh:
                overlap_conf = min(1.0, best_iou / 0.6)
                evidences["vehicle_overlap"] = overlap_conf

        # ── C. Sudden cluster-stop (v3: with speed-before-stop gate) ────
        current_speeds = [v.speed for v in vehicles] if vehicles else []
        mean_speed = float(np.mean(current_speeds)) if current_speeds else 0.0
        self._vehicle_speed_hist.append(mean_speed)

        stopped = sum(1 for v in vehicles if v.speed < 1.5)
        self._vehicle_stop_hist.append(stopped)
        if len(self._vehicle_stop_hist) >= 8 and len(self._vehicle_speed_hist) >= 8:
            prev_stopped = list(self._vehicle_stop_hist)[-8]
            delta_stopped = stopped - prev_stopped

            # v3: Speed-before-stop gate — vehicles must have been moving fast
            # in the recent past before the sudden stop counts as evidence.
            recent_speeds = list(self._vehicle_speed_hist)[-8:-2]
            prev_avg_speed = float(np.mean(recent_speeds)) if recent_speeds else 0.0

            if delta_stopped >= 4 and stopped >= 4 and prev_avg_speed > 12.0:
                evidences["cluster_stop"] = min(1.0, delta_stopped / 8.0)

        # ── v3: Require ≥2 evidence channels, with quality gate ──────────
        # cluster_stop alone is weak — require it to pair with vehicle_overlap,
        # NOT with flow_spike (which can be triggered by any large movement).
        if len(evidences) < 2:
            return None

        # If only cluster_stop + flow_spike (no overlap), suppress.
        evidence_keys = set(evidences.keys())
        if evidence_keys == {"cluster_stop", "flow_spike"}:
            return None

        # ── Fusion ───────────────────────────────────────────────────────
        weights = {
            "flow_spike":      0.45,
            "vehicle_overlap": 0.35,
            "cluster_stop":    0.20,
        }
        total_w = sum(weights.get(k, 0) for k in evidences)
        fused = sum(evidences[k] * weights.get(k, 0.1) for k in evidences) / (total_w + 1e-8)

        desc_parts = []
        if "flow_spike"      in evidences: desc_parts.append("sudden motion surge")
        if "vehicle_overlap" in evidences: desc_parts.append(f"vehicle overlap IoU={overlap_conf:.2f}")
        if "cluster_stop"    in evidences: desc_parts.append(f"{stopped} vehicles stopped suddenly")

        return AccidentSignal(
            confidence=round(fused, 3),
            bbox=crash_bbox,
            description="; ".join(desc_parts)
        )

    # ──────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────

    @staticmethod
    def _bbox_iou(a: Tuple, b: Tuple) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter + 1e-8)
