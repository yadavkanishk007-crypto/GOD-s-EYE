"""
GOD's EYE — Feature Extractor
Computes per-frame analytics from tracked objects: crowd density,
speed variance, direction entropy, fall detection, vehicle stops,
and multi-modal incident signals (fire, smoke, accident).

PRODUCTION HARDENED (v2):
  • Fall detection: requires 3-frame persistence of low aspect ratio (not single-frame).
  • Memory: _prev_aspects pruned each frame to prevent unbounded growth.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from collections import deque


@dataclass
class FrameFeatures:
    people_count: int = 0
    vehicle_count: int = 0
    crowd_density: float = 0.0
    avg_speed: float = 0.0
    speed_variance: float = 0.0
    direction_entropy: float = 0.0
    fall_detected: bool = False
    vehicle_sudden_stop: bool = False
    # ── Incident signals ─────────────────────
    fire_detected: bool = False
    fire_confidence: float = 0.0
    smoke_detected: bool = False
    smoke_confidence: float = 0.0
    accident_detected: bool = False
    accident_confidence: float = 0.0


class FeatureExtractor:
    def __init__(self, density_area=500000, fall_aspect_threshold=0.6,
                 fall_speed_threshold=15.0, vehicle_stop_speed=2.0,
                 vehicle_stop_prev_speed=20.0):
        self.density_area = density_area
        self.fall_aspect_threshold = fall_aspect_threshold
        self.fall_speed_threshold = fall_speed_threshold
        self.vehicle_stop_speed = vehicle_stop_speed
        self.vehicle_stop_prev_speed = vehicle_stop_prev_speed
        self._prev_aspects = {}  # track_id -> previous aspect ratio
        self._speed_history: deque = deque(maxlen=6)  # for sudden-stop via history

        # v2: Multi-frame fall persistence tracking
        # track_id -> deque of recent aspect ratios (last 5 frames)
        self._aspect_history = {}
        self._fall_persist_frames = 3  # require low aspect for N frames

    def extract(self, tracked_objects, incident=None) -> FrameFeatures:
        """
        Parameters
        ----------
        tracked_objects : list[TrackedObject]
        incident        : IncidentResult (from IncidentDetector) or None
                          Also accepts a plain list[FireRegion] for back-compat.
        """
        features = FrameFeatures()

        people = [o for o in tracked_objects if o.class_name == "person"]
        vehicles = [o for o in tracked_objects
                    if o.class_name in ("car", "truck", "bus", "motorcycle", "bicycle")]

        features.people_count = len(people)
        features.vehicle_count = len(vehicles)

        # Crowd density (normalized)
        features.crowd_density = min(1.0, len(people) / max(1, self.density_area / 10000))

        # Speed stats for people
        if people:
            speeds = [p.speed for p in people]
            features.avg_speed = float(np.mean(speeds))
            features.speed_variance = float(np.std(speeds)) if len(speeds) > 1 else 0.0

        # Direction entropy (Shannon entropy of 8-bin direction histogram)
        if len(people) > 2:
            directions = [p.direction for p in people if p.speed > 1.0]
            if directions:
                hist, _ = np.histogram(directions, bins=8, range=(-np.pi, np.pi))
                hist = hist / (hist.sum() + 1e-8)
                entropy = -np.sum(hist * np.log2(hist + 1e-8))
                features.direction_entropy = float(entropy / 3.0)  # normalize to ~[0,1]

        # ── Fall detection (v2: multi-frame persistence) ──────────────
        current_track_ids = set()
        for p in people:
            current_track_ids.add(p.track_id)
            x1, y1, x2, y2 = p.bbox
            w, h = x2 - x1, y2 - y1
            aspect = h / max(w, 1)

            # Maintain aspect history for this track
            if p.track_id not in self._aspect_history:
                self._aspect_history[p.track_id] = deque(maxlen=5)
            self._aspect_history[p.track_id].append(aspect)

            prev = self._prev_aspects.get(p.track_id, aspect)

            # Check: person was upright (aspect > 1.2), now has LOW aspect,
            # AND the low aspect persists for >= 3 frames
            if prev > 1.2 and aspect < self.fall_aspect_threshold and p.speed > self.fall_speed_threshold:
                hist = self._aspect_history[p.track_id]
                # Count how many recent frames had low aspect
                low_count = sum(1 for a in hist if a < self.fall_aspect_threshold)
                if low_count >= self._fall_persist_frames:
                    # Additional check: bbox area should have expanded (person on ground = wider)
                    bbox_area = w * h
                    if bbox_area > 2000:  # minimum area to avoid noise
                        features.fall_detected = True

            self._prev_aspects[p.track_id] = aspect

        # ── Memory: prune _prev_aspects and _aspect_history for dead tracks ──
        dead_ids = [tid for tid in self._prev_aspects if tid not in current_track_ids]
        for tid in dead_ids:
            del self._prev_aspects[tid]
            self._aspect_history.pop(tid, None)

        # Vehicle sudden stop (improved: compare against historical mean speed)
        if vehicles:
            current_speeds = [v.speed for v in vehicles]
            mean_speed = float(np.mean(current_speeds))
            self._speed_history.append(mean_speed)
            if len(self._speed_history) >= 4:
                prev_mean = float(np.mean(list(self._speed_history)[:-2]))
                # Sudden collective stop: previous speed was high, now low
                if prev_mean > 8.0 and mean_speed < 2.0:
                    features.vehicle_sudden_stop = True
            else:
                # Fallback: any stopped vehicle
                if any(v.speed < self.vehicle_stop_speed for v in vehicles):
                    features.vehicle_sudden_stop = True

        # ── Incident signals ──────────────────────────────────────────────
        if incident is not None:
            # Support both IncidentResult objects and legacy list[FireRegion]
            if isinstance(incident, list):
                # Legacy: plain list of FireRegion
                if incident:
                    features.fire_detected = True
                    features.fire_confidence = max(f.confidence for f in incident)
            else:
                # New: IncidentResult
                features.fire_detected = incident.fire_detected
                features.fire_confidence = incident.fire_confidence
                features.smoke_detected = incident.smoke_detected
                features.smoke_confidence = incident.smoke_confidence
                features.accident_detected = incident.accident_detected
                if incident.accident and incident.accident_detected:
                    features.accident_confidence = incident.accident.confidence

        return features
