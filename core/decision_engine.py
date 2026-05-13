"""
GOD's EYE — Decision Engine
Translates extracted features and risk scores into discrete actionable events.

PRODUCTION HARDENED (v2):
  • Global (per-camera) cooldowns increased significantly to prevent alert fatigue.
  • Short cooldowns cause operators to ignore alerts.
"""

import time
from dataclasses import dataclass
from typing import List


@dataclass
class Event:
    event_type: str
    description: str
    risk_level: str  # "high", "medium", "low"
    camera_id: str
    timestamp: float


class DecisionEngine:
    def __init__(self, crowd_risk_high=0.7, crowd_risk_medium=0.4):
        self.crowd_risk_high = crowd_risk_high
        self.crowd_risk_medium = crowd_risk_medium

        self._last_event_times = {}

        # v2: Increased cooldowns significantly to prevent alert fatigue
        self._cooldowns = {
            "Fire/Smoke Emergency": 30.0,
            "Vehicle Incident": 60.0,
            "Person Fall Detected": 20.0,
            "Anomalous Behavior": 30.0,
            "High Crowd Risk": 30.0,
            "Elevated Crowd Activity": 60.0,
            "Predictive Pre-Alert": 120.0
        }

    def evaluate(self, features, risk, camera_id, prediction_risk=None) -> List[Event]:
        events = []
        now = time.time()

        def add_event(evt_type, desc, level):
            cooldown = self._cooldowns.get(evt_type, 10.0)
            last_time = self._last_event_times.get(evt_type, 0.0)
            if now - last_time > cooldown:
                events.append(Event(evt_type, desc, level, camera_id, now))
                self._last_event_times[evt_type] = now

        # ── 1. Critical Life Safety ────────────────────────────────────
        if features.fire_detected:
            add_event("Fire/Smoke Emergency",
                      f"Fire signature detected ({features.fire_confidence:.0%})", "high")
        elif features.smoke_detected:
            add_event("Fire/Smoke Emergency",
                      f"Smoke signature detected ({features.smoke_confidence:.0%})", "high")

        # ── 2. Severe Incidents ────────────────────────────────────────
        if features.accident_detected:
            add_event("Vehicle Incident",
                      f"Collision/sudden stop detected ({features.accident_confidence:.0%})", "high")

        if features.fall_detected:
            add_event("Person Fall Detected",
                      "Individual detected falling/collapsed", "high")

        # ── 3. Behavioral Anomalies ────────────────────────────────────
        if features.vehicle_sudden_stop and not features.accident_detected:
            add_event("Anomalous Behavior", "Sudden collective vehicle stop", "medium")

        if features.speed_variance > 10.0:
            add_event("Anomalous Behavior", "Chaotic movement patterns detected", "medium")

        # ── 4. Crowd Risk Thresholds ───────────────────────────────────
        if risk.risk_score > self.crowd_risk_high:
            add_event("High Crowd Risk",
                      f"Critical threshold exceeded ({risk.risk_score:.2f})", "high")
        elif risk.risk_score > self.crowd_risk_medium:
            add_event("Elevated Crowd Activity",
                      f"Density/speed forming risk ({risk.risk_score:.2f})", "medium")

        # ── 5. Predictive Pre-Alerts ───────────────────────────────────
        if prediction_risk is not None:
            if prediction_risk > self.crowd_risk_high:
                add_event("Predictive Pre-Alert",
                          f"Forecasted critical risk ({prediction_risk:.2f}) in next 5m", "medium")

        return events
