"""
GOD's EYE — Risk Scorer
Weighted crowd risk scoring + anomaly detection using z-score.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass


@dataclass
class RiskScore:
    risk_score: float = 0.0
    anomaly_score: float = 0.0
    risk_level: str = "low"  # low, medium, high


class RiskScorer:
    def __init__(self, w_density=0.4, w_speed_var=0.3, w_direction_ent=0.3,
                 high_threshold=0.7, medium_threshold=0.4,
                 anomaly_z_threshold=2.0, rolling_window=50):
        self.w1 = w_density
        self.w2 = w_speed_var
        self.w3 = w_direction_ent
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.anomaly_z = anomaly_z_threshold
        self._history = deque(maxlen=rolling_window)
        # Running normalization
        self._density_max = 1.0
        self._speed_var_max = 1.0
        self._entropy_max = 1.0

    def score(self, features) -> RiskScore:
        # Update running maxes for normalization
        self._density_max = max(self._density_max, features.crowd_density + 0.01)
        self._speed_var_max = max(self._speed_var_max, features.speed_variance + 0.01)
        self._entropy_max = max(self._entropy_max, features.direction_entropy + 0.01)

        # Normalize to [0, 1]
        d = min(1.0, features.crowd_density / self._density_max)
        s = min(1.0, features.speed_variance / self._speed_var_max)
        e = min(1.0, features.direction_entropy / self._entropy_max)

        # Weighted risk
        risk = self.w1 * d + self.w2 * s + self.w3 * e
        risk = float(np.clip(risk, 0, 1))

        # Anomaly detection via z-score
        self._history.append(risk)
        anomaly = 0.0
        if len(self._history) > 10:
            mean = np.mean(self._history)
            std = np.std(self._history) + 1e-8
            anomaly = (risk - mean) / std

        # Risk level
        if risk >= self.high_threshold:
            level = "high"
        elif risk >= self.medium_threshold:
            level = "medium"
        else:
            level = "low"

        return RiskScore(risk_score=risk, anomaly_score=float(anomaly), risk_level=level)
