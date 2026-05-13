"""
GOD's EYE — Prediction Preprocessor
Data loading, feature scaling, and sequence generation for LSTM.
"""

import numpy as np
import pandas as pd
import json
import os
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List


class PredictionPreprocessor:
    FEATURE_COLS = ["people_count", "vehicle_count", "crowd_density",
                    "avg_speed", "speed_variance", "risk_score"]
    TARGET_COL = "risk_score"

    def __init__(self, sequence_length=15):
        self.sequence_length = sequence_length
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

    def load_from_db(self, db, camera_id="cam_01", limit=500):
        rows = db.get_recent_metrics(camera_id=camera_id, limit=limit)
        if not rows:
            return None
        return pd.DataFrame(rows)

    def load_from_csv(self, filepath):
        return pd.read_csv(filepath)

    def prepare_data(self, df) -> Tuple:
        """Prepare scaled sequences and targets for LSTM training."""
        cols_available = [c for c in self.FEATURE_COLS if c in df.columns]
        if not cols_available or len(df) < self.sequence_length + 1:
            return None, None, None, None

        features = df[cols_available].values.astype(np.float32)
        target = df[self.TARGET_COL].values.astype(np.float32).reshape(-1, 1)

        # Scale
        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.target_scaler.fit_transform(target)

        # Create sequences
        X, y = [], []
        for i in range(len(features_scaled) - self.sequence_length):
            X.append(features_scaled[i:i + self.sequence_length])
            y.append(target_scaled[i + self.sequence_length])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # Train/test split (80/20)
        split = int(len(X) * 0.8)
        return X[:split], y[:split], X[split:], y[split:]

    def prepare_inference_sequence(self, recent_data: List[dict]):
        """Prepare a single sequence from recent data for inference."""
        if len(recent_data) < self.sequence_length:
            return None

        cols = [c for c in self.FEATURE_COLS if c in recent_data[0]]
        data = recent_data[-self.sequence_length:]
        features = np.array([[row.get(c, 0) for c in cols] for row in data], dtype=np.float32)

        # Use fitted scaler if available, otherwise just normalize
        try:
            features_scaled = self.feature_scaler.transform(features)
        except Exception:
            features_scaled = features / (features.max(axis=0) + 1e-8)

        return features_scaled.reshape(1, self.sequence_length, len(cols))

    def inverse_scale_target(self, value):
        try:
            return float(self.target_scaler.inverse_transform([[value]])[0][0])
        except Exception:
            return float(value)

    @staticmethod
    def generate_synthetic_data(n_points=500, camera_id="cam_01"):
        """Generate synthetic time-series data for demo/training."""
        np.random.seed(42)
        timestamps = pd.date_range("2025-01-01", periods=n_points, freq="5min")
        data = []
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            # Simulate daily patterns
            base_people = 20 + 30 * np.sin(np.pi * hour / 12) + np.random.randn() * 5
            base_vehicles = 10 + 15 * np.sin(np.pi * hour / 12) + np.random.randn() * 3
            density = max(0, min(1, base_people / 80 + np.random.randn() * 0.05))
            speed = max(0, 5 + np.random.randn() * 2)
            speed_var = max(0, np.random.exponential(1.5))
            risk = 0.4 * density + 0.3 * min(1, speed_var/5) + 0.3 * np.random.random() * 0.3
            event = 1 if risk > 0.6 and np.random.random() > 0.7 else 0
            data.append({
                "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
                "camera_id": camera_id,
                "people_count": int(max(0, base_people)),
                "vehicle_count": int(max(0, base_vehicles)),
                "incident_count": event,
                "crowd_density": round(density, 3),
                "avg_speed": round(speed, 2),
                "speed_variance": round(speed_var, 3),
                "direction_entropy": round(np.random.random() * 0.5, 3),
                "risk_score": round(min(1, max(0, risk)), 3),
                "event_flag": event
            })
        return pd.DataFrame(data)
