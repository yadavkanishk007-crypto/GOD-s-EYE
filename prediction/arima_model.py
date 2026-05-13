"""
GOD's EYE — ARIMA Model
Linear trend and seasonality prediction using statsmodels ARIMA.
"""

import os
import pickle
import warnings
import numpy as np
from typing import Optional

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class ARIMAPredictor:
    def __init__(self, order=(2, 1, 2), model_dir="data/models"):
        self.order = order
        self.model_dir = model_dir
        self._fitted = None
        self._last_series = None
        os.makedirs(model_dir, exist_ok=True)

    def fit(self, series):
        """Fit ARIMA on a 1D time series of risk scores."""
        from statsmodels.tsa.arima.model import ARIMA
        try:
            self._last_series = np.array(series, dtype=np.float64)
            # Clamp to avoid numerical issues
            self._last_series = np.clip(self._last_series, 0, 1)
            model = ARIMA(self._last_series, order=self.order)
            self._fitted = model.fit()
            return True
        except Exception as e:
            print(f"  ARIMA fit warning: {e}")
            self._fitted = None
            return False

    def predict(self, steps=1) -> float:
        """Predict next step(s). Returns value normalized to [0, 1]."""
        if self._fitted is None:
            return 0.5  # fallback
        try:
            forecast = self._fitted.forecast(steps=steps)
            value = float(forecast.iloc[-1]) if hasattr(forecast, 'iloc') else float(forecast[-1])
            return float(np.clip(value, 0, 1))
        except Exception as e:
            print(f"  ARIMA predict warning: {e}")
            return 0.5

    def save_model(self, filename="arima_model.pkl"):
        path = os.path.join(self.model_dir, filename)
        with open(path, "wb") as f:
            pickle.dump({
                "fitted": self._fitted,
                "order": self.order,
                "last_series": self._last_series
            }, f)
        print(f"  ARIMA model saved to {path}")

    def load_model(self, filename="arima_model.pkl") -> bool:
        # Security: Resolve path and ensure file is inside the expected model directory.
        # WARNING: pickle.load() can execute arbitrary code if the .pkl file is replaced
        # by an attacker. This is safe ONLY because data/models/ is local and access-controlled.
        abs_model_dir = os.path.realpath(os.path.abspath(self.model_dir))
        path = os.path.join(abs_model_dir, filename)
        
        # Block path traversal: ensure the resolved path is still inside model_dir
        if not os.path.realpath(path).startswith(abs_model_dir + os.sep):
            print(f"[SECURITY] Path traversal attempt blocked for ARIMA model: {filename}")
            return False

        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)  # noqa: S301 — local file, access-controlled dir
                if not isinstance(data, dict):
                    print("[SECURITY] ARIMA model file format invalid — possible tampering.")
                    return False
                self._fitted = data.get("fitted")
                self.order = data.get("order", self.order)
                self._last_series = data.get("last_series")
                print(f"  ARIMA model loaded from {path}")
                return True
            except Exception as e:
                print(f"[SECURITY] ARIMA model load failed (corrupted/tampered file?): {e}")
                return False
        return False
