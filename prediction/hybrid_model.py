"""
GOD's EYE — Hybrid Prediction Model
Combines LSTM (60%) + ARIMA (40%) for robust risk forecasting.
"""

from dataclasses import dataclass
from typing import Optional
from prediction.lstm_model import LSTMPredictor
from prediction.arima_model import ARIMAPredictor


@dataclass
class PredictionResult:
    predicted_risk: float
    risk_level: str  # low, medium, high
    recommended_action: str
    lstm_prediction: float = 0.0
    arima_prediction: float = 0.0


class HybridPredictor:
    def __init__(self, lstm_weight=0.6, arima_weight=0.4,
                 input_size=6, hidden_size=64, sequence_length=15,
                 arima_order=(2, 1, 2), model_dir="data/models"):
        self.lstm_weight = lstm_weight
        self.arima_weight = arima_weight
        self.lstm = LSTMPredictor(input_size=input_size, hidden_size=hidden_size,
                                  model_dir=model_dir)
        self.arima = ARIMAPredictor(order=arima_order, model_dir=model_dir)
        self.model_dir = model_dir

    def predict(self, lstm_sequence=None, arima_series=None) -> PredictionResult:
        """
        Run hybrid prediction.
        lstm_sequence: shape (1, seq_len, features) — for LSTM
        arima_series: 1D array of recent risk scores — for ARIMA
        """
        lstm_pred = 0.5
        arima_pred = 0.5

        # LSTM prediction
        if lstm_sequence is not None:
            try:
                lstm_pred = self.lstm.predict(lstm_sequence)
            except Exception:
                lstm_pred = 0.5

        # ARIMA prediction
        if arima_series is not None and len(arima_series) > 10:
            try:
                self.arima.fit(arima_series)
                arima_pred = self.arima.predict(steps=1)
            except Exception:
                arima_pred = 0.5

        # Hybrid fusion
        final = self.lstm_weight * lstm_pred + self.arima_weight * arima_pred
        final = max(0.0, min(1.0, final))

        # Classify
        if final > 0.7:
            level = "high"
            action = "Pre-deploy police and emergency units. Maximum alert."
        elif final > 0.4:
            level = "medium"
            action = "Increase monitoring. Alert standby patrol units."
        else:
            level = "low"
            action = "Normal operations. Continue standard surveillance."

        return PredictionResult(
            predicted_risk=round(final, 3),
            risk_level=level,
            recommended_action=action,
            lstm_prediction=round(lstm_pred, 3),
            arima_prediction=round(arima_pred, 3)
        )

    def load_models(self) -> bool:
        lstm_ok = self.lstm.load_model()
        arima_ok = self.arima.load_model()
        return lstm_ok  # ARIMA can be fitted on-the-fly

    def save_models(self):
        self.lstm.save_model()
        self.arima.save_model()
