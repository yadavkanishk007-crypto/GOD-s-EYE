import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prediction.hybrid_model import HybridPredictor

def test_hybrid_predictor_initialization():
    predictor = HybridPredictor(lstm_weight=0.6, arima_weight=0.4, model_dir="data/models_test")
    assert predictor.lstm_weight == 0.6
    assert predictor.arima_weight == 0.4
    assert predictor.lstm is not None
    assert predictor.arima is not None

def test_hybrid_predict_fallback():
    predictor = HybridPredictor(model_dir="data/models_test")
    
    # Empty inputs should use fallback 0.5
    res = predictor.predict(lstm_sequence=None, arima_series=None)
    
    # 0.6 * 0.5 + 0.4 * 0.5 = 0.5
    assert np.isclose(res.predicted_risk, 0.5)
    assert res.risk_level == "medium"

def test_hybrid_predict_with_data():
    predictor = HybridPredictor(model_dir="data/models_test")
    
    # Create dummy sequences
    lstm_seq = np.random.rand(1, 15, 6).astype(np.float32)
    arima_series = np.linspace(0.1, 0.9, 15).astype(np.float32)
    
    # This should run without crashing
    res = predictor.predict(lstm_sequence=lstm_seq, arima_series=arima_series)
    
    assert 0.0 <= res.predicted_risk <= 1.0
    assert res.risk_level in ["low", "medium", "high"]
