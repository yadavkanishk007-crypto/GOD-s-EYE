"""
GOD's EYE — Training Pipeline
End-to-end training for LSTM + ARIMA models.
Can be run locally or on Google Colab.
"""

import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prediction.preprocess import PredictionPreprocessor
from prediction.lstm_model import LSTMPredictor
from prediction.arima_model import ARIMAPredictor


def train_pipeline(data_path=None, model_dir="data/models", epochs=50):
    """
    Full training pipeline.
    If no data_path, generates synthetic data for demo.
    """
    print("=" * 60)
    print("GOD's EYE — Training Pipeline")
    print("=" * 60)

    preprocessor = PredictionPreprocessor(sequence_length=15)

    # Load or generate data
    if data_path and os.path.exists(data_path):
        print(f"\n[1/5] Loading data from {data_path}")
        df = preprocessor.load_from_csv(data_path)
    else:
        print("\n[1/5] Generating synthetic training data...")
        df = PredictionPreprocessor.generate_synthetic_data(n_points=500)
        os.makedirs("data/logs", exist_ok=True)
        df.to_csv("data/logs/synthetic_training_data.csv", index=False)
        print(f"  Saved synthetic data: {len(df)} rows")

    # Prepare sequences
    print("\n[2/5] Preparing sequences...")
    X_train, y_train, X_test, y_test = preprocessor.prepare_data(df)
    if X_train is None:
        print("  ERROR: Not enough data for training.")
        return
    print(f"  Train: {X_train.shape}, Test: {X_test.shape if X_test is not None else 'N/A'}")

    # Train LSTM
    print(f"\n[3/5] Training LSTM ({epochs} epochs)...")
    input_size = X_train.shape[2]
    lstm = LSTMPredictor(input_size=input_size, hidden_size=64, model_dir=model_dir)
    losses = lstm.train(X_train, y_train, epochs=epochs)
    lstm.save_model()

    # Test LSTM
    if X_test is not None and len(X_test) > 0:
        test_pred = lstm.predict(X_test[:1])
        print(f"  Sample LSTM prediction: {test_pred:.4f}")

    # Fit ARIMA
    print("\n[4/5] Fitting ARIMA...")
    risk_col = "risk_score" if "risk_score" in df.columns else "crowd_risk"
    risk_series = df[risk_col].values
    arima = ARIMAPredictor(order=(2, 1, 2), model_dir=model_dir)
    arima.fit(risk_series)
    arima_pred = arima.predict(steps=1)
    arima.save_model()
    print(f"  ARIMA next-step prediction: {arima_pred:.4f}")

    # Summary
    print("\n[5/5] Training complete!")
    print(f"  Models saved to: {model_dir}/")
    print(f"  LSTM final loss: {losses[-1]:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GOD's EYE Training Pipeline")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV data")
    parser.add_argument("--model-dir", type=str, default="data/models")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    train_pipeline(args.data, args.model_dir, args.epochs)
