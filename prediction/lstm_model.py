"""
GOD's EYE — LSTM Model (PyTorch)
Temporal risk prediction using LSTM neural network.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class LSTMNet(nn.Module):
    """LSTM network for time-series risk prediction."""
    def __init__(self, input_size=6, hidden_size=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        out = self.dropout(last)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        return out


class LSTMPredictor:
    def __init__(self, input_size=6, hidden_size=64, dropout=0.2,
                 model_dir="data/models"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.model_dir = model_dir
        self.model = LSTMNet(input_size, hidden_size, dropout)
        self.device = torch.device("cpu")
        self.model.to(self.device)
        os.makedirs(model_dir, exist_ok=True)

    def train(self, X_train, y_train, epochs=50, lr=0.001, batch_size=32):
        """Train LSTM on prepared sequences."""
        self.model.train()
        X = torch.FloatTensor(X_train).to(self.device)
        y = torch.FloatTensor(y_train).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
            if (epoch + 1) % 10 == 0:
                print(f"  LSTM Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.6f}")
        return losses

    def predict(self, sequence) -> float:
        """Predict risk from a single sequence. Input shape: (1, seq_len, features)."""
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(sequence).to(self.device)
            pred = self.model(X)
            return float(pred.cpu().numpy()[0][0])

    def save_model(self, filename="lstm_model.pt"):
        path = os.path.join(self.model_dir, filename)
        torch.save(self.model.state_dict(), path)
        print(f"  LSTM model saved to {path}")

    def load_model(self, filename="lstm_model.pt") -> bool:
        # Security: Resolve to absolute path and check for path traversal
        abs_model_dir = os.path.realpath(os.path.abspath(self.model_dir))
        path = os.path.realpath(os.path.join(abs_model_dir, filename))
        
        if not path.startswith(abs_model_dir + os.sep):
            print(f"[SECURITY] Path traversal attempt blocked for LSTM model: {filename}")
            return False

        if os.path.exists(path):
            try:
                # weights_only=True prevents arbitrary code execution from malicious .pt files
                state_dict = torch.load(path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print(f"  LSTM model loaded from {path}")
                return True
            except Exception as e:
                print(f"[SECURITY] LSTM model load failed (corrupted/tampered file?): {e}")
                return False
        return False
