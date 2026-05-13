"""
GOD's EYE — Prediction Panel
Displays hybrid prediction output with risk gauge and level indicators.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QProgressBar, QPushButton)
from PyQt5.QtCore import Qt


class PredictionPanel(QWidget):
    """Displays LSTM+ARIMA hybrid prediction results."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("🔮 PREDICTIVE ANALYSIS")
        title.setObjectName("title")
        layout.addWidget(title)

        # Risk gauge
        self.risk_bar = QProgressBar()
        self.risk_bar.setRange(0, 100)
        self.risk_bar.setValue(0)
        self.risk_bar.setFormat("Risk: %v%")
        self.risk_bar.setMinimumHeight(20)
        layout.addWidget(self.risk_bar)

        # Risk level
        self.risk_label = QLabel("—")
        self.risk_label.setObjectName("risk_low")
        self.risk_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.risk_label)

        # Model breakdown
        model_layout = QHBoxLayout()
        self.lstm_label = QLabel("LSTM: —")
        self.lstm_label.setStyleSheet("color: #8b5cf6; font-size: 12px;")
        model_layout.addWidget(self.lstm_label)
        self.arima_label = QLabel("ARIMA: —")
        self.arima_label.setStyleSheet("color: #06b6d4; font-size: 12px;")
        model_layout.addWidget(self.arima_label)
        layout.addLayout(model_layout)

        # Recommended action from prediction
        self.action_label = QLabel("Awaiting prediction data...")
        self.action_label.setWordWrap(True)
        self.action_label.setStyleSheet(
            "color: #94a3b8; font-size: 12px; padding: 6px; "
            "background-color: #0f1729; border-radius: 6px;"
        )
        layout.addWidget(self.action_label)

        # Pre-alert indicator
        self.pre_alert = QLabel("")
        self.pre_alert.setAlignment(Qt.AlignCenter)
        self.pre_alert.setStyleSheet("font-size: 12px; font-weight: bold;")
        layout.addWidget(self.pre_alert)

        layout.addStretch()

    def update_prediction(self, result):
        """Update display from a PredictionResult."""
        risk_pct = int(result.predicted_risk * 100)
        self.risk_bar.setValue(risk_pct)

        # Risk level styling
        level = result.risk_level
        self.risk_label.setText(f"{level.upper()} ({risk_pct}%)")
        style_map = {
            "high": "risk_high",
            "medium": "risk_medium",
            "low": "risk_low"
        }
        self.risk_label.setObjectName(style_map.get(level, "risk_low"))
        # Force style refresh
        self.risk_label.setStyleSheet(self.risk_label.styleSheet())
        self.style().polish(self.risk_label)

        # Model breakdown
        self.lstm_label.setText(f"LSTM: {result.lstm_prediction:.3f}")
        self.arima_label.setText(f"ARIMA: {result.arima_prediction:.3f}")

        # Recommendation
        self.action_label.setText(result.recommended_action)

        # Pre-alert
        if result.predicted_risk > 0.7:
            self.pre_alert.setText("⚠️ PRE-ALERT ACTIVE")
            self.pre_alert.setStyleSheet(
                "color: #ef4444; font-size: 13px; font-weight: bold; "
                "background-color: rgba(239,68,68,0.15); padding: 6px; "
                "border-radius: 6px; border: 1px solid #ef4444;"
            )
        elif result.predicted_risk > 0.4:
            self.pre_alert.setText("📡 ELEVATED MONITORING")
            self.pre_alert.setStyleSheet(
                "color: #f59e0b; font-size: 12px; font-weight: bold; "
                "background-color: rgba(245,158,11,0.1); padding: 6px; "
                "border-radius: 6px;"
            )
        else:
            self.pre_alert.setText("✅ Normal Operations")
            self.pre_alert.setStyleSheet(
                "color: #10b981; font-size: 12px; padding: 6px;"
            )

        # Dynamic progress bar color
        if risk_pct > 70:
            bar_style = "QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #ef4444, stop:1 #dc2626); border-radius: 6px; }"
        elif risk_pct > 40:
            bar_style = "QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #f59e0b, stop:1 #d97706); border-radius: 6px; }"
        else:
            bar_style = "QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #06b6d4, stop:1 #8b5cf6); border-radius: 6px; }"
        self.risk_bar.setStyleSheet(
            "QProgressBar { background-color: #1e293b; border: none; border-radius: 6px; "
            "height: 16px; text-align: center; color: #e2e8f0; font-size: 11px; } " + bar_style
        )
