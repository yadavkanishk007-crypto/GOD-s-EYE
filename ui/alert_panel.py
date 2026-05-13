"""
GOD's EYE — Alert Panel
Real-time alert display with severity-coded styling.
"""

import time
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QScrollArea,
                             QFrame, QSizePolicy)
from PyQt5.QtCore import Qt


class AlertPanel(QWidget):
    """Displays real-time alerts with color-coded severity."""

    def __init__(self, max_alerts=50, parent=None):
        super().__init__(parent)
        self.max_alerts = max_alerts
        self._alerts = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Title
        title = QLabel("⚡ LIVE ALERTS")
        title.setObjectName("title")
        layout.addWidget(title)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.alert_container = QWidget()
        self.alert_layout = QVBoxLayout(self.alert_container)
        self.alert_layout.setContentsMargins(0, 0, 0, 0)
        self.alert_layout.setSpacing(6)
        self.alert_layout.addStretch()

        scroll.setWidget(self.alert_container)
        layout.addWidget(scroll)

    def add_alert(self, event, camera_id="cam_01", camera_name=None):
        """Add a new alert from a detected event."""
        # Fix: event has risk_level, not severity_str
        severity = getattr(event, "risk_level", "low").lower()
        ts = time.strftime("%H:%M:%S", time.localtime(event.timestamp))
        display_name = camera_name or camera_id

        style_map = {
            "critical": "alert_critical",
            "high": "alert_high",
            "medium": "alert_medium",
            "low": "alert_low",
        }

        icon_map = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢",
        }

        label = QLabel(
            f"{icon_map.get(severity, '⚪')} [{ts}] 📹 {display_name}\n"
            f"   ⚠️ {event.event_type} — {severity.upper()} "
            f"(conf: {getattr(event, 'confidence', 0.0):.2f})\n"
            f"   🔔 Action required by operator"
        )
        label.setObjectName(style_map.get(severity, "alert_low"))
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        # Insert at top (before stretch)
        self.alert_layout.insertWidget(0, label)
        self._alerts.append(label)

        # Limit alerts
        while len(self._alerts) > self.max_alerts:
            old = self._alerts.pop(0)
            self.alert_layout.removeWidget(old)
            old.deleteLater()

    def clear(self):
        for a in self._alerts:
            a.deleteLater()
        self._alerts.clear()
