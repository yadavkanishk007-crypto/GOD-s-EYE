"""
GOD's EYE — Recommendation Panel
Shows resource deployment suggestions with Confirm/Reject actions.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QSizePolicy, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal


class RecommendationPanel(QWidget):
    """Displays recommendations and provides human-in-the-loop controls."""
    action_confirmed = pyqtSignal(int, str)   # event_row, "confirmed"
    action_rejected = pyqtSignal(int, str)    # event_row, "rejected"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_event_row = -1
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("🎯 RECOMMENDED ACTION")
        title.setObjectName("title")
        layout.addWidget(title)

        # Recommendation display
        self.icon_label = QLabel("")
        self.icon_label.setStyleSheet("font-size: 32px;")
        self.icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.icon_label)

        self.action_label = QLabel("No active recommendation")
        self.action_label.setWordWrap(True)
        self.action_label.setStyleSheet(
            "font-size: 14px; padding: 12px; background-color: #F8FAFC; "
            "border-radius: 8px; border: 1px solid #E5E7EB; color: #1F2937;"
        )
        layout.addWidget(self.action_label)

        self.units_label = QLabel("")
        self.units_label.setStyleSheet("color: #94a3b8; font-size: 12px;")
        layout.addWidget(self.units_label)

        self.priority_label = QLabel("")
        self.priority_label.setStyleSheet("color: #64748b; font-size: 11px;")
        layout.addWidget(self.priority_label)

        # Confirm / Reject buttons
        btn_layout = QHBoxLayout()

        self.confirm_btn = QPushButton("✅ CONFIRM DEPLOYMENT")
        self.confirm_btn.setObjectName("confirm_btn")
        self.confirm_btn.setMinimumHeight(40)
        self.confirm_btn.clicked.connect(self._on_confirm)
        btn_layout.addWidget(self.confirm_btn)

        self.reject_btn = QPushButton("❌ REJECT")
        self.reject_btn.setObjectName("reject_btn")
        self.reject_btn.setMinimumHeight(40)
        self.reject_btn.clicked.connect(self._on_reject)
        btn_layout.addWidget(self.reject_btn)

        layout.addLayout(btn_layout)
        layout.addStretch()

    def show_recommendation(self, recommendation, event_row=-1, camera_name=None):
        """Display a recommendation from the resource recommender."""
        if recommendation is None:
            self.action_label.setText("No action required")
            self.units_label.setText("")
            self.priority_label.setText("")
            self.icon_label.setText("")
            return

        self._current_event_row = event_row
        self.icon_label.setText(recommendation.icon)

        # Show camera name prominently if available
        cam_prefix = f"📹 {camera_name}:\n" if camera_name else ""
        self.action_label.setText(f"{cam_prefix}{recommendation.action}")
        self.units_label.setText(f"Units: {recommendation.units}")

        priority_colors = {
            "immediate": "#DC2626",
            "high": "#D97706",
            "standard": "#15803D"
        }
        color = priority_colors.get(recommendation.priority, "#64748b")
        self.priority_label.setText(f"Priority: {recommendation.priority.upper()}")
        self.priority_label.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: bold;")

    def _on_confirm(self):
        if self._current_event_row >= 0:
            self.action_confirmed.emit(self._current_event_row, "confirmed")
            self.icon_label.setText("✅")
            self.action_label.setText("Deployment confirmed. Waiting for new alerts.")
            self.units_label.setText("")
            self.priority_label.setText("")
            self._current_event_row = -1

    def _on_reject(self):
        if self._current_event_row >= 0:
            self.action_rejected.emit(self._current_event_row, "rejected")
            self.icon_label.setText("❌")
            self.action_label.setText("Recommendation rejected. No action taken.")
            self.units_label.setText("")
            self.priority_label.setText("")
            self._current_event_row = -1
