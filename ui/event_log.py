"""
GOD's EYE — Event Log
Scrollable timeline table of all detected events with status tracking.
"""

import time
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QTableWidget,
                             QTableWidgetItem, QHeaderView, QAbstractItemView)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor


class EventLog(QWidget):
    """Event log timeline with table view."""
    clip_requested = pyqtSignal(str)  # clip_path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._events = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("📋 EVENT LOG")
        title.setObjectName("title")
        layout.addWidget(title)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["Time", "Camera", "Event", "Severity", "Confidence", "Status"]
        )
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(False)
        self.table.cellClicked.connect(self._on_row_clicked)
        layout.addWidget(self.table)

    def add_event(self, event, clip_path=""):
        ts = time.strftime("%H:%M:%S", time.localtime(event.timestamp))
        row = self.table.rowCount()
        self.table.insertRow(0)  # Insert at top

        items = [
            ts,
            event.camera_id,
            event.event_type,
            event.severity_str.upper(),
            f"{event.confidence:.2f}",
            "Pending"
        ]

        severity_colors = {
            "critical": QColor(239, 68, 68),
            "high": QColor(245, 158, 11),
            "medium": QColor(6, 182, 212),
            "low": QColor(16, 185, 129),
        }
        color = severity_colors.get(event.severity_str, QColor(200, 200, 200))

        for col, text in enumerate(items):
            item = QTableWidgetItem(text)
            if col == 3:  # severity column
                item.setForeground(color)
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(0, col, item)

        self._events.insert(0, {"event": event, "clip_path": clip_path})

    def _on_row_clicked(self, row, col):
        if row < len(self._events):
            clip = self._events[row].get("clip_path", "")
            if clip:
                self.clip_requested.emit(clip)

    def update_event_status(self, row, status):
        if row < self.table.rowCount():
            item = QTableWidgetItem(status)
            color = QColor(16, 185, 129) if status == "Confirmed" else QColor(239, 68, 68)
            item.setForeground(color)
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 5, item)
