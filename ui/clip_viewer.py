"""
GOD's EYE — Clip Viewer
OpenCV-based event clip playback widget.
"""

import os
import cv2
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QSizePolicy, QSlider)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap


class ClipViewer(QWidget):
    """Plays saved event clips using OpenCV + QLabel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cap = None
        self._timer = QTimer()
        self._timer.timeout.connect(self._next_frame)
        self._playing = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("🎬 CLIP VIEWER")
        title.setObjectName("title")
        layout.addWidget(title)

        self.video_label = QLabel("Select an event to view clip")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumHeight(150)
        self.video_label.setStyleSheet(
            "background-color: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 6px;"
        )
        layout.addWidget(self.video_label)

        # Controls
        ctrl = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setObjectName("action_btn")
        self.play_btn.clicked.connect(self._toggle_play)
        ctrl.addWidget(self.play_btn)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self._stop)
        ctrl.addWidget(self.stop_btn)

        self.clip_label = QLabel("No clip loaded")
        self.clip_label.setObjectName("subtitle")
        ctrl.addWidget(self.clip_label, 1)

        layout.addLayout(ctrl)

    def load_clip(self, clip_path):
        """Load a clip file for playback."""
        self._stop()
        if not clip_path:
            return

        # Security: Resolve absolute path and ensure file is inside data/events/
        abs_events_dir = os.path.realpath(os.path.abspath("data/events"))
        abs_clip_path = os.path.realpath(os.path.abspath(clip_path))
        ALLOWED_CLIP_EXTS = {".mp4", ".avi", ".mkv", ".mov"}
        
        if not abs_clip_path.startswith(abs_events_dir + os.sep):
            self.clip_label.setText("Access denied: invalid clip path")
            return
        if os.path.splitext(abs_clip_path)[1].lower() not in ALLOWED_CLIP_EXTS:
            self.clip_label.setText("Access denied: invalid file type")
            return

        self._cap = cv2.VideoCapture(abs_clip_path)
        if self._cap.isOpened():
            self.clip_label.setText(os.path.basename(abs_clip_path))
            self._show_first_frame()
        else:
            self.clip_label.setText("Failed to load clip")

    def _show_first_frame(self):
        if self._cap:
            ret, frame = self._cap.read()
            if ret:
                self._display_frame(frame)
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _toggle_play(self):
        if self._playing:
            self._timer.stop()
            self._playing = False
            self.play_btn.setText("▶ Play")
        else:
            if self._cap and self._cap.isOpened():
                self._timer.start(33)  # ~30fps playback
                self._playing = True
                self.play_btn.setText("⏸ Pause")

    def _stop(self):
        self._timer.stop()
        self._playing = False
        self.play_btn.setText("▶ Play")
        if self._cap:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _next_frame(self):
        if not self._cap:
            return
        ret, frame = self._cap.read()
        if ret:
            self._display_frame(frame)
        else:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._stop()

    def _display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    def cleanup(self):
        self._stop()
        if self._cap:
            self._cap.release()
