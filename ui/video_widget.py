"""
GOD's EYE — Video Widget
Displays live video feed with detection overlays, scaled to fit.
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap


class VideoWidget(QWidget):
    """Displays annotated video frames from the CV pipeline."""

    def __init__(self, camera_id="cam_01", parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.video_label = QLabel("Waiting for video feed...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #F8FAFC;
                color: #64748B;
                font-size: 16px;
                border: 1px dashed #CBD5E1;
                border-radius: 8px;
            }
        """)
        layout.addWidget(self.video_label)

    def update_frame(self, q_image: QImage, camera_id: str):
        """Update the displayed frame."""
        if camera_id != self.camera_id:
            return
        pixmap = QPixmap.fromImage(q_image)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)

    def clear(self):
        self.video_label.clear()
        self.video_label.setText("No feed")
