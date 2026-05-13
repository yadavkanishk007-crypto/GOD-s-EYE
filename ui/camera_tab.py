"""
Smart City Command Center — Camera Tab Widget
Each camera tab shows either a live feed, a connection panel, or a loading/error state.
Operators can connect RTSP, webcam, or video file from within the tab itself.
"""

import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QLineEdit, QComboBox, QFileDialog,
                              QStackedWidget, QFrame, QSizePolicy, QInputDialog,
                              QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

from ui.video_widget import VideoWidget


class CameraTab(QWidget):
    """
    A single camera tab that shows either:
    - Page 0: 'Connect Feed' panel (when no source is active)
    - Page 1: Live video feed (when connected)
    - Page 2: Loading/connecting indicator
    """
    connect_requested = pyqtSignal(str, str)   # source, camera_id
    disconnect_requested = pyqtSignal(str)      # camera_id
    close_requested = pyqtSignal(str)           # camera_id

    def __init__(self, camera_id, camera_name="Camera", parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.is_connected = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        
        # Header with name, rename, and close button
        header = QHBoxLayout()
        self._name_label = QLabel(f"📹 {self.camera_name}")
        self._name_label.setStyleSheet("font-weight: bold; color: #475569;")
        header.addWidget(self._name_label)

        # Rename button
        rename_btn = QPushButton("✏️")
        rename_btn.setFixedSize(22, 22)
        rename_btn.setToolTip("Rename this camera")
        rename_btn.setStyleSheet(
            "background: #F0F9FF; color: #0369A1; border: 1px solid #BAE6FD; "
            "border-radius: 4px; font-size: 12px; padding: 0;"
        )
        rename_btn.clicked.connect(self._rename_camera)
        header.addWidget(rename_btn)

        header.addStretch()
        
        # Disconnect button (hidden initially)
        self.disconnect_btn = QPushButton("⏹ Stop")
        self.disconnect_btn.setFixedHeight(20)
        self.disconnect_btn.setStyleSheet(
            "background: #FEF3C7; color: #D97706; border: 1px solid #FDE68A; "
            "border-radius: 4px; font-size: 11px; padding: 0 6px;"
        )
        self.disconnect_btn.clicked.connect(lambda: self.disconnect_requested.emit(self.camera_id))
        self.disconnect_btn.hide()
        header.addWidget(self.disconnect_btn)
        
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(20, 20)
        close_btn.setStyleSheet("background: #FEF2F2; color: #DC2626; border: 1px solid #FECACA; border-radius: 4px;")
        close_btn.clicked.connect(lambda: self.close_requested.emit(self.camera_id))
        header.addWidget(close_btn)
        layout.addLayout(header)

        # Content stack
        self.stack = QStackedWidget()
        self.stack.setStyleSheet("border: 1px solid #E5E7EB; border-radius: 4px; background: #FFFFFF;")
        layout.addWidget(self.stack)

        # Page 0: Connect panel
        self.stack.addWidget(self._build_connect_panel())

        # Page 1: Live feed
        self.video_widget = VideoWidget(camera_id=self.camera_id)
        self.stack.addWidget(self.video_widget)

        # Page 2: Loading state
        self.stack.addWidget(self._build_loading_panel())

        self.stack.setCurrentIndex(0)

    def _build_connect_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(16)

        # Icon + title
        icon = QLabel("📡")
        icon.setAlignment(Qt.AlignCenter)
        icon.setStyleSheet("font-size: 48px;")
        layout.addWidget(icon)

        title = QLabel(f"{self.camera_name}")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: 600; color: #374151;")
        layout.addWidget(title)

        self._status_label = QLabel("No feed connected. Choose a source below.")
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setStyleSheet("font-size: 13px; color: #6B7280;")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("border: 1px solid #E5E7EB;")
        layout.addWidget(line)

        # RTSP URL input
        rtsp_layout = QHBoxLayout()
        self.rtsp_input = QLineEdit()
        self.rtsp_input.setPlaceholderText("rtsp://... or https://... (Stream URL)")
        self.rtsp_input.setMinimumHeight(38)
        self.rtsp_input.setStyleSheet(
            "border: 1px solid #D1D5DB; border-radius: 6px; "
            "padding: 6px 12px; font-size: 13px; background: #FFFFFF; color: #1F2937;"
        )
        self.rtsp_input.returnPressed.connect(self._connect_rtsp)
        rtsp_btn = QPushButton("🔗 Connect Stream")
        rtsp_btn.setObjectName("action_btn")
        rtsp_btn.setMinimumHeight(38)
        rtsp_btn.clicked.connect(self._connect_rtsp)
        rtsp_layout.addWidget(self.rtsp_input, 3)
        rtsp_layout.addWidget(rtsp_btn, 1)
        layout.addLayout(rtsp_layout)

        # Webcam + File buttons
        alt_layout = QHBoxLayout()

        webcam_btn = QPushButton("📷 Open Webcam (Index 0)")
        webcam_btn.setMinimumHeight(38)
        webcam_btn.setStyleSheet(
            "background: #F0FDF4; border: 1px solid #22C55E; "
            "color: #15803D; border-radius: 6px; font-weight: 500;"
        )
        webcam_btn.clicked.connect(lambda: self._connect_source("0"))
        alt_layout.addWidget(webcam_btn)

        file_btn = QPushButton("📁 Open Video File")
        file_btn.setMinimumHeight(38)
        file_btn.setStyleSheet(
            "background: #FFF7ED; border: 1px solid #F97316; "
            "color: #C2410C; border-radius: 6px; font-weight: 500;"
        )
        file_btn.clicked.connect(self._open_file)
        alt_layout.addWidget(file_btn)

        layout.addLayout(alt_layout)

        # RTSP format hint
        hint = QLabel(
            "💡 Common RTSP formats:\n"
            "  Hikvision: rtsp://admin:pass@IP:554/Streaming/Channels/101\n"
            "  Dahua/CP Plus: rtsp://admin:pass@IP:554/cam/realmonitor?channel=1\n"
            "  Generic: rtsp://admin:pass@IP:554/stream1"
        )
        hint.setStyleSheet(
            "font-size: 11px; color: #6B7280; padding: 12px; "
            "background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 6px;"
        )
        layout.addWidget(hint)

        return panel

    def _build_loading_panel(self):
        """Loading indicator shown while connecting to a stream."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(12)

        self._loading_icon = QLabel("⏳")
        self._loading_icon.setAlignment(Qt.AlignCenter)
        self._loading_icon.setStyleSheet("font-size: 48px;")
        layout.addWidget(self._loading_icon)

        self._loading_label = QLabel("Connecting to stream...")
        self._loading_label.setAlignment(Qt.AlignCenter)
        self._loading_label.setStyleSheet(
            "font-size: 16px; font-weight: 600; color: #D97706;"
        )
        layout.addWidget(self._loading_label)

        self._loading_detail = QLabel("Resolving video source. This may take a few seconds.")
        self._loading_detail.setAlignment(Qt.AlignCenter)
        self._loading_detail.setWordWrap(True)
        self._loading_detail.setStyleSheet("font-size: 12px; color: #6B7280;")
        layout.addWidget(self._loading_detail)

        # Cancel button
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedWidth(120)
        cancel_btn.setMinimumHeight(32)
        cancel_btn.setStyleSheet(
            "background: #FEF2F2; border: 1px solid #FECACA; "
            "color: #DC2626; border-radius: 6px;"
        )
        cancel_btn.clicked.connect(lambda: self.disconnect_requested.emit(self.camera_id))
        layout.addWidget(cancel_btn, alignment=Qt.AlignCenter)

        return panel

    def _rename_camera(self):
        """Let the operator rename this camera via a simple input dialog."""
        new_name, ok = QInputDialog.getText(
            self, "Rename Camera",
            "Enter a new name for this camera:",
            text=self.camera_name
        )
        if ok and new_name.strip():
            self.camera_name = new_name.strip()
            self._name_label.setText(f"📹 {self.camera_name}")

    def _connect_rtsp(self):
        url = self.rtsp_input.text().strip()
        if not url:
            return
        if not any(url.lower().startswith(prefix) for prefix in ("rtsp://", "rtsps://", "http://", "https://")):
            QMessageBox.warning(self, "Invalid URL",
                "Please enter a valid stream URL starting with rtsp://, rtsps://, http://, or https://")
            return
        self._connect_source(url)

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv);;All Files (*)"
        )
        if path:
            self._connect_source(path)

    def _connect_source(self, source):
        # Show loading state immediately
        self.show_loading(f"Connecting to: {source[:60]}...")
        self.connect_requested.emit(source, self.camera_id)

    def show_loading(self, detail=""):
        """Switch to loading/connecting view."""
        self._loading_label.setText("Connecting to stream...")
        if detail:
            self._loading_detail.setText(detail)
        self.stack.setCurrentIndex(2)

    def on_connected(self):
        """Switch to the live video view."""
        self.is_connected = True
        self.disconnect_btn.show()
        self._name_label.setStyleSheet("font-weight: bold; color: #15803D;")
        self.stack.setCurrentIndex(1)

    def on_disconnected(self):
        """Switch back to the connect panel."""
        self.is_connected = False
        self.disconnect_btn.hide()
        self._name_label.setStyleSheet("font-weight: bold; color: #475569;")
        self.video_widget.clear()
        self.stack.setCurrentIndex(0)

    def on_connection_error(self, error_msg):
        """Show an error on the connect panel."""
        self.is_connected = False
        self.disconnect_btn.hide()
        self._name_label.setStyleSheet("font-weight: bold; color: #DC2626;")
        self._status_label.setText(f"❌ Connection failed: {error_msg}")
        self._status_label.setStyleSheet("font-size: 13px; color: #DC2626; font-weight: bold;")
        self.stack.setCurrentIndex(0)
        # Reset status color after 10 seconds
        QTimer.singleShot(10000, lambda: (
            self._status_label.setText("No feed connected. Choose a source below."),
            self._status_label.setStyleSheet("font-size: 13px; color: #6B7280;"),
            self._name_label.setStyleSheet("font-weight: bold; color: #475569;"),
        ))

    def update_frame(self, q_image: QImage, camera_id: str):
        """Pass frame down to the video widget."""
        self.video_widget.update_frame(q_image, camera_id)
