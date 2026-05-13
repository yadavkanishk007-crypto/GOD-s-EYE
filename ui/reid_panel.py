"""
Smart City Command Center — Person Re-ID & Target Tracking Panel
Allows operators to upload a suspect photo and track them across all camera feeds.
Uses the lightweight MobileNetV2 feature extractor from core/reid_tracker.py.
"""

import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QScrollArea, QFrame, QSlider,
                              QFileDialog, QSizePolicy, QMessageBox, QListWidget,
                              QListWidgetItem)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QPixmap, QImage, QColor, QFont


class TargetCard(QWidget):
    """Displays a single tracked target with their reference photo and last seen info."""
    remove_requested = pyqtSignal(int)  # target_index

    def __init__(self, target_index, image_path, label="Target", threshold=0.65, parent=None):
        super().__init__(parent)
        self.target_index = target_index
        self.threshold = threshold
        self._build_ui(image_path, label)

    def _build_ui(self, image_path, label):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.setStyleSheet(
            "background: #FFFFFF; border: 1px solid #E5E7EB; "
            "border-radius: 8px; margin-bottom: 4px;"
        )

        # Reference photo thumbnail
        img_label = QLabel()
        img_label.setFixedSize(56, 72)
        img_label.setStyleSheet(
            "border: 1px solid #D1D5DB; border-radius: 4px; background: #F3F4F6;"
        )
        img_label.setAlignment(Qt.AlignCenter)
        if image_path and os.path.exists(image_path):
            pix = QPixmap(image_path).scaled(56, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            img_label.setPixmap(pix)
        else:
            img_label.setText("👤")
            img_label.setStyleSheet(img_label.styleSheet() + "font-size: 24px;")
        layout.addWidget(img_label)

        # Info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)

        name_label = QLabel(label)
        name_label.setStyleSheet("font-weight: 600; color: #1F2937; font-size: 13px;")
        info_layout.addWidget(name_label)

        self.status_label = QLabel("🔍 Searching all feeds...")
        self.status_label.setStyleSheet("color: #6B7280; font-size: 11px;")
        info_layout.addWidget(self.status_label)

        self.last_seen_label = QLabel("Last seen: —")
        self.last_seen_label.setStyleSheet("color: #9CA3AF; font-size: 11px;")
        info_layout.addWidget(self.last_seen_label)

        layout.addLayout(info_layout, 1)

        # Remove button
        remove_btn = QPushButton("✕")
        remove_btn.setFixedSize(24, 24)
        remove_btn.setStyleSheet(
            "background: #FEF2F2; border: 1px solid #FECACA; "
            "color: #DC2626; border-radius: 4px; font-size: 12px;"
        )
        remove_btn.clicked.connect(lambda: self.remove_requested.emit(self.target_index))
        layout.addWidget(remove_btn)

    def set_found(self, camera_id, similarity):
        """Update card to show target was spotted."""
        self.status_label.setText(f"🚨 SPOTTED — {camera_id}  ({similarity:.0%} match)")
        self.status_label.setStyleSheet(
            "color: #DC2626; font-size: 11px; font-weight: 600;"
        )
        import time
        self.last_seen_label.setText(f"Last seen: {time.strftime('%H:%M:%S')}")

    def set_searching(self):
        self.status_label.setText("🔍 Searching all feeds...")
        self.status_label.setStyleSheet("color: #6B7280; font-size: 11px;")


class ReIDPanel(QWidget):
    """
    Person Re-Identification & Target Tracking Panel.
    Operators upload a photo; the system searches for that person across all camera feeds.
    """
    target_added = pyqtSignal(object, str, float)    # embedding (np.array), label, threshold
    target_removed = pyqtSignal(int)          # target_index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.targets = []           # List of dicts: {embedding, label, image_path, card}
        self._reid_extractor = None
        self._setup_ui()

    def _get_extractor(self):
        """Lazy-load the Re-ID extractor so startup stays fast."""
        if self._reid_extractor is None:
            try:
                from core.reid_tracker import ReIDExtractor
                self._reid_extractor = ReIDExtractor(device="cpu")
            except Exception as e:
                QMessageBox.warning(self, "Re-ID Not Available",
                    f"Could not load tracking model:\n{e}\n\n"
                    "Make sure torchvision is installed:\n  pip install torchvision")
                return None
        return self._reid_extractor

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Header ──────────────────────────────────────────────
        header = QLabel("🎯 TARGET TRACKING")
        header.setObjectName("title")
        layout.addWidget(header)

        desc = QLabel(
            "Upload a photo of a person to track them across all connected camera feeds. "
            "The AI will search every frame in real-time and alert you when found."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(
            "font-size: 11px; color: #6B7280; padding: 8px; "
            "background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 6px;"
        )
        layout.addWidget(desc)

        # ── Add Target ───────────────────────────────────────────
        add_frame = QFrame()
        add_frame.setStyleSheet(
            "background: #FFF7ED; border: 1px solid #FED7AA; border-radius: 8px; padding: 4px;"
        )
        add_layout = QVBoxLayout(add_frame)
        add_layout.setSpacing(6)

        add_title = QLabel("➕ Add New Target")
        add_title.setStyleSheet("font-weight: 600; color: #C2410C; font-size: 13px;")
        add_layout.addWidget(add_title)

        # Photo path display
        self.photo_path_label = QLabel("No photo selected")
        self.photo_path_label.setStyleSheet("color: #6B7280; font-size: 11px;")
        add_layout.addWidget(self.photo_path_label)

        # Photo preview
        self.photo_preview = QLabel()
        self.photo_preview.setFixedHeight(120)
        self.photo_preview.setAlignment(Qt.AlignCenter)
        self.photo_preview.setStyleSheet(
            "background: #F3F4F6; border: 1px dashed #D1D5DB; border-radius: 6px;"
        )
        self.photo_preview.setText("👤\nNo photo")
        add_layout.addWidget(self.photo_preview)

        # Buttons row
        btn_row = QHBoxLayout()
        browse_btn = QPushButton("📁 Browse Photo")
        browse_btn.setObjectName("action_btn")
        browse_btn.setMinimumHeight(36)
        browse_btn.clicked.connect(self._browse_photo)
        btn_row.addWidget(browse_btn)

        self.add_target_btn = QPushButton("🎯 Start Tracking")
        self.add_target_btn.setMinimumHeight(36)
        self.add_target_btn.setEnabled(False)
        self.add_target_btn.setStyleSheet(
            "background: #F0FDF4; border: 1px solid #22C55E; "
            "color: #15803D; border-radius: 6px; font-weight: 600;"
        )
        self.add_target_btn.clicked.connect(self._add_target)
        btn_row.addWidget(self.add_target_btn)
        add_layout.addLayout(btn_row)

        # Threshold slider
        thresh_row = QHBoxLayout()
        thresh_label = QLabel("Match Sensitivity:")
        thresh_label.setStyleSheet("font-size: 11px; color: #374151;")
        thresh_row.addWidget(thresh_label)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(50, 95)
        self.threshold_slider.setValue(70)
        self.threshold_slider.setStyleSheet("margin: 0 8px;")
        thresh_row.addWidget(self.threshold_slider, 1)

        self.thresh_val_label = QLabel("70%")
        self.thresh_val_label.setStyleSheet("font-size: 11px; color: #C2410C; font-weight: 600;")
        thresh_row.addWidget(self.thresh_val_label)
        self.threshold_slider.valueChanged.connect(
            lambda v: self.thresh_val_label.setText(f"{v}%")
        )
        add_layout.addLayout(thresh_row)

        layout.addWidget(add_frame)

        # ── Active Targets ────────────────────────────────────────
        targets_header = QLabel("📋 Active Targets")
        targets_header.setStyleSheet(
            "font-weight: 600; color: #374151; font-size: 13px; margin-top: 4px;"
        )
        layout.addWidget(targets_header)

        self.no_targets_label = QLabel("No targets being tracked.\nUpload a photo above to begin.")
        self.no_targets_label.setAlignment(Qt.AlignCenter)
        self.no_targets_label.setStyleSheet(
            "color: #9CA3AF; font-size: 12px; padding: 24px; "
            "background: #F9FAFB; border: 1px dashed #E5E7EB; border-radius: 6px;"
        )
        layout.addWidget(self.no_targets_label)

        # Scrollable target cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self.targets_container = QWidget()
        self.targets_layout = QVBoxLayout(self.targets_container)
        self.targets_layout.setContentsMargins(0, 0, 0, 0)
        self.targets_layout.setSpacing(4)
        self.targets_layout.addStretch()
        scroll.setWidget(self.targets_container)
        layout.addWidget(scroll, 1)

        # Compatibility note
        compat = QLabel(
            "✅ Compatible with: Hikvision · Dahua · CP Plus · Axis · Bosch · Reolink · "
            "Amcrest · generic ONVIF cameras · any NVR/DVR with RTSP output"
        )
        compat.setWordWrap(True)
        compat.setStyleSheet(
            "font-size: 10px; color: #6B7280; padding: 6px; "
            "background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 6px;"
        )
        layout.addWidget(compat)

        self._selected_photo_path = None

    def _browse_photo(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Person Photo", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All Files (*)"
        )
        if not path:
            return
        self._selected_photo_path = path
        self.photo_path_label.setText(os.path.basename(path))

        # Show preview
        pix = QPixmap(path).scaled(100, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.photo_preview.setPixmap(pix)
        self.add_target_btn.setEnabled(True)

    def _add_target(self):
        if not self._selected_photo_path:
            return

        extractor = self._get_extractor()
        if extractor is None:
            return

        # Load image and extract embedding from the whole frame
        # (assuming the uploaded photo is a cropped person image)
        try:
            frame = cv2.imread(self._selected_photo_path)
            if frame is None:
                QMessageBox.warning(self, "Error", "Could not read the selected image file.")
                return

            h, w = frame.shape[:2]
            embedding = extractor.extract_feature(frame, [0, 0, w, h])
            if embedding is None:
                QMessageBox.warning(self, "Error",
                    "Could not extract features from this image.\n"
                    "Please use a clear, well-lit photo showing the full person.")
                return
        except Exception as e:
            QMessageBox.warning(self, "Extraction Error", str(e))
            return

        idx = len(self.targets)
        threshold = self.threshold_slider.value() / 100.0
        label = f"Target {idx + 1}"

        card = TargetCard(idx, self._selected_photo_path, label, threshold)
        card.remove_requested.connect(self._remove_target)

        target_data = {
            "embedding": embedding,
            "label": label,
            "image_path": self._selected_photo_path,
            "threshold": threshold,
            "card": card
        }
        self.targets.append(target_data)

        # Insert card above the stretch
        self.targets_layout.insertWidget(self.targets_layout.count() - 1, card)
        self.no_targets_label.setVisible(False)

        # Emit signal so main window can wire this into CV workers
        self.target_added.emit(embedding, label, threshold)

        # Reset UI
        self._selected_photo_path = None
        self.photo_preview.setText("👤\nNo photo")
        self.photo_path_label.setText("No photo selected")
        self.add_target_btn.setEnabled(False)

    def _remove_target(self, index):
        if 0 <= index < len(self.targets):
            card = self.targets[index]["card"]
            self.targets_layout.removeWidget(card)
            card.deleteLater()
            self.targets.pop(index)
            self.target_removed.emit(index)

            if not self.targets:
                self.no_targets_label.setVisible(True)

    def notify_match(self, target_index, camera_id, similarity):
        """Called by the CV worker when a target is found in a camera feed."""
        if 0 <= target_index < len(self.targets):
            self.targets[target_index]["card"].set_found(camera_id, similarity)

    def get_active_targets(self):
        """Returns list of {embedding, threshold, label} for the CV pipeline."""
        return [
            {"embedding": t["embedding"], "threshold": t["threshold"], "label": t["label"]}
            for t in self.targets
        ]
