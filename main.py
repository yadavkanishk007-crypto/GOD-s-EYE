"""
GOD's EYE — AI-Powered Urban Intelligence System
Main application entry point.

A city-scale augmented intelligence platform for real-time surveillance
analysis, incident detection, crowd risk scoring, and decision support.
"""

import sys
import os
import warnings
import yaml

# ── Suppress known-harmless warnings ──────────────────────────────────────────
# statsmodels ARIMA may not converge with short sequences — not an error.
# The warning is a ConvergenceWarning (subclass of UserWarning) — catch both.
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization failed.*")
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

# Suppress OpenCV FFMPEG codec-tag warning spam (mp4v fallback is intentional).
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")     # AV_LOG_QUIET
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

# Lazy-import torch only if available (allows running without GPU drivers)
try:
    import torch  # noqa: F401
except ImportError:
    pass

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from ui.main_window import MainWindow


def load_config(path="config.yaml"):
    """Load configuration from YAML file."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    # Default config
    return {
        "video": {"source": "0", "target_fps": 8},
        "cameras": [
            {"id": "cam_01", "name": "Camera 1", "source": "0"},
            {"id": "cam_02", "name": "Camera 2", "source": ""},
            {"id": "cam_03", "name": "Camera 3", "source": ""},
            {"id": "cam_04", "name": "Camera 4", "source": ""},
        ],
        "detection": {
            "model": "yolov8n.pt", "confidence": 0.4,
            "device": "cpu", "classes": [0, 1, 2, 3, 5, 7]
        },
        "tracking": {"history_length": 30},
        "features": {"density_area": 500000},
        "risk": {
            "weights": {"density": 0.4, "speed_variance": 0.3, "direction_entropy": 0.3},
            "thresholds": {"high": 0.7, "medium": 0.4}
        },
        "decision": {"crowd_risk_high": 0.7, "crowd_risk_medium": 0.4},
        "buffer": {"duration_seconds": 20},
        "prediction": {
            "lstm_weight": 0.6, "arima_weight": 0.4,
            "sequence_length": 15, "lstm_hidden_size": 64,
            "update_interval_seconds": 60
        },
        "storage": {
            "events_dir": "data/events",
            "models_dir": "data/models",
            "database": "data/gods_eye.db"
        }
    }


def main():
    import multiprocessing as mp
    mp.freeze_support()
    
    # Create required directories
    for d in ["data/events", "data/models", "data/logs"]:
        os.makedirs(d, exist_ok=True)

    # Load config
    config = load_config()

    # Init Qt
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Smart City Command Center")
    app.setFont(QFont("Segoe UI", 10))

    # Launch main window
    window = MainWindow(config)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
