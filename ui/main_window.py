"""
GOD's EYE — Main Window
Primary application window with multi-camera tabbed layout.
"""

import os
import yaml
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSplitter, QTabWidget, QLabel, QAction, QPushButton,
                             QFileDialog, QInputDialog, QStatusBar,
                             QMessageBox, QFrame, QScrollArea, QGridLayout)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QIcon
from core.model_optimizer import ModelOptimizer

from ui.styles import LIGHT_THEME
from ui.camera_tab import CameraTab
from ui.alert_panel import AlertPanel
from ui.event_log import EventLog
from ui.clip_viewer import ClipViewer
from ui.recommendation_panel import RecommendationPanel
from ui.prediction_panel import PredictionPanel
from ui.reid_panel import ReIDPanel
from ui.workers import PredictionWorker
from core.camera_process import CameraProcess, CameraProxy
import multiprocessing as mp

from core.video_pipeline import VideoPipeline
from core.tracker import ObjectTracker
from core.feature_extractor import FeatureExtractor
from core.risk_scorer import RiskScorer
from core.decision_engine import DecisionEngine
from core.resource_recommender import ResourceRecommender
from core.event_buffer import EventBuffer


from prediction.hybrid_model import HybridPredictor
from prediction.preprocess import PredictionPreprocessor

from storage.database import Database


class MainWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setWindowTitle("Smart City Command Center — Indian Urban Intelligence")
        self.setMinimumSize(1280, 800)
        self.resize(1600, 900)

        # Apply light theme
        self.setStyleSheet(LIGHT_THEME)

        # Init database
        raw_db_path = config.get("storage", {}).get("database", "data/gods_eye.db")
        db_path = self._jail_path(raw_db_path)
        self.db = Database(db_path)
        
        # Production safeguard: Prune data older than 30 days to prevent disk exhaustion
        self.db.prune_old_data(days=30)

        # Camera workers and tabs dicts
        self.camera_workers = {}
        self.camera_tabs_map = {}   # cam_id -> CameraTab widget
        self.prediction_workers = {}
        # _next_cam_index is set after config cameras are created (see _setup_ui)

        # GPU override state
        self._force_gpu_active = False

        # Init prediction
        pred_cfg = config.get("prediction", {})
        self.predictor = HybridPredictor(
            lstm_weight=pred_cfg.get("lstm_weight", 0.6),
            arima_weight=pred_cfg.get("arima_weight", 0.4),
            input_size=6,
            hidden_size=pred_cfg.get("lstm_hidden_size", 64),
            sequence_length=pred_cfg.get("sequence_length", 15),
            arima_order=tuple(pred_cfg.get("arima_order", [2, 1, 2])),
            model_dir=self._jail_path(config.get("storage", {}).get("models_dir", "data/models"))
        )
        self.preprocessor = PredictionPreprocessor(
            sequence_length=pred_cfg.get("sequence_length", 15)
        )
        # Try loading pre-trained models
        self.predictor.load_models()

        self._setup_statusbar()   # Must come first — _setup_ui may trigger pipelines
        self._setup_ui()
        self._setup_menu()

    def _jail_path(self, path, default_dir="data"):
        """Security: Jails a config path to ensure it cannot traverse outside the workspace."""
        base = os.path.realpath(os.path.abspath(default_dir))
        target = os.path.realpath(os.path.abspath(path))
        if not target.startswith(base + os.sep) and target != base:
            print(f"[SECURITY] Path traversal blocked: {path}. Defaulting to {base}")
            return os.path.join(base, os.path.basename(path))
        return target

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)

        # === LEFT: Dynamic Camera tabs ===
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        # Header
        header_row = QHBoxLayout()
        header = QLabel("  👁️  Smart City Command Center  —  Urban Intelligence System")
        header.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #ea580c; "
            "padding: 8px; background-color: #ffffff; "
            "border-radius: 8px; border: 1px solid #E5E7EB;"
        )
        header_row.addWidget(header, 1)

        # Add Camera button
        add_cam_btn = QPushButton("+  Add Camera")
        add_cam_btn.setObjectName("action_btn")
        add_cam_btn.setMinimumHeight(38)
        add_cam_btn.setFixedWidth(130)
        add_cam_btn.clicked.connect(self._add_camera_tab)
        header_row.addWidget(add_cam_btn)

        # ── Force GPU Button ────────────────────────────────────────────
        self.gpu_btn = QPushButton("⚡ Force GPU: OFF")
        self.gpu_btn.setObjectName("gpu_force_btn")
        self.gpu_btn.setMinimumHeight(38)
        self.gpu_btn.setFixedWidth(160)
        self.gpu_btn.setCheckable(True)
        self.gpu_btn.setToolTip(
            "Force all camera pipelines to use CUDA GPU.\n"
            "Use this if automatic GPU detection fails.\n"
            "Active cameras will restart in GPU mode."
        )
        self.gpu_btn.setStyleSheet(
            "QPushButton#gpu_force_btn {"
            "  background-color: #e2e8f0; color: #334155;"
            "  border: 2px solid #94a3b8; border-radius: 6px;"
            "  font-weight: bold; font-size: 12px;"
            "}"
            "QPushButton#gpu_force_btn:checked {"
            "  background-color: #16a34a; color: #ffffff;"
            "  border: 2px solid #15803d;"
            "}"
            "QPushButton#gpu_force_btn:hover {"
            "  background-color: #cbd5e1;"
            "}"
            "QPushButton#gpu_force_btn:checked:hover {"
            "  background-color: #15803d;"
            "}"
        )
        self.gpu_btn.clicked.connect(self._toggle_force_gpu)
        header_row.addWidget(self.gpu_btn)
        left_layout.addLayout(header_row)

        # Camera Grid widget
        self.camera_scroll = QScrollArea()
        self.camera_scroll.setWidgetResizable(True)
        self.camera_scroll.setFrameShape(QFrame.NoFrame)
        self.camera_container = QWidget()
        self.camera_grid = QGridLayout(self.camera_container)
        self.camera_grid.setContentsMargins(0, 0, 0, 0)
        self.camera_grid.setSpacing(8)
        self.camera_scroll.setWidget(self.camera_container)
        left_layout.addWidget(self.camera_scroll, 1)
        
        # Pagination controls
        pagination_layout = QHBoxLayout()
        self.prev_page_btn = QPushButton("◀ Previous")
        self.prev_page_btn.clicked.connect(self._prev_page)
        self.next_page_btn = QPushButton("Next ▶")
        self.next_page_btn.clicked.connect(self._next_page)
        self.page_label = QLabel("Page 1 / 1")
        self.page_label.setAlignment(Qt.AlignCenter)
        self.page_label.setStyleSheet("font-weight: bold; color: #475569;")
        
        pagination_layout.addWidget(self.prev_page_btn)
        pagination_layout.addWidget(self.page_label, 1)
        pagination_layout.addWidget(self.next_page_btn)
        left_layout.addLayout(pagination_layout)

        # Create initial camera tabs from config
        cameras = self.config.get("cameras", [{"id": "cam_01", "name": "Camera 1", "source": ""}])
        self.video_widgets = {}  # kept for signal compatibility
        self._auto_connect_queue = []   # collect sources to connect after full init
        max_cam_idx = 0
        for cam in cameras:
            cam_id = cam.get("id", "cam_01")
            cam_name = cam.get("name", cam_id)
            self._create_camera_tab(cam_id, cam_name)
            # Track highest cam index to avoid ID collisions
            try:
                idx = int(cam_id.split("_")[1])
                max_cam_idx = max(max_cam_idx, idx)
            except (IndexError, ValueError):
                pass
            # Queue auto-connect if source is pre-configured
            source = cam.get("source", "").strip()
            if source:
                self._auto_connect_queue.append((source, cam_id))

        # Set next index AFTER all config cameras to avoid ID collision
        self._next_cam_index = max_cam_idx + 1

        # Defer auto-connect until after full UI is initialised
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, self._run_auto_connect)

        # Stats bar
        self.stats_bar = QLabel("People: 0 | Vehicles: 0 | Density: 0.00 | Risk: 0.00")
        self.stats_bar.setStyleSheet(
            "color: #475569; font-size: 12px; padding: 6px; "
            "background-color: #ffffff; border-radius: 6px; border: 1px solid #E5E7EB;"
        )
        left_layout.addWidget(self.stats_bar)
        splitter.addWidget(left)

        # === RIGHT: Control panels ===
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        self.panel_tabs = QTabWidget()

        # Tab 1: Alerts + Recommendations
        alerts_tab = QWidget()
        alerts_layout = QVBoxLayout(alerts_tab)
        alerts_layout.setContentsMargins(0, 0, 0, 0)
        self.alert_panel = AlertPanel()
        alerts_layout.addWidget(self.alert_panel, 2)
        self.recommendation_panel = RecommendationPanel()
        self.recommendation_panel.action_confirmed.connect(self._on_action_confirmed)
        self.recommendation_panel.action_rejected.connect(self._on_action_rejected)
        alerts_layout.addWidget(self.recommendation_panel, 1)
        self.panel_tabs.addTab(alerts_tab, "⚡ Alerts")

        # Tab 2: Event Log
        self.event_log = EventLog()
        self.event_log.clip_requested.connect(self._on_clip_requested)
        self.panel_tabs.addTab(self.event_log, "📋 Log")

        # Tab 3: Clip Viewer
        self.clip_viewer = ClipViewer()
        self.panel_tabs.addTab(self.clip_viewer, "🎬 Clips")

        # Tab 4: Person Re-ID / Target Tracking
        self.reid_panel = ReIDPanel()
        self.reid_panel.target_added.connect(self._on_target_added)
        self.reid_panel.target_removed.connect(self._on_target_removed)
        self.panel_tabs.addTab(self.reid_panel, "🎯 Track")

        # Tab 5: Prediction
        self.prediction_panel = PredictionPanel()
        self.panel_tabs.addTab(self.prediction_panel, "🔮 Predict")

        right_layout.addWidget(self.panel_tabs)
        splitter.addWidget(right)

        splitter.setSizes([960, 640])
        main_layout.addWidget(splitter)

    def _setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        add_cam = QAction("➕ Add Camera Tab", self)
        add_cam.triggered.connect(self._add_camera_tab)
        file_menu.addAction(add_cam)

        file_menu.addSeparator()

        train_action = QAction("🧠 Train Prediction Models", self)
        train_action.triggered.connect(self._train_models)
        file_menu.addAction(train_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = menubar.addMenu("View")
        gen_data = QAction("📊 Generate Synthetic Data", self)
        gen_data.triggered.connect(self._generate_synthetic_data)
        view_menu.addAction(gen_data)

    def _setup_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("color: #0f172a;")
        self.status_label = QLabel("Status: Idle")
        self.status_bar.addWidget(self.status_label, 1)
        self.status_bar.addPermanentWidget(self.fps_label)

    def _run_auto_connect(self):
        """Connect cameras that had a source pre-configured in config.yaml."""
        for source, cam_id in getattr(self, "_auto_connect_queue", []):
            self._start_pipeline(source, cam_id)
        self._auto_connect_queue = []

    # === Camera Tab Management ===
    
    def _prev_page(self):
        if not hasattr(self, 'current_page'): self.current_page = 0
        if self.current_page > 0:
            self.current_page -= 1
            self._rearrange_grid()
            
    def _next_page(self):
        if not hasattr(self, 'current_page'): self.current_page = 0
        self.max_per_page = getattr(self, 'max_per_page', 16)
        total_pages = max(1, (len(self.camera_tabs_map) + self.max_per_page - 1) // self.max_per_page)
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self._rearrange_grid()

    def _rearrange_grid(self):
        """Rearranges the camera widgets in the grid layout dynamically with pagination."""
        # Remove all widgets from layout
        while self.camera_grid.count():
            item = self.camera_grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.hide() # Hide instead of removing
                self.camera_grid.removeWidget(widget)
        
        # Add back active widgets for current page
        widgets = list(self.camera_tabs_map.values())
        if not widgets:
            if hasattr(self, 'page_label'):
                self.page_label.setText("Page 1 / 1")
            return
            
        # Pagination logic
        self.max_per_page = 16
        if not hasattr(self, 'current_page'):
            self.current_page = 0
            
        total_pages = max(1, (len(widgets) + self.max_per_page - 1) // self.max_per_page)
        # Ensure current_page is valid
        if self.current_page >= total_pages:
            self.current_page = total_pages - 1
            
        start_idx = self.current_page * self.max_per_page
        end_idx = min(start_idx + self.max_per_page, len(widgets))
        
        page_widgets = widgets[start_idx:end_idx]
        
        if hasattr(self, 'page_label'):
            self.page_label.setText(f"Page {self.current_page + 1} / {total_pages}")
            
        import math
        cols = min(4, math.ceil(math.sqrt(len(page_widgets))))
        
        for i, widget in enumerate(page_widgets):
            row = i // cols
            col = i % cols
            self.camera_grid.addWidget(widget, row, col)
            widget.show()
            
            # Wake up active cameras
            cam_id = widget.camera_id
            if cam_id in self.camera_workers:
                self.camera_workers[cam_id].wake()
        
        # Hibernate cameras NOT on the current page
        active_ids = [w.camera_id for w in page_widgets]
        for cam_id, proxy in self.camera_workers.items():
            if cam_id not in active_ids:
                proxy.hibernate()

    def _create_camera_tab(self, cam_id, cam_name):
        """Create and register a new Camera widget."""
        tab = CameraTab(camera_id=cam_id, camera_name=cam_name)
        tab.connect_requested.connect(self._start_pipeline)
        tab.disconnect_requested.connect(self._stop_camera)
        tab.close_requested.connect(self._close_camera_tab)
        self.camera_tabs_map[cam_id] = tab
        self.video_widgets[cam_id] = tab  # alias for signal compatibility
        self._rearrange_grid()
        return tab

    def _add_camera_tab(self):
        """Dynamically add a new camera."""
        idx = self._next_cam_index
        self._next_cam_index += 1
        cam_id = f"cam_{idx:02d}"
        cam_name = f"Camera {idx}"
        self._create_camera_tab(cam_id, cam_name)

    # === GPU Force Override ===

    def _toggle_force_gpu(self, checked):
        """Toggle forced GPU mode — overrides ModelOptimizer's auto-detection."""
        self._force_gpu_active = checked

        if checked:
            # Override cached device to CUDA, bypassing auto-detection
            ModelOptimizer._cached_device = "cuda"
            self.gpu_btn.setText("⚡ Force GPU: ON")
            self.status_label.setText("Status: GPU mode FORCED — restarting active cameras...")
            print("[GPU OVERRIDE] Force GPU activated — CUDA set as device.")

            # Ask user before restarting active cameras
            active = list(self.camera_workers.keys())
            if active:
                reply = QMessageBox.question(
                    self, "Restart Cameras in GPU Mode?",
                    f"{len(active)} active camera(s) will be restarted in GPU mode.\n"
                    "This will briefly interrupt streams.\n\nContinue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                if reply == QMessageBox.Yes:
                    self._restart_all_cameras_with_mode()
                else:
                    self.status_label.setText(
                        "Status: GPU mode ON — new cameras will use GPU. Existing unchanged."
                    )
            else:
                self.status_label.setText(
                    "Status: GPU mode FORCED — new camera streams will use GPU."
                )
        else:
            # Revert to auto-detection
            ModelOptimizer._cached_device = None  # Clear cache → next call will auto-detect
            self.gpu_btn.setText("⚡ Force GPU: OFF")
            self.status_label.setText("Status: GPU override OFF — reverted to auto-detection.")
            print("[GPU OVERRIDE] Force GPU deactivated — auto-detection restored.")

    def _restart_all_cameras_with_mode(self):
        """Stop and restart every active camera pipeline with the current device setting."""
        # Collect current sources before stopping
        camera_sources = {}
        for cam_id, proxy in list(self.camera_workers.items()):
            # The source is stored on the process object
            proc = getattr(proxy, '_process', None)
            if proc and hasattr(proc, 'source'):
                camera_sources[cam_id] = proc.source

        if not camera_sources:
            self.status_label.setText("Status: No active cameras to restart.")
            return

        print(f"[GPU OVERRIDE] Restarting {len(camera_sources)} camera(s): {list(camera_sources.keys())}")

        # Stop all workers
        for cam_id in list(camera_sources.keys()):
            self._stop_camera(cam_id)

        # Small delay so threads can clean up
        device_label = "GPU (CUDA)" if self._force_gpu_active else "CPU (Auto)"
        self.status_label.setText(f"Status: Restarting cameras on {device_label}...")

        # Restart each camera after a brief pause
        def _do_restart():
            for cam_id, source in camera_sources.items():
                print(f"[GPU OVERRIDE] Restarting {cam_id} on {'cuda' if self._force_gpu_active else 'auto'}")
                self._start_pipeline(source, cam_id)
            self.status_label.setText(
                f"Status: All cameras restarted on {device_label}."
            )

        QTimer.singleShot(800, _do_restart)

    def _close_camera_tab(self, cam_id):
        """Stop and remove a camera."""
        self._stop_camera(cam_id)
        widget = self.camera_tabs_map.pop(cam_id, None)
        self.video_widgets.pop(cam_id, None)
        if widget:
            widget.deleteLater()
        self._rearrange_grid()

    def _stop_camera(self, camera_id):
        """Stop the worker for a specific camera."""
        if camera_id in self.camera_workers:
            proxy = self.camera_workers[camera_id]
            proxy.stop()
            if hasattr(proxy, '_process'):
                # Wait for the thread/process to exit cleanly
                proxy._process.join(timeout=3.0)
            del self.camera_workers[camera_id]
        if camera_id in self.camera_tabs_map:
            self.camera_tabs_map[camera_id].on_disconnected()

    def _get_active_camera_id(self):
        if self.camera_tabs_map:
            return next(iter(self.camera_tabs_map.keys()))
        return "cam_01"

    def _start_pipeline(self, source, camera_id):
        """Initialize and start the CV pipeline for a camera."""
        print(f"[DEBUG] Starting pipeline for {camera_id} from _start_pipeline")
        # Stop existing worker for this camera
        if camera_id in self.camera_workers:
            self.camera_workers[camera_id].stop()

        cfg = self.config
        det_cfg = cfg.get("detection", {})
        feat_cfg = cfg.get("features", {})
        risk_cfg = cfg.get("risk", {})
        dec_cfg = cfg.get("decision", {})
        buf_cfg = cfg.get("buffer", {})

        # If Force GPU is active, ensure the override is applied before model selection
        if self._force_gpu_active:
            ModelOptimizer._cached_device = "cuda"
            model_file = det_cfg.get("model", "yolov8n.pt")
            # Use .pt model directly — Ultralytics handles CUDA natively
            opt_model_path = model_file if model_file.endswith(".pt") else model_file.replace(".onnx", ".pt")
            opt_device = "cuda"
            print(f"[GPU OVERRIDE] Using forced CUDA device for {camera_id}")
        else:
            # Use ModelOptimizer to get the best model (auto-detection)
            opt_model_path, opt_device = ModelOptimizer.get_optimized_model(
                model_path=det_cfg.get("model", "yolov8n.pt")
            )

        import queue
        # Create queues
        out_q = queue.Queue(maxsize=30)  # Larger buffer; non-blocking puts in camera thread prevent stalls
        cmd_q = queue.Queue()

        # Create and start worker
        import copy
        worker_process = CameraProcess(
            camera_id=camera_id,
            source=source,
            config=copy.deepcopy(self.config),
            opt_model_path=opt_model_path,
            opt_device=opt_device,
            output_queue=out_q,
            command_queue=cmd_q
        )
        
        proxy = CameraProxy(camera_id, out_q, cmd_q)

        # Connect signals
        if camera_id in self.camera_tabs_map:
            tab = self.camera_tabs_map[camera_id]
            proxy.frame_ready.connect(tab.update_frame)
            proxy.pipeline_started.connect(self._on_pipeline_started)
            proxy.connection_failed.connect(self._on_connection_failed)
        elif camera_id in self.video_widgets:
            proxy.frame_ready.connect(self.video_widgets[camera_id].update_frame)
            
        proxy.features_ready.connect(self._on_features)
        proxy.risk_ready.connect(self._on_risk)
        proxy.event_detected.connect(self._on_event)
        proxy.fps_update.connect(self._on_fps)
        proxy.reid_match_found.connect(self.reid_panel.notify_match)
        
        # New DB logging connection
        proxy.db_log_ready.connect(self._on_db_log)

        # Start process and proxy thread
        worker_process.start()
        proxy.start()
        
        # Initialize worker with existing active targets
        for t in self.reid_panel.get_active_targets():
            proxy.add_reid_target(t["embedding"], t["label"], t.get("threshold", 0.7))

        # Store a tuple of (process, proxy) in camera_workers so we can stop both later
        self.camera_workers[camera_id] = proxy
        self.camera_workers[camera_id]._process = worker_process
        
        self.status_label.setText(f"Status: Connecting — {camera_id}")

        # Start prediction worker if not running
        if camera_id not in self.prediction_workers:
            self._start_prediction_worker(camera_id)

    def _start_prediction_worker(self, camera_id):
        pred_cfg = self.config.get("prediction", {})
        worker = PredictionWorker(
            hybrid_predictor=self.predictor,
            preprocessor=self.preprocessor,
            database=self.db,
            camera_id=camera_id,
            interval=pred_cfg.get("update_interval_seconds", 60)
        )
        worker.prediction_ready.connect(self._on_prediction)
        self.prediction_workers[camera_id] = worker
        worker.start()

    # === Re-ID / Target Tracking ===

    def _on_target_added(self, embedding, label, threshold):
        """Pass new tracking target to all active CV workers."""
        for worker in self.camera_workers.values():
            if hasattr(worker, 'add_reid_target'):
                worker.add_reid_target(embedding, label, threshold)
        self.status_label.setText(f"Status: Tracking target '{label}' across all cameras")

    def _on_target_removed(self, index):
        for worker in self.camera_workers.values():
            if hasattr(worker, 'remove_reid_target'):
                worker.remove_reid_target(index)

    # === Signal Handlers ===

    def _on_features(self, features, camera_id):
        if features is None:
            return
        self.stats_bar.setText(
            f"[{camera_id}]  People: {features.people_count} | "
            f"Vehicles: {features.vehicle_count} | "
            f"Density: {features.crowd_density:.2f} | "
            f"Speed: {features.avg_speed:.1f} | "
            f"Entropy: {features.direction_entropy:.2f}"
        )

    def _on_risk(self, risk, camera_id):
        pass  # Risk is shown in HUD overlay

    def _get_camera_name(self, camera_id):
        """Resolve camera_id to its human-readable name."""
        tab = self.camera_tabs_map.get(camera_id)
        if tab:
            return tab.camera_name
        return camera_id

    def _on_event(self, event, recommendation, camera_id):
        cam_name = self._get_camera_name(camera_id)
        
        # Log to DB
        rec_str = recommendation.action if recommendation else ""
        self.db.log_event(
            timestamp=event.timestamp if isinstance(event.timestamp, str) else __import__('time').strftime("%Y-%m-%d %H:%M:%S", __import__('time').localtime(event.timestamp)),
            camera_id=camera_id,
            event_type=event.event_type,
            risk_level=event.risk_level,
            recommendation=rec_str,
            clip_path="" # Can be added later if needed
        )

        self.alert_panel.add_alert(event, camera_id, cam_name)
        row = self.event_log.table.rowCount()
        self.event_log.add_event(event)
        self.recommendation_panel.show_recommendation(recommendation, event_row=row, camera_name=cam_name)
        # Switch to alerts tab
        self.panel_tabs.setCurrentIndex(0)

    def _on_db_log(self, log_dict):
        """Handle time-series DB logging emitted from camera threads."""
        self.db.log_time_series(
            timestamp=log_dict["timestamp"],
            camera_id=log_dict["camera_id"],
            features=log_dict["features"],
            risk=log_dict["risk"],
            event_flag=log_dict["event_flag"]
        )

    def _on_prediction(self, result):
        self.prediction_panel.update_prediction(result)
        # Feed prediction risk back to CV workers
        for worker in self.camera_workers.values():
            worker.set_prediction_risk(result.predicted_risk)

    def _on_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def _on_pipeline_started(self, camera_id):
        if camera_id in self.camera_tabs_map:
            self.camera_tabs_map[camera_id].on_connected()
        cam_name = self._get_camera_name(camera_id)
        self.status_label.setText(f"Status: Live — {cam_name} ({camera_id})")

    def _on_db_log(self, log_msg):
        """Handle time-series DB logging emitted from camera threads."""
        self.db.log_time_series(
            timestamp=log_msg["timestamp"],
            camera_id=log_msg["camera_id"],
            features=log_msg["features"],
            risk=log_msg["risk"],
            event_flag=log_msg["event_flag"]
        )

    def _on_clip_requested(self, clip_path):
        self.clip_viewer.load_clip(clip_path)
        self.panel_tabs.setCurrentIndex(2)  # Switch to clips tab

    def _on_connection_failed(self, camera_id, error_msg):
        """Handle failed stream connections — show error in the camera tab."""
        if camera_id in self.camera_tabs_map:
            self.camera_tabs_map[camera_id].on_connection_error(error_msg)
        # Clean up the dead worker
        if camera_id in self.camera_workers:
            del self.camera_workers[camera_id]
        self.status_label.setText(f"Status: Connection failed — {camera_id}")

    def _on_action_confirmed(self, row, status):
        self.event_log.update_event_status(row, "Confirmed")
        self.status_label.setText("Status: Action CONFIRMED by operator")

    def _on_action_rejected(self, row, status):
        self.event_log.update_event_status(row, "Rejected")
        self.status_label.setText("Status: Action REJECTED by operator")

    # === Training ===

    def _train_models(self):
        from prediction.trainer import train_pipeline
        self.status_label.setText("Status: Training models...")
        try:
            model_dir = self._jail_path(self.config.get("storage", {}).get("models_dir", "data/models"))
            train_pipeline(model_dir=model_dir, epochs=30)
            self.predictor.load_models()
            QMessageBox.information(self, "Training Complete",
                                    "Prediction models trained successfully!")
        except Exception as e:
            QMessageBox.warning(self, "Training Error", str(e))
        self.status_label.setText("Status: Idle")

    def _generate_synthetic_data(self):
        df = PredictionPreprocessor.generate_synthetic_data(n_points=500)
        os.makedirs("data/logs", exist_ok=True)
        path = "data/logs/synthetic_training_data.csv"
        df.to_csv(path, index=False)
        # Also insert into DB
        for _, row in df.iterrows():
            self.db.log_time_series(
                timestamp=row["timestamp"], camera_id=row["camera_id"],
                people_count=int(row["people_count"]),
                vehicle_count=int(row["vehicle_count"]),
                crowd_density=float(row["crowd_density"]),
                avg_speed=float(row["avg_speed"]),
                speed_variance=float(row.get("speed_variance", 0)),
                direction_entropy=float(row.get("direction_entropy", 0)),
                risk_score=float(row.get("risk_score", row.get("crowd_risk", 0.0))),
                event_flag=int(row["event_flag"])
            )
        QMessageBox.information(self, "Data Generated",
                                f"Generated 500 synthetic data points.\nSaved to {path}")

    # === Cleanup ===

    def closeEvent(self, event):
        for camera_id in list(self.camera_workers.keys()):
            self._stop_camera(camera_id)
        for pw in self.prediction_workers.values():
            if pw is not None:
                pw.stop()
        self.clip_viewer.cleanup()
        event.accept()
