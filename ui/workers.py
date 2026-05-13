"""
GOD's EYE — QThread Workers
Background workers for CV pipeline and prediction inference.
"""

import time
import threading
import numpy as np
import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage


class CVPipelineWorker(QThread):
    """
    Main CV processing loop running in a background thread.
    Grabs frames, runs detection+tracking, extracts features, scores risk,
    evaluates decisions, and emits signals to the UI.
    """
    frame_ready = pyqtSignal(QImage, str)  # annotated frame, camera_id
    features_ready = pyqtSignal(object, str)  # FrameFeatures, camera_id
    risk_ready = pyqtSignal(object, str)  # RiskScore, camera_id
    event_detected = pyqtSignal(object, object, str)  # Event, Recommendation, camera_id
    fps_update = pyqtSignal(float)
    reid_match_found = pyqtSignal(int, str, float)  # target_index, camera_id, similarity
    pipeline_started = pyqtSignal(str)  # camera_id — emitted once on first valid frame
    connection_failed = pyqtSignal(str, str)  # camera_id, error_message

    def __init__(self, video_pipeline, tracker, feature_extractor,
                 risk_scorer, decision_engine, resource_recommender,
                 event_buffer, database, camera_id="cam_01"):
        super().__init__()
        self.pipeline = video_pipeline
        self.tracker = tracker
        self.extractor = feature_extractor
        self.scorer = risk_scorer
        self.engine = decision_engine
        self.recommender = resource_recommender
        self.buffer = event_buffer
        self.db = database
        self.camera_id = camera_id

        self._running = False
        self._lock = threading.Lock()   # Use native Python lock — supports 'with' syntax
        self._prediction_risk = None
        self._log_interval = 5  # log to DB every N frames
        self._frame_counter = 0
        self._started_emitted = False  # guard for pipeline_started one-shot signal

        # Re-ID variables
        self.active_targets = []
        self._reid_extractor = None

    def add_reid_target(self, embedding, label, threshold=0.7):
        with self._lock:
            self.active_targets.append({
                "embedding": embedding,
                "label": label,
                "threshold": threshold,
                "index": len(self.active_targets)
            })

    def remove_reid_target(self, index):
        with self._lock:
            if 0 <= index < len(self.active_targets):
                self.active_targets.pop(index)
                for i, target in enumerate(self.active_targets):
                    target["index"] = i

    def set_prediction_risk(self, risk):
        self._prediction_risk = risk

    def run(self):
        self._running = True
        if not self.pipeline.open():
            err = getattr(self.pipeline, 'last_error', None) or "Failed to open video source."
            self.connection_failed.emit(self.camera_id, err)
            return
        fps_timer = time.time()
        fps_count = 0

        try:
            while self._running and self.pipeline.is_running():
                packet = self.pipeline.get_frame()
                if packet is None:
                    time.sleep(0.01)
                    continue

                frame = packet.frame
                self._frame_counter += 1

                try:
                    # Track objects
                    tracked = self.tracker.track(frame, timestamp=packet.timestamp)

                    # Re-ID target matching
                    with self._lock:
                        targets = list(self.active_targets)

                    if targets:
                        if self._reid_extractor is None:
                            from core.reid_tracker import ReIDExtractor
                            self._reid_extractor = ReIDExtractor(device="cpu")

                        for obj in tracked:
                            if obj.class_name == "person":
                                emb = self._reid_extractor.extract_feature(frame, obj.bbox)
                                if emb is not None:
                                    for t in targets:
                                        sim = self._reid_extractor.compute_similarity(emb, t["embedding"])
                                        if sim >= t.get("threshold", 0.7):
                                            # Match found
                                            self.reid_match_found.emit(t["index"], self.camera_id, float(sim))
                                            # Put a visual indicator
                                            obj.class_name = f"TARGET: {t['label']} ({sim:.0%})"

                    # Detect fire separately (tracker doesn't detect fire)
                    fire_regions = self._detect_fire(frame)

                    # Extract features
                    features = self.extractor.extract(tracked, fire_regions)

                    # Score risk
                    risk = self.scorer.score(features)

                    # Decision engine
                    events = self.engine.evaluate(
                        features, risk, camera_id=self.camera_id,
                        prediction_risk=self._prediction_risk
                    )

                    # Annotate frame
                    annotated = self.tracker.get_annotated_frame(frame, tracked)
                    # Draw fire regions
                    for fr in fire_regions:
                        x1, y1, x2, y2 = fr.bbox
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(annotated, f"FIRE {fr.confidence:.2f}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 255), 2)

                    # Draw HUD overlay
                    self._draw_hud(annotated, features, risk)

                    # Buffer frame
                    self.buffer.add_frame(frame)

                    # Convert to QImage and emit
                    q_img = self._frame_to_qimage(annotated)
                    if not self._started_emitted:
                        self.pipeline_started.emit(self.camera_id)
                        self._started_emitted = True
                    self.frame_ready.emit(q_img, self.camera_id)
                    self.features_ready.emit(features, self.camera_id)
                    self.risk_ready.emit(risk, self.camera_id)
                except Exception as e:
                    print(f"[{self.camera_id}] CV Pipeline Error: {e}")
                    continue

                # Handle events
                for event in events:
                    clip_path = self.buffer.save_clip(event.event_type)
                    recommendation = self.recommender.recommend(event)
                    # Save to DB
                    self.db.insert_event(
                        timestamp=event.timestamp,
                        camera_id=event.camera_id,
                        event_type=event.event_type,
                        severity=event.severity_str,
                        confidence=event.confidence,
                        description=event.description,
                        clip_path=clip_path
                    )
                    self.event_detected.emit(event, recommendation, self.camera_id)

                # Log time series
                if self._frame_counter % self._log_interval == 0:
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    self.db.log_time_series(
                        timestamp=ts,
                        camera_id=self.camera_id,
                        features=features,
                        risk=risk,
                        event_flag=1 if events else 0
                    )

                # FPS calculation
                fps_count += 1
                elapsed = time.time() - fps_timer
                if elapsed >= 1.0:
                    self.fps_update.emit(fps_count / elapsed)
                    fps_count = 0
                    fps_timer = time.time()
        finally:
            self.pipeline.release()

    def _detect_fire(self, frame):
        """HSV fire detection — runs alongside tracker."""
        from core.detector import FireRegion
        regions = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 120, 200])
        upper = np.array([25, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area >= 3000:
                x, y, w, h = cv2.boundingRect(c)
                conf = min(1.0, area / 15000)
                regions.append(FireRegion(bbox=(x, y, x+w, y+h), area=area, confidence=conf))
        return regions

    def _draw_hud(self, frame, features, risk):
        """Draw heads-up display overlay."""
        h, w = frame.shape[:2]
        # Semi-transparent HUD bar at top (Light Theme)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 45), (241, 245, 249), -1) # Light slate gray BGR
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Risk indicator (using Indian Saffron/Green BGR)
        risk_color = (8, 136, 19) if risk.risk_level == "low" else \
                     (51, 153, 255) if risk.risk_level == "medium" else (34, 34, 239) # Red for high
        cv2.putText(frame, f"RISK: {risk.risk_score:.2f} [{risk.risk_level.upper()}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_color, 2)

        # Stats (Dark text for light HUD)
        stats = f"People: {features.people_count}  Vehicles: {features.vehicle_count}  Density: {features.crowd_density:.2f}"
        cv2.putText(frame, stats, (w // 3, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (51, 65, 85), 1)

        # Camera ID
        cv2.putText(frame, self.camera_id, (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (51, 153, 255), 2)


    def _frame_to_qimage(self, frame):
        """Convert OpenCV BGR frame to QImage."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

    def stop(self):
        self._running = False
        self.wait(3000)


class PredictionWorker(QThread):
    """Runs hybrid prediction periodically."""
    prediction_ready = pyqtSignal(object)  # PredictionResult

    def __init__(self, hybrid_predictor, preprocessor, database,
                 camera_id="cam_01", interval=60):
        super().__init__()
        self.predictor = hybrid_predictor
        self.preprocessor = preprocessor
        self.db = database
        self.camera_id = camera_id
        self.interval = interval
        self._running = False

    def run(self):
        self._running = True
        # Wait a bit for data to accumulate
        time.sleep(10)

        while self._running:
            try:
                # get_recent_metrics returns dicts with new column names
                logs = self.db.get_recent_metrics(self.camera_id, limit=200)
                if len(logs) >= 20:
                    # Prepare LSTM input
                    seq = self.preprocessor.prepare_inference_sequence(logs)
                    # Prepare ARIMA input - now uses risk_score key
                    risk_series = [row.get("risk_score", 0.5) for row in logs]

                    result = self.predictor.predict(
                        lstm_sequence=seq,
                        arima_series=risk_series
                    )

                    # Removed self.db.insert_prediction as it's not in the new schema.
                    # UI will receive the signal and handle it.
                    self.prediction_ready.emit(result)
            except Exception as e:
                print(f"[PredictionWorker {self.camera_id}] Error: {e}")

            # Wait for next interval
            for _ in range(self.interval):
                if not self._running:
                    break
                time.sleep(1)

    def stop(self):
        self._running = False
        self.wait(3000)
