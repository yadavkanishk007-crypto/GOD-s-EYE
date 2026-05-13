"""
GOD's EYE — Camera Multiprocessing Architecture
Isolates each camera's CV pipeline into a separate OS Thread to bypass the Python GIL.
Uses compressed frame transmission via Queues to prevent memory bottlenecks.

PRODUCTION HARDENED (v3):
  • GPU auto-detection: CUDA → Ultralytics+ByteTrack; CPU → ONNX-only (<100 MB).
  • All DB writes routed through proxy→main thread (no SQLite contention).
  • Heartbeat signals for silent-hang detection.
  • Optimized HUD drawing (no frame.copy() per frame).
"""

import time
import cv2
import numpy as np
import threading
import queue
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage


class CameraProcess(threading.Thread):
    """
    Runs the complete CV Pipeline for a single camera in an isolated Thread.
    Auto-selects tracker backend based on detected hardware:
      • GPU (cuda/mps) → Ultralytics YOLO + ByteTrack (full power)
      • CPU            → OnnxTracker (lightweight, <100 MB RAM)
    """
    def __init__(self, camera_id, source, config, opt_model_path, opt_device, output_queue, command_queue):
        super().__init__()
        self.camera_id = camera_id
        self.source = source
        self.config = config
        self.opt_model_path = opt_model_path
        self.opt_device = opt_device
        self.out_q = output_queue
        self.cmd_q = command_queue
        self.daemon = True # Kill process if main GUI dies

    def run(self):
        # Local imports inside run to avoid issues
        from core.video_pipeline import VideoPipeline
        from core.feature_extractor import FeatureExtractor
        from core.risk_scorer import RiskScorer
        from core.decision_engine import DecisionEngine
        from core.resource_recommender import ResourceRecommender
        from core.event_buffer import EventBuffer
        from core.incident_detector import IncidentDetector

        cfg = self.config
        det_cfg = cfg.get("detection", {})
        feat_cfg = cfg.get("features", {})
        risk_cfg = cfg.get("risk", {})
        dec_cfg = cfg.get("decision", {})
        buf_cfg = cfg.get("buffer", {})

        # Instantiate objects inside the new thread
        pipeline = VideoPipeline(
            source=self.source,
            camera_id=self.camera_id,
            target_fps=cfg.get("video", {}).get("target_fps", 8)
        )
        
        # ── Tracker selection: GPU vs CPU ───────────────────────────────
        use_gpu = self.opt_device in ("cuda", "mps")
        
        if use_gpu:
            # GPU MODE: Full Ultralytics + ByteTrack for maximum accuracy
            from core.tracker import ObjectTracker
            tracker = ObjectTracker(
                model_path=self.opt_model_path,
                confidence=det_cfg.get("confidence", 0.4),
                iou_threshold=det_cfg.get("iou_threshold", 0.45),
                device=self.opt_device,
                target_classes=det_cfg.get("classes", [0, 1, 2, 3, 5, 7]),
                history_length=cfg.get("tracking", {}).get("history_length", 30)
            )
            print(f"[{self.camera_id}] GPU MODE: Using Ultralytics + ByteTrack on {self.opt_device}")
        else:
            # CPU MODE: Lightweight ONNX tracker (no PyTorch overhead)
            from core.onnx_tracker import OnnxTracker
            onnx_path = self.opt_model_path
            if not onnx_path.endswith(".onnx"):
                onnx_path = onnx_path.replace(".pt", ".onnx")
            tracker = OnnxTracker(
                model_path=onnx_path,
                confidence=det_cfg.get("confidence", 0.4),
                target_classes=det_cfg.get("classes", [0, 1, 2, 3, 5, 7]),
                history_length=cfg.get("tracking", {}).get("history_length", 30)
            )
            print(f"[{self.camera_id}] CPU MODE: Using ONNX tracker ({onnx_path})")

        extractor = FeatureExtractor(
            density_area=feat_cfg.get("density_area", 500000),
            fall_aspect_threshold=feat_cfg.get("fall_aspect_ratio_threshold", 0.6),
            fall_speed_threshold=feat_cfg.get("fall_speed_threshold", 15.0)
        )

        weights = risk_cfg.get("weights", {})
        thresholds = risk_cfg.get("thresholds", {})
        scorer = RiskScorer(
            w_density=weights.get("density", 0.4),
            w_speed_var=weights.get("speed_variance", 0.3),
            w_direction_ent=weights.get("direction_entropy", 0.3),
            high_threshold=thresholds.get("high", 0.7),
            medium_threshold=thresholds.get("medium", 0.4)
        )

        engine = DecisionEngine(
            crowd_risk_high=dec_cfg.get("crowd_risk_high", 0.7),
            crowd_risk_medium=dec_cfg.get("crowd_risk_medium", 0.4)
        )
        
        recommender = ResourceRecommender()

        # Multi-modal incident detector (fire + smoke + accident)
        incident_detector = IncidentDetector(
            temporal_window=12,
            accident_flow_thresh=18.0,
            accident_overlap_thresh=0.15
        )

        # Event buffer — no DB here, all DB writes go through proxy
        events_dir = cfg.get("storage", {}).get("events_dir", "data/events")
        buffer = EventBuffer(
            fps=cfg.get("video", {}).get("target_fps", 8),
            duration_seconds=buf_cfg.get("duration_seconds", 20),
            events_dir=events_dir
        )

        if not pipeline.open():
            err = getattr(pipeline, 'last_error', None) or "Failed to open video source."
            self._try_put({"type": "error", "error": err, "camera_id": self.camera_id})
            return

        self._try_put({"type": "started", "camera_id": self.camera_id})
        
        running = True
        prediction_risk = None
        active_targets = []
        hibernated = False # When hibernated, we do almost nothing
        
        # Efficiency settings
        inference_interval = cfg.get("performance", {}).get("inference_interval", 3) 
        motion_threshold = cfg.get("performance", {}).get("motion_threshold", 500)
        
        reid_extractor = None
        frame_counter = 0
        fps_timer = time.time()
        fps_count = 0
        prev_gray = None
        last_heartbeat = time.time()
        heartbeat_interval = 10.0  # Send heartbeat every 10s

        try:
            while running and pipeline.is_running():
                # Check for commands
                while not self.cmd_q.empty():
                    try:
                        cmd = self.cmd_q.get_nowait()
                    except queue.Empty:
                        break
                    if cmd["action"] == "stop":
                        running = False
                    elif cmd["action"] == "hibernate":
                        if not hibernated:
                            hibernated = True
                            print(f"[{self.camera_id}] Hibernating (freeing resources)...")
                            if hasattr(tracker, 'unload_model'):
                                tracker.unload_model()
                            buffer.clear()
                            pipeline.release()
                    elif cmd["action"] == "wake":
                        if hibernated:
                            hibernated = False
                            print(f"[{self.camera_id}] Waking up (reloading resources)...")
                            pipeline.open()
                            if hasattr(tracker, 'load_model'):
                                tracker.load_model()
                    elif cmd["action"] == "set_prediction":
                        prediction_risk = cmd["risk"]
                    elif cmd["action"] == "add_target":
                        active_targets.append(cmd["target"])
                    elif cmd["action"] == "remove_target":
                        idx = cmd["index"]
                        if 0 <= idx < len(active_targets):
                            active_targets.pop(idx)
                            for i, t in enumerate(active_targets): t["index"] = i

                if not running: break
                
                # If hibernated, we sleep longer and don't process frames
                if hibernated:
                    time.sleep(1.0)
                    continue

                packet = pipeline.get_frame()
                if packet is None:
                    time.sleep(0.01)
                    # Send heartbeat to main thread so it knows we're alive
                    now = time.time()
                    if now - last_heartbeat >= heartbeat_interval:
                        self._try_put({"type": "heartbeat", "camera_id": self.camera_id})
                        last_heartbeat = now
                    continue

                frame = packet.frame
                frame_counter += 1
                
                # --- Step 1: Motion Gating (Lightweight) ---
                has_motion = True
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                if prev_gray is not None:
                    delta = cv2.absdiff(prev_gray, gray)
                    thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
                    motion_score = np.sum(thresh) / 255
                    has_motion = motion_score > motion_threshold
                prev_gray = gray

                # --- Step 2: Temporal Skipping ---
                # On GPU, run inference more frequently (every frame with motion)
                effective_interval = 1 if use_gpu else inference_interval
                should_run_inference = (frame_counter % effective_interval == 0) and has_motion
                
                if not should_run_inference:
                    # Send raw frame to UI occasionally (non-blocking)
                    if frame_counter % 4 == 0:
                        display_frame = cv2.resize(frame, (640, 360))
                        _, buffer_img = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                        self._try_put({
                            "type": "frame", "camera_id": self.camera_id,
                            "image_bytes": buffer_img.tobytes(),
                            "features": None, "risk": None
                        })
                    continue

                try:
                    # Tracking (Heavy)
                    tracked = tracker.track(frame, timestamp=packet.timestamp)

                    # Re-ID logic
                    if active_targets:
                        if reid_extractor is None:
                            from core.reid_tracker import ReIDExtractor
                            reid_extractor = ReIDExtractor(device=self.opt_device)

                        for obj in tracked:
                            if obj.class_name == "person":
                                emb = reid_extractor.extract_feature(frame, obj.bbox)
                                if emb is not None:
                                    for t in active_targets:
                                        sim = reid_extractor.compute_similarity(emb, t["embedding"])
                                        if sim >= t.get("threshold", 0.7):
                                            self.out_q.put({
                                                "type": "reid_match",
                                                "target_index": t["index"],
                                                "camera_id": self.camera_id,
                                                "similarity": float(sim)
                                            })
                                            obj.class_name = f"TARGET: {t['label']} ({sim:.0%})"

                    # Multi-modal incident detection (fire + smoke + accident)
                    incident = incident_detector.detect(frame, tracked_objects=tracked)

                    # Feature Extraction
                    features = extractor.extract(tracked, incident)

                    # Risk Scoring
                    risk = scorer.score(features)

                    # Decision Engine
                    events = engine.evaluate(
                        features, risk, camera_id=self.camera_id,
                        prediction_risk=prediction_risk
                    )

                    # Annotation
                    annotated = tracker.get_annotated_frame(frame, tracked)

                    # Draw fire regions
                    for fr in incident.fire_regions:
                        x1, y1, x2, y2 = fr.bbox
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 50, 255), 3)
                        cv2.putText(annotated, f"FIRE {fr.confidence:.0%}",
                                    (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 50, 255), 2)

                    # Draw smoke regions
                    for sr in incident.smoke_regions:
                        x1, y1, x2, y2 = sr.bbox
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (180, 180, 180), 2)
                        cv2.putText(annotated, f"SMOKE {sr.confidence:.0%}",
                                    (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (220, 220, 220), 2)

                    # Draw accident signal
                    if incident.accident_detected and incident.accident.bbox:
                        ax1, ay1, ax2, ay2 = incident.accident.bbox
                        cv2.rectangle(annotated, (ax1, ay1), (ax2, ay2), (0, 0, 255), 4)
                        cv2.putText(annotated,
                                    f"ACCIDENT {incident.accident.confidence:.0%}",
                                    (ax1, max(0, ay1 - 12)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 0, 255), 2)

                    self._draw_hud(annotated, features, risk, pipeline)
                    buffer.add_frame(frame)

                    # Compress frame for IPC
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                    # Resize to UI-friendly resolution to save massive IPC bandwidth
                    display_frame = cv2.resize(annotated, (640, 360))
                    _, buffer_img = cv2.imencode('.jpg', display_frame, encode_param)
                    
                    self._try_put({
                        "type": "frame",
                        "camera_id": self.camera_id,
                        "image_bytes": buffer_img.tobytes(),
                        "features": features,
                        "risk": risk
                    })

                except Exception as e:
                    print(f"[{self.camera_id}] CV Pipeline Error: {e}")
                    continue

                # Handle Events — route through proxy for DB writes (no direct DB here)
                for event in events:
                    clip_path = buffer.save_clip(event.event_type)
                    recommendation = recommender.recommend(event)
                    
                    self._try_put({
                        "type": "event",
                        "camera_id": self.camera_id,
                        "event": event,
                        "recommendation": recommendation,
                        "clip_path": clip_path
                    })

                # DB Logging — route through proxy (rate-limited: every 10 frames)
                if frame_counter % 10 == 0:
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    self._try_put({
                        "type": "db_log",
                        "camera_id": self.camera_id,
                        "timestamp": ts,
                        "features": features,
                        "risk": risk,
                        "event_flag": 1 if events else 0
                    })

                # FPS tracking
                fps_count += 1
                elapsed = time.time() - fps_timer
                if elapsed >= 1.0:
                    self._try_put({"type": "fps", "fps": fps_count / elapsed, "camera_id": self.camera_id})
                    fps_count = 0
                    fps_timer = time.time()

                # Heartbeat
                now = time.time()
                if now - last_heartbeat >= heartbeat_interval:
                    self._try_put({"type": "heartbeat", "camera_id": self.camera_id})
                    last_heartbeat = now

        except Exception as e:
            self._try_put({"type": "error", "error": str(e), "camera_id": self.camera_id})
        finally:
            pipeline.release()

    def _try_put(self, msg):
        """Non-blocking put to the output queue. Drops message if queue is full
        instead of blocking the camera thread (which causes the stall)."""
        try:
            self.out_q.put_nowait(msg)
        except Exception:
            # Queue full — drop this message to keep the pipeline flowing.
            # Frame drops are acceptable; thread deadlock is not.
            pass

    def _draw_hud(self, frame, features, risk, pipeline=None):
        """Draw heads-up display — optimized: no frame.copy() per call."""
        h, w = frame.shape[:2]
        # Semi-transparent bar using direct drawing (avoids 2.7 MB copy per frame)
        cv2.rectangle(frame, (0, 0), (w, 45), (241, 245, 249), -1)

        risk_color = (8, 136, 19) if risk.risk_level == "low" else \
                     (51, 153, 255) if risk.risk_level == "medium" else (34, 34, 239)
        cv2.putText(frame, f"RISK: {risk.risk_score:.2f} [{risk.risk_level.upper()}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_color, 2)
        flags = []
        if features.fire_detected:     flags.append(f"FIRE({features.fire_confidence:.0%})")
        if features.smoke_detected:    flags.append(f"SMOKE({features.smoke_confidence:.0%})")
        if features.accident_detected: flags.append(f"ACCIDENT({features.accident_confidence:.0%})")
        incident_str = "  ".join(flags) if flags else ""
        stats = (f"People:{features.people_count} Veh:{features.vehicle_count} "
                 f"Den:{features.crowd_density:.2f}  {incident_str}")
        cv2.putText(frame, stats, (w // 4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (51, 65, 85), 1)

        # Network quality badge (top-right)
        if pipeline is not None:
            bw = pipeline.get_bandwidth_mbps()
            tier = pipeline.get_quality_tier()
            fps_label = f"{pipeline.target_fps}fps"
            if bw is not None:
                bw_str = f"{bw:.1f}Mbps {tier} {fps_label}"
                # Yellow if degraded from default (target_fps < 8)
                net_color = (0, 200, 255) if pipeline.target_fps < 8 else (51, 153, 255)
            else:
                bw_str = f"{tier} {fps_label}"
                net_color = (51, 153, 255)
            txt_size, _ = cv2.getTextSize(bw_str, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(frame, bw_str, (w - txt_size[0] - 8, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, net_color, 1)
        else:
            cv2.putText(frame, self.camera_id, (w - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (51, 153, 255), 2)


class CameraProxy(QThread):
    """
    Runs in the main Qt Thread. Reads messages from the `CameraProcess` queue
    and emits Qt Signals to update the UI.
    """
    frame_ready = pyqtSignal(QImage, str)  
    features_ready = pyqtSignal(object, str)  
    risk_ready = pyqtSignal(object, str)  
    event_detected = pyqtSignal(object, object, str) # event, recommendation, camera_id 
    fps_update = pyqtSignal(float)
    reid_match_found = pyqtSignal(int, str, float) 
    pipeline_started = pyqtSignal(str)  
    connection_failed = pyqtSignal(str, str) 
    db_log_ready = pyqtSignal(dict) # To pass logs back to main thread for DB writing
    heartbeat_received = pyqtSignal(str)  # camera_id

    def __init__(self, camera_id, out_q, cmd_q):
        super().__init__()
        self.camera_id = camera_id
        self.out_q = out_q
        self.cmd_q = cmd_q
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            try:
                # Use a small timeout so the thread can exit cleanly
                msg = self.out_q.get(timeout=0.1)
                msg_type = msg.get("type")

                if msg_type == "frame":
                    # Decode jpeg bytes to QImage
                    np_arr = np.frombuffer(msg["image_bytes"], np.uint8)
                    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        continue
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    h, w, ch = img_rgb.shape
                    bpl = ch * w
                    qimg = QImage(img_rgb.data, w, h, bpl, QImage.Format_RGB888).copy()
                    
                    self.frame_ready.emit(qimg, self.camera_id)
                    self.features_ready.emit(msg["features"], self.camera_id)
                    self.risk_ready.emit(msg["risk"], self.camera_id)
                
                elif msg_type == "event":
                    self.event_detected.emit(msg["event"], msg["recommendation"], self.camera_id)
                elif msg_type == "fps":
                    self.fps_update.emit(msg["fps"])
                elif msg_type == "reid_match":
                    self.reid_match_found.emit(msg["target_index"], self.camera_id, msg["similarity"])
                elif msg_type == "started":
                    self.pipeline_started.emit(self.camera_id)
                elif msg_type == "error":
                    self.connection_failed.emit(self.camera_id, msg["error"])
                elif msg_type == "db_log":
                    self.db_log_ready.emit(msg)
                elif msg_type == "heartbeat":
                    self.heartbeat_received.emit(self.camera_id)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Proxy {self.camera_id}] Error: {e}")

    def stop(self):
        self._running = False
        self.cmd_q.put({"action": "stop"})
        self.wait(1000)

    def set_prediction_risk(self, risk):
        self.cmd_q.put({"action": "set_prediction", "risk": risk})
        
    def add_reid_target(self, embedding, label, threshold=0.7):
        self.cmd_q.put({
            "action": "add_target",
            "target": {"embedding": embedding, "label": label, "threshold": threshold, "index": -1}
        })
        
    def remove_reid_target(self, index):
        self.cmd_q.put({"action": "remove_target", "index": index})

    def hibernate(self):
        self.cmd_q.put({"action": "hibernate"})

    def wake(self):
        self.cmd_q.put({"action": "wake"})
