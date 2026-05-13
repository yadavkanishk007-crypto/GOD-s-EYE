"""
GOD's EYE — Object Tracker
ByteTrack integration via ultralytics for persistent object tracking.
"""

import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class TrackedObject:
    track_id: int
    bbox: Tuple[int, int, int, int]
    class_id: int
    class_name: str
    confidence: float
    center: Tuple[int, int] = field(default=(0, 0))
    velocity: Tuple[float, float] = field(default=(0.0, 0.0))
    speed: float = 0.0
    direction: float = 0.0

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)


CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class ObjectTracker:
    def __init__(self, model_path="yolov8n.pt", confidence=0.4, iou_threshold=0.45,
                 device="cpu", target_classes=None, history_length=30):
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device
        self.target_classes = target_classes or [0, 1, 2, 3, 5, 7]
        self.history_length = history_length
        self._model = None
        self._track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=history_length))

    def load_model(self):
        from ultralytics import YOLO
        self._model = YOLO(self.model_path)

    def track(self, frame, timestamp=0.0):
        if self._model is None:
            self.load_model()
        tracked = []
        results = self._model.track(frame, conf=self.confidence, iou=self.iou_threshold,
                                    device=self.device, classes=self.target_classes,
                                    tracker="bytetrack.yaml", persist=True, verbose=False)
        if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            r = results[0]
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            ids = r.boxes.id.cpu().numpy().astype(int)
            cls = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for bbox, tid, c, cf in zip(boxes, ids, cls, confs):
                x1, y1, x2, y2 = bbox
                cx, cy = (x1+x2)//2, (y1+y2)//2
                self._track_history[tid].append((cx, cy, timestamp))
                vel, spd, dirn = (0.0, 0.0), 0.0, 0.0
                h = self._track_history[tid]
                if len(h) >= 2:
                    dx, dy = h[-1][0]-h[-2][0], h[-1][1]-h[-2][1]
                    vel = (float(dx), float(dy))
                    spd = float(np.sqrt(dx**2 + dy**2))
                    dirn = float(np.arctan2(dy, dx))
                tracked.append(TrackedObject(
                    track_id=int(tid), bbox=(int(x1),int(y1),int(x2),int(y2)),
                    class_id=int(c), class_name=CLASS_NAMES.get(c, f"cls_{c}"),
                    confidence=float(cf), center=(cx,cy), velocity=vel, speed=spd, direction=dirn
                ))
        return tracked

    def get_track_history(self, track_id):
        return list(self._track_history.get(track_id, []))

    def get_annotated_frame(self, frame, tracked_objects):
        import cv2
        ann = frame.copy()
        colors = {"person":(0,255,200),"car":(255,200,0),"truck":(255,200,0),
                  "bus":(255,200,0),"motorcycle":(255,150,0),"bicycle":(200,255,0)}
        for obj in tracked_objects:
            x1,y1,x2,y2 = obj.bbox
            c = colors.get(obj.class_name, (200,200,200))
            cv2.rectangle(ann,(x1,y1),(x2,y2),c,2)
            lbl = f"ID:{obj.track_id} {obj.class_name} {obj.confidence:.2f}"
            sz,_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(ann,(x1,y1-sz[1]-8),(x1+sz[0]+4,y1),c,-1)
            cv2.putText(ann,lbl,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,0),1)
            hist = self._track_history.get(obj.track_id,[])
            if len(hist) > 1:
                pts = [(int(p[0]),int(p[1])) for p in hist]
                for i in range(1,len(pts)):
                    a = i/len(pts)
                    cv2.line(ann,pts[i-1],pts[i],tuple(int(v*a) for v in c),max(1,int(a*3)))
        return ann

    def reset(self):
        self._track_history.clear()
