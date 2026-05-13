"""
GOD's EYE — Event Buffer
Maintains a rolling buffer of recent frames in RAM (JPEG compressed to save memory).
When an event occurs, it writes the buffered frames to disk as an MP4 clip.

PRODUCTION HARDENED (v2):
  • Uses ThreadPoolExecutor to prevent thread explosion under rapid event triggers.
"""

import os
import cv2
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# Global thread pool for writing clips to disk to prevent thread explosion
_clip_writer_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ClipWriter")


class EventBuffer:
    def __init__(self, fps=8, duration_seconds=20, events_dir="data/events"):
        self.fps = fps
        self.duration_seconds = duration_seconds
        self.events_dir = events_dir
        self.max_frames = fps * duration_seconds
        
        # Store frames as JPEG-compressed bytes in RAM to save memory
        # (A 720p raw numpy array is 2.7MB. JPEG is ~150KB. Reduces buffer from 432MB -> 24MB)
        self._buffer: deque = deque(maxlen=self.max_frames)

    def add_frame(self, frame):
        """Compress frame to JPEG and add to ring buffer."""
        # Lower quality slightly for buffer to save more RAM (80% is fine for reference clips)
        success, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if success:
            self._buffer.append(encoded.tobytes())

    def save_clip(self, event_name: str) -> str:
        """
        Saves the current buffer to disk asynchronously.
        Returns the path where the clip will be saved.
        """
        if len(self._buffer) == 0:
            return ""
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_name = event_name.replace(" ", "_").replace("/", "_")
        filename = f"{timestamp}_{safe_name}.mp4"
        filepath = os.path.join(self.events_dir, filename)
        
        # Take a snapshot of current frames
        frames_snapshot = list(self._buffer)
        
        # Submit to pool instead of spawning unbounded threads
        _clip_writer_pool.submit(self._write_clip_task, filepath, frames_snapshot, self.fps)
        return filepath

    def clear(self):
        self._buffer.clear()

    @staticmethod
    def _write_clip_task(filepath, frames, fps):
        """Background task to write JPEG bytes to MP4."""
        if not frames:
            return
            
        try:
            import numpy as np
            # Decode first frame to get dimensions
            first_frame = cv2.imdecode(np.frombuffer(frames[0], np.uint8), cv2.IMREAD_COLOR)
            h, w = first_frame.shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
            
            for frame_bytes in frames:
                frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    out.write(frame)
                    
            out.release()
            print(f"[EventBuffer] Saved clip: {filepath}")
        except Exception as e:
            print(f"[EventBuffer] Error saving clip {filepath}: {e}")
