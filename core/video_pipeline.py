"""
GOD's EYE — Video Pipeline
Handles video ingestion from RTSP streams, video files, web streams, or camera indices.
Provides configurable FPS frame extraction with thread-safe access.
Supports EarthCam, YouTube, and other web streams via yt-dlp extraction.
Auto-degrades stream quality on slow connections.
"""

import os
import cv2
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

# Suppress OpenCV/FFMPEG codec warning spam in the console.
# The fallback to mp4v is intentional and harmless.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")   # AV_LOG_QUIET
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")


@dataclass
class FramePacket:
    """A single captured frame with metadata."""
    frame: object  # numpy ndarray
    frame_id: int
    timestamp: float
    camera_id: str
    source_fps: float
    width: int
    height: int


class VideoPipeline:
    """
    OpenCV-based video capture pipeline.
    Supports RTSP streams, local video files, camera indices, and web streams.
    Web URLs (http/https) are automatically resolved via yt-dlp when needed.
    Implements frame skipping for configurable target FPS.
    """

    def __init__(self, source, camera_id: str = "cam_01",
                 target_fps: int = 8, resolution: Tuple[int, int] = (1280, 720)):
        self.source = source
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.resolution = resolution

        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._frame_count = 0
        self._running = False
        self._source_fps = 30.0
        self._skip_interval = 1
        self._resolved_url = None   # Actual URL after yt-dlp resolution
        self.last_error = None      # Human-readable error for UI feedback

        # Reconnect state
        self._consecutive_failures = 0
        self._max_consecutive_failures = 30   # ~3-4 s at 8 fps before we attempt reconnect
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 5.0           # seconds between reconnect tries
        self._last_reconnect_time = 0.0
        self._is_live = False                  # True when source is a network stream

        # Adaptive quality state
        self._quality_tier = "best"           # current streamlink quality preference
        self._measured_bw_mbps = None         # last measured bandwidth (Mbps)
        self._total_frames_attempted = 0
        self._total_frames_failed = 0
        self._adaptive_check_interval = 60    # re-check every N frames
        self._min_target_fps = 2              # never go below this

    @staticmethod
    def _validate_source(src) -> bool:
        """
        Security: Validate video source before opening.
        Blocks crafted file paths, non-RTSP URLs, and injection attempts.
        """
        if isinstance(src, int):
            return 0 <= src <= 10  # Only allow webcam indices 0-10
        if isinstance(src, str):
            s = src.strip()
            # Allow camera indices expressed as digits
            if s.isdigit():
                return 0 <= int(s) <= 10
            # Whitelist permitted URL schemes only
            ALLOWED_SCHEMES = ("rtsp://", "rtsps://", "http://", "https://")
            if any(s.lower().startswith(scheme) for scheme in ALLOWED_SCHEMES):
                return True
            # Allow local video files — resolve to absolute path to detect traversal
            abs_path = os.path.realpath(os.path.abspath(s))
            # Must be a real file with a safe video extension
            ALLOWED_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"}
            if os.path.isfile(abs_path) and os.path.splitext(abs_path)[1].lower() in ALLOWED_EXTS:
                return True
            return False
        return False

    def _resolve_web_url(self, url: str) -> str:
        """
        Resolve a web URL to a direct video/stream URL using yt-dlp.
        Works for YouTube, EarthCam, and many other sites.
        Falls back to the original URL if resolution fails.
        """
        try:
            # Strategy 1: EarthCam-specific extraction
            if 'earthcam.com' in url.lower():
                resolved = self._resolve_earthcam(url)
                if resolved:
                    return resolved

            # Strategy 2: yt-dlp for YouTube, Twitch, and other supported sites
            import yt_dlp
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'quiet': True,
                'no_warnings': True,
                'socket_timeout': 15,
                'retries': 2,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info:
                    # Check for direct URL
                    resolved = info.get('url', '')
                    if resolved and 'example' not in resolved:
                        print(f"[{self.camera_id}] Stream resolved via yt-dlp")
                        return resolved
                    # Check for entries (playlists / multi-stream pages)
                    entries = info.get('entries')
                    if entries:
                        entries = list(entries)
                        if entries and 'url' in entries[0]:
                            print(f"[{self.camera_id}] Playlist stream resolved to first entry")
                            return entries[0]['url']
        except ImportError:
            print(f"[{self.camera_id}] yt-dlp not installed. Cannot resolve web streams.")
        except Exception as e:
            print(f"[{self.camera_id}] Stream extraction failed: {e}")
        return url  # Fallback to raw URL

    def _resolve_earthcam(self, url: str):
        """
        EarthCam-specific stream resolver.
        EarthCam hides HLS streams behind a 3-layer embed chain:
          Page HTML → container.php (gets cam ID) → embed.php → m3u8 token URL
        """
        import urllib.request
        import re

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Referer': 'https://www.earthcam.com/'
        }

        try:
            # Step 1: Get the main page to find the container.php reference
            req = urllib.request.Request(url, headers=headers)
            html = urllib.request.urlopen(req, timeout=15).read().decode('utf-8', errors='ignore')

            # Find the container.php reference with the cam FLV name
            container_match = re.search(r'container\.php\?name=([^&"\']+\.flv)', html)
            if not container_match:
                # Try to find the cam ID from JS config directly
                flv_match = re.search(r'["\']([0-9]+\.flv)["\']', html)
                if not flv_match:
                    print(f"[{self.camera_id}] EarthCam: Could not find cam ID in page")
                    return None
                cam_flv = flv_match.group(1)
            else:
                cam_flv = container_match.group(1)

            print(f"[{self.camera_id}] EarthCam: Found cam ID: {cam_flv}")

            # Step 2: Get the embed.php JS which contains the actual m3u8 URL with auth token
            embed_url = f'https://www.earthcam.com/js/video/embed.php?vid={cam_flv}&type=h264&w=auto&requested_version=current'
            req = urllib.request.Request(embed_url, headers=headers)
            js = urllib.request.urlopen(req, timeout=15).read().decode('utf-8', errors='ignore')

            # Step 3: Extract the m3u8 URL (it's escaped with \/ in JSON)
            m3u8_match = re.search(r'(https?:\\/\\/[^\s"\']*?\.m3u8[^\s"\'\\]*)', js)
            if m3u8_match:
                m3u8_url = m3u8_match.group(1).replace('\\/', '/')
                print(f"[{self.camera_id}] EarthCam: Live stream resolved → {m3u8_url[:80]}...")
                return m3u8_url

            print(f"[{self.camera_id}] EarthCam: No m3u8 stream found in embed response")
            return None

        except Exception as e:
            print(f"[{self.camera_id}] EarthCam extraction error: {e}")
            return None

    def open(self) -> bool:
        """Open the video source. Returns True on success."""
        self.last_error = None

        # Determine source type
        src = self.source
        if isinstance(src, str):
            src = src.strip()
            if src.isdigit():
                src = int(src)

        # Security: Validate source before opening
        if not self._validate_source(src if isinstance(src, int) else self.source):
            self.last_error = f"Blocked unsafe video source: {self.source!r}"
            print(f"[SECURITY] {self.last_error}")
            return False

        # Resolve source to an actual openable URL/path
        if isinstance(src, str):
            if any(src.lower().startswith(s) for s in ("http://", "https://")):
                # Web stream — resolve via yt-dlp
                src = self._resolve_web_url(src)
                self._resolved_url = src
            elif not any(src.lower().startswith(s) for s in ("rtsp://", "rtsps://")):
                # Local file path
                src = os.path.realpath(os.path.abspath(src))

        # Open with OpenCV
        # For RTSP streams, use TCP transport for reliability
        if isinstance(src, str) and src.lower().startswith("rtsp://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        # Use FFMPEG backend explicitly for network streams (RTSP, HTTP, HLS m3u8)
        is_network = isinstance(src, str) and any(src.lower().startswith(s) for s in ("rtsp://", "rtsps://", "http://", "https://"))
        if is_network:
            self._cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        else:
            self._cap = cv2.VideoCapture(src)

        # Fallback: If OpenCV can't open the stream, try streamlink for HLS streams
        if not self._cap.isOpened() and is_network:
            print(f"[{self.camera_id}] OpenCV FFMPEG failed, trying streamlink fallback...")
            streamlink_url = self._try_streamlink(src)
            if streamlink_url:
                self._cap = cv2.VideoCapture(streamlink_url, cv2.CAP_FFMPEG)

        if not self._cap.isOpened():
            self.last_error = f"Could not open video source. Check the URL or file path."
            print(f"[{self.camera_id}] {self.last_error} (source: {str(src)[:80]})")
            return False

        # Get source FPS
        self._source_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self._source_fps <= 0 or self._source_fps > 120:
            self._source_fps = 30.0

        # Calculate frame skip interval
        self._skip_interval = max(1, int(self._source_fps / self.target_fps))

        # Set resolution if camera
        if isinstance(src, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        self._running = True
        self._frame_count = 0
        self._consecutive_failures = 0
        self._reconnect_attempts = 0
        # Mark as live stream so reconnect is allowed
        self._is_live = isinstance(src, str) and any(
            src.lower().startswith(s) for s in ("rtsp://", "rtsps://", "http://", "https://")
        )
        print(f"[{self.camera_id}] Pipeline opened. Source FPS: {self._source_fps:.1f}, "
              f"Target FPS: {self.target_fps}, Skip: {self._skip_interval}")
        return True

    def _try_streamlink(self, url: str):
        """
        Use streamlink to resolve a direct stream URL for OpenCV.
        Probes bandwidth first and picks the appropriate quality tier:
          ≥ 4 Mbps  → 1080p (best)
          ≥ 1.5 Mbps → 720p
          ≥ 0.6 Mbps → 480p
          < 0.6 Mbps → worst (lowest available)
        """
        try:
            import streamlink as sl

            streams = sl.streams(url)
            if not streams:
                print(f"[{self.camera_id}] Streamlink: No streams found")
                return None

            available = list(streams.keys())
            print(f"[{self.camera_id}] Streamlink: Available qualities: {available}")

            # Probe bandwidth once
            bw = self._probe_bandwidth_mbps()
            self._measured_bw_mbps = bw

            # Pick quality tier based on measured bandwidth
            tier = self._select_quality_tier(bw, available)
            self._quality_tier = tier

            stream_obj = streams.get(tier) or streams.get("best") or list(streams.values())[0]
            stream_url = stream_obj.url
            print(f"[{self.camera_id}] Streamlink: Selected quality='{tier}' "
                  f"(bandwidth={bw:.1f} Mbps)")
            return stream_url

        except ImportError:
            print(f"[{self.camera_id}] streamlink not installed")
        except Exception as e:
            print(f"[{self.camera_id}] Streamlink fallback failed: {e}")
        return None

    @staticmethod
    def _probe_bandwidth_mbps(test_url: str = "https://speed.cloudflare.com/__down?bytes=500000",
                              timeout: float = 3.0) -> float:
        """
        Downloads ~500 KB from Cloudflare's speed test endpoint and measures Mbps.
        Returns 0.0 on failure (treat as low bandwidth → conservative quality).
        Timeout is capped at 3 s to avoid slowing pipeline startup.
        """
        try:
            import urllib.request
            start = time.time()
            req = urllib.request.Request(test_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            elapsed = time.time() - start
            if elapsed <= 0:
                return 0.0
            mbps = (len(data) * 8) / (elapsed * 1_000_000)
            return round(mbps, 2)
        except Exception:
            return 0.0  # Assume worst-case on any error

    @staticmethod
    def _select_quality_tier(bw_mbps: float, available: list) -> str:
        """
        Map bandwidth (Mbps) to a streamlink quality label.
        Falls back gracefully if the ideal tier isn't in the stream.
        """
        # Preference order: highest fitting quality first
        if bw_mbps >= 4.0:
            preference = ["1080p", "720p", "480p", "worst", "best"]
        elif bw_mbps >= 1.5:
            preference = ["720p", "480p", "360p", "worst", "best"]
        elif bw_mbps >= 0.6:
            preference = ["480p", "360p", "240p", "worst", "best"]
        else:
            preference = ["worst", "240p", "360p", "480p", "best"]

        for q in preference:
            if q in available:
                return q
        # Last resort: first available
        return available[0] if available else "best"

    def get_frame(self) -> Optional[FramePacket]:
        """
        Get next frame, respecting target FPS via frame skipping.
        Returns None if no frame available.
        Automatically attempts reconnection for live streams on repeated failures.
        Monitors frame delivery rate and auto-degrades quality on slow connections.
        """
        if not self._running:
            return None

        # If cap is gone (post-hibernate), return None
        if not self._cap:
            return None

        with self._lock:
            self._total_frames_attempted += 1

            # Adaptive quality check every N attempts
            if (self._is_live and
                    self._total_frames_attempted % self._adaptive_check_interval == 0):
                self._maybe_downgrade_quality()

            # Skip frames to match target FPS
            grab_ok = True
            for _ in range(self._skip_interval - 1):
                if not self._cap.grab():
                    grab_ok = False
                    break
                self._frame_count += 1

            if not grab_ok:
                self._consecutive_failures += 1
                self._total_frames_failed += 1
                self._maybe_reconnect()
                return None

            ret, frame = self._cap.read()
            self._frame_count += 1

            if not ret or frame is None:
                self._consecutive_failures += 1
                self._total_frames_failed += 1
                if self._is_live:
                    self._maybe_reconnect()
                else:
                    # For local video files, end of stream = stop
                    self._running = False
                return None

            # Good frame received — reset failure counter
            self._consecutive_failures = 0
            self._reconnect_attempts = 0

            h, w = frame.shape[:2]

            # --- Performance Optimization: Resolution Capping ---
            # Processing 4K or 1080p is overkill for most city-scale tracking.
            # Downscaling here saves massive CPU in tracking, feature extraction, and HUD drawing.
            max_h = 720
            if h > max_h:
                new_w = int(w * (max_h / h))
                frame = cv2.resize(frame, (new_w, max_h), interpolation=cv2.INTER_AREA)
                h, w = max_h, new_w

            return FramePacket(
                frame=frame,
                frame_id=self._frame_count,
                timestamp=time.time(),
                camera_id=self.camera_id,
                source_fps=self._source_fps,
                width=w,
                height=h
            )

    def _maybe_reconnect(self):
        """Attempt to reconnect a dropped live stream after enough consecutive failures."""
        if not self._is_live:
            return
        if self._consecutive_failures < self._max_consecutive_failures:
            return  # Not yet — give it more chances
        now = time.time()
        if now - self._last_reconnect_time < self._reconnect_delay:
            return  # Throttle reconnects
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            print(f"[{self.camera_id}] Max reconnect attempts reached. Stopping pipeline.")
            self._running = False
            return
        self._reconnect_attempts += 1
        self._last_reconnect_time = now
        self._consecutive_failures = 0
        print(f"[{self.camera_id}] Stream dropped. Reconnecting "
              f"(attempt {self._reconnect_attempts}/{self._max_reconnect_attempts})...")
        try:
            if self._cap:
                self._cap.release()
                self._cap = None
            # Re-resolve and reopen with current (possibly degraded) quality
            src = self._resolved_url or self.source
            if isinstance(src, str) and any(src.lower().startswith(s)
                                             for s in ("http://", "https://")):
                src = self._resolve_web_url(src)
                self._resolved_url = src
            self._cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            if self._cap.isOpened():
                print(f"[{self.camera_id}] Reconnected successfully (quality='{self._quality_tier}').")
                self._reconnect_attempts = 0  # Reset on success
            else:
                print(f"[{self.camera_id}] Reconnect failed (cap not opened).")
                self._cap = None
        except Exception as e:
            print(f"[{self.camera_id}] Reconnect error: {e}")
            self._cap = None

    def _maybe_downgrade_quality(self):
        """
        If frame failure rate over the last check window exceeds 30%,
        reduce target_fps (lowers bandwidth demand without dropping the stream).
        Also re-probes bandwidth and logs the result.
        """
        if self._total_frames_attempted == 0:
            return
        failure_rate = self._total_frames_failed / self._total_frames_attempted
        # Reset counters for next window
        self._total_frames_attempted = 0
        self._total_frames_failed = 0

        if failure_rate > 0.30:
            # Re-probe bandwidth
            bw = self._probe_bandwidth_mbps()
            self._measured_bw_mbps = bw
            old_fps = self.target_fps
            # Drop target FPS by 25%, floor at min
            new_fps = max(self._min_target_fps, int(self.target_fps * 0.75))
            if new_fps < self.target_fps:
                self.target_fps = new_fps
                # Recalculate skip interval
                if self._source_fps > 0:
                    self._skip_interval = max(1, int(self._source_fps / self.target_fps))
                print(f"[{self.camera_id}] ⬇ Slow connection detected "
                      f"(failure={failure_rate:.0%}, bw={bw:.1f} Mbps). "
                      f"Reducing target FPS: {old_fps} → {self.target_fps}")
        elif failure_rate < 0.05 and self.target_fps < 8:
            # Connection improved — try stepping FPS back up
            old_fps = self.target_fps
            self.target_fps = min(8, self.target_fps + 1)
            if self._source_fps > 0:
                self._skip_interval = max(1, int(self._source_fps / self.target_fps))
            print(f"[{self.camera_id}] ⬆ Connection stable. "
                  f"Increasing target FPS: {old_fps} → {self.target_fps}")

    def get_bandwidth_mbps(self) -> Optional[float]:
        """Return last measured bandwidth in Mbps, or None if not yet measured."""
        return self._measured_bw_mbps

    def get_quality_tier(self) -> str:
        """Return the current stream quality tier label."""
        return self._quality_tier

    def is_running(self) -> bool:
        """Check if the pipeline is still active.
        Uses _running flag (not cap.isOpened) so transient HLS segment drops
        don't kill the detection loop — reconnect logic handles those.
        """
        return self._running

    def get_source_fps(self) -> float:
        return self._source_fps

    def get_frame_count(self) -> int:
        return self._frame_count

    def release(self):
        """Release the video capture resource."""
        self._running = False
        with self._lock:
            if self._cap:
                self._cap.release()
                self._cap = None

    def __del__(self):
        self.release()
