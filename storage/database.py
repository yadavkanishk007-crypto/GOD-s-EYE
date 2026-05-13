"""
GOD's EYE — SQLite Database Manager
Logs time-series metrics and discrete events.

PRODUCTION HARDENED (v2):
  • Uses thread-local persistent connections to avoid constant connect/disconnect overhead.
  • Created indexes on (camera_id, timestamp) for much faster prediction queries.
  • Uses WAL (Write-Ahead Logging) mode to prevent "database is locked" errors during concurrent reads/writes.
"""

import sqlite3
import threading
import os
from typing import List, Dict


class Database:
    """
    Manages SQLite connections for time-series logging and event storage.
    Uses thread-local storage so that each thread (main UI thread, prediction thread, etc.)
    has exactly ONE persistent connection, eliminating connection churn and lock contention.
    """
    _local = threading.local()

    def __init__(self, db_path="data/gods_eye.db"):
        self.db_path = db_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_schema()

    def _get_conn(self):
        """Get or create a persistent thread-local connection."""
        if not hasattr(self._local, "conn"):
            # check_same_thread=False is technically not needed since we use thread-local,
            # but it's safe to have. isolation_level=None enables autocommit for WAL mode.
            self._local.conn = sqlite3.connect(
                self.db_path,
                timeout=15.0,
                check_same_thread=False,
                isolation_level=None 
            )
            # Enable Write-Ahead Logging for vastly improved concurrency
            self._local.conn.execute('PRAGMA journal_mode=WAL;')
            self._local.conn.execute('PRAGMA synchronous=NORMAL;')
        return self._local.conn

    def _init_schema(self):
        """Create tables and indexes if they don't exist."""
        conn = self._get_conn()
        # Time-series log
        conn.execute("""
            CREATE TABLE IF NOT EXISTS time_series_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                camera_id TEXT,
                people_count INTEGER,
                vehicle_count INTEGER,
                crowd_density REAL,
                avg_speed REAL,
                speed_variance REAL,
                direction_entropy REAL,
                risk_score REAL,
                event_flag INTEGER
            )
        """)
        # Discrete events log
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                camera_id TEXT,
                event_type TEXT,
                risk_level TEXT,
                recommendation TEXT,
                clip_path TEXT
            )
        """)
        # Indexes for fast querying (crucial for prediction module)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ts_cam_time ON time_series_log(camera_id, timestamp);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_cam_time ON events_log(camera_id, timestamp);")

        # Migration: Ensure risk_score exists if table was created with old crowd_risk name
        try:
            cursor = conn.execute("PRAGMA table_info(time_series_log)")
            columns = [info[1] for info in cursor.fetchall()]
            if "risk_score" not in columns:
                if "crowd_risk" in columns:
                    conn.execute("ALTER TABLE time_series_log RENAME COLUMN crowd_risk TO risk_score")
                    print("[Database] Migrated crowd_risk to risk_score.")
                else:
                    conn.execute("ALTER TABLE time_series_log ADD COLUMN risk_score REAL DEFAULT 0.0")
                    print("[Database] Added missing risk_score column.")
        except sqlite3.Error as e:
            print(f"[Database] Migration error: {e}")
        
        # We don't commit here because isolation_level=None (autocommit)

    def log_time_series(self, timestamp: str, camera_id: str, features, risk, event_flag: int = 0):
        """Insert a row into the time-series log."""
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT INTO time_series_log (
                    timestamp, camera_id, people_count, vehicle_count,
                    crowd_density, avg_speed, speed_variance, direction_entropy,
                    risk_score, event_flag
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, camera_id,
                features.people_count, features.vehicle_count,
                features.crowd_density, features.avg_speed,
                features.speed_variance, features.direction_entropy,
                risk.risk_score, event_flag
            ))
        except sqlite3.Error as e:
            print(f"[Database] Error logging time series: {e}")

    def log_event(self, timestamp: str, camera_id: str, event_type: str, 
                  risk_level: str, recommendation: str, clip_path: str = ""):
        """Insert a row into the discrete events log."""
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT INTO events_log (
                    timestamp, camera_id, event_type, risk_level,
                    recommendation, clip_path
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                timestamp, camera_id, event_type, risk_level,
                recommendation, clip_path
            ))
        except sqlite3.Error as e:
            print(f"[Database] Error logging event: {e}")

    def get_recent_metrics(self, camera_id: str, limit: int = 60) -> List[Dict]:
        """Fetch the most recent N metrics for a specific camera (used by Prediction Engine)."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT timestamp, people_count, vehicle_count, crowd_density,
                       avg_speed, speed_variance, direction_entropy, risk_score
                FROM time_series_log
                WHERE camera_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (camera_id, limit))
            
            rows = cursor.fetchall()
            
            metrics = []
            # We want them in chronological order, so reverse the DESC results
            for row in reversed(rows):
                metrics.append({
                    "timestamp": row[0],
                    "people_count": row[1],
                    "vehicle_count": row[2],
                    "crowd_density": row[3],
                    "avg_speed": row[4],
                    "speed_variance": row[5],
                    "direction_entropy": row[6],
                    "risk_score": row[7]
                })
            return metrics
        except sqlite3.Error as e:
            print(f"[Database] Error fetching metrics: {e}")
            return []

    def close(self):
        """Close the thread-local connection if it exists."""
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn

    def prune_old_data(self, days: int = 30):
        """Delete time-series and event logs older than the specified number of days."""
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM time_series_log WHERE timestamp < datetime('now', ?)", (f"-{days} days",))
            conn.execute("DELETE FROM events_log WHERE timestamp < datetime('now', ?)", (f"-{days} days",))
            print(f"[Database] Pruned data older than {days} days.")
        except sqlite3.Error as e:
            print(f"[Database] Error pruning old data: {e}")
