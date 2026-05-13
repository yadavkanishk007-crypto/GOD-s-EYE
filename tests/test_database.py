import pytest
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from storage.database import Database

@pytest.fixture
def test_db():
    db_path = "data/test_gods_eye.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    db = Database(db_path)
    yield db
    if os.path.exists(db_path):
        os.remove(db_path)

def test_insert_and_get_events(test_db):
    test_db.insert_event(
        timestamp=time.time(),
        camera_id="cam_01",
        event_type="fire",
        severity="high",
        confidence=0.95,
        description="Fire detected",
        clip_path="dummy.mp4"
    )
    
    events = test_db.get_recent_events(limit=10)
    assert len(events) == 1
    assert events[0]["event_type"] == "fire"
    assert events[0]["camera_id"] == "cam_01"

def test_log_time_series(test_db):
    test_db.log_time_series(
        timestamp="2026-01-01 12:00:00",
        camera_id="cam_02",
        people_count=10,
        vehicle_count=5,
        crowd_density=0.8,
        avg_speed=1.5,
        speed_variance=0.2,
        direction_entropy=0.5,
        risk_score=0.6,
        event_flag=0,
        incident_count=0
    )
    
    logs = test_db.get_recent_logs(camera_id="cam_02", limit=10)
    assert len(logs) == 1
    assert logs[0]["people_count"] == 10
    assert logs[0]["camera_id"] == "cam_02"

def test_sql_injection_safety(test_db):
    # Try an injection in the description
    malicious_desc = "fire'); DROP TABLE events; --"
    test_db.insert_event(
        timestamp=time.time(),
        camera_id="cam_01",
        event_type="fire",
        severity="high",
        confidence=0.9,
        description=malicious_desc,
        clip_path=""
    )
    
    # Table should still exist and contain the event
    events = test_db.get_recent_events(limit=10)
    assert len(events) == 1
    assert events[0]["description"] == malicious_desc
