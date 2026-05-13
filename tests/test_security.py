import pytest
import os
import sys

# Add root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.video_pipeline import VideoPipeline
from ui.main_window import MainWindow

def test_video_pipeline_validate_source():
    # Safe sources
    assert VideoPipeline._validate_source(0) == True
    assert VideoPipeline._validate_source("rtsp://admin:pass@192.168.1.100:554/stream") == True
    assert VideoPipeline._validate_source("https://example.com/video.mp4") == True
    
    # Create a dummy valid file
    safe_file = "test_safe.mp4"
    with open(safe_file, "w") as f:
        f.write("dummy")
    assert VideoPipeline._validate_source(safe_file) == True
    os.remove(safe_file)

    # Unsafe sources
    assert VideoPipeline._validate_source("../../etc/passwd") == False
    assert VideoPipeline._validate_source("file:///C:/Windows/System32/cmd.exe") == False

def test_jail_path():
    # Create a dummy config and main window without UI initialization
    class DummyConfig(dict):
        def get(self, k, d=None):
            return d
            
    # We bypass MainWindow's full __init__ for testing the method
    mw = MainWindow.__new__(MainWindow)
    
    # Test valid sub-path
    expected_valid = os.path.realpath(os.path.abspath("data/gods_eye.db"))
    assert mw._jail_path("data/gods_eye.db", default_dir="data") == expected_valid
    
    # Test path traversal attempt
    traversal_path = "../../windows/system32/evil.dll"
    jailed = mw._jail_path(traversal_path, default_dir="data")
    expected_jailed = os.path.realpath(os.path.abspath(os.path.join("data", "evil.dll")))
    assert jailed == expected_jailed

    # Test absolute outside path
    abs_traversal = "C:\\evil.db" if os.name == 'nt' else "/etc/passwd"
    jailed_abs = mw._jail_path(abs_traversal, default_dir="data")
    expected_abs = os.path.realpath(os.path.abspath(os.path.join("data", os.path.basename(abs_traversal))))
    assert jailed_abs == expected_abs
