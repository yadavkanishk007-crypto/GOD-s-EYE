import pytest
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.reid_tracker import ReIDExtractor

def test_reid_extractor_initialization():
    extractor = ReIDExtractor(device="cpu")
    assert extractor is not None
    assert extractor.feature_extractor is not None

def test_reid_feature_extraction():
    extractor = ReIDExtractor(device="cpu")
    
    # Create a dummy frame (RGB)
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Valid bbox [x1, y1, x2, y2]
    bbox = [100, 100, 200, 300]
    
    emb = extractor.extract_feature(dummy_frame, bbox)
    assert emb is not None
    assert isinstance(emb, np.ndarray)
    assert len(emb.shape) == 1
    assert emb.shape[0] > 0
    
    # Norm should be close to 1 (cosine similarity prep)
    norm = np.linalg.norm(emb)
    assert np.isclose(norm, 1.0, atol=1e-5)

def test_reid_too_small_crop():
    extractor = ReIDExtractor(device="cpu")
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Too small bbox
    bbox = [10, 10, 15, 15]
    emb = extractor.extract_feature(dummy_frame, bbox)
    assert emb is None

def test_compute_similarity():
    extractor = ReIDExtractor(device="cpu")
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([1.0, 0.0, 0.0])
    emb3 = np.array([0.0, 1.0, 0.0])
    
    # Same vector -> sim = 1.0
    assert np.isclose(extractor.compute_similarity(emb1, emb2), 1.0)
    # Orthogonal -> sim = 0.0
    assert np.isclose(extractor.compute_similarity(emb1, emb3), 0.0)
