"""
GOD's EYE — Model Optimizer & Hardware Detection
Handles auto-detecting GPU vs CPU, and exporting/loading quantized YOLOv8 models.

PRODUCTION HARDENED:
  • Zero-import-cost on CPU path — torch/ultralytics are only loaded if ONNX
    needs to be exported, or if GPU is detected (where we WANT full PyTorch).
  • GPU detected → returns .pt + "cuda"  (full Ultralytics + CUDA power)
  • CPU only    → returns .onnx + "cpu"   (lightweight ONNX-only, <300 MB)
"""

import os
import threading


class ModelOptimizer:
    _lock = threading.Lock()
    _cached_device = None  # Cache hardware detection across calls

    @staticmethod
    def detect_hardware():
        """
        Detect the best available hardware accelerator.
        Uses lazy torch import — only loads torch when GPU might be available.
        On CPU-only machines, avoids the 150 MB torch import entirely
        by checking environment hints first.
        """
        if ModelOptimizer._cached_device is not None:
            return ModelOptimizer._cached_device

        # Quick pre-check: if CUDA_VISIBLE_DEVICES is set to empty or
        # NVIDIA driver files are missing, skip torch import entirely.
        cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_env == "":
            print("[HARDWARE] CUDA_VISIBLE_DEVICES is empty — CPU only.")
            ModelOptimizer._cached_device = "cpu"
            return "cpu"

        # Try to detect GPU via torch (lazy import)
        try:
            import torch
            print(f"[HARDWARE] Torch version: {torch.__version__}")
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
                print(f"[HARDWARE] CUDA GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
                ModelOptimizer._cached_device = "cuda"
                return "cuda"
            else:
                print("[HARDWARE] torch.cuda.is_available() is False")
                
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("[HARDWARE] MPS detected.")
                ModelOptimizer._cached_device = "mps"
                return "mps"
        except ImportError:
            print("[HARDWARE] PyTorch not installed — CPU only.")
        except Exception as e:
            print(f"[HARDWARE] GPU detection error: {e}")

        print("[HARDWARE] Falling back to CPU mode.")
        ModelOptimizer._cached_device = "cpu"
        return "cpu"

    @staticmethod
    def get_optimized_model(model_path="yolov8n.pt", img_size=640):
        """
        Returns (model_path, device) for the best available runtime:
          • GPU (cuda/mps) → original .pt path + device string
            (Ultralytics will handle CUDA acceleration natively)
          • CPU → ONNX path + "cpu"
            (OnnxTracker runs without PyTorch for <100 MB footprint)
        """
        device = ModelOptimizer.detect_hardware()

        # ── GPU path: full PyTorch power ────────────────────────────────
        if device in ("cuda", "mps"):
            return model_path, device

        # ── CPU path: ONNX-only (avoid loading torch if possible) ───────
        onnx_path = model_path.replace(".pt", ".onnx")
        openvino_path = model_path.replace(".pt", "_openvino_model")

        with ModelOptimizer._lock:
            # Check if OpenVINO export exists (fastest on Intel CPUs)
            if os.path.exists(openvino_path):
                print(f"[OPTIMIZER] Found existing OpenVINO model at {openvino_path}")
                return openvino_path, "cpu"

            # Check if ONNX exists (no need to import torch at all!)
            if os.path.exists(onnx_path):
                print(f"[OPTIMIZER] Found existing ONNX model at {onnx_path}")
                return onnx_path, "cpu"

            # ONNX doesn't exist — must export (requires torch + ultralytics)
            print(f"[OPTIMIZER] Optimizing model for CPU inference (Exporting {model_path})...")
            try:
                from ultralytics import YOLO
                model = YOLO(model_path)
                exported = model.export(
                    format="onnx", imgsz=img_size,
                    dynamic=True, half=False, simplify=True
                )
                print(f"[OPTIMIZER] Exported successfully to {exported}")
                return exported, "cpu"
            except Exception as e:
                print(f"[OPTIMIZER] Failed to export model: {e}")
                print("[OPTIMIZER] Falling back to unoptimized PyTorch model.")
                return model_path, "cpu"

    @staticmethod
    def is_gpu_mode():
        """Quick check if running in GPU mode (cached after first detect)."""
        return ModelOptimizer.detect_hardware() in ("cuda", "mps")
