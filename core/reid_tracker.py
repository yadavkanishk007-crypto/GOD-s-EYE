"""
GOD's EYE — Person Re-Identification Tracker
Supports GPU (PyTorch MobileNetV2) and CPU (ONNX export) modes.

PRODUCTION HARDENED:
  • Lazy imports — torch/torchvision are only loaded if GPU mode or ONNX export needed.
  • On CPU, auto-exports to ONNX and uses onnxruntime for <5 MB overhead vs 150 MB.
  • On GPU, uses full PyTorch with CUDA for maximum speed.
"""

import os
import cv2
import numpy as np


class ReIDExtractor:
    """
    Lightweight Person Re-Identification feature extractor.
    Auto-selects backend:
      • GPU (cuda) → PyTorch MobileNetV2 (full CUDA acceleration)
      • CPU         → ONNX Runtime MobileNetV2 (avoids loading torch entirely)
    """
    _ONNX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "models", "reid_mobilenetv2.onnx")

    def __init__(self, device="cpu"):
        self.device_str = device
        self._use_onnx = (device == "cpu")
        self._ort_session = None
        self._torch_model = None
        self._torch_transform = None
        self._torch_pool = None
        self._torch_device = None
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy-load the model on first use."""
        if self._loaded:
            return

        if self._use_onnx and os.path.exists(self._ONNX_PATH):
            # CPU path: pure ONNX — no torch import at all
            self._load_onnx()
        elif self._use_onnx:
            # ONNX doesn't exist yet — export it (one-time cost, requires torch)
            self._export_and_load_onnx()
        else:
            # GPU path: full PyTorch power
            self._load_pytorch()

        self._loaded = True

    def _load_onnx(self):
        """Load ONNX model using onnxruntime — zero torch overhead."""
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._ort_session = ort.InferenceSession(
            self._ONNX_PATH, opts, providers=['CPUExecutionProvider']
        )
        print(f"[ReID] Loaded ONNX model: {self._ONNX_PATH}")

    def _export_and_load_onnx(self):
        """One-time export: PyTorch → ONNX, then use ONNX going forward."""
        print("[ReID] Exporting MobileNetV2 to ONNX (one-time)...")
        try:
            import torch
            from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

            model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            features = model.features
            pool = torch.nn.AdaptiveAvgPool2d((1, 1))

            class _FeatureModel(torch.nn.Module):
                def __init__(self, feat, p):
                    super().__init__()
                    self.feat = feat
                    self.pool = p
                def forward(self, x):
                    x = self.feat(x)
                    x = self.pool(x)
                    return x.flatten(1)

            export_model = _FeatureModel(features, pool)
            export_model.eval()

            os.makedirs(os.path.dirname(self._ONNX_PATH), exist_ok=True)
            dummy = torch.randn(1, 3, 256, 128)
            torch.onnx.export(
                export_model, dummy, self._ONNX_PATH,
                input_names=["input"], output_names=["embedding"],
                dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
                opset_version=11
            )
            print(f"[ReID] ONNX export successful: {self._ONNX_PATH}")

            # Now load the exported ONNX
            self._load_onnx()
        except Exception as e:
            print(f"[ReID] ONNX export failed ({e}), falling back to PyTorch")
            self._use_onnx = False
            self._load_pytorch()

    def _load_pytorch(self):
        """GPU path: full PyTorch MobileNetV2 with CUDA."""
        import torch
        import torchvision.transforms as T
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

        self._torch_device = torch.device(self.device_str)
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self._torch_model = model.features
        self._torch_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self._torch_model.to(self._torch_device)
        self._torch_model.eval()

        self._torch_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print(f"[ReID] Loaded PyTorch MobileNetV2 on {self.device_str}")

    def _preprocess_crop(self, frame, bbox):
        """Extract and validate person crop from frame."""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return crop_rgb

    def extract_feature(self, frame, bbox):
        """
        Extract a 1280-dimensional feature vector for a person crop.
        bbox is [x1, y1, x2, y2].
        """
        self._ensure_loaded()

        crop_rgb = self._preprocess_crop(frame, bbox)
        if crop_rgb is None:
            return None

        if self._use_onnx and self._ort_session is not None:
            return self._extract_onnx(crop_rgb)
        else:
            return self._extract_pytorch(crop_rgb)

    def _extract_onnx(self, crop_rgb):
        """ONNX inference — no torch required."""
        # Manual preprocessing (replaces torchvision transforms)
        img = cv2.resize(crop_rgb, (128, 256))  # W, H for cv2
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        # HWC → CHW → NCHW
        blob = img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        outputs = self._ort_session.run(None, {"input": blob})
        embedding = outputs[0][0]
        # L2 normalize
        norm = np.linalg.norm(embedding) + 1e-8
        return embedding / norm

    def _extract_pytorch(self, crop_rgb):
        """PyTorch inference (GPU path)."""
        import torch

        tensor = self._torch_transform(crop_rgb).unsqueeze(0).to(self._torch_device)

        with torch.no_grad():
            features = self._torch_model(tensor)
            pooled = self._torch_pool(features).flatten()
            embedding = torch.nn.functional.normalize(pooled, p=2, dim=0)

        return embedding.cpu().numpy()

    def compute_similarity(self, emb1, emb2):
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(emb1, emb2))
