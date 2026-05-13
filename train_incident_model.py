"""
GOD's EYE — YOLO Fine-Tuner for Incident Detection
====================================================
Trains a YOLOv8n model on custom incident data (fire, smoke, accident).

USAGE
-----
1. Prepare your dataset (see README below).
2. Run:
       python train_incident_model.py --epochs 50 --data data/incident_dataset/dataset.yaml

The trained model will be exported as:
   data/models/incident_yolov8n.onnx
   data/models/incident_yolov8n.pt

DATASET STRUCTURE
-----------------
data/incident_dataset/
  dataset.yaml          (class definitions + paths, auto-created by this script)
  images/train/         (put your training images here, JPG/PNG)
  images/val/           (validation images)
  labels/train/         (YOLO format .txt files matching image names)
  labels/val/

YOLO LABEL FORMAT (one object per line):
  <class_id> <cx> <cy> <w> <h>   (all normalized 0-1)

Classes:
  0 = fire
  1 = smoke
  2 = accident / crash
  3 = person_fallen

FREE DATASET SOURCES
--------------------
  Fire/Smoke: https://github.com/ultralytics/yolov5  (fire dataset)
              https://www.kaggle.com/datasets/phylake1337/fire-dataset
  Accidents:  https://www.kaggle.com/datasets/anuragbantu/accident-detection-dataset
              https://universe.roboflow.com/ (search "car accident")

AUTO-ANNOTATION (no labels yet?)
---------------------------------
Run: python train_incident_model.py --auto-annotate images/train
This uses the base COCO model + fire/smoke HSV heuristics to bootstrap labels.
"""

import argparse
import os
import sys
import shutil
import yaml
from pathlib import Path


# ─── Class definitions ────────────────────────────────────────────────────────

INCIDENT_CLASSES = {
    0: "fire",
    1: "smoke",
    2: "accident",
    3: "person_fallen",
}

DATASET_YAML_TEMPLATE = """\
# GOD's EYE Incident Detection Dataset
path: {dataset_path}
train: images/train
val:   images/val

nc: {num_classes}
names: {class_names}
"""


# ─── Dataset scaffolding ─────────────────────────────────────────────────────

def scaffold_dataset(base_dir: Path):
    """Create dataset directory structure if it doesn't exist."""
    for split in ("train", "val"):
        (base_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (base_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    yaml_path = base_dir / "dataset.yaml"
    if not yaml_path.exists():
        content = DATASET_YAML_TEMPLATE.format(
            dataset_path=str(base_dir.resolve()),
            num_classes=len(INCIDENT_CLASSES),
            class_names=list(INCIDENT_CLASSES.values())
        )
        yaml_path.write_text(content)
        print(f"[Scaffold] Created dataset.yaml at {yaml_path}")
    else:
        print(f"[Scaffold] Dataset YAML already exists: {yaml_path}")

    return yaml_path


# ─── Auto-annotation bootstrapper ────────────────────────────────────────────

def auto_annotate(image_dir: str, output_label_dir: str):
    """
    Bootstrap YOLO labels using HSV fire/smoke detection + YOLO person detection.
    Saves .txt label files next to images.
    Useful when you have images but no labels yet.
    """
    import cv2
    import numpy as np
    from ultralytics import YOLO

    print("[AutoAnnotate] Loading base YOLOv8n model for person/vehicle detection...")
    model = YOLO("yolov8n.pt")

    img_dir = Path(image_dir)
    lbl_dir = Path(output_label_dir)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    print(f"[AutoAnnotate] Found {len(images)} images in {img_dir}")

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        labels = []

        # ── Fire detection via HSV ────────────────────────────────────────
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        fire_mask = cv2.inRange(hsv,
                                np.array([0,   120, 180]),
                                np.array([25,  255, 255]))
        fire_mask2 = cv2.inRange(hsv,
                                 np.array([170, 120, 180]),
                                 np.array([179, 255, 255]))
        fire_mask = cv2.bitwise_or(fire_mask, fire_mask2)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 800:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            labels.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")   # class 0 = fire

        # ── Smoke detection via HSV ───────────────────────────────────────
        smoke_mask = cv2.inRange(hsv,
                                 np.array([0,   0,  100]),
                                 np.array([179, 55, 210]))
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 3000:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            labels.append(f"1 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")   # class 1 = smoke

        # ── Person fallen heuristic ───────────────────────────────────────
        results = model(frame, verbose=False, classes=[0])  # class 0 = person
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bw_px = x2 - x1
                bh_px = y2 - y1
                aspect = bh_px / max(bw_px, 1)
                if aspect < 0.6:  # wide bounding box = possibly fallen
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    nw = bw_px / w
                    nh = bh_px / h
                    labels.append(f"3 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")  # class 3 = fallen

        if labels:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            lbl_path.write_text("\n".join(labels))
            print(f"[AutoAnnotate] {img_path.name} → {len(labels)} labels")
        else:
            print(f"[AutoAnnotate] {img_path.name} → no detections (skipping)")

    print("[AutoAnnotate] Done.")


# ─── Training ─────────────────────────────────────────────────────────────────

def train(data_yaml: str, epochs: int, batch: int, img_size: int, output_dir: str):
    """Fine-tune YOLOv8n on incident dataset."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print("=" * 60)
    print("GOD's EYE — YOLO Incident Model Training")
    print("=" * 60)
    print(f"  Dataset : {data_yaml}")
    print(f"  Epochs  : {epochs}")
    print(f"  Batch   : {batch}")
    print(f"  ImgSize : {img_size}")
    print(f"  Output  : {output_dir}")
    print("=" * 60)

    # Start from pretrained nano model (transfer learning)
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=img_size,
        device="cpu",          # Change to 0 if you have GPU
        workers=2,
        patience=15,           # Early stopping
        project=output_dir,
        name="incident_run",
        exist_ok=True,
        augment=True,          # Auto-augmentation (mosaic, flip, HSV shifts)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        verbose=True
    )

    # Export best checkpoint to ONNX for deployment
    best_pt = Path(output_dir) / "incident_run" / "weights" / "best.pt"
    if best_pt.exists():
        print(f"\n[Export] Exporting {best_pt} → ONNX...")
        best_model = YOLO(str(best_pt))
        best_model.export(format="onnx", imgsz=img_size, simplify=True)

        # Copy to GOD's EYE model directory
        onnx_src = best_pt.with_suffix(".onnx")
        onnx_dst = Path("data/models/incident_yolov8n.onnx")
        pt_dst   = Path("data/models/incident_yolov8n.pt")
        onnx_dst.parent.mkdir(parents=True, exist_ok=True)

        if onnx_src.exists():
            shutil.copy2(onnx_src, onnx_dst)
            print(f"[Export] Saved ONNX → {onnx_dst}")
        shutil.copy2(best_pt, pt_dst)
        print(f"[Export] Saved PT   → {pt_dst}")
        print("\n✅ Training complete! Update config.yaml to use 'incident_yolov8n.onnx'")
    else:
        print("[WARNING] best.pt not found — training may have failed.")

    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GOD's EYE YOLO Incident Model Trainer"
    )
    parser.add_argument("--data", type=str,
                        default="data/incident_dataset/dataset.yaml",
                        help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (reduce if RAM is limited)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--output", type=str, default="data/models_test",
                        help="Output directory for training runs")
    parser.add_argument("--scaffold", action="store_true",
                        help="Create dataset folder structure and exit")
    parser.add_argument("--auto-annotate", type=str, metavar="IMAGE_DIR",
                        help="Auto-generate YOLO labels for images in this directory")
    parser.add_argument("--label-out", type=str, default=None,
                        help="Output dir for auto-annotation labels (default: IMAGE_DIR/../labels/train)")
    args = parser.parse_args()

    # Scaffold dataset structure
    if args.scaffold:
        dataset_dir = Path(args.data).parent
        yaml_path = scaffold_dataset(dataset_dir)
        print(f"\n[Scaffold] Populate the following folders with your images and labels:")
        print(f"   {dataset_dir / 'images' / 'train'}")
        print(f"   {dataset_dir / 'labels' / 'train'}")
        print(f"\nClass IDs: {INCIDENT_CLASSES}")
        return

    # Auto-annotate
    if args.auto_annotate:
        img_dir = Path(args.auto_annotate)
        lbl_dir = Path(args.label_out) if args.label_out else \
                  img_dir.parent.parent / "labels" / "train"
        auto_annotate(str(img_dir), str(lbl_dir))
        return

    # Validate dataset YAML exists
    if not Path(args.data).exists():
        print(f"[ERROR] Dataset YAML not found: {args.data}")
        print("Run with --scaffold to create the folder structure first.")
        sys.exit(1)

    # Train
    train(
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        img_size=args.imgsz,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
