"""
Train a YOLOv8 segmentation model on the SC2079 MDP symbol dataset.

A segmentation model is used (rather than a plain detector) because the
dataset's labels are polygon masks, and a -seg model natively outputs both
bounding boxes and masks -- covering the "detection and segmentation"
requirement from a single trained model.

Usage:
    python train.py
    python train.py --model s --epochs 150 --imgsz 640
    python train.py --model n --epochs 5 --name smoke_test   # quick sanity run
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path(r"C:\Users\ma0011in\mdp\data\data.yaml")
RUNS_DIR = Path(__file__).parent / "runs"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="YOLOv8 segmentation model size (default: n)",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=-1, help="batch size, -1 = auto")
    p.add_argument("--device", default=0, help="cuda device index, or 'cpu'")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=30, help="early-stop patience")
    p.add_argument("--data", type=Path, default=DATA_YAML)
    p.add_argument("--name", default=None, help="run name under image_rec/runs/segment/")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    weights = f"yolov8{args.model}-seg.pt"
    model = YOLO(weights)

    model.train(
        data=str(args.data),
        task="segment",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(RUNS_DIR),
        name=args.name,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
