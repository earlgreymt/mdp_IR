"""
Run a trained YOLOv8-seg model on an image or a whole folder of images.
Saves annotated images (boxes + masks drawn) and a detections.json keyed by
filename, in the shape the Image Recognition server would send back to the
RPi: class name, Image ID, confidence, and box.

Usage:
    python predict.py --weights weights/best.pt --source path/to/image.jpg
    python predict.py --weights weights/best.pt --source ../data/test/images --name test_check
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from image_id_map import get_image_id

RUNS_DIR = Path(__file__).parent / "runs"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True, type=str)
    p.add_argument("--source", required=True, type=str, help="image file OR folder of images")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--name", default="predict", help="output subfolder under image_rec/runs/")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        verbose=False,
        save=True,  # writes annotated images (boxes + masks drawn) alongside detections.json
        project=str(RUNS_DIR),
        name=args.name,
    )

    all_detections = {}
    for result in results:
        detections = []
        for box in result.boxes:
            cls_idx = int(box.cls[0])
            cls_name = model.names[cls_idx]
            try:
                image_id = get_image_id(cls_name)
            except KeyError:
                image_id = None  # e.g. "Bullseye" -- no Image ID defined yet
            detections.append(
                {
                    "class_name": cls_name,
                    "image_id": image_id,
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox_xyxy": [round(v, 1) for v in box.xyxy[0].tolist()],
                }
            )
        all_detections[Path(result.path).name] = detections

    if results:
        save_dir = Path(results[0].save_dir)
        out_json = save_dir / "detections.json"
        out_json.write_text(json.dumps(all_detections, indent=2))
        print(f"Annotated images + detections.json saved to: {save_dir}")

    print(json.dumps(all_detections, indent=2))


if __name__ == "__main__":
    main()
