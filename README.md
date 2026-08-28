# image_rec — YOLOv8 training for SC2079 MDP symbol recognition

Trains a YOLOv8 **segmentation** model on `../data` (31 classes: digits,
letters, arrows/stop, Bullseye). Segmentation is used instead of plain
detection because the dataset's labels are polygon masks, and a `-seg` model
gives both bounding boxes and masks from one trained model.

## Setup

A conda env `yolov8` (Python 3.11) already exists. Your GPU (RTX 5060) is a
Blackwell-generation card and needs a CUDA 12.8 PyTorch build:

```powershell
conda activate yolov8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Verify the GPU is visible before training:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Train

```powershell
python train.py                              # yolov8n-seg, 100 epochs, imgsz 640
python train.py --model s --epochs 150        # larger model
python train.py --model n --epochs 5 --name smoke_test   # quick sanity run first
```

Runs (weights, curves, val predictions) are written to `runs/segment/<name>/`.
Best weights end up at `runs/segment/<name>/weights/best.pt`.

## Predict / sanity-check a trained model

```powershell
python predict.py --weights runs/segment/train/weights/best.pt --source path\to\image.jpg
```

Prints each detection's class name, confidence, box, and the arena **Image
ID** (see `image_id_map.py`) -- i.e. the same shape the Image Recognition
server should hand back to the RPi.

## Files

- `train.py` — training entry point (CLI args for model size, epochs, imgsz, device, etc.)
- `predict.py` — single-image inference + Image ID lookup, for sanity-checking a checkpoint
- `image_id_map.py` — YOLO class name → arena Image ID (11-40) lookup table. This
  is what the IR server uses to translate model output into what RPi/Android expect.
  **Open item:** the `Bullseye` class has no Image ID in the arena reference chart —
  decide how it should be reported before wiring this into the real server.
- `runs/` — training outputs (gitignored)

## Notes

- `data.yaml` lives in `../data` and is referenced directly (not copied) —
  train/valid/test image and label counts there already match 1:1.
- Class index order comes from `data.yaml`'s `names` list; it is **not**
  stable across dataset re-exports, which is why `image_id_map.py` keys off
  class *name*, not raw index.
