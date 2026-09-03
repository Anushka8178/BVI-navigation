# RTX 3050 Training Guide (Windows)

## Recommended scope

Fine-tune a pretrained YOLOv8 nano model. Do not train from random initialization. With 4 GB VRAM, start with `imgsz=640` and `batch=4`; if CUDA runs out of memory, reduce to `batch=2`, then `imgsz=512` only if needed.

## 1. Create the environment

Install Python 3.11 and an up-to-date NVIDIA driver. Open PowerShell inside the project:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Use the command generated for Windows/Pip/Python/CUDA on the official PyTorch “Start Locally” page. Then install the remaining packages:

```powershell
pip install -r requirements-ml.txt
```

Verify that PyTorch sees the GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Do not begin training unless this prints `True` and `NVIDIA GeForce RTX 3050 Laptop GPU`.

## 2. Download and prepare the dataset

Download the Obstacle Dataset from the link supplied by its official repository. Extract the VOC-format directory containing:

```text
JPEGImages/
Annotations/
ImageSets/Main/train.txt
ImageSets/Main/val.txt
ImageSets/Main/test.txt
```

Convert and validate it:

```powershell
python tools/prepare_obstacle_dataset.py --source "D:\path\to\Obstacle-Dataset" --output datasets\obstacle_yolo
```

Open `dataset/obstacle_dataset.yaml` and replace its `path` value with the absolute output path, using forward slashes, for example:

```yaml
path: D:/BVI-project/datasets/obstacle_yolo
```

## 3. Run a short smoke test

```powershell
python train_detector.py --epochs 2 --batch 2 --name smoke_test
```

Confirm that training uses `CUDA:0`, finishes two epochs, and produces `runs/detect/smoke_test/weights/best.pt`.

## 4. Run the review-quality training

```powershell
python train_detector.py --epochs 60 --batch 4 --imgsz 640 --name obstacle_yolov8n
```

The training script uses pretrained weights, mixed precision, early stopping, deterministic seed 42, plots, and validation after each epoch. Keep the laptop plugged in, set Windows power mode to Best Performance, and ensure ventilation.

If you receive a CUDA out-of-memory error:

```powershell
python train_detector.py --epochs 60 --batch 2 --imgsz 640 --name obstacle_yolov8n_b2
```

## 5. Evaluate on the untouched test split

```powershell
python evaluate_detector.py --weights runs\detect\obstacle_yolov8n\weights\best.pt --split test
```

Present these artifacts in the review:

- `results.png`: training/validation loss and metric curves
- `confusion_matrix.png`: class confusions
- `PR_curve.png`: precision-recall behavior
- `evaluation_test.json`: mAP50-95, mAP50, precision, and recall
- `weights/best.pt`: selected checkpoint

Do not select settings based on the test set. Use validation results while developing and run the test evaluation only for the final report.

## 6. Demonstrate real inference

Webcam:

```powershell
python live_detector_demo.py --weights runs\detect\obstacle_yolov8n\weights\best.pt --source 0
```

Recorded video:

```powershell
python live_detector_demo.py --weights runs\detect\obstacle_yolov8n\weights\best.pt --source "D:\demo\campus_walk.mp4"
```

The displayed distance is a monocular approximation based on bounding-box size. Call it “estimated distance,” not RGB-D depth. The detector and tracker are real; calibrated depth and SLAM remain the next hardware-dependent stage.

## What makes this good enough for the review

You can now show three distinct proofs:

1. Dataset proof: conversion statistics and class counts.
2. ML proof: loss curves, precision, recall, mAP, confusion matrix, and a saved checkpoint.
3. Architecture proof: detections use the same `Detection` interface consumed by hazard triage, memory, attention, planning, and audio modules.

