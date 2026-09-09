from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an Ultralytics obstacle detector")
    parser.add_argument("--data", type=Path, default=Path("dataset/obstacle_dataset.yaml"))
    parser.add_argument("--model", default="yolo11n.pt", help="Use yolov8n.pt for the comparison run")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="obstacle_yolo11n")
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Install CUDA-enabled PyTorch or use --device cpu.")
    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project="runs/detect",
        name=args.name,
        pretrained=True,
        amp=True,
        cache=False,
        patience=15,
        save=True,
        plots=True,
        seed=42,
        deterministic=True,
    )


if __name__ == "__main__":
    main()

