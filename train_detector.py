from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO for obstacle detection")
    parser.add_argument("--data", type=Path, default=Path("dataset/obstacle_dataset.yaml"))
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="obstacle_yolov8n")
    args = parser.parse_args()

    from ultralytics import YOLO
    import torch

    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Verify the NVIDIA driver and CUDA-enabled PyTorch installation.")
    if torch.cuda.is_available():
        print(f"Training on: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

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

