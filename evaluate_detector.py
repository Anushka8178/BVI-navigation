from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained obstacle detector")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("dataset/obstacle_dataset.yaml"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()

    from ultralytics import YOLO

    metrics = YOLO(str(args.weights)).val(data=str(args.data.resolve()), split=args.split, device=0, plots=True)
    summary = {
        "split": args.split,
        "mAP50-95": float(metrics.box.map),
        "mAP50": float(metrics.box.map50),
        "precision_mean": float(metrics.box.mp),
        "recall_mean": float(metrics.box.mr),
    }
    output = args.weights.parent / f"evaluation_{args.split}.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

