from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO-only video diagnostic")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {args.source.resolve()}")
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = model.track(frame, persist=True, conf=args.confidence, device=args.device, verbose=False)[0]
        cv2.imshow("Detector diagnostic - Q to exit", result.plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

