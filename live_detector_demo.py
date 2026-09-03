from __future__ import annotations

import argparse
from pathlib import Path

from navigation.models import FramePacket, Pose
from navigation.yolo_perception import YOLOPerception


def main() -> None:
    parser = argparse.ArgumentParser(description="Real webcam/video obstacle inference")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", default="0", help="0 for webcam or path to video")
    args = parser.parse_args()
    import cv2

    source: int | str = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Cannot open source: {source}")
    perception = YOLOPerception(args.weights)
    frame_id = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            frame_id += 1
            packet = FramePacket(frame_id, float(frame_id), Pose(0, 0, 0, False), image)
            detections = perception.process(packet)
            display = perception.last_result.plot() if perception.last_result is not None else image
            for detection in detections:
                text = f"{detection.label} {detection.confidence:.2f} ~{detection.distance:.1f}m {detection.motion.value}"
                cv2.putText(display, text, (15, 30 + 25 * detections.index(detection)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("BVI Obstacle Detector - press Q to exit", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
