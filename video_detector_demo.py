from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from navigation.models import FramePacket, Pose
from navigation.yolo_perception import YOLOPerception


def main() -> None:
    parser = argparse.ArgumentParser(description="Run obstacle inference on a recorded video")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True, help="Path to a recorded video file")
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video file: {args.video}")

    perception = YOLOPerception(args.weights)
    frame_id = 0
    frames_per_second = capture.get(cv2.CAP_PROP_FPS)
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break

            frame_id += 1
            timestamp = frame_id / frames_per_second if frames_per_second > 0 else float(frame_id)
            packet = FramePacket(frame_id, timestamp, Pose(0, 0, 0, False), image)
            detections = perception.process(packet)
            display = perception.last_result.plot() if perception.last_result is not None else image
            for index, detection in enumerate(detections):
                label = (
                    f"{detection.label} {detection.confidence:.2f} "
                    f"~{detection.distance:.1f}m {detection.motion.value}"
                )
                cv2.putText(
                    display,
                    label,
                    (15, 30 + 25 * index),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("BVI Obstacle Detector - press Q to exit", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
