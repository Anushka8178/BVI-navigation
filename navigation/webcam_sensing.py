from __future__ import annotations

import time
from collections.abc import Iterator

from .models import FramePacket, Pose


class WebcamSensor:
    """Camera/video adapter. Pose is stationary until real SLAM is attached."""

    def __init__(self, source: int | str = 0) -> None:
        self.source = source

    def __iter__(self) -> Iterator[FramePacket]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is not installed") from exc
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open camera/video source: {self.source}")
        frame_id = 0
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                frame_id += 1
                yield FramePacket(
                    frame_id=frame_id,
                    timestamp=time.monotonic(),
                    pose=Pose(0.0, 0.0, 0.0, localization_valid=False),
                    image=image,
                )
        finally:
            capture.release()
