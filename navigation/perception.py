from __future__ import annotations

from .models import Detection, FramePacket, Motion


class ScriptedPerception:
    """A stand-in for parallel YOLO, segmentation, depth, and SLAM nodes."""

    def __init__(self) -> None:
        self._frames: dict[int, list[Detection]] = {
            1: [
                Detection("chair-a", "chair", -1.2, 3.5, 0.91),
                Detection("stairs-a", "stairs", 0.1, 5.0, 0.96),
            ],
            2: [
                Detection("chair-a", "chair", -1.2, 2.8, 0.93),
                Detection("person-a", "person", 0.2, 1.4, 0.90, Motion.APPROACHING),
            ],
            3: [
                Detection("person-a", "person", 0.1, 0.75, 0.94, Motion.APPROACHING),
                Detection("door-a", "door", 1.5, 2.5, 0.88),
            ],
            4: [
                Detection("pothole-a", "pothole", -0.15, 0.65, 0.97),
                Detection("door-a", "door", 1.3, 1.8, 0.90),
            ],
            5: [Detection("box-a", "box", 0.4, 0.8, 0.89)],
            6: [],
        }

    def process(self, frame: FramePacket) -> list[Detection]:
        return list(self._frames.get(frame.frame_id, []))

