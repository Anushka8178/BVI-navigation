from __future__ import annotations

from collections.abc import Iterator

from .models import FramePacket, Pose


class SimulatedSensor:
    """Deterministic camera/pose source used in place of hardware."""

    def __iter__(self) -> Iterator[FramePacket]:
        poses = [
            Pose(0.0, 0.0, 0.0),
            Pose(0.0, 0.7, 0.0),
            Pose(0.0, 1.4, 0.0),
            Pose(0.0, 2.1, 5.0),
            Pose(0.0, 2.8, 5.0, localization_valid=False),
            Pose(0.1, 3.5, 8.0),
        ]
        for frame_id, pose in enumerate(poses, start=1):
            yield FramePacket(frame_id, float(frame_id), pose)
