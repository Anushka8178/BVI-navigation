"""
Localization stand-in for tomorrow's demo.

Real architecture (not built yet, not needed for this demo):

    RealSense RGB-D + IMU
            |
    RTAB-Map / ORB-SLAM3
            |
          Pose
            |
          Memory

Demo architecture (this file):

    SimulatedSensor
            |
          Pose
            |
          Memory

`SimulatedSensor` below replaces the "RealSense + RTAB-Map/ORB-SLAM3" box with
a fixed, hand-written list of six Pose objects. There is no camera, no IMU,
and no actual SLAM math - just deterministic numbers, so every run of the demo
(and every re-run of the tests) produces identical output.

`local_to_world()` is the other half of the localization contract: given the
current Pose and a Detection (which is expressed relative to the camera),
it produces the position that persistent memory should actually store.
"""

from __future__ import annotations

from math import cos, radians, sin
from typing import NamedTuple

from .models import Detection, Pose


class SimulatedSensor:
    """
    Deterministic stand-in for real SLAM.

    Represents a person walking a few steps roughly forward with a slight
    rightward drift and turn. Call `get_poses()` to get all six poses at
    once, or `get_pose(frame_id)` to get just one (frame_id is 1-based, the
    same convention `FramePacket.frame_id` already uses elsewhere).
    """

    _POSES: tuple[Pose, ...] = (
        Pose(x=0.0, y=0.0, heading_deg=0.0, timestamp=0.0),
        Pose(x=0.5, y=0.0, heading_deg=0.0, timestamp=1.0),
        Pose(x=1.0, y=0.0, heading_deg=0.0, timestamp=2.0),
        Pose(x=1.5, y=0.1, heading_deg=5.0, timestamp=3.0),
        Pose(x=2.0, y=0.2, heading_deg=5.0, timestamp=4.0),
        Pose(x=2.5, y=0.3, heading_deg=10.0, timestamp=5.0),
    )

    def get_poses(self) -> list[Pose]:
        """Return all six poses, in walking order. Always the same values."""
        return list(self._POSES)

    def get_pose(self, frame_id: int) -> Pose:
        """Return the pose for one frame. frame_id is 1-based (1 through 6)."""
        return self._POSES[frame_id - 1]


class WorldPoint(NamedTuple):
    """
    A detection's position after converting it into world coordinates.

    This is the hand-off value: localization (this file) produces it, and
    persistent memory (navigation/memory.py, the other teammate's file) is
    what's expected to consume it - e.g. store (world_x, world_y) per hazard
    in SQLite.
    """

    x: float
    y: float


def local_to_world(pose: Pose, detection: Detection) -> WorldPoint:
    """
    Convert a detection from camera-relative coordinates into world coordinates.

    A Detection's (relative_x, relative_y) is only meaningful relative to
    wherever the camera currently is and which way it's currently facing.
    To remember a hazard across frames (memory's job), it first has to be
    expressed in one fixed world frame - that's what this function does,
    using the current Pose.
    """
    theta = radians(pose.heading_deg)

    world_x = pose.x + detection.relative_x * cos(theta) - detection.relative_y * sin(theta)
    world_y = pose.y + detection.relative_x * sin(theta) + detection.relative_y * cos(theta)

    return WorldPoint(world_x, world_y)