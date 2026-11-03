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

Frame convention (see CLAUDE.md):
    world x = right of the start heading, world y = forward at the start (m).
    heading = degrees CLOCKWISE from +y, so forward = (sin h, cos h) and
    right = (cos h, -sin h). Detections use relative_x = right,
    relative_y = forward.

`local_to_world()` is the other half of the localization contract: given the
current Pose and a Detection (which is expressed relative to the camera),
it produces the position that persistent memory should actually store.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin
from typing import NamedTuple

from .models import Detection, Pose


class SimulatedSensor:
    """
    Deterministic stand-in for real SLAM.

    Represents a person walking a few steps forward (+y at heading 0) with a
    slight rightward drift (+x) and a slight right (clockwise) turn. Call `get_poses()` to get all six poses at
    once, or `get_pose(frame_id)` to get just one (frame_id is 1-based, the
    same convention `FramePacket.frame_id` already uses elsewhere).
    """

    _POSES: tuple[Pose, ...] = (
        Pose(x=0.0, y=0.0, heading_deg=0.0, timestamp=0.0),
        Pose(x=0.0, y=0.5, heading_deg=0.0, timestamp=1.0),
        Pose(x=0.0, y=1.0, heading_deg=0.0, timestamp=2.0),
        Pose(x=0.1, y=1.5, heading_deg=5.0, timestamp=3.0),
        Pose(x=0.2, y=2.0, heading_deg=5.0, timestamp=4.0),
        Pose(x=0.3, y=2.5, heading_deg=10.0, timestamp=5.0),
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
    rx, ry = detection.relative_x, detection.relative_y

    # Heading is clockwise from +y: forward = (sin h, cos h), right = (cos h, -sin h).
    world_x = pose.x + rx * cos(theta) + ry * sin(theta)
    world_y = pose.y - rx * sin(theta) + ry * cos(theta)

    return WorldPoint(world_x, world_y)


def world_to_local(pose: Pose, world_x: float, world_y: float) -> tuple[float, float]:
    """Inverse of `local_to_world()`: world point -> (relative_x, relative_y).

    relative_x = right of the user (m), relative_y = ahead of the user (m).
    """
    theta = radians(pose.heading_deg)
    dx, dy = world_x - pose.x, world_y - pose.y

    relative_x = dx * cos(theta) - dy * sin(theta)
    relative_y = dx * sin(theta) + dy * cos(theta)

    return relative_x, relative_y


def simulated_pose(
    timestamp: float,
    walking_speed_m_s: float = 0.5,
    lateral_drift_m_s: float = 0.05,
    turn_rate_deg_s: float = 1.0,
) -> Pose:
    """Deterministic simulated walking pose used instead of real SLAM.

    The user walks FORWARD along the current heading, drifts slightly to
    their right, and turns slowly clockwise (to the right). Position is the
    exact integral of that motion, so it always agrees with the heading.
    """
    heading_deg = turn_rate_deg_s * timestamp
    h = radians(heading_deg)
    omega = radians(turn_rate_deg_s)

    if abs(omega) < 1e-9:
        x = lateral_drift_m_s * timestamp
        y = walking_speed_m_s * timestamp
    else:
        # Integral of v*(sin wt, cos wt) + d*(cos wt, -sin wt) from 0 to t.
        x = (walking_speed_m_s * (1.0 - cos(h)) + lateral_drift_m_s * sin(h)) / omega
        y = (walking_speed_m_s * sin(h) + lateral_drift_m_s * (cos(h) - 1.0)) / omega

    return Pose(x=x, y=y, heading_deg=heading_deg, timestamp=timestamp)


@dataclass(frozen=True)
class LocalizedDetection:
    """
    The actual handoff object for persistent memory (navigation/memory.py).

    `WorldPoint` alone is just an (x, y) pair - memory also needs to know
    *which* hazard that point belongs to and how much to trust it. This
    bundles the `WorldPoint` produced by `local_to_world()` together with
    the rest of `Detection` that memory is expected to care about, so the
    handoff is one object instead of several loose arguments:

        memory.save(localize_detection(pose, detection))
    """

    object_id: str
    label: str
    world_x: float
    world_y: float
    confidence: float
    motion: str
    timestamp: float


def localize_detection(pose: Pose, detection: Detection) -> LocalizedDetection:
    """
    Full localization handoff for one detection: run `local_to_world()` and
    package the result with the detection metadata memory needs to store it.

    This is the one function the memory module actually needs to call.
    """
    world = local_to_world(pose, detection)

    return LocalizedDetection(
        object_id=detection.object_id,
        label=detection.label,
        world_x=world.x,
        world_y=world.y,
        confidence=detection.confidence,
        motion=detection.motion.value,
        timestamp=pose.timestamp,
    )