from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import atan2, degrees, hypot

import numpy as np
from numpy.typing import NDArray


ImageArray = NDArray[np.uint8]


class Motion(str, Enum):
    STATIC = "static"
    APPROACHING = "approaching"


@dataclass(frozen=True)
class Detection:
    object_id: str
    label: str
    relative_x: float
    relative_y: float
    confidence: float
    motion: Motion = Motion.STATIC

    @property
    def distance(self) -> float:
        return hypot(self.relative_x, self.relative_y)

    @property
    def azimuth_deg(self) -> float:
        return degrees(atan2(self.relative_x, self.relative_y))


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    timestamp: float
    image: ImageArray

@dataclass(frozen=True)
class Pose:
    """
    Where the person/robot is in the world, at one point in time.

    For tomorrow's demo this is produced by `navigation.sensing.SimulatedSensor`
    (a fixed, hand-authored walking path). Later it will be produced by a real
    SLAM system instead:

        RealSense RGB-D + IMU -> RTAB-Map / ORB-SLAM3 -> Pose

    Nothing downstream (this class included) needs to know or care which of
    those two produced it - that's the whole point of having this type.
    """

    x: float
    y: float
    heading_deg: float
    timestamp: float


@dataclass(frozen=True)
class WarningEvent:
    object_id: str
    label: str
    direction: str
    azimuth_deg: float
    distance_band: str
    distance_m: float
    score: float
    motion: str
    reason: str