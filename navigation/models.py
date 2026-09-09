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
class WarningEvent:
    object_id: str
    label: str
    direction: str
    distance_band: str
    distance_m: float
    score: float
    motion: str
    reason: str