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
        """Approximate distance from the camera/user."""
        return hypot(
            self.relative_x,
            self.relative_y,
        )

    @property
    def azimuth_deg(self) -> float:
        """Physical camera-relative azimuth.

        Project convention:
            negative = left
            0       = straight ahead
            positive = right
        """
        return degrees(
            atan2(
                self.relative_x,
                self.relative_y,
            )
        )

    @property
    def audio_azimuth_deg(self) -> float:
        """Perceptually expanded PHYSICAL azimuth for HRTF audio.

        Important:
            This property keeps the same physical convention as
            azimuth_deg.

            negative = left
            0       = ahead
            positive = right

        The conversion from this physical convention to the
        SOFA/MIT-KEMAR convention is performed ONLY inside
        HRTFRenderer.

        This prevents the urgent-warning and path-guidance
        branches from using different coordinate conventions.
        """

        expanded = self.azimuth_deg * 2.5

        return max(
            -90.0,
            min(
                90.0,
                expanded,
            ),
        )


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    timestamp: float
    image: ImageArray


@dataclass(frozen=True)
class Pose:
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
    category: str
    ttc_s: float | None = None