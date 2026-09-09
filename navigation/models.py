from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import atan2, degrees, hypot
from typing import Any, Optional


class Motion(str, Enum):
    STATIC = "static"
    APPROACHING = "approaching"
    CROSSING = "crossing"


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    heading_deg: float
    localization_valid: bool = True


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


@dataclass
class HazardEntry:
    key: str
    label: str
    world_x: float
    world_y: float
    confidence: float
    last_seen: float
    motion: Motion
    decay_weight: float = 1.0
    priority: float = 0.0


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    timestamp: float
    pose: Pose
    image: Any = None


@dataclass(frozen=True)
class RouteCandidate:
    route_id: str
    name: str
    length_m: float
    base_risk: float


@dataclass(frozen=True)
class PlanDecision:
    selected: RouteCandidate
    score: float
    rationale: str
    alternatives: list[tuple[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class AudioEvent:
    kind: str
    label: str
    direction: str
    distance_band: str
    urgency: str
    message: Optional[str] = None
