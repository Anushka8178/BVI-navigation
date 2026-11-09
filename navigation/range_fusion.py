"""Fuse the ESP32 ultrasonic range with this frame's detections (pure logic).

The HC-SR04 sits next to the chest camera and sees a ~15 deg cone straight
ahead (+-cone_deg). Each frame, with a fresh reading (RangeSensor.latest_m):

1. Distance fix: the nearest detection inside the cone gets the measured
   distance (its relative_x / relative_y rescaled along its bearing), if
   the reading is in [min_m, max_m] and roughly agrees with the camera's
   monocular estimate (within a factor agree_ratio). A reading far nearer
   than the camera estimate is something else (a wall, glass, a low
   object) and is not forced onto that detection.
2. Unknown obstacle: if the reading is below unknown_obstacle_m and no
   detection in the cone was matched to it, a synthetic
   Detection(label="unknown_obstacle", straight ahead) is added, so the
   hazard classifier and planner react to obstacles YOLO cannot see.
   It must not be stored in persistent memory (persist=False).

No reading (None: stale, no echo, sensor missing or --no-esp32) changes
nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Mapping, Sequence

from .models import Detection, Motion

UNKNOWN_OBSTACLE = "unknown_obstacle"
UNKNOWN_OBSTACLE_ID = f"{UNKNOWN_OBSTACLE}-0"


@dataclass(frozen=True)
class RangeFusionConfig:
    """config.yaml `range_fusion:`."""

    cone_deg: float = 7.5             # half-angle of the ultrasonic cone
    min_m: float = 0.2                # distance fix only for readings in [min_m, max_m]
    max_m: float = 4.0
    unknown_obstacle_m: float = 1.2   # nearer than this and unexplained -> unknown obstacle
    agree_ratio: float = 2.0          # camera and ultrasonic distances within this factor

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> "RangeFusionConfig":
        unknown = set(section) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"range_fusion: unknown key(s) {sorted(unknown)}")
        return cls(**{name: float(value) for name, value in section.items()})


@dataclass(frozen=True)
class FusionResult:
    detections: list[Detection]
    corrected_id: str | None = None   # detection whose distance was replaced
    added_unknown: bool = False
    range_m: float | None = None


def describe_fusion(result: FusionResult | None, connected: bool) -> str:
    """One status-panel line: what the ultrasonic sensor did this frame.

    result None = the range sensor is off (--no-esp32).
    """
    if result is None:
        return "Range: off (--no-esp32)"
    if result.range_m is None or result.range_m <= 0.0:
        return "Range: no reading" if connected else "Range: ESP32 not connected"
    text = f"Range: {result.range_m:.2f} m"
    if result.corrected_id is not None:
        return f"{text} -> {result.corrected_id} (ultrasonic distance)"
    if result.added_unknown:
        return f"{text} UNKNOWN OBSTACLE ahead"
    return f"{text} (no match)"


def with_distance(detection: Detection, distance_m: float) -> Detection:
    """Same detection and bearing, at a new distance."""
    current = detection.distance
    if current <= 1e-9:
        return replace(detection, relative_x=0.0, relative_y=distance_m)
    scale = distance_m / current
    return replace(
        detection,
        relative_x=detection.relative_x * scale,
        relative_y=detection.relative_y * scale,
    )


def fuse_range(
    detections: Sequence[Detection],
    range_m: float | None,
    cfg: RangeFusionConfig,
) -> FusionResult:
    """Apply one ultrasonic reading (metres, or None) to this frame."""
    detections = list(detections)
    if range_m is None or range_m <= 0.0:
        return FusionResult(detections, range_m=range_m)

    ahead = [
        d for d in detections
        if abs(d.azimuth_deg) <= cfg.cone_deg and d.relative_y > 0.0
    ]
    if ahead and cfg.min_m <= range_m <= cfg.max_m:
        nearest = min(ahead, key=lambda d: d.distance)
        ratio = nearest.distance / range_m
        if 1.0 / cfg.agree_ratio <= ratio <= cfg.agree_ratio:
            fused = [
                with_distance(d, range_m) if d is nearest else d for d in detections
            ]
            return FusionResult(fused, corrected_id=nearest.object_id, range_m=range_m)

    if range_m < cfg.unknown_obstacle_m:
        unknown = Detection(
            object_id=UNKNOWN_OBSTACLE_ID,
            label=UNKNOWN_OBSTACLE,
            relative_x=0.0,
            relative_y=range_m,
            confidence=1.0,
            motion=Motion.STATIC,
        )
        return FusionResult(detections + [unknown], added_unknown=True, range_m=range_m)

    return FusionResult(detections, range_m=range_m)
