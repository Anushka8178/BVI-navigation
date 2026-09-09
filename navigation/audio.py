from __future__ import annotations

from .models import Detection, WarningEvent


def direction_for(angle_deg: float) -> str:
    if angle_deg < -20:
        return "left"
    if angle_deg > 20:
        return "right"
    return "ahead"


def distance_band_for(distance_m: float) -> str:
    if distance_m <= 2.5:
        return "very-near"
    if distance_m <= 5.0:
        return "near"
    if distance_m <= 10.0:
        return "medium"
    return "far"


def make_warning(detection: Detection, score: float, reason: str) -> WarningEvent:
    return WarningEvent(
        object_id=detection.object_id,
        label=detection.label,
        direction=direction_for(detection.azimuth_deg),
        distance_band=distance_band_for(detection.distance),
        distance_m=detection.distance,
        score=score,
        motion=detection.motion.value,
        reason=reason,
    )

