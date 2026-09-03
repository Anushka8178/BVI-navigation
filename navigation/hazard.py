from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians

from .models import Detection, Motion


@dataclass(frozen=True)
class SeverityResult:
    score: float
    urgent: bool
    reason: str


class HazardSeverityClassifier:
    """Transparent baseline classifier suitable for explaining in a review."""

    def __init__(self, urgency_threshold: float = 0.72) -> None:
        self.urgency_threshold = urgency_threshold

    def classify(self, detection: Detection) -> SeverityResult:
        proximity = 1.0 / (1.0 + detection.distance)
        path_alignment = max(cos(radians(detection.azimuth_deg)), 0.0)
        motion_risk = {
            Motion.STATIC: 0.0,
            Motion.CROSSING: 0.15,
            Motion.APPROACHING: 0.30,
        }[detection.motion]
        class_risk = 0.18 if detection.label in {"pothole", "stairs", "curb"} else 0.0
        score = min(
            1.0,
            0.50 * proximity + 0.25 * path_alignment + motion_risk + class_risk,
        )
        urgent = score >= self.urgency_threshold
        reason = (
            f"distance={detection.distance:.2f}m, angle={detection.azimuth_deg:+.0f}°, "
            f"motion={detection.motion.value}, class={detection.label}"
        )
        return SeverityResult(score, urgent, reason)
