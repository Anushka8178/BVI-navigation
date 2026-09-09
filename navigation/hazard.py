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
    """Explainable prototype rule; its constants are hypotheses, not learned values."""

    def __init__(
        self,
        urgency_threshold: float = 0.72,
        directly_ahead_deg: float = 20.0,
        very_close_m: float = 2.5,
    ) -> None:
        self.urgency_threshold = urgency_threshold
        self.directly_ahead_deg = directly_ahead_deg
        self.very_close_m = very_close_m

    def classify(self, detection: Detection) -> SeverityResult:
        proximity = 1.0 / (1.0 + detection.distance)
        alignment = max(cos(radians(detection.azimuth_deg)), 0.0)
        motion_risk = 0.30 if detection.motion is Motion.APPROACHING else 0.0

        # The OD dataset does not contain pothole/stairs/curb, so no unsupported
        # class bonus is included in this video prototype.
        score = min(1.0, 0.50 * proximity + 0.25 * alignment + motion_risk)
        safety_override = (
            abs(detection.azimuth_deg) <= self.directly_ahead_deg
            and detection.distance <= self.very_close_m
        )
        urgent = score >= self.urgency_threshold or safety_override
        reason = (
            f"distance={detection.distance:.2f}m, "
            f"angle={detection.azimuth_deg:+.1f}deg, "
            f"motion={detection.motion.value}, "
            f"score={score:.2f}, override={safety_override}"
        )
        return SeverityResult(score=score, urgent=urgent, reason=reason)

