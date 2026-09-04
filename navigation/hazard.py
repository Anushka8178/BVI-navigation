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
    """Calculate hazard severity using distance, direction, motion and class."""

    def __init__(self, urgency_threshold: float = 0.72) -> None:
        self.urgency_threshold = urgency_threshold

    def classify(self, detection: Detection) -> SeverityResult:
        # A closer object receives a higher proximity value.
        proximity = 1.0 / (1.0 + detection.distance)

        # An object directly ahead receives a higher path-alignment value.
        path_alignment = max(
            cos(radians(detection.azimuth_deg)),
            0.0,
        )

        # Moving objects receive additional risk.
        motion_risk = {
            Motion.STATIC: 0.0,
            Motion.CROSSING: 0.15,
            Motion.APPROACHING: 0.30,
        }[detection.motion]

        # Ground-level hazards receive additional risk.
        class_risk = (
            0.18
            if detection.label in {"pothole", "stairs", "curb"}
            else 0.0
        )

        # Calculate the normal weighted hazard score.
        score = min(
            1.0,
            0.50 * proximity
            + 0.25 * path_alignment
            + motion_risk
            + class_risk,
        )

        # Safety rule: anything very close and directly ahead is urgent,
        # even if it is not moving.
        directly_ahead = abs(detection.azimuth_deg) <= 20
        very_close = detection.distance <= 2.5
        close_ahead_hazard = directly_ahead and very_close

        urgent = (
            score >= self.urgency_threshold
            or close_ahead_hazard
        )

        # Avoid output such as "critical:0.44".
        # A hazard classified as urgent must display at least the threshold.
        if close_ahead_hazard:
            score = max(score, self.urgency_threshold)

        reason = (
            f"distance={detection.distance:.2f}m, "
            f"angle={detection.azimuth_deg:+.0f}°, "
            f"motion={detection.motion.value}, "
            f"class={detection.label}"
        )

        return SeverityResult(
            score=score,
            urgent=urgent,
            reason=reason,
        )