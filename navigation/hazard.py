from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians

from .models import Detection, Motion


@dataclass(frozen=True)
class SeverityResult:
    score: float
    urgent: bool
    reason: str
    category: str
    ttc_s: float | None = None


class HazardSeverityClassifier:
    """Context-aware hazard triage.

    Important design rule:
        A path obstacle is NOT automatically an urgent collision.

    Urgent warnings are reserved for:
      - a genuinely close obstacle in front of the user
      - a predicted collision with sufficiently small TTC
      - a high-risk object that is strongly aligned with the user

    Static objects at the side of the walking path can therefore remain
    non-urgent even when the local planner is aware of them.
    """

    def __init__(
        self,
        urgency_threshold: float = 0.72,
        directly_ahead_deg: float = 20.0,
        very_close_m: float = 2.5,
        immediate_close_m: float = 1.5,
        critical_ttc_s: float = 1.5,
        high_risk_ttc_s: float = 2.5,
    ) -> None:
        self.urgency_threshold = urgency_threshold
        self.directly_ahead_deg = directly_ahead_deg
        self.very_close_m = very_close_m
        self.immediate_close_m = immediate_close_m
        self.critical_ttc_s = critical_ttc_s
        self.high_risk_ttc_s = high_risk_ttc_s

    def classify(
        self,
        detection: Detection,
        *,
        path_blocking: bool = False,
        ttc_s: float | None = None,
    ) -> SeverityResult:

        distance = detection.distance
        angle = detection.azimuth_deg

        alignment = max(
            cos(radians(angle)),
            0.0,
        )

        approaching = detection.motion is Motion.APPROACHING

        motion_risk = (
            0.30
            if approaching
            else 0.0
        )

        proximity = 1.0 / (
            1.0 + distance
        )

        score = min(
            1.0,
            0.50 * proximity
            + 0.25 * alignment
            + motion_risk,
        )

        # --------------------------------------------------
        # FRONT / ALIGNMENT
        # --------------------------------------------------
        #
        # Keep this deliberately narrow.
        #
        # Example:
        #   +29.7 degrees = right side
        #   +10 degrees  = approximately ahead
        #
        directly_ahead = (
            abs(angle)
            <= self.directly_ahead_deg
        )

        # --------------------------------------------------
        # IMMEDIATE FRONT OBSTACLE
        # --------------------------------------------------

        immediate_front = (
            directly_ahead
            and distance <= self.very_close_m
        )

        # --------------------------------------------------
        # VERY CLOSE OBJECT
        # --------------------------------------------------
        #
        # A very close object can still be dangerous even
        # if it is slightly off-center.
        #
        # But a 2 m side object should NOT be treated as
        # an immediate collision.
        #
        immediate_close = (
            distance <= self.immediate_close_m
            and abs(angle) <= 45.0
        )

        # --------------------------------------------------
        # TTC
        # --------------------------------------------------

        imminent_collision = (
            ttc_s is not None
            and ttc_s <= self.critical_ttc_s
        )

        predicted_collision = (
            ttc_s is not None
            and ttc_s <= self.high_risk_ttc_s
        )

        # --------------------------------------------------
        # SCORE-BASED WARNING
        # --------------------------------------------------
        #
        # Score alone must NOT make a side object urgent.
        #
        # TTC is responsible for dynamic collision prediction.
        # Alignment is required for a score-based warning.
        #
        score_trigger = (
            score >= self.urgency_threshold
            and directly_ahead
        )

        # --------------------------------------------------
        # FINAL URGENCY DECISION
        # --------------------------------------------------

        urgent = bool(
            score_trigger
            or immediate_front
            or immediate_close
            or imminent_collision
            or predicted_collision
        )

        # --------------------------------------------------
        # CATEGORY
        # --------------------------------------------------

        if imminent_collision:
            category = "IMMINENT_COLLISION"

        elif predicted_collision:
            category = "PREDICTED_COLLISION"

        elif immediate_front:
            category = "IMMEDIATE_FRONT_OBSTACLE"

        elif immediate_close:
            category = "IMMEDIATE_CLOSE_OBSTACLE"

        elif path_blocking:
            # IMPORTANT:
            #
            # Being inside the local walking corridor does
            # NOT automatically mean emergency.
            #
            # The planner can use this information to give
            # soft path guidance.
            category = "PATH_OBSTACLE"

        else:
            category = "NON_ACTIONABLE"

        # --------------------------------------------------
        # COMPATIBILITY / DEBUG INFO
        # --------------------------------------------------

        override = (
            immediate_front
            or immediate_close
        )

        reason = (
            f"category={category}, "
            f"distance={distance:.2f}m, "
            f"angle={angle:+.1f}deg, "
            f"motion={detection.motion.value}, "
            f"path_blocking={path_blocking}, "
            f"ttc={('none' if ttc_s is None else f'{ttc_s:.2f}s')}, "
            f"score={score:.2f}, "
            f"override={override}"
        )

        return SeverityResult(
            score=score,
            urgent=urgent,
            reason=reason,
            category=category,
            ttc_s=ttc_s,
        )