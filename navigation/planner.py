from __future__ import annotations

from dataclasses import dataclass
from math import hypot, radians, sin, cos
from typing import Iterable

from .memory import HazardEntry
from .models import Pose


OBSTACLE_RADIUS = {
    "person": 0.35,
    "bicycle": 0.50,
    "motorbike": 0.50,
    "car": 0.90,
    "truck": 1.20,
    "bus": 1.40,
    "pole": 0.30,
    "dog": 0.35,
    "tricycle": 0.60,
    "fire_hydrant": 0.30,
    "stop_sign": 0.30,
    "ashcan": 0.45,
    "trashcan": 0.45,
    "reflective_cone": 0.25,
    "warning_column": 0.40,
    "spherical_roadblock": 0.45,
}


@dataclass(frozen=True)
class PathCandidate:
    name: str
    lateral_offset: float
    cost: float
    safe: bool
    minimum_clearance: float


@dataclass(frozen=True)
class PathDecision:
    current_path_safe: bool
    selected_path: str
    selected_offset: float
    reason: str
    candidates: list[PathCandidate]
    blocking_hazards: list[str]
    guidance_changed: bool = False


class LocalPathPlanner:
    """
    Conservative local path planner for the BVI prototype.

    Important design choice:

    * The planner may CONSIDER objects several metres ahead.
    * An object only becomes an actionable blocker when it is close enough
      to require a response, or when its predicted motion creates a
      near-term collision risk.
    * Path guidance is debounced so one noisy monocular estimate does not
      immediately change the user's instruction.

    This keeps perception rich while keeping user-facing guidance sparse.
    """

    def __init__(
        self,
        horizon_m: float = 5.0,
        path_width_m: float = 0.65,
        safety_margin_m: float = 0.50,
        walking_speed_m_s: float = 0.5,
        prediction_horizon_s: float = 4.0,
        prediction_step_s: float = 0.25,
        static_action_distance_m: float = 2.75,
        dynamic_action_distance_m: float = 6.0,
        dynamic_ttc_s: float = 4.0,
        confirm_frames: int = 3,
        clear_frames: int = 3,
    ) -> None:
        self.horizon_m = horizon_m
        self.path_width_m = path_width_m
        self.safety_margin_m = safety_margin_m
        self.walking_speed_m_s = walking_speed_m_s
        self.prediction_horizon_s = prediction_horizon_s
        self.prediction_step_s = prediction_step_s
        self.static_action_distance_m = static_action_distance_m
        self.dynamic_action_distance_m = dynamic_action_distance_m
        self.dynamic_ttc_s = dynamic_ttc_s
        self.confirm_frames = max(1, confirm_frames)
        self.clear_frames = max(1, clear_frames)

        self.candidate_offsets = (-1.5, -0.75, 0.0, 0.75, 1.5)

        self._unsafe_streak = 0
        self._clear_streak = 0
        self._guidance_active = False
        self._last_selected_path = "ahead"

    def plan(
        self,
        pose: Pose,
        hazards: Iterable[HazardEntry],
    ) -> PathDecision:
        hazards = list(hazards)

        candidates: list[PathCandidate] = []
        blocking_by_offset: dict[float, list[str]] = {}

        for offset in self.candidate_offsets:
            clearance, blockers = self._evaluate_candidate(
                pose, offset, hazards
            )

            safe = len(blockers) == 0

            if safe:
                cost = abs(offset) * 0.5
            else:
                cost = (
                    100.0
                    + max(0.0, self.path_width_m - clearance) * 10.0
                    + abs(offset) * 0.5
                )

            candidates.append(
                PathCandidate(
                    name=self._offset_name(offset),
                    lateral_offset=offset,
                    cost=cost,
                    safe=safe,
                    minimum_clearance=clearance,
                )
            )
            blocking_by_offset[offset] = blockers

        current = next(
            candidate for candidate in candidates
            if abs(candidate.lateral_offset) < 0.01
        )

        raw_unsafe = not current.safe

        # Debounce path activation. A single noisy frame is not enough to
        # tell the user to move. Once active, keep the instruction stable
        # until the path has been clear for several consecutive frames.
        if raw_unsafe:
            self._unsafe_streak += 1
            self._clear_streak = 0
        else:
            self._clear_streak += 1
            self._unsafe_streak = 0

        was_active = self._guidance_active

        if not self._guidance_active and self._unsafe_streak >= self.confirm_frames:
            self._guidance_active = True
        elif self._guidance_active and self._clear_streak >= self.clear_frames:
            self._guidance_active = False
            self._last_selected_path = "ahead"

        safe_candidates = [c for c in candidates if c.safe]

        if not self._guidance_active:
            selected = current
            reason = "Current walking path is clear or obstacle is not yet actionable."
            guidance_changed = was_active
        else:
            if safe_candidates:
                # Prefer the smallest lateral change.
                selected = min(
                    safe_candidates,
                    key=lambda candidate: abs(candidate.lateral_offset),
                )
                blockers = blocking_by_offset.get(0.0, [])
                reason = (
                    "Current path is unsafe because of "
                    f"{', '.join(blockers) or 'a predicted obstacle'}. "
                    f"Safer path: {selected.name}."
                )
            else:
                selected = current
                reason = (
                    "Current path is unsafe and no safe local "
                    "alternative was found."
                )

            guidance_changed = (
                not was_active
                or selected.name != self._last_selected_path
            )
            self._last_selected_path = selected.name

        return PathDecision(
            current_path_safe=not self._guidance_active,
            selected_path=selected.name,
            selected_offset=selected.lateral_offset,
            reason=reason,
            candidates=candidates,
            blocking_hazards=blocking_by_offset.get(0.0, [])
            if self._guidance_active else [],
            guidance_changed=guidance_changed,
        )

    def _evaluate_candidate(
        self,
        pose: Pose,
        lateral_offset: float,
        hazards: list[HazardEntry],
    ) -> tuple[float, list[str]]:
        """
        Evaluate only hazards that are actionable for this candidate.

        Static objects use a close-distance threshold. Moving objects may
        activate earlier when their predicted position intersects the
        candidate path within the configured time-to-collision window.
        """
        minimum_clearance = float("inf")
        blockers: set[str] = set()

        theta = radians(pose.heading_deg)
        forward_x = sin(theta)
        forward_y = cos(theta)
        right_x = cos(theta)
        right_y = -sin(theta)

        corridor_radius = self.path_width_m / 2.0 + self.safety_margin_m

        for hazard in hazards:
            obstacle_radius = OBSTACLE_RADIUS.get(hazard.label, 0.50)
            required = corridor_radius + obstacle_radius

            distance_now = hypot(
                hazard.world_x - pose.x,
                hazard.world_y - pose.y,
            )

            is_dynamic = (
                abs(hazard.velocity_x) > 0.05
                or abs(hazard.velocity_y) > 0.05
            )

            if not is_dynamic:
                # Distant static objects remain known to the planner, but
                # they do not generate immediate user guidance.
                if distance_now > self.static_action_distance_m + required:
                    continue

                clearance = self._static_clearance(
                    pose, lateral_offset, hazard, required
                )
                minimum_clearance = min(minimum_clearance, clearance)

                if clearance < 0.0:
                    blockers.add(hazard.object_id)
                continue

            # Moving hazards get an earlier look-ahead because a vehicle or
            # person can enter the walking corridor before reaching it.
            if distance_now > self.dynamic_action_distance_m:
                continue

            steps = int(self.dynamic_ttc_s / self.prediction_step_s)
            for step in range(steps + 1):
                t = step * self.prediction_step_s
                walking_progress = min(
                    self.horizon_m,
                    self.walking_speed_m_s * t,
                )

                path_x = (
                    pose.x
                    + forward_x * walking_progress
                    + right_x * lateral_offset
                )
                path_y = (
                    pose.y
                    + forward_y * walking_progress
                    + right_y * lateral_offset
                )

                predicted_x = hazard.world_x + hazard.velocity_x * t
                predicted_y = hazard.world_y + hazard.velocity_y * t

                clearance = hypot(
                    predicted_x - path_x,
                    predicted_y - path_y,
                ) - required

                minimum_clearance = min(minimum_clearance, clearance)

                if clearance < 0.0:
                    blockers.add(hazard.object_id)
                    break

        if minimum_clearance == float("inf"):
            minimum_clearance = 100.0

        return minimum_clearance, sorted(blockers)

    def _static_clearance(
        self,
        pose: Pose,
        lateral_offset: float,
        hazard: HazardEntry,
        required: float,
    ) -> float:
        theta = radians(pose.heading_deg)
        forward_x = sin(theta)
        forward_y = cos(theta)
        right_x = cos(theta)
        right_y = -sin(theta)

        minimum_clearance = float("inf")
        samples = 25

        # Only the actionable portion of the route is relevant for an
        # immediate instruction. This prevents a 5 m planning horizon from
        # turning every distant roadside object into an alert.
        path_horizon = min(
            self.horizon_m,
            self.static_action_distance_m,
        )

        for index in range(samples + 1):
            progress = path_horizon * index / samples
            path_x = (
                pose.x
                + forward_x * progress
                + right_x * lateral_offset
            )
            path_y = (
                pose.y
                + forward_y * progress
                + right_y * lateral_offset
            )

            clearance = hypot(
                hazard.world_x - path_x,
                hazard.world_y - path_y,
            ) - required
            minimum_clearance = min(minimum_clearance, clearance)

        return minimum_clearance

    @staticmethod
    def _offset_name(offset: float) -> str:
        if offset < -1.0:
            return "left"
        if offset < -0.01:
            return "slight-left"
        if offset > 1.0:
            return "right"
        if offset > 0.01:
            return "slight-right"
        return "ahead"
