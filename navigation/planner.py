from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, radians, sin
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
    current_path_ttc_s: float | None = None
    collision_ttc_by_hazard: dict[str, float] | None = None

    @property
    def guidance_azimuth_deg(self) -> float:
        """HRTF direction representing the recommended walking direction."""

        return {
            "left": -60.0,
            "slight-left": -30.0,
            "ahead": 0.0,
            "slight-right": 30.0,
            "right": 60.0,
        }.get(
            self.selected_path,
            0.0,
        )


class LocalPathPlanner:
    """Local obstacle-aware planner with prediction and hysteresis.

    The planner is intentionally more conservative than the hazard alarm,
    but it does not turn every path obstacle into an urgent warning.
    """

    def __init__(
        self,
        horizon_m: float = 5.0,
        path_width_m: float = 0.65,
        safety_margin_m: float = 0.25,
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

        self.prediction_horizon_s = (
            prediction_horizon_s
        )

        self.prediction_step_s = (
            prediction_step_s
        )

        self.static_action_distance_m = (
            static_action_distance_m
        )

        self.dynamic_action_distance_m = (
            dynamic_action_distance_m
        )

        self.dynamic_ttc_s = dynamic_ttc_s

        self.confirm_frames = max(
            1,
            confirm_frames,
        )

        self.clear_frames = max(
            1,
            clear_frames,
        )

        # Negative = left
        # Positive = right
        self.candidate_offsets = (
            -1.5,
            -0.75,
            0.0,
            0.75,
            1.5,
        )

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

        blocking_by_offset: dict[
            float,
            list[str],
        ] = {}

        ttc_by_hazard: dict[
            str,
            float,
        ] = {}

        # --------------------------------------------------
        # Evaluate each possible walking path
        # --------------------------------------------------

        for offset in self.candidate_offsets:

            clearance, blockers, candidate_ttc = (
                self._evaluate_candidate(
                    pose,
                    offset,
                    hazards,
                )
            )

            if not blockers:
                cost = abs(offset) * 0.5

            else:
                cost = (
                    100.0
                    + max(
                        0.0,
                        self.path_width_m - clearance,
                    ) * 10.0
                    + abs(offset) * 0.5
                )

            candidates.append(
                PathCandidate(
                    name=self._offset_name(offset),
                    lateral_offset=offset,
                    cost=cost,
                    safe=not blockers,
                    minimum_clearance=clearance,
                )
            )

            blocking_by_offset[offset] = blockers

            # The zero-offset candidate represents the user's
            # current walking corridor.
            if abs(offset) < 0.01:
                ttc_by_hazard = candidate_ttc

        # --------------------------------------------------
        # Current path
        # --------------------------------------------------

        current = next(
            candidate
            for candidate in candidates
            if abs(candidate.lateral_offset) < 0.01
        )

        raw_unsafe = not current.safe

        # --------------------------------------------------
        # HYSTERESIS
        # --------------------------------------------------

        if raw_unsafe:

            self._unsafe_streak += 1
            self._clear_streak = 0

        else:

            self._clear_streak += 1
            self._unsafe_streak = 0

        was_active = self._guidance_active

        # Activate guidance only after several consecutive
        # unsafe frames.
        if (
            not self._guidance_active
            and self._unsafe_streak
            >= self.confirm_frames
        ):

            self._guidance_active = True

        # Clear guidance only after several consecutive
        # clear frames.
        elif (
            self._guidance_active
            and self._clear_streak
            >= self.clear_frames
        ):

            self._guidance_active = False
            self._last_selected_path = "ahead"

        safe_candidates = [
            candidate
            for candidate in candidates
            if candidate.safe
        ]

        # --------------------------------------------------
        # NO ACTIVE GUIDANCE
        # --------------------------------------------------

        if not self._guidance_active:

            selected = current

            reason = (
                "Current walking path is clear or "
                "the obstacle is not yet actionable."
            )

            guidance_changed = was_active

        # --------------------------------------------------
        # ACTIVE GUIDANCE
        # --------------------------------------------------

        else:

            if safe_candidates:

                # Prefer the smallest lateral movement
                # that is actually safe.
                selected = min(
                    safe_candidates,
                    key=lambda candidate: (
                        abs(candidate.lateral_offset),
                        candidate.cost,
                    ),
                )

                blockers = (
                    blocking_by_offset.get(
                        0.0,
                        [],
                    )
                )

                reason = (
                    "Current path is unsafe because of "
                    f"{', '.join(blockers) or 'a predicted obstacle'}. "
                    f"Safer path: {selected.name}."
                )

            else:

                selected = current

                reason = (
                    "Current path is unsafe and "
                    "no safe local alternative was found."
                )

            guidance_changed = (
                not was_active
                or selected.name
                != self._last_selected_path
            )

            self._last_selected_path = (
                selected.name
            )

        # --------------------------------------------------
        # RETURN DECISION
        # --------------------------------------------------

        return PathDecision(
            current_path_safe=(
                not self._guidance_active
            ),

            selected_path=selected.name,

            selected_offset=(
                selected.lateral_offset
            ),

            reason=reason,

            candidates=candidates,

            blocking_hazards=(
                blocking_by_offset.get(
                    0.0,
                    [],
                )
                if self._guidance_active
                else []
            ),

            guidance_changed=guidance_changed,

            current_path_ttc_s=(
                min(ttc_by_hazard.values())
                if ttc_by_hazard
                else None
            ),

            collision_ttc_by_hazard=(
                ttc_by_hazard
            ),
        )

    def time_to_collision(
        self,
        pose: Pose,
        hazard: HazardEntry,
    ) -> float | None:
        """Estimate collision time with the current walking corridor.

        This is mainly used for tests and diagnostics.

        The main planner uses the same future-position model
        in _evaluate_candidate so that it can also compare
        alternative walking paths.
        """

        vx_user = (
            sin(radians(pose.heading_deg))
            * self.walking_speed_m_s
        )

        vy_user = (
            cos(radians(pose.heading_deg))
            * self.walking_speed_m_s
        )

        rx = (
            hazard.world_x
            - pose.x
        )

        ry = (
            hazard.world_y
            - pose.y
        )

        rvx = (
            hazard.velocity_x
            - vx_user
        )

        rvy = (
            hazard.velocity_y
            - vy_user
        )

        radius = (
            self.path_width_m / 2.0
            + self.safety_margin_m
            + OBSTACLE_RADIUS.get(
                hazard.label,
                0.50,
            )
        )

        a = (
            rvx * rvx
            + rvy * rvy
        )

        b = (
            2.0
            * (
                rx * rvx
                + ry * rvy
            )
        )

        c = (
            rx * rx
            + ry * ry
            - radius * radius
        )

        if c <= 0.0:
            return 0.0

        if a < 1e-9:
            return None

        discriminant = (
            b * b
            - 4.0 * a * c
        )

        if discriminant < 0.0:
            return None

        root = discriminant ** 0.5

        t1 = (
            -b - root
        ) / (
            2.0 * a
        )

        t2 = (
            -b + root
        ) / (
            2.0 * a
        )

        candidates = [
            t
            for t in (t1, t2)
            if t >= 0.0
        ]

        if not candidates:
            return None

        ttc = min(candidates)

        if ttc <= self.dynamic_ttc_s:
            return ttc

        return None

    def _evaluate_candidate(
        self,
        pose: Pose,
        lateral_offset: float,
        hazards: list[HazardEntry],
    ) -> tuple[
        float,
        list[str],
        dict[str, float],
    ]:

        minimum_clearance = float("inf")

        blockers: set[str] = set()

        candidate_ttc: dict[
            str,
            float,
        ] = {}

        theta = radians(
            pose.heading_deg
        )

        forward_x = sin(theta)
        forward_y = cos(theta)

        right_x = cos(theta)
        right_y = -sin(theta)

        # --------------------------------------------------
        # WALKING CORRIDOR
        # --------------------------------------------------
        #
        # The previous 0.50 m safety margin was too wide for
        # side objects. A 0.25 m margin still gives personal
        # clearance without treating every nearby side object
        # as occupying the walking corridor.
        #
        corridor_radius = (
            self.path_width_m / 2.0
            + self.safety_margin_m
        )

        for hazard in hazards:

            obstacle_radius = OBSTACLE_RADIUS.get(
                hazard.label,
                0.50,
            )

            required = (
                corridor_radius
                + obstacle_radius
            )

            distance_now = hypot(
                hazard.world_x - pose.x,
                hazard.world_y - pose.y,
            )

            is_dynamic = (
                abs(hazard.velocity_x)
                > 0.05
                or abs(hazard.velocity_y)
                > 0.05
            )

            # ==================================================
            # STATIC OBSTACLE
            # ==================================================

            if not is_dynamic:

                if distance_now > (
                    self.static_action_distance_m
                    + required
                ):
                    continue

                clearance = (
                    self._static_clearance(
                        pose,
                        lateral_offset,
                        hazard,
                        required,
                    )
                )

                minimum_clearance = min(
                    minimum_clearance,
                    clearance,
                )

                if clearance < 0.0:
                    blockers.add(
                        hazard.object_id
                    )

                continue

            # ==================================================
            # DYNAMIC OBSTACLE
            # ==================================================

            if distance_now > (
                self.dynamic_action_distance_m
            ):
                continue

            steps = int(
                self.prediction_horizon_s
                / self.prediction_step_s
            )

            for step in range(
                steps + 1
            ):

                t = (
                    step
                    * self.prediction_step_s
                )

                # ------------------------------------------
                # Predicted USER position
                # ------------------------------------------

                progress = min(
                    self.horizon_m,
                    self.walking_speed_m_s
                    * t,
                )

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

                # ------------------------------------------
                # Predicted OBJECT position
                # ------------------------------------------

                predicted_x = (
                    hazard.world_x
                    + hazard.velocity_x * t
                )

                predicted_y = (
                    hazard.world_y
                    + hazard.velocity_y * t
                )

                # ------------------------------------------
                # Clearance
                # ------------------------------------------

                clearance = (
                    hypot(
                        predicted_x - path_x,
                        predicted_y - path_y,
                    )
                    - required
                )

                minimum_clearance = min(
                    minimum_clearance,
                    clearance,
                )

                # ------------------------------------------
                # Collision prediction
                # ------------------------------------------

                if clearance < 0.0:

                    blockers.add(
                        hazard.object_id
                    )

                    candidate_ttc[
                        hazard.object_id
                    ] = t

                    break

        if (
            minimum_clearance
            == float("inf")
        ):
            minimum_clearance = 100.0

        return (
            minimum_clearance,
            sorted(blockers),
            candidate_ttc,
        )

    def _static_clearance(
        self,
        pose: Pose,
        lateral_offset: float,
        hazard: HazardEntry,
        required: float,
    ) -> float:

        theta = radians(
            pose.heading_deg
        )

        forward_x = sin(theta)
        forward_y = cos(theta)

        right_x = cos(theta)
        right_y = -sin(theta)

        minimum_clearance = float(
            "inf"
        )

        path_horizon = min(
            self.horizon_m,
            self.static_action_distance_m,
        )

        # Sample the user's future walking path.
        for index in range(26):

            progress = (
                path_horizon
                * index
                / 25
            )

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

            clearance = (
                hypot(
                    hazard.world_x
                    - path_x,
                    hazard.world_y
                    - path_y,
                )
                - required
            )

            minimum_clearance = min(
                minimum_clearance,
                clearance,
            )

        return minimum_clearance

    @staticmethod
    def _offset_name(
        offset: float,
    ) -> str:

        if offset < -1.0:
            return "left"

        if offset < -0.01:
            return "slight-left"

        if offset > 1.0:
            return "right"

        if offset > 0.01:
            return "slight-right"

        return "ahead"