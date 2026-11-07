"""Reasoning over persistent memory (pure logic: no camera, audio or GPU).

Used by video_navigation_demo_slam_memory.py and tools/eval_memory.py.

Evidence-based forgetting (`update_negative_evidence`)
    Time decay alone keeps a removed obstacle in memory for its whole
    half-life, even while the camera looks straight at the empty spot.
    Here, a remembered static hazard that *should* be visible right now
    (trusted pose, comfortable range, well inside the field of view, not
    hidden behind a nearer detection) but is not detected gets a "miss":
    misses += 1 and belief *= miss_factor, at most once per
    miss_interval_s. After forget_after_misses misses (or belief below
    forget_belief) it is deleted. A real sighting resets misses and
    restores belief (RouteMemory.remember, hit_gain).

Frames (CLAUDE.md): world x right / y forward; heading clockwise from +y;
camera-relative (relative_x right, relative_y forward); azimuth negative =
left, positive = right.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Iterable, Mapping, Sequence

from .memory import HazardEntry, RouteMemory
from .models import Detection, Pose
from .path_overlay import nearest_visible_m
from .sensing import local_to_world, world_to_local

SLAM_TRACKING = "TRACKING"  # navigation.vslam.TRACKING (not imported: keeps this light)


@dataclass(frozen=True)
class MemoryReasoningConfig:
    """config.yaml `memory_reasoning:` (defaults = the repo config)."""

    # -- evidence-based forgetting ---------------------------------------
    forget_min_m: float = 1.5           # expected visible: forward distance range
    forget_max_m: float = 6.0
    forget_edge_margin_deg: float = 8.0  # stay this far inside the field of view
    occlusion_deg: float = 6.0          # a nearer detection within this azimuth ...
    occlusion_margin_m: float = 0.5     # ... and nearer by this much hides the hazard
    miss_interval_s: float = 0.5        # at most one miss per hazard this often
    miss_factor: float = 0.7            # belief *= miss_factor per miss
    forget_after_misses: int = 4
    forget_belief: float = 0.2
    hit_gain: float = 0.3               # belief += hit_gain per sighting (RouteMemory)

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> "MemoryReasoningConfig":
        known = {f.name: f.type for f in fields(cls)}
        unknown = set(section) - set(known)
        if unknown:
            raise ValueError(f"memory_reasoning: unknown key(s) {sorted(unknown)}")
        values: dict[str, Any] = {}
        for f in fields(cls):
            if f.name in section:
                values[f.name] = int(section[f.name]) if f.type == "int" else float(section[f.name])
        return cls(**values)


@dataclass(frozen=True)
class CameraModel:
    """What the camera can see (same model as perception / path_overlay)."""

    hfov_deg: float = 70.0
    camera_height_m: float = 1.3
    image_h: int = 720
    ground_labels: frozenset[str] = frozenset({"pothole", "stairs", "curb", "drain"})

    @property
    def nearest_ground_m(self) -> float:
        """Nearest ground point in view (bottom image row)."""
        return nearest_visible_m(self.image_h, self.camera_height_m)


def pose_is_trusted(pose: Pose, slam_state: str | None = None) -> bool:
    """Valid pose, and TRACKING when the pose comes from the visual SLAM."""
    return bool(pose.valid) and (slam_state is None or slam_state == SLAM_TRACKING)


def relative_position(pose: Pose, hazard: HazardEntry) -> tuple[float, float, float, float]:
    """(right_m, forward_m, range_m, azimuth_deg) of a hazard from the user."""
    right, forward = world_to_local(pose, hazard.world_x, hazard.world_y)
    return right, forward, math.hypot(right, forward), math.degrees(math.atan2(right, forward))


def detections_in_world(pose: Pose, detections: Iterable[Detection]) -> list[tuple[str, float, float]]:
    """(label, world_x, world_y) of this frame's detections."""
    return [(d.label, *local_to_world(pose, d)) for d in detections]


def is_seen(
    hazard: HazardEntry,
    seen_hazard_ids: set[str],
    detections_world: Sequence[tuple[str, float, float]],
    match_radius_m: float,
) -> bool:
    """True if a detection in this frame matched the hazard.

    Same rule as RouteMemory: the detection was resolved to this hazard_id,
    or a detection with the same label lies within match_radius_m.
    """
    if hazard.hazard_id in seen_hazard_ids:
        return True
    return any(
        label == hazard.label
        and math.hypot(x - hazard.world_x, y - hazard.world_y) <= match_radius_m
        for label, x, y in detections_world
    )


def expected_visible(
    hazard: HazardEntry, pose: Pose, cfg: MemoryReasoningConfig, camera: CameraModel
) -> bool:
    """Would the camera see this hazard now, if it were still there?"""
    _right, forward, _rng, azimuth = relative_position(pose, hazard)
    if not cfg.forget_min_m <= forward <= cfg.forget_max_m:
        return False
    if hazard.label in camera.ground_labels and forward < camera.nearest_ground_m:
        return False
    return abs(azimuth) <= camera.hfov_deg / 2.0 - cfg.forget_edge_margin_deg


def is_occluded(
    hazard: HazardEntry, pose: Pose, detections: Iterable[Detection], cfg: MemoryReasoningConfig
) -> bool:
    """A detection near the hazard's azimuth and clearly nearer hides it."""
    _right, _forward, rng, azimuth = relative_position(pose, hazard)
    return any(
        abs(d.azimuth_deg - azimuth) <= cfg.occlusion_deg
        and d.distance < rng - cfg.occlusion_margin_m
        for d in detections
    )


@dataclass(frozen=True)
class ForgetEvent:
    hazard_id: str
    label: str
    misses: int
    belief: float
    time: float

    @property
    def message(self) -> str:
        return (
            f"MEMORY | forgot {self.label} {self.hazard_id[:8]}: "
            f"not seen in clear view {self.misses} times"
        )


@dataclass
class NegativeEvidenceState:
    """Per-run state: time of each hazard's last miss, and what was forgotten."""

    last_miss: dict[str, float] = field(default_factory=dict)
    misses_recorded: int = 0
    forgotten: list[ForgetEvent] = field(default_factory=list)


def update_negative_evidence(
    memory: RouteMemory,
    hazards: Iterable[HazardEntry],
    pose: Pose,
    detections: Sequence[Detection],
    seen_hazard_ids: set[str],
    now: float,
    cfg: MemoryReasoningConfig,
    camera: CameraModel,
    state: NegativeEvidenceState,
    pose_trusted: bool = True,
    log: Callable[[str], None] | None = print,
) -> list[ForgetEvent]:
    """Record misses for remembered static hazards that should be in view.

    `hazards`: remembered hazards (memory.get_hazards()); `detections`:
    this frame's camera-relative detections; `seen_hazard_ids`: hazard_ids
    that this frame's detections were resolved to by memory.remember();
    `now`: memory clock (epoch s). Dynamic labels are never touched, and
    nothing happens without a trusted pose. Returns the hazards forgotten
    in this call (already deleted from memory).
    """
    if not pose_trusted:
        return []
    world = detections_in_world(pose, detections)
    events: list[ForgetEvent] = []
    for hazard in hazards:
        if hazard.label in memory.dynamic_labels:
            continue
        if not expected_visible(hazard, pose, cfg, camera):
            continue
        if is_seen(hazard, seen_hazard_ids, world, memory.match_radius_m):
            continue
        if is_occluded(hazard, pose, detections, cfg):
            continue
        last = state.last_miss.get(hazard.hazard_id)
        if last is not None and now - last < cfg.miss_interval_s:
            continue
        state.last_miss[hazard.hazard_id] = now
        updated = memory.record_miss(hazard.hazard_id, cfg.miss_factor)
        if updated is None:
            continue
        state.misses_recorded += 1
        if updated.misses >= cfg.forget_after_misses or updated.belief < cfg.forget_belief:
            memory.forget(hazard.hazard_id)
            state.last_miss.pop(hazard.hazard_id, None)
            event = ForgetEvent(
                hazard.hazard_id, hazard.label, updated.misses, updated.belief, now
            )
            events.append(event)
            state.forgotten.append(event)
            if log is not None:
                log(event.message)
    return events
