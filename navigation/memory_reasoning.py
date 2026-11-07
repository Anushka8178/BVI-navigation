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

Memory early warning (`select_memory_cues`, `MemoryEarlyWarning`)
    The sonified part of top-k acoustic attention. Remembered static
    hazards that the camera does NOT see right now but that lie ahead
    (cue_min_m..cue_max_m forward, within cue_max_bearing_deg), were
    confirmed (seen in >= cue_min_sightings frames) and are still trusted
    (decay_weight x belief >= cue_min_trust) are ranked with
    AcousticAttention; the best one not on cooldown gets a soft "remembered
    hazard" HRTF cue. At most one per frame, per-hazard and global
    cooldowns, and an urgent or path-guidance cue in the same frame
    suppresses it. Memory cues never reach the planner (CLAUDE.md: the
    planner sees current-frame hazards only). `CueLog` measures lead time:
    how long before the camera confirmed a cued hazard.

Frames (CLAUDE.md): world x right / y forward; heading clockwise from +y;
camera-relative (relative_x right, relative_y forward); azimuth negative =
left, positive = right.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .memory import (
    DEFAULT_DYNAMIC_LABELS,
    DEFAULT_MATCH_RADIUS_M,
    HazardEntry,
    RouteMemory,
)
from .models import Detection, Pose, expand_audio_azimuth
from .path_overlay import nearest_visible_m
from .reasoning import AcousticAttention
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

    # -- memory early warning (sonified top-k attention) -------------------
    cue_min_m: float = 2.5              # forward distance window for a cue
    cue_max_m: float = 8.0
    cue_max_bearing_deg: float = 40.0   # |bearing from heading|
    cue_min_trust: float = 0.3          # decay_weight x belief
    cue_min_sightings: int = 3          # confirmed hazards only (as planner.confirm_frames)
    cue_cooldown_s: float = 8.0         # per hazard
    cue_global_cooldown_s: float = 2.0  # any memory cue
    cue_skip_seen_within_s: float = 1.0  # seen this recently = still in view
    cue_display_s: float = 3.0          # status-panel line stays this long

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


# ---------------------------------------------------------------------------
# Memory early warning
# ---------------------------------------------------------------------------


def direction_words(azimuth_deg: float) -> str:
    """Spoken-style direction of a physical azimuth (negative = left)."""
    magnitude = abs(azimuth_deg)
    side = "left" if azimuth_deg < 0 else "right"
    if magnitude <= 10.0:
        return "ahead"
    if magnitude <= 25.0:
        return f"ahead-{side}"
    return side


def format_age(seconds: float) -> str:
    """Rounded-down age: "35 s", "4 min", "3 h", "2 days"."""
    seconds = max(seconds, 0.0)
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds // 60:.0f} min"
    if seconds < 2 * 86400:
        return f"{seconds // 3600:.0f} h"
    return f"{seconds // 86400:.0f} days"


@dataclass(frozen=True)
class MemoryCue:
    """One memory early-warning cue for a remembered, currently unseen hazard."""

    hazard_id: str
    label: str
    time: float          # memory clock (epoch s)
    azimuth_deg: float   # physical, relative to the heading (negative = left)
    distance_m: float
    right_m: float
    forward_m: float
    world_x: float
    world_y: float
    trust: float         # decay_weight x belief
    priority: float      # AcousticAttention priority
    earlier_walks: int   # sessions before this one in which it was seen
    last_seen: float     # memory clock (epoch s)

    @property
    def audio_azimuth_deg(self) -> float:
        """Physical azimuth for the HRTF cue (same expansion as detections)."""
        return expand_audio_azimuth(self.azimuth_deg)

    def describe(self) -> str:
        """E.g. "pothole ahead-left 4.2 m | seen in 2 earlier walks, last 2 days ago"."""
        where = f"{self.label} {direction_words(self.azimuth_deg)} {self.distance_m:.1f} m"
        age = format_age(self.time - self.last_seen)
        if self.earlier_walks > 0:
            walks = "walk" if self.earlier_walks == 1 else "walks"
            return f"{where} | seen in {self.earlier_walks} earlier {walks}, last {age} ago"
        return f"{where} | seen earlier in this walk, {age} ago"


@dataclass
class CueState:
    """Cooldown state for memory cues (one per run)."""

    last_by_hazard: dict[str, float] = field(default_factory=dict)
    last_cue_time: float | None = None


def select_memory_cues(
    hazards: Iterable[HazardEntry],
    pose: Pose,
    detections: Sequence[Detection],
    seen_hazard_ids: set[str],
    now: float,
    cfg: MemoryReasoningConfig,
    attention: AcousticAttention,
    state: CueState,
    *,
    pose_trusted: bool = True,
    suppressed: bool = False,
    session_id: str = "",
    dynamic_labels: frozenset[str] = DEFAULT_DYNAMIC_LABELS,
    match_radius_m: float = DEFAULT_MATCH_RADIUS_M,
) -> MemoryCue | None:
    """Pick at most one memory cue for this frame (None = stay quiet).

    `suppressed`: an urgent or path-guidance cue is due this frame. A
    returned cue has already been booked on the cooldowns in `state`.
    """
    if not pose_trusted or suppressed:
        return None
    if state.last_cue_time is not None and now - state.last_cue_time < cfg.cue_global_cooldown_s:
        return None

    world = detections_in_world(pose, detections)
    candidates: list[HazardEntry] = []
    for hazard in hazards:
        if hazard.label in dynamic_labels:
            continue
        if is_seen(hazard, seen_hazard_ids, world, match_radius_m):
            continue
        if now - hazard.timestamp < cfg.cue_skip_seen_within_s:
            continue  # seen a moment ago: still in view, not a memory
        _right, forward, _rng, azimuth = relative_position(pose, hazard)
        if not cfg.cue_min_m <= forward <= cfg.cue_max_m:
            continue
        if abs(azimuth) > cfg.cue_max_bearing_deg:
            continue
        if hazard.decay_weight * hazard.belief < cfg.cue_min_trust:
            continue
        if hazard.times_seen < cfg.cue_min_sightings:
            continue  # one-frame detection (often a false positive): not confirmed
        candidates.append(dataclasses.replace(hazard))  # rank() writes .priority

    for hazard in attention.rank(candidates, pose):
        last = state.last_by_hazard.get(hazard.hazard_id)
        if last is not None and now - last < cfg.cue_cooldown_s:
            continue
        right, forward, rng, azimuth = relative_position(pose, hazard)
        this_walk = 1 if session_id and hazard.last_session == session_id else 0
        state.last_by_hazard[hazard.hazard_id] = now
        state.last_cue_time = now
        return MemoryCue(
            hazard_id=hazard.hazard_id,
            label=hazard.label,
            time=now,
            azimuth_deg=azimuth,
            distance_m=rng,
            right_m=right,
            forward_m=forward,
            world_x=hazard.world_x,
            world_y=hazard.world_y,
            trust=hazard.decay_weight * hazard.belief,
            priority=hazard.priority,
            earlier_walks=max(hazard.sessions_seen - this_walk, 0),
            last_seen=hazard.timestamp,
        )
    return None


@dataclass
class CueRecord:
    cue: MemoryCue
    audio_played: bool | None = None  # None = audio disabled
    confirmed_at: float | None = None

    @property
    def lead_time_s(self) -> float | None:
        return None if self.confirmed_at is None else self.confirmed_at - self.cue.time

    def as_dict(self, start_epoch: float = 0.0) -> dict[str, Any]:
        cue = self.cue
        confirmed = self.confirmed_at is not None
        return {
            "time_s": round(cue.time - start_epoch, 3),
            "hazard_id": cue.hazard_id,
            "label": cue.label,
            "direction": direction_words(cue.azimuth_deg),
            "azimuth_deg": round(cue.azimuth_deg, 2),
            "distance_m": round(cue.distance_m, 2),
            "world_x": round(cue.world_x, 3),
            "world_y": round(cue.world_y, 3),
            "trust": round(cue.trust, 3),
            "priority": round(cue.priority, 3),
            "earlier_walks": cue.earlier_walks,
            "last_seen_s_ago": round(cue.time - cue.last_seen, 1),
            "description": cue.describe(),
            "audio_played": self.audio_played,
            "confirmed_by_camera": confirmed,
            "lead_time_s": round(self.lead_time_s, 3) if confirmed else None,
            "lead_distance_m": round(cue.distance_m, 2) if confirmed else None,
        }


class CueLog:
    """Every memory cue of a run, and whether the camera later confirmed it.

    A cue is confirmed by the first later frame whose detections are
    resolved to the cued hazard: lead_time_s = that time - cue time,
    lead_distance_m = distance to the hazard when the cue played.
    """

    def __init__(self) -> None:
        self.records: list[CueRecord] = []
        self._pending: dict[str, list[CueRecord]] = {}

    def add(self, cue: MemoryCue, audio_played: bool | None = None) -> CueRecord:
        record = CueRecord(cue, audio_played)
        self.records.append(record)
        self._pending.setdefault(cue.hazard_id, []).append(record)
        return record

    def confirm(self, seen_hazard_ids: Iterable[str], now: float) -> list[CueRecord]:
        """Mark pending cues of hazards seen at `now` (memory clock)."""
        confirmed: list[CueRecord] = []
        for hazard_id in seen_hazard_ids:
            pending = self._pending.get(hazard_id, [])
            done = [record for record in pending if now > record.cue.time]
            for record in done:
                record.confirmed_at = now
                confirmed.append(record)
            remaining = [record for record in pending if record not in done]
            if remaining:
                self._pending[hazard_id] = remaining
            else:
                self._pending.pop(hazard_id, None)
        return confirmed

    def summary(self, hazards_forgotten: int = 0) -> dict[str, Any]:
        confirmed = [r for r in self.records if r.confirmed_at is not None]
        lead_times = [r.lead_time_s for r in confirmed if r.lead_time_s is not None]
        return {
            "cues_played": len(self.records),
            "cues_with_audio": sum(1 for r in self.records if r.audio_played),
            "cues_later_confirmed": len(confirmed),
            "mean_lead_time_s": (
                round(sum(lead_times) / len(lead_times), 3) if lead_times else None
            ),
            "mean_lead_distance_m": (
                round(sum(r.cue.distance_m for r in confirmed) / len(confirmed), 3)
                if confirmed
                else None
            ),
            "hazards_forgotten_by_negative_evidence": hazards_forgotten,
        }


class MemoryCueRenderer(Protocol):
    def play_memory_cue(self, azimuth_deg: float, distance_m: float) -> bool: ...


class MemoryEarlyWarning:
    """Per-run memory early warning: confirm earlier cues, pick this frame's
    cue, play it (if a renderer is given) and log it."""

    def __init__(
        self,
        cfg: MemoryReasoningConfig,
        attention: AcousticAttention,
        session_id: str = "",
        dynamic_labels: frozenset[str] = DEFAULT_DYNAMIC_LABELS,
        match_radius_m: float = DEFAULT_MATCH_RADIUS_M,
        renderer: MemoryCueRenderer | None = None,
        log: Callable[[str], None] | None = print,
    ) -> None:
        self.cfg = cfg
        self.attention = attention
        self.session_id = session_id
        self.dynamic_labels = frozenset(dynamic_labels)
        self.match_radius_m = match_radius_m
        self.renderer = renderer
        self.log = log
        self.state = CueState()
        self.cue_log = CueLog()
        self.last_cue: MemoryCue | None = None

    def update(
        self,
        hazards: Iterable[HazardEntry],
        pose: Pose,
        detections: Sequence[Detection],
        seen_hazard_ids: set[str],
        now: float,
        *,
        pose_trusted: bool = True,
        suppressed: bool = False,
    ) -> MemoryCue | None:
        """One frame. `seen_hazard_ids` = hazard_ids this frame's detections
        were resolved to; `suppressed` = urgent/path cue due this frame."""
        self.cue_log.confirm(seen_hazard_ids, now)
        cue = select_memory_cues(
            hazards, pose, detections, seen_hazard_ids, now, self.cfg, self.attention,
            self.state, pose_trusted=pose_trusted, suppressed=suppressed,
            session_id=self.session_id, dynamic_labels=self.dynamic_labels,
            match_radius_m=self.match_radius_m,
        )
        if cue is None:
            return None
        played = (
            None
            if self.renderer is None
            else bool(self.renderer.play_memory_cue(cue.audio_azimuth_deg, cue.distance_m))
        )
        self.cue_log.add(cue, played)
        self.last_cue = cue
        if self.log is not None:
            self.log(f"MEMORY CUE | {cue.describe()}")
        return cue

    def display_text(self, now: float) -> str | None:
        """Status-panel text while the last cue is recent, else None."""
        if self.last_cue is None or now - self.last_cue.time > self.cfg.cue_display_s:
            return None
        return f"Memory cue: {self.last_cue.describe()}"
