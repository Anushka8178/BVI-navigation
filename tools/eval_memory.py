"""Memory ablation on a simulated campus walk (hardware-free, deterministic).

Evidence for the memory claims: persistent cross-journey memory with
class-aware decay, evidence-based forgetting and memory early-warning cues
(sonified top-k acoustic attention). Uses the REAL modules: RouteMemory
(temp SQLite DB, fake clock), AcousticAttention, memory_reasoning
(negative evidence + MemoryEarlyWarning), LocalPathPlanner,
HazardSeverityClassifier and HazardTrajectoryTracker, all configured
from config.yaml. Only the camera (a simulated detector), the pose error
and the audio timing are simulated.

World: a square campus loop (default 20 m sides = 80 m) walked clockwise
from a start mark, with 2 potholes, stairs, a curb, 2 poles, 2 cones, a
barrier and passing people. Sessions: S1 day 0 (first walk), S2 day 2
(a cone removed, a new barrier), S3 day 9 (a pothole repaired).

Variants (ablation):
    A  no memory (camera only)
    B  time-only decay, one 30 s half-life, no memory cues (system before this work)
    C  class-aware decay + reinforcement + memory cues, no negative evidence
    D  full: C + evidence-based forgetting

Output (runs/ is git-ignored): runs/memory_eval/results.csv, results.md,
results.png, and the table on the console.

    python tools/eval_memory.py                    # 5 seeded runs pooled (~2 min)
    python tools/eval_memory.py --pose-model phone --out runs/memory_eval_phone
    python tools/eval_memory.py --render-slam --runs 1 --out runs/memory_eval_vslam  # slow

This is a simulation, not a user study: it shows what the memory logic
does with an imperfect detector and pose, not how people respond to it.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from navigation.config import load_config  # noqa: E402
from navigation.hazard import HazardSeverityClassifier  # noqa: E402
from navigation.memory import HazardEntry, RouteMemory  # noqa: E402
from navigation.memory_reasoning import (  # noqa: E402
    CameraModel,
    MemoryEarlyWarning,
    MemoryReasoningConfig,
    NegativeEvidenceState,
    update_negative_evidence,
)
from navigation.models import Detection, Motion, Pose  # noqa: E402
from navigation.planner import LocalPathPlanner  # noqa: E402
from navigation.reasoning import AcousticAttention  # noqa: E402
from navigation.sensing import local_to_world, world_to_local  # noqa: E402
from navigation.trajectory import HazardTrajectoryTracker  # noqa: E402

DAY_S = 86_400.0
EPOCH0 = 1_794_000_000.0  # simulated "day 0, 09:00" (any epoch works)
VARIANTS = ("A", "B", "C", "D")
VARIANT_NAMES = {
    "A": "no memory (camera only)",
    "B": "time-only decay 30 s, no cues",
    "C": "class decay + reinforcement + cues",
    "D": "full (C + negative evidence)",
}

# (name, label, leg, fraction along the leg, lateral offset m (+ = right), sessions present)
HAZARDS: tuple[tuple[str, str, int, float, float, frozenset[int]], ...] = (
    ("pothole-1", "pothole", 0, 0.30, 0.3, frozenset({1, 2, 3})),
    ("pole-1", "pole", 0, 0.55, 1.2, frozenset({1, 2, 3})),
    ("cone-1", "reflective_cone", 0, 0.78, -0.5, frozenset({1})),  # removed before S2
    ("pothole-2", "pothole", 1, 0.22, -0.2, frozenset({1, 2})),  # repaired before S3
    ("curb-1", "curb", 1, 0.55, 0.0, frozenset({1, 2, 3})),
    ("pole-2", "pole", 1, 0.80, 1.2, frozenset({1, 2, 3})),
    ("stairs-1", "stairs", 2, 0.35, -0.2, frozenset({1, 2, 3})),
    ("cone-2", "reflective_cone", 2, 0.65, 0.6, frozenset({1, 2, 3})),
    ("barrier-1", "barrier", 3, 0.35, -0.6, frozenset({1, 2, 3})),
    ("barrier-2", "barrier", 3, 0.70, 0.4, frozenset({2, 3})),  # new in S2
)
SESSION_DAYS = {1: 0.0, 2: 2.0, 3: 9.0}
STATIC_FALSE_LABELS = ("pothole", "pole", "reflective_cone", "barrier", "curb")
PERSON_RADIUS_M = 0.3
CUE_SECONDS = {"urgent": 0.5, "path": 0.45, "memory": 0.26}  # nominal sound lengths


@dataclass
class Params:
    """Simulation parameters (chosen once; see WORKLOG.md, Task 6)."""

    seed: int = 7
    side_m: float = 20.0
    fps: float = 10.0
    speed_m_s: float = 1.0
    turn_deg_s: float = 60.0
    sessions: tuple[int, ...] = (1, 2, 3)
    hfov_deg: float = 70.0
    min_range_m: float = 1.0
    max_range_m: float = 8.0
    p_detect: float = 0.85
    distance_noise: float = 0.10  # fraction (1 sigma)
    bearing_noise_deg: float = 2.0
    false_positive_rate: float = 0.01  # per frame
    people_per_min: float = 4.0
    person_speed_m_s: float = 1.2
    pose_model: str = "slam"  # "slam" | "phone"
    slam_sigma_m: float = 0.15
    slam_sigma_deg: float = 1.0
    slam_tau_s: float = 5.0
    phone_stride_error: float = 0.07
    phone_drift_deg_s: float = 0.3
    phone_noise_deg: float = 1.0
    render_slam: bool = False


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------


def legs(side_m: float) -> list[tuple[tuple[float, float], float]]:
    """(start corner, heading) of the four legs of the clockwise square."""
    s = side_m
    return [((0.0, 0.0), 0.0), ((0.0, s), 90.0), ((s, s), 180.0), ((s, 0.0), 270.0)]


def route(params: Params) -> list[tuple[float, float, float, float]]:
    """True (t, x, y, heading) at params.fps: walk each leg, turn right in place."""
    from navigation.synthetic_scene import square_walk

    half = params.side_m / 2.0
    return [
        (t, x + half, y + half, h)
        for t, x, y, h in square_walk(
            side_m=params.side_m, laps=1, speed_m_s=params.speed_m_s,
            turn_deg_s=params.turn_deg_s, fps=params.fps,
        )
    ]


@dataclass(frozen=True)
class TruthObject:
    name: str
    label: str
    x: float
    y: float
    sessions: frozenset[int]


def truth_objects(params: Params) -> list[TruthObject]:
    out = []
    for name, label, leg, frac, lateral, sessions in HAZARDS:
        (x0, y0), heading = legs(params.side_m)[leg]
        h = math.radians(heading)
        along = frac * params.side_m
        x = x0 + along * math.sin(h) + lateral * math.cos(h)
        y = y0 + along * math.cos(h) - lateral * math.sin(h)
        out.append(TruthObject(name, label, x, y, sessions))
    return out


@dataclass
class Person:
    name: str
    x0: float
    y0: float
    vx: float
    vy: float
    t0: float
    t1: float

    def at(self, t: float) -> tuple[float, float] | None:
        if not self.t0 <= t <= self.t1:
            return None
        return self.x0 + self.vx * (t - self.t0), self.y0 + self.vy * (t - self.t0)


def spawn_people(walk: Sequence[tuple[float, float, float, float]], params: Params,
                 rng: np.random.Generator) -> list[Person]:
    """Pedestrians coming towards the user along the path, 12 m ahead."""
    people: list[Person] = []
    duration = walk[-1][0]
    t = float(rng.exponential(60.0 / params.people_per_min))
    while t < duration:
        _t, x, y, heading = min(walk, key=lambda row: abs(row[0] - t))
        h = math.radians(heading)
        lateral = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.2))
        px = x + 12.0 * math.sin(h) + lateral * math.cos(h)
        py = y + 12.0 * math.cos(h) - lateral * math.sin(h)
        people.append(Person(
            f"person-{len(people) + 1}", px, py,
            -params.person_speed_m_s * math.sin(h), -params.person_speed_m_s * math.cos(h),
            t, t + 14.0,
        ))
        t += float(rng.exponential(60.0 / params.people_per_min))
    return people


# ---------------------------------------------------------------------------
# Simulated detector (sees the TRUE world from the TRUE pose)
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    t: float
    true_pose: Pose
    detections: list[Detection]
    truth: list[str | None]  # truth object name per detection (None = false positive)


def polar(pose: Pose, x: float, y: float) -> tuple[float, float, float, float]:
    right, forward = world_to_local(pose, x, y)
    return right, forward, math.hypot(right, forward), math.degrees(math.atan2(right, forward))


def simulate_frames(params: Params, session: int, camera: CameraModel,
                    rng: np.random.Generator) -> list[Frame]:
    walk = route(params)
    objects = [o for o in truth_objects(params) if session in o.sessions]
    people = spawn_people(walk, params, rng)
    track_ids: dict[str, int] = {}
    next_fp = [1000]
    frames: list[Frame] = []
    ground = camera.ground_labels

    def track(name: str) -> int:
        return track_ids.setdefault(name, len(track_ids) + 1)

    for t, x, y, heading in walk:
        pose = Pose(x=x, y=y, heading_deg=heading, timestamp=t)
        visible_people = [(p, p.at(t)) for p in people]
        visible_people = [(p, xy) for p, xy in visible_people if xy is not None]
        people_polar = [(p, polar(pose, *xy)) for p, xy in visible_people]
        detections: list[Detection] = []
        truth: list[str | None] = []

        def occluded(rng_m: float, az: float, name: str) -> bool:
            for person, (_r, _f, p_rng, p_az) in people_polar:
                if person.name == name or p_rng >= rng_m - 0.5:
                    continue
                half_width = math.degrees(math.atan2(PERSON_RADIUS_M, max(p_rng, 0.1)))
                if abs(p_az - az) <= half_width:
                    return True
            return False

        candidates: list[tuple[str, str, float, float, Motion]] = [
            (o.name, o.label, o.x, o.y, Motion.STATIC) for o in objects
        ] + [(p.name, "person", *xy, Motion.APPROACHING) for p, xy in visible_people]
        for name, label, ox, oy, motion in candidates:
            _r, forward, rng_m, az = polar(pose, ox, oy)
            if abs(az) > params.hfov_deg / 2.0 or forward <= 0.0:
                continue
            if label in ground:
                if not camera.nearest_ground_m <= forward <= params.max_range_m:
                    continue
            elif not params.min_range_m <= rng_m <= params.max_range_m:
                continue
            if occluded(rng_m, az, name) or rng.random() > params.p_detect:
                continue
            measured = rng_m * (1.0 + params.distance_noise * float(rng.normal()))
            bearing = math.radians(az + params.bearing_noise_deg * float(rng.normal()))
            detections.append(Detection(
                object_id=f"{label}-{track(name)}", label=label,
                relative_x=measured * math.sin(bearing), relative_y=measured * math.cos(bearing),
                confidence=float(rng.uniform(0.5, 0.95)), motion=motion,
                ground_level=label in ground,
            ))
            truth.append(name)

        if rng.random() < params.false_positive_rate:
            label = str(rng.choice(STATIC_FALSE_LABELS))
            distance = float(rng.uniform(2.5, params.max_range_m))
            az = math.radians(float(rng.uniform(-30.0, 30.0)))
            next_fp[0] += 1
            detections.append(Detection(
                object_id=f"{label}-{next_fp[0]}", label=label,
                relative_x=distance * math.sin(az), relative_y=distance * math.cos(az),
                confidence=float(rng.uniform(0.35, 0.6)), ground_level=label in ground,
            ))
            truth.append(None)

        frames.append(Frame(t, pose, detections, truth))
    return frames


# ---------------------------------------------------------------------------
# Pose estimate (what the system believes)
# ---------------------------------------------------------------------------


def estimate_poses(frames: Sequence[Frame], params: Params, rng: np.random.Generator,
                   session: int) -> tuple[list[Pose], list[bool]]:
    """Estimated pose per frame + whether it is trusted (SLAM TRACKING)."""
    if params.render_slam:
        return rendered_slam_poses(frames, params, session)
    poses: list[Pose] = []
    dt = 1.0 / params.fps
    if params.pose_model == "slam":
        # Gauss-Markov error around the truth: bounded (loop closure / map),
        # correlated over slam_tau_s, a new offset each session (relocalization).
        a = math.exp(-dt / params.slam_tau_s)
        b = math.sqrt(1.0 - a * a)
        ex, ey = rng.normal(0.0, params.slam_sigma_m, 2)
        eh = float(rng.normal(0.0, params.slam_sigma_deg))
        for frame in frames:
            p = frame.true_pose
            poses.append(Pose(p.x + ex, p.y + ey, p.heading_deg + eh, p.timestamp))
            ex = a * ex + b * float(rng.normal(0.0, params.slam_sigma_m))
            ey = a * ey + b * float(rng.normal(0.0, params.slam_sigma_m))
            eh = a * eh + b * float(rng.normal(0.0, params.slam_sigma_deg))
        return poses, [True] * len(poses)

    if params.pose_model != "phone":
        raise ValueError(f"unknown pose model {params.pose_model!r}")
    # Phone dead reckoning from the start mark: stride-length error and a
    # heading drift (random sign each session) accumulate along the walk.
    scale = 1.0 + float(rng.choice([-1.0, 1.0])) * params.phone_stride_error
    drift = float(rng.choice([-1.0, 1.0])) * params.phone_drift_deg_s
    x, y = frames[0].true_pose.x, frames[0].true_pose.y
    previous = frames[0].true_pose
    for frame in frames:
        p = frame.true_pose
        err = drift * p.timestamp
        dx, dy = p.x - previous.x, p.y - previous.y
        e = math.radians(err)
        x += scale * (dx * math.cos(e) + dy * math.sin(e))
        y += scale * (-dx * math.sin(e) + dy * math.cos(e))
        noise = float(rng.normal(0.0, params.phone_noise_deg))
        poses.append(Pose(x, y, p.heading_deg + err + noise, p.timestamp))
        previous = p
    return poses, [True] * len(poses)


def rendered_slam_poses(frames: Sequence[Frame], params: Params,
                        session: int) -> tuple[list[Pose], list[bool]]:
    """Run the real VisualSLAM on frames rendered along the true walk
    (navigation/synthetic_scene.py room 4 m larger than the loop on each
    side; simulated phone steps for scale). A new map each session, started
    at the start mark (so its frame is the world frame)."""
    from navigation.synthetic_scene import FakePhone, Room, render
    from navigation.vslam import TRACKING, SlamConfig, VisualSLAM, camera_matrix

    size = (640, 360)
    cfg = SlamConfig.from_config(load_config()["slam"])
    K = camera_matrix(cfg, *size)
    half = params.side_m / 2.0
    room = Room(half_size_m=half + 4.0)
    phone = FakePhone(stride_m=0.7, true_stride_m=0.7 * (1.0 + params.phone_stride_error),
                      drift_deg_s=params.phone_drift_deg_s, seed=params.seed * 10 + session)
    slam = VisualSLAM(cfg, size, helper=phone, log=lambda _msg: None)
    poses, trusted = [], []
    for frame in frames:
        p = frame.true_pose
        phone.update(p.timestamp, p.x - half, p.y - half, p.heading_deg)
        est = slam.process(render(room, p.x - half, p.y - half, p.heading_deg, K, size), p.timestamp)
        start = frames[0].true_pose
        poses.append(Pose(est.x + start.x, est.y + start.y, est.heading_deg, p.timestamp, est.valid))
        trusted.append(bool(est.valid) and slam.state == TRACKING)
    return poses, trusted


# ---------------------------------------------------------------------------
# One variant over all sessions
# ---------------------------------------------------------------------------


class SimAudio:
    """Cue timing like HRTFRenderer: each kind has a cooldown counted from
    the end of its (nominal-length) sound."""

    def __init__(self, audio_cfg: dict[str, Any]) -> None:
        self.cooldown = {
            "urgent": float(audio_cfg["warning_cooldown_s"]),
            "path": float(audio_cfg["path_guidance_cooldown_s"]),
            "memory": float(audio_cfg["memory_cue_cooldown_s"]),
        }
        self.free_at = {kind: -math.inf for kind in self.cooldown}
        self.now = 0.0

    def play(self, kind: str) -> bool:
        if self.now < self.free_at[kind]:
            return False
        self.free_at[kind] = self.now + CUE_SECONDS[kind] + self.cooldown[kind]
        return True

    def play_memory_cue(self, azimuth_deg: float, distance_m: float) -> bool:
        return self.play("memory")


@dataclass
class SessionResult:
    variant: str
    session: int
    day: float
    minutes: float
    static_detected: int = 0
    warned: int = 0
    lead_times: list[float] = field(default_factory=list)
    lead_distances: list[float] = field(default_factory=list)
    alerted: int = 0
    warned_before_alert: int = 0
    lead_vs_alert: list[float] = field(default_factory=list)
    memory_cues: int = 0
    stale_cues: int = 0
    false_cues: int = 0
    urgent_cues: int = 0
    path_cues: int = 0
    memory_size: int = 0
    static_rows: int = 0
    false_in_memory: int = 0
    position_errors: list[float] = field(default_factory=list)
    forgotten: int = 0
    forgot_false: int = 0    # forgotten rows of false positives / removed objects (correct)
    forgot_present: int = 0  # forgotten rows of objects still there (wrong, or a duplicate)
    lost_hazards: int = 0    # objects seen this walk with no memory row at the end
    memory_at_start: int = 0  # static rows remembered from earlier walks when this one starts
    # |cue azimuth - true azimuth of the object| for cues of present objects (deg)
    cue_direction_errors: list[float] = field(default_factory=list)
    removed_still_remembered: int = 0
    # (t, label, truth object or None, kind ok/stale/false, true distance or None)
    cue_events: list[tuple[float, str, str | None, str, float | None]] = field(default_factory=list)
    seed: int = 0

    def row(self) -> dict[str, Any]:
        return summarize([self])


def summarize(group: Sequence[SessionResult]) -> dict[str, Any]:
    """One table row for a (variant, session) over one or more seeded runs.

    Percentages and means are pooled over all hazards of all runs; counts
    are means per run; rates are pooled per minute walked.
    """
    first = group[0]
    runs = len(group)

    def total(name: str) -> float:
        return float(sum(getattr(r, name) for r in group))

    def pooled(name: str) -> list[float]:
        return [v for r in group for v in getattr(r, name)]

    def mean(values: list[float]) -> float | None:
        return round(float(np.mean(values)), 2) if values else None

    def pct(a: str, b: str) -> float | None:
        return round(100.0 * total(a) / total(b), 1) if total(b) else None

    def per_run(name: str) -> float:
        return round(total(name) / runs, 1)

    minutes = total("minutes")
    cues_total = total("urgent_cues") + total("path_cues") + total("memory_cues")
    return {
        "variant": first.variant,
        "session": f"S{first.session}",
        "day": first.day,
        "runs": runs,
        "memory_at_start": per_run("memory_at_start"),
        "warned_before_detection_pct": pct("warned", "static_detected"),
        "mean_lead_time_s": mean(pooled("lead_times")),
        "mean_lead_distance_m": mean(pooled("lead_distances")),
        "warned_before_camera_alert_pct": pct("warned_before_alert", "alerted"),
        "mean_lead_vs_alert_s": mean(pooled("lead_vs_alert")),
        "memory_cues": per_run("memory_cues"),
        "stale_cues": per_run("stale_cues"),
        "false_cues": per_run("false_cues"),
        "false_hazards_in_memory": per_run("false_in_memory"),
        "removed_still_remembered": per_run("removed_still_remembered"),
        "memory_cues_per_min": round(total("memory_cues") / minutes, 2),
        "all_cues_per_min": round(cues_total / minutes, 2),
        "urgent_cues": per_run("urgent_cues"),
        "path_cues": per_run("path_cues"),
        "memory_size": per_run("memory_size"),
        "static_rows": per_run("static_rows"),
        "mean_position_error_m": mean(pooled("position_errors")),
        "mean_cue_direction_error_deg": mean(pooled("cue_direction_errors")),
        "forgotten_by_negative_evidence": per_run("forgotten"),
        "forgot_false_or_removed": per_run("forgot_false"),
        "forgot_present_object_row": per_run("forgot_present"),
        "lost_hazards": per_run("lost_hazards"),
    }


def pooled_rows(results: Sequence[SessionResult]) -> list[dict[str, Any]]:
    """Rows in variant, session order, pooled over seeds."""
    groups: dict[tuple[str, int], list[SessionResult]] = {}
    for result in results:
        groups.setdefault((result.variant, result.session), []).append(result)
    return [summarize(groups[key]) for key in sorted(groups)]


def make_memory(variant: str, config: dict[str, Any], db_path: str, session_id: str,
                clock: Any, reasoning: MemoryReasoningConfig) -> RouteMemory | None:
    memory_cfg = config["memory"]
    if variant == "A":
        return None
    if variant == "B":  # the system before this work: one 30 s half-life for every label
        legacy = {k: v for k, v in memory_cfg.items() if k not in
                  ("categories", "default_half_life_s", "reinforce_per_session", "reinforce_max")}
        legacy["decay_half_life_s"] = 30.0
        return RouteMemory.from_config(legacy, db_path=db_path, session_id=session_id, now_fn=clock)
    return RouteMemory.from_config(memory_cfg, db_path=db_path, session_id=session_id,
                                   now_fn=clock, hit_gain=reasoning.hit_gain)


class Clock:
    def __init__(self) -> None:
        self.t = EPOCH0

    def __call__(self) -> float:
        return self.t


def run_variant(variant: str, sessions: dict[int, tuple[list[Frame], list[Pose], list[bool]]],
                params: Params, config: dict[str, Any], workdir: Path,
                camera: CameraModel) -> list[SessionResult]:
    reasoning = MemoryReasoningConfig.from_config(config["memory_reasoning"])
    objects = {o.name: o for o in truth_objects(params)}
    clock = Clock()
    # memory hazard_id -> how often each truth object (None = false positive)
    # updated it; the most frequent one is what that memory row "is".
    hazard_votes: dict[str, Counter[str | None]] = {}
    results: list[SessionResult] = []

    def truth_of(hazard_id: str) -> str | None:
        votes = hazard_votes.get(hazard_id)
        return votes.most_common(1)[0][0] if votes else None
    db_path = str(workdir / f"memory_{variant}.db")

    for session, (frames, poses, trusted) in sessions.items():
        day = SESSION_DAYS.get(session, float(session))
        start = EPOCH0 + day * DAY_S
        clock.t = start
        session_id = f"{variant}-S{session}"
        memory = make_memory(variant, config, db_path, session_id, clock, reasoning)
        planner = LocalPathPlanner(**config["planner"])
        severity = HazardSeverityClassifier(**config["hazard"])
        tracker = HazardTrajectoryTracker(**config["trajectory"])
        audio = SimAudio(config["audio"])
        attention = AcousticAttention(top_k=int(config["attention"]["top_k"]),
                                      weights=tuple(float(w) for w in config["attention"]["weights"]))
        cues_on = variant in ("C", "D")
        early = MemoryEarlyWarning(
            reasoning, attention, session_id=session_id,
            dynamic_labels=memory.dynamic_labels if memory else frozenset(),
            match_radius_m=memory.match_radius_m if memory else 1.0,
            renderer=audio, log=None,
        ) if cues_on else None
        negative = NegativeEvidenceState()

        result = SessionResult(variant, session, day, minutes=frames[-1].t / 60.0, seed=params.seed)
        if memory is not None:
            result.memory_at_start = sum(
                1 for h in memory.get_hazards() if h.label not in memory.dynamic_labels
            )
        first_detection: dict[str, float] = {}
        first_alert: dict[str, float] = {}
        first_cue: dict[str, tuple[float, float]] = {}  # truth -> (time, true distance)

        for frame, pose, pose_ok in zip(frames, poses, trusted):
            now = start + frame.t
            clock.t = now
            audio.now = frame.t
            entries: list[HazardEntry] = []
            for det, truth in zip(frame.detections, frame.truth):
                if truth is not None:
                    first_detection.setdefault(truth, frame.t)
                wx, wy = local_to_world(pose, det)
                track = tracker.update(object_id=det.object_id, x=wx, y=wy, timestamp=frame.t)
                if memory is not None:
                    entry = memory.remember(det.object_id, det.label, wx, wy, det.confidence,
                                            timestamp=now, velocity_x=track.velocity_x,
                                            velocity_y=track.velocity_y, persist=pose.valid)
                    if entry.hazard_id and pose.valid:
                        hazard_votes.setdefault(entry.hazard_id, Counter())[truth] += 1
                else:
                    entry = HazardEntry(det.object_id, det.label, wx, wy, det.confidence, now,
                                        velocity_x=track.velocity_x, velocity_y=track.velocity_y)
                entries.append(entry)
            seen_ids = {e.hazard_id for e in entries if e.hazard_id}
            truth_by_object = dict(zip((d.object_id for d in frame.detections), frame.truth))

            remembered = memory.get_hazards() if memory is not None else []
            if variant == "D":
                gone = update_negative_evidence(
                    memory, remembered, pose, frame.detections, seen_ids, now, reasoning,
                    camera, negative, pose_trusted=pose_ok, log=None,
                )
                if gone:
                    ids = {g.hazard_id for g in gone}
                    remembered = [h for h in remembered if h.hazard_id not in ids]
                    for hazard_id in ids:
                        truth = truth_of(hazard_id)
                        if truth in objects and session in objects[truth].sessions:
                            result.forgot_present += 1
                        else:
                            result.forgot_false += 1

            # Camera-based audio, exactly as the demo decides it.
            decision = planner.plan(pose, entries)
            urgent_truths = []
            for det in frame.detections:
                verdict = severity.classify(
                    det, path_blocking=det.object_id in decision.blocking_hazards,
                    ttc_s=(decision.collision_ttc_by_hazard or {}).get(det.object_id),
                )
                if verdict.urgent:
                    urgent_truths.append(truth_by_object.get(det.object_id))
            played_this_frame = False
            if urgent_truths and audio.play("urgent"):
                result.urgent_cues += 1
                played_this_frame = True
                for truth in urgent_truths:
                    if truth is not None:
                        first_alert.setdefault(truth, frame.t)
            path_due = False
            if not decision.current_path_safe:
                blocking = [d for d in frame.detections if d.object_id in decision.blocking_hazards]
                path_due = bool(blocking) and decision.selected_path != "ahead" and decision.guidance_changed
                if path_due and not played_this_frame and audio.play("path"):
                    result.path_cues += 1
                    for d in blocking:
                        truth = truth_by_object.get(d.object_id)
                        if truth is not None:
                            first_alert.setdefault(truth, frame.t)

            if early is not None:
                cue = early.update(remembered, pose, frame.detections,
                                   seen_ids if pose.valid else set(), now,
                                   pose_trusted=pose_ok, suppressed=bool(urgent_truths) or path_due)
                if cue is not None and early.cue_log.records[-1].audio_played:
                    result.memory_cues += 1
                    truth = truth_of(cue.hazard_id)
                    true_distance = None
                    if truth not in objects:
                        result.false_cues += 1  # remembered false positive
                        kind = "false"
                    elif session not in objects[truth].sessions:
                        result.stale_cues += 1  # object removed since it was remembered
                        kind = "stale"
                    else:
                        kind = "ok"
                        _r, _f, true_distance, true_azimuth = polar(
                            frame.true_pose, objects[truth].x, objects[truth].y
                        )
                        first_cue.setdefault(truth, (frame.t, true_distance))
                        result.cue_direction_errors.append(
                            abs((cue.azimuth_deg - true_azimuth + 180.0) % 360.0 - 180.0)
                        )
                    result.cue_events.append((frame.t, cue.label, truth, kind, true_distance))

        # -- session metrics -------------------------------------------------
        present_static = {n for n, o in objects.items() if session in o.sessions}
        detected = present_static & set(first_detection)
        result.static_detected = len(detected)
        for name in detected:
            if name in first_cue and first_cue[name][0] < first_detection[name]:
                result.warned += 1
                result.lead_times.append(first_detection[name] - first_cue[name][0])
                result.lead_distances.append(first_cue[name][1])
        alerted = present_static & set(first_alert)
        result.alerted = len(alerted)
        for name in alerted:
            if name in first_cue and first_cue[name][0] < first_alert[name]:
                result.warned_before_alert += 1
                result.lead_vs_alert.append(first_alert[name] - first_cue[name][0])
        if memory is not None:
            rows = memory.get_hazards()
            result.memory_size = len(rows)
            remembered_truths = {truth_of(row.hazard_id) for row in rows}
            result.lost_hazards = len(detected - remembered_truths)
            for row in rows:
                if row.label in memory.dynamic_labels:
                    continue
                result.static_rows += 1
                truth = truth_of(row.hazard_id)
                if truth not in objects or session not in objects[truth].sessions:
                    result.false_in_memory += 1
                    if truth in objects:
                        result.removed_still_remembered += 1
                else:
                    obj = objects[truth]
                    result.position_errors.append(math.hypot(row.world_x - obj.x, row.world_y - obj.y))
        result.forgotten = len(negative.forgotten)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Driver + output
# ---------------------------------------------------------------------------


def run(params: Params, variants: Iterable[str] = VARIANTS,
        workdir: Path | None = None) -> list[SessionResult]:
    """Simulate every session once, then replay it for each variant."""
    config = load_config()
    perception = config["perception"]
    camera = CameraModel(hfov_deg=params.hfov_deg,
                         camera_height_m=float(perception["camera_height_m"]),
                         image_h=720, ground_labels=frozenset(perception["ground_labels"]))
    sessions: dict[int, tuple[list[Frame], list[Pose], list[bool]]] = {}
    for session in params.sessions:
        frames = simulate_frames(params, session, camera,
                                 np.random.default_rng([params.seed, session, 1]))
        poses, trusted = estimate_poses(frames, params,
                                        np.random.default_rng([params.seed, session, 2]), session)
        sessions[session] = (frames, poses, trusted)

    results: list[SessionResult] = []
    with tempfile.TemporaryDirectory(prefix="eval_memory_") as tmp:
        base = Path(tmp) if workdir is None else workdir
        for variant in variants:
            results.extend(run_variant(variant, sessions, params, config, base, camera))
    return results


def run_seeds(params: Params, runs: int, variants: Iterable[str] = VARIANTS) -> list[SessionResult]:
    """`runs` independent runs with seeds params.seed, params.seed + 1, ..."""
    variants = tuple(variants)
    results: list[SessionResult] = []
    for k in range(runs):
        results.extend(run(dataclasses.replace(params, seed=params.seed + k), variants))
    return results


COLUMNS = (
    ("variant", "Variant"),
    ("session", "Session"),
    ("memory_at_start", "Remembered at start"),
    ("warned_before_detection_pct", "Warned before 1st detection %"),
    ("mean_lead_time_s", "Lead s"),
    ("mean_lead_distance_m", "Lead m"),
    ("warned_before_camera_alert_pct", "Warned before camera alert %"),
    ("mean_lead_vs_alert_s", "Lead vs alert s"),
    ("stale_cues", "Stale cues"),
    ("false_cues", "False cues"),
    ("false_hazards_in_memory", "False hazards in memory"),
    ("removed_still_remembered", "Removed still remembered"),
    ("memory_cues_per_min", "Memory cues/min"),
    ("all_cues_per_min", "All cues/min"),
    ("memory_size", "Memory size"),
    ("mean_position_error_m", "Pos. error m"),
    ("mean_cue_direction_error_deg", "Cue direction error deg"),
    ("forgotten_by_negative_evidence", "Forgotten (neg. evidence)"),
    ("forgot_present_object_row", "of which present-object rows"),
    ("lost_hazards", "Objects lost from memory"),
)


def markdown_table(rows: Sequence[dict[str, Any]]) -> str:
    def cell(value: Any) -> str:
        return "-" if value is None else str(value)

    lines = ["| " + " | ".join(title for _key, title in COLUMNS) + " |",
             "|" + "|".join("---" for _ in COLUMNS) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(cell(row[key]) for key, _title in COLUMNS) + " |")
    return "\n".join(lines)


def describe_setup(params: Params, runs: int) -> str:
    pose = "real VisualSLAM on rendered frames" if params.render_slam else f"`{params.pose_model}` noise model"
    seeds = f"seed {params.seed}" if runs == 1 else f"seeds {params.seed}-{params.seed + runs - 1}"
    return (
        f"Simulated {4 * params.side_m:.0f} m campus loop at {params.speed_m_s:g} m/s, "
        f"{params.fps:g} Hz, pose: {pose}, {runs} run(s) ({seeds}). Detector: "
        f"FOV {params.hfov_deg:g} deg, range {params.min_range_m:g}-{params.max_range_m:g} m "
        f"(ground-level classes from the nearest visible ground), p_detect {params.p_detect}, "
        f"distance noise {100 * params.distance_noise:.0f} %, bearing noise "
        f"{params.bearing_noise_deg:g} deg, false positives {params.false_positive_rate}/frame, "
        f"{params.people_per_min:g} passing people/min (they occlude). "
        f"S1 day 0, S2 day 2 (cone-1 removed, barrier-2 added), S3 day 9 (pothole-2 removed)."
    )


def write_outputs(rows: Sequence[dict[str, Any]], params: Params, out: Path,
                  runs: int = 1) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    md_path = out / "results.md"
    legend = "\n".join(f"- **{v}**: {VARIANT_NAMES[v]}" for v in VARIANTS)
    md_path.write_text(
        "# Memory ablation (tools/eval_memory.py)\n\n"
        f"{describe_setup(params, runs)}\n\n{legend}\n\n{markdown_table(rows)}\n\n"
        "Percentages and lead means are pooled over all hazards of all runs; counts are "
        "means per run (session). *Warned before 1st detection*: a memory cue for an object "
        "before the camera first detected it in that session. *Warned before camera alert*: "
        "a memory cue before the first urgent/path cue caused by that object. Lead distances "
        "are true distances at cue time. *Stale* = cue for an object removed since; *false* = "
        "cue for a remembered false positive. *Cue direction error* = angle between the "
        "direction the memory cue points to and the object's true direction (cues of objects "
        "still present). *Remembered at start* = static rows in memory "
        "when the walk begins. *Present-object rows* forgotten are usually noisy duplicates; "
        "*objects lost* counts objects seen in the walk with no memory row at its end. "
        "Simulation only (not a user study).\n",
        encoding="utf-8",
    )
    png_path = out / "results.png"
    pose = "rendered VisualSLAM" if params.render_slam else f"{params.pose_model} pose model"
    footer = (
        f"Values and full setup: results.md / results.csv.  {4 * params.side_m:.0f} m loop, "
        f"{pose}, {runs} seeded run(s), p_detect {params.p_detect}, "
        f"false positives {params.false_positive_rate}/frame. Simulation, not a user study."
    )
    draw_chart(rows, png_path, footer)
    return {"csv": csv_path, "md": md_path, "png": png_path}


# Chart: small multiples, one measure (one y-axis) per panel, grouped bars
# variant x session. Reference palette slots 1-4 in fixed order (validated:
# scripts/validate_palette.js, light surface). Aqua/yellow are below 3:1 on
# the surface, so results.md / results.csv are the table view.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
PANELS = (
    ("warned_before_camera_alert_pct", "Warned before the camera alert (%)"),
    ("warned_before_detection_pct", "Warned before first detection (%)"),
    ("stale_false_cues", "Stale + false memory cues per walk"),
    ("false_hazards_in_memory", "False hazards in memory at end of walk"),
    ("all_cues_per_min", "All cues per minute (urgent + path + memory)"),
    ("memory_at_start", "Static hazards remembered when the walk starts"),
)


def panel_value(row: dict[str, Any], key: str) -> float:
    if key == "stale_false_cues":
        return float(row["stale_cues"]) + float(row["false_cues"])
    value = row.get(key)
    return 0.0 if value is None else float(value)


def draw_chart(rows: Sequence[dict[str, Any]], path: Path, subtitle: str = "") -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        draw_chart_opencv(rows, path)
        return

    sessions = sorted({row["session"] for row in rows})
    variants = [v for v in VARIANTS if any(row["variant"] == v for row in rows)]
    by_key = {(row["variant"], row["session"]): row for row in rows}
    width = 0.8 / max(len(variants), 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), facecolor=SURFACE)
    for ax, (key, title) in zip(axes.flat, PANELS):
        ax.set_facecolor(SURFACE)
        for i, variant in enumerate(variants):
            xs = [s + (i - (len(variants) - 1) / 2) * width for s in range(len(sessions))]
            ys = [panel_value(by_key[(variant, session)], key)
                  if (variant, session) in by_key else 0.0 for session in sessions]
            # Surface-coloured edge = the 2 px gap between adjacent bars.
            ax.bar(xs, ys, width, color=SERIES[variant], edgecolor=SURFACE, linewidth=2,
                   label=f"{variant}  {VARIANT_NAMES[variant]}", zorder=3)
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
        ax.set_xticks(range(len(sessions)))
        ax.set_xticklabels(sessions, color=INK_2)
        ax.tick_params(axis="y", colors=MUTED, labelsize=9)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.set_ylim(bottom=0)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(variants), frameon=False,
               fontsize=10, labelcolor=INK_2, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Memory ablation: simulated campus loop, S1 day 0 / S2 day 2 / S3 day 9",
                 x=0.012, y=0.995, ha="left", fontsize=13, color=INK)
    fig.text(0.012, 0.012, subtitle, fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def draw_chart_opencv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """Same panels with OpenCV when matplotlib is not installed."""
    import cv2

    def bgr(hex_colour: str) -> tuple[int, int, int]:
        r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
        return b, g, r

    sessions = sorted({row["session"] for row in rows})
    variants = [v for v in VARIANTS if any(row["variant"] == v for row in rows)]
    by_key = {(row["variant"], row["session"]): row for row in rows}
    pw, ph, top = 480, 300, 60
    image = np.full((top + 2 * ph, 3 * pw, 3), bgr(SURFACE), np.uint8)
    for i, variant in enumerate(variants):
        x = 20 + i * 330
        cv2.rectangle(image, (x, 18), (x + 14, 32), bgr(SERIES[variant]), -1)
        cv2.putText(image, f"{variant} {VARIANT_NAMES[variant]}", (x + 20, 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr(INK_2), 1, cv2.LINE_AA)
    for n, (key, title) in enumerate(PANELS):
        ox, oy = (n % 3) * pw, top + (n // 3) * ph
        cv2.putText(image, title, (ox + 12, oy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    bgr(INK), 1, cv2.LINE_AA)
        values = [panel_value(by_key[(v, s)], key) if (v, s) in by_key else 0.0
                  for s in sessions for v in variants]
        vmax = max(max(values), 1e-9)
        base, height = oy + ph - 40, ph - 80
        cv2.line(image, (ox + 20, base), (ox + pw - 20, base), bgr(AXIS), 1)
        group_w = (pw - 60) / len(sessions)
        bar_w = group_w * 0.8 / len(variants)
        for s_i, session in enumerate(sessions):
            gx = ox + 30 + s_i * group_w
            cv2.putText(image, session, (int(gx + group_w * 0.35), base + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr(INK_2), 1, cv2.LINE_AA)
            for v_i, variant in enumerate(variants):
                value = panel_value(by_key[(variant, session)], key) if (variant, session) in by_key else 0.0
                x0 = int(gx + v_i * bar_w)
                y0 = int(base - height * value / vmax)
                cv2.rectangle(image, (x0 + 1, y0), (int(x0 + bar_w) - 1, base), bgr(SERIES[variant]), -1)
        cv2.putText(image, f"max {vmax:.1f}", (ox + pw - 110, oy + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, bgr(MUTED), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), image)


def parse_args(argv: Sequence[str] | None = None) -> tuple[Params, Path, int]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = Params()
    parser.add_argument("--seed", type=int, default=defaults.seed, help="first seed")
    parser.add_argument("--runs", type=int, default=5,
                        help="independent seeded runs to pool (seeds seed .. seed+runs-1)")
    parser.add_argument("--pose-model", choices=("slam", "phone"), default=defaults.pose_model)
    parser.add_argument("--render-slam", action="store_true",
                        help="run the real VisualSLAM on rendered frames instead of the slam noise model (slow)")
    parser.add_argument("--side", type=float, default=defaults.side_m, help="loop side (m)")
    parser.add_argument("--fps", type=float, default=defaults.fps)
    parser.add_argument("--p-detect", type=float, default=defaults.p_detect)
    parser.add_argument("--max-range", type=float, default=defaults.max_range_m,
                        help="camera detection range (m)")
    parser.add_argument("--out", type=Path, default=Path("runs") / "memory_eval")
    args = parser.parse_args(argv)
    params = Params(seed=args.seed, pose_model=args.pose_model, render_slam=args.render_slam,
                    side_m=args.side, fps=args.fps, p_detect=args.p_detect,
                    max_range_m=args.max_range)
    if params.render_slam and params.pose_model != "slam":
        parser.error("--render-slam needs --pose-model slam")
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    return params, args.out, args.runs


def main(argv: Sequence[str] | None = None) -> None:
    params, out, runs = parse_args(argv)
    print(f"Simulating {len(params.sessions)} sessions x {len(VARIANTS)} variants x {runs} run(s) "
          f"(pose: {'rendered SLAM' if params.render_slam else params.pose_model}) ...")
    rows = pooled_rows(run_seeds(params, runs))
    paths = write_outputs(rows, params, out, runs)
    print()
    print(markdown_table(rows))
    print()
    for kind, path in paths.items():
        print(f"{kind}: {path.resolve()}")


if __name__ == "__main__":
    main()

