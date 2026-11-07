"""Memory early warning: sonified top-k attention (fake renderer, no audio)."""

import math

import pytest

from navigation.config import load_config
from navigation.memory import RouteMemory
from navigation.memory_reasoning import (
    MemoryEarlyWarning,
    MemoryReasoningConfig,
    direction_words,
    format_age,
)
from navigation.models import Detection, Pose
from navigation.reasoning import AcousticAttention

DAY = 86_400.0
T0 = 1_750_000_000.0
POSE = Pose(x=0.0, y=0.0, heading_deg=0.0, timestamp=0.0)
CFG = MemoryReasoningConfig()


class Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class FakeRenderer:
    def __init__(self, accept: bool = True) -> None:
        self.calls: list[tuple[float, float]] = []
        self.accept = accept

    def play_memory_cue(self, azimuth_deg: float, distance_m: float) -> bool:
        self.calls.append((azimuth_deg, distance_m))
        return self.accept


def at_polar(distance: float, azimuth_deg: float) -> tuple[float, float]:
    a = math.radians(azimuth_deg)
    return distance * math.sin(a), distance * math.cos(a)


class Walks:
    """Walk 1 (two days ago) builds memory; walk 2 (now) gets the cues."""

    def __init__(self, tmp_path, hazards=(("pothole", 5.0, 0.0),)):
        self.clock = Clock(T0 - 2 * DAY)
        cfg = load_config()["memory"]
        db = str(tmp_path / "m.db")
        walk1 = RouteMemory.from_config(cfg, db_path=db, now_fn=self.clock, session_id="walk-1")
        self.ids = []
        for n, (label, distance, azimuth) in enumerate(hazards):
            x, y = at_polar(distance, azimuth)
            self.ids.append(walk1.remember(f"{label}-{n}", label, x, y, 0.9, timestamp=self.clock.t).hazard_id)
        self.clock.t = T0
        self.memory = RouteMemory.from_config(cfg, db_path=db, now_fn=self.clock, session_id="walk-2")
        self.renderer = FakeRenderer()
        self.lines: list[str] = []
        self.warning = MemoryEarlyWarning(
            CFG, AcousticAttention(top_k=3), session_id="walk-2",
            dynamic_labels=self.memory.dynamic_labels, match_radius_m=self.memory.match_radius_m,
            renderer=self.renderer, log=self.lines.append,
        )

    def frame(self, dt=0.0, detections=(), seen=frozenset(), pose=POSE, **kwargs):
        self.clock.t += dt
        return self.warning.update(
            self.memory.get_hazards(), pose, list(detections), set(seen), self.clock.t, **kwargs
        )


def test_remembered_pothole_ahead_and_unseen_gets_a_cue(tmp_path):
    w = Walks(tmp_path)
    cue = w.frame()

    assert cue is not None and cue.hazard_id == w.ids[0]
    assert cue.distance_m == pytest.approx(5.0)
    assert cue.earlier_walks == 1
    assert w.renderer.calls == [(pytest.approx(0.0), pytest.approx(5.0))]
    assert w.lines == ["MEMORY CUE | pothole ahead 5.0 m | seen in 1 earlier walk, last 2 days ago"]
    assert w.warning.display_text(w.clock.t).startswith("Memory cue: pothole ahead 5.0 m")
    assert w.warning.display_text(w.clock.t + CFG.cue_display_s + 0.1) is None


def test_currently_detected_hazard_gets_no_cue(tmp_path):
    w = Walks(tmp_path)
    assert w.frame(seen={w.ids[0]}) is None
    # Same label close to the remembered spot counts as detected too.
    x, y = at_polar(5.3, 2.0)
    pothole = Detection("pothole-9", "pothole", relative_x=x, relative_y=y, confidence=0.8)
    assert w.frame(detections=[pothole]) is None
    assert w.renderer.calls == []


@pytest.mark.parametrize(
    "distance, azimuth",
    [(5.0, 180.0), (5.0, 45.0), (9.0, 0.0), (2.0, 0.0)],
    ids=["behind", "beyond-40deg", "too-far", "too-close"],
)
def test_hazard_outside_the_cue_window_gets_no_cue(tmp_path, distance, azimuth):
    w = Walks(tmp_path, hazards=(("pothole", distance, azimuth),))
    assert w.frame() is None


def test_hazard_behind_after_turning_gets_no_cue_but_ahead_does(tmp_path):
    w = Walks(tmp_path, hazards=(("pole", 5.0, 0.0),))  # world (0, 5)
    assert w.frame(pose=Pose(0.0, 0.0, 180.0, 0.0)) is None  # facing away
    assert w.frame(dt=0.1, pose=Pose(0.0, 0.0, 10.0, 0.0)) is not None


def test_cooldowns_hold(tmp_path):
    w = Walks(tmp_path, hazards=(("pothole", 5.0, 0.0), ("pole", 6.0, -20.0)))
    first = w.frame()
    assert first.hazard_id == w.ids[0]  # nearer and straight ahead ranks first
    assert w.frame(dt=1.0) is None  # global cooldown (2 s)
    second = w.frame(dt=1.1)
    assert second.hazard_id == w.ids[1]  # pothole on its own 8 s cooldown
    assert w.frame(dt=2.1) is None  # both hazards on cooldown
    third = w.frame(dt=4.0)  # 8.2 s after the first cue
    assert third.hazard_id == w.ids[0]
    assert len(w.renderer.calls) == 3


def test_urgent_or_path_cue_suppresses_memory_cue(tmp_path):
    w = Walks(tmp_path)
    assert w.frame(suppressed=True) is None
    assert w.renderer.calls == []
    # Suppression books no cooldown: the next free frame gets the cue.
    assert w.frame(dt=0.1) is not None


def test_untrusted_pose_or_low_trust_gets_no_cue(tmp_path):
    w = Walks(tmp_path)
    assert w.frame(pose_trusted=False) is None  # e.g. SLAM relocalizing

    for _ in range(4):  # belief 0.7 ** 4 = 0.24 < cue_min_trust
        w.memory.record_miss(w.ids[0], CFG.miss_factor)
    assert w.frame() is None
    assert w.renderer.calls == []


def test_hazard_seen_a_moment_ago_is_still_in_view_not_a_memory(tmp_path):
    w = Walks(tmp_path)
    x, y = at_polar(5.0, 0.0)
    w.memory.remember("pothole-3", "pothole", x, y, 0.9, timestamp=T0)  # seen in this walk
    assert w.frame(dt=0.5) is None  # one-frame dropout
    cue = w.frame(dt=0.6)  # unseen for 1.1 s -> memory cue
    assert cue is not None and cue.hazard_id == w.ids[0]
    assert cue.earlier_walks == 1  # walk 1; this walk is not "earlier"


def test_dynamic_hazard_gets_no_cue(tmp_path):
    clock = Clock(T0)
    memory = RouteMemory.from_config(
        load_config()["memory"], db_path=str(tmp_path / "m.db"), now_fn=clock
    )
    memory.remember("person-1", "person", 0.0, 5.0, 0.9, timestamp=T0)
    clock.t = T0 + 3.0  # still remembered (10 s half-life)
    warning = MemoryEarlyWarning(CFG, AcousticAttention(), log=None)
    assert len(memory.get_hazards()) == 1
    assert warning.update(memory.get_hazards(), POSE, [], set(), clock.t) is None


def test_hazard_seen_only_earlier_in_this_walk(tmp_path):
    clock = Clock(T0)
    memory = RouteMemory.from_config(
        load_config()["memory"], db_path=str(tmp_path / "m.db"), now_fn=clock, session_id="walk-1"
    )
    memory.remember("curb-1", "curb", 0.0, 6.0, 0.9, timestamp=T0)
    warning = MemoryEarlyWarning(CFG, AcousticAttention(), session_id="walk-1", log=None)
    clock.t = T0 + 40.0
    cue = warning.update(memory.get_hazards(), POSE, [], set(), clock.t)
    assert cue.earlier_walks == 0
    assert cue.describe() == "curb ahead 6.0 m | seen earlier in this walk, 40 s ago"


def test_lead_time_is_measured_when_the_camera_confirms(tmp_path):
    w = Walks(tmp_path)
    cue = w.frame()
    walked = Pose(x=0.0, y=1.4, heading_deg=0.0, timestamp=0.0)
    w.frame(dt=0.7, pose=walked)  # not yet seen
    w.frame(dt=0.8, pose=walked, seen={w.ids[0]})  # camera sees it 1.5 s after the cue

    record = w.warning.cue_log.records[0]
    assert record.cue == cue
    assert record.lead_time_s == pytest.approx(1.5)
    summary = w.warning.cue_log.summary(hazards_forgotten=2)
    assert summary == {
        "cues_played": 1,
        "cues_with_audio": 1,
        "cues_later_confirmed": 1,
        "mean_lead_time_s": 1.5,
        "mean_lead_distance_m": 5.0,
        "hazards_forgotten_by_negative_evidence": 2,
    }
    row = record.as_dict(start_epoch=T0)
    assert row["confirmed_by_camera"] and row["lead_distance_m"] == 5.0
    assert row["time_s"] == 0.0 and row["direction"] == "ahead"


def test_unconfirmed_cue_and_no_renderer():
    warning = MemoryEarlyWarning(CFG, AcousticAttention(), log=None)
    assert warning.cue_log.summary()["mean_lead_time_s"] is None
    assert warning.renderer is None


def test_direction_words_and_age_format():
    assert direction_words(0.0) == "ahead"
    assert direction_words(-18.0) == "ahead-left"
    assert direction_words(18.0) == "ahead-right"
    assert direction_words(-35.0) == "left"
    assert format_age(35) == "35 s"
    assert format_age(250) == "4 min"
    assert format_age(3 * 3600 + 5) == "3 h"
    assert format_age(2 * DAY + 10) == "2 days"


def test_config_cue_section():
    cfg = MemoryReasoningConfig.from_config(load_config()["memory_reasoning"])
    assert (cfg.cue_min_m, cfg.cue_max_m, cfg.cue_max_bearing_deg) == (2.5, 8.0, 40.0)
    assert (cfg.cue_min_trust, cfg.cue_cooldown_s, cfg.cue_global_cooldown_s) == (0.3, 8.0, 2.0)
    assert load_config()["audio"]["memory_cue_cooldown_s"] == 2.0
