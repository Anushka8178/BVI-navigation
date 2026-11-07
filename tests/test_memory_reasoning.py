"""Reasoning over memory: evidence-based forgetting (negative evidence)."""

import math

import pytest

from navigation.config import load_config
from navigation.memory import RouteMemory
from navigation.memory_reasoning import (
    CameraModel,
    MemoryReasoningConfig,
    NegativeEvidenceState,
    pose_is_trusted,
    update_negative_evidence,
)
from navigation.models import Detection, Pose

T0 = 1_750_000_000.0
POSE = Pose(x=0.0, y=0.0, heading_deg=0.0, timestamp=0.0)
CAMERA = CameraModel(hfov_deg=70.0, camera_height_m=1.3, image_h=720)
CFG = MemoryReasoningConfig()


class Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def at_polar(distance: float, azimuth_deg: float) -> tuple[float, float]:
    """World point at this range/azimuth from POSE (heading 0)."""
    a = math.radians(azimuth_deg)
    return distance * math.sin(a), distance * math.cos(a)


def det(label: str, distance: float, azimuth_deg: float, oid: str = "d-1") -> Detection:
    x, y = at_polar(distance, azimuth_deg)
    return Detection(object_id=oid, label=label, relative_x=x, relative_y=y, confidence=0.9)


class World:
    def __init__(self, tmp_path):
        self.clock = Clock(T0)
        self.memory = RouteMemory.from_config(
            load_config()["memory"], db_path=str(tmp_path / "m.db"), now_fn=self.clock,
            session_id="walk-1",
        )
        self.state = NegativeEvidenceState()
        self.logs: list[str] = []

    def add(self, label: str, distance: float, azimuth_deg: float, oid: str = "h-1"):
        x, y = at_polar(distance, azimuth_deg)
        return self.memory.remember(oid, label, x, y, 0.9, timestamp=self.clock.t)

    def step(self, dt: float, detections=(), seen=frozenset(), pose=POSE, trusted=True):
        self.clock.t += dt
        return update_negative_evidence(
            self.memory, self.memory.get_hazards(), pose, list(detections), set(seen),
            self.clock.t, CFG, CAMERA, self.state, pose_trusted=trusted, log=self.logs.append,
        )

    def hazard(self, hazard_id: str):
        return next((h for h in self.memory.get_hazards() if h.hazard_id == hazard_id), None)


def test_hazard_in_clear_view_is_forgotten_after_four_spaced_misses(tmp_path):
    w = World(tmp_path)
    pothole = w.add("pothole", 4.0, 0.0)

    assert w.step(0.6) == []  # miss 1
    assert w.step(0.2) == []  # within miss_interval_s: not counted
    assert w.hazard(pothole.hazard_id).misses == 1
    assert w.step(0.4) == []  # miss 2 (0.6 s after miss 1)
    assert w.step(0.6) == []  # miss 3
    remembered = w.hazard(pothole.hazard_id)
    assert remembered.misses == 3
    assert remembered.belief == pytest.approx(0.7 ** 3)

    events = w.step(0.6)  # miss 4 -> forgotten
    assert [e.hazard_id for e in events] == [pothole.hazard_id]
    assert w.memory.get_hazards() == []
    assert w.logs == [f"MEMORY | forgot pothole {pothole.hazard_id[:8]}: not seen in clear view 4 times"]
    assert w.state.forgotten == events


def test_occluded_hazard_gets_no_miss(tmp_path):
    w = World(tmp_path)
    pole = w.add("pole", 4.0, 0.0)
    person_in_front = det("person", 2.0, 3.0)  # 3 deg off, 2 m nearer
    for _ in range(6):
        w.step(0.6, [person_in_front])
    assert w.hazard(pole.hazard_id).misses == 0

    # A detection 10 deg away does not hide it.
    w.step(0.6, [det("person", 2.0, 10.0)])
    assert w.hazard(pole.hazard_id).misses == 1


@pytest.mark.parametrize(
    "label, distance, azimuth, expect_miss",
    [
        ("pole", 4.0, 30.0, False),   # outside hfov/2 - edge margin (27 deg)
        ("pole", 4.0, -25.0, True),   # inside
        ("pole", 7.0, 0.0, False),    # too far (forget_max_m 6)
        ("pole", 1.0, 0.0, False),    # too close (forget_min_m 1.5)
        ("pole", 2.0, 0.0, True),
        ("pothole", 2.0, 0.0, False),  # ground-level: below the image bottom (2.47 m)
        ("pothole", 3.0, 0.0, True),
    ],
)
def test_expected_visible_limits(tmp_path, label, distance, azimuth, expect_miss):
    w = World(tmp_path)
    hazard = w.add(label, distance, azimuth)
    w.step(0.6)
    assert w.hazard(hazard.hazard_id).misses == (1 if expect_miss else 0)


def test_dynamic_hazard_is_never_touched(tmp_path):
    w = World(tmp_path)
    person = w.add("person", 3.0, 0.0)
    for _ in range(3):
        w.step(0.6)
    # Still there (10 s half-life) and untouched by negative evidence.
    assert w.hazard(person.hazard_id).misses == 0
    assert w.state.misses_recorded == 0


def test_untrusted_pose_touches_nothing(tmp_path):
    w = World(tmp_path)
    pole = w.add("pole", 4.0, 0.0)
    invalid = Pose(x=0.0, y=0.0, heading_deg=0.0, timestamp=0.0, valid=False)
    for _ in range(5):
        w.step(0.6, pose=invalid, trusted=pose_is_trusted(invalid))
        w.step(0.6, trusted=pose_is_trusted(POSE, slam_state="RELOCALIZING"))
    assert w.hazard(pole.hazard_id).misses == 0

    assert pose_is_trusted(POSE)
    assert pose_is_trusted(POSE, slam_state="TRACKING")
    assert not pose_is_trusted(POSE, slam_state="LOST")
    assert not pose_is_trusted(invalid, slam_state="TRACKING")


def test_matched_detection_is_not_a_miss_and_sighting_restores_belief(tmp_path):
    w = World(tmp_path)
    cone = w.add("reflective_cone", 4.0, 0.0)
    w.step(0.6)
    w.step(0.6)
    assert w.hazard(cone.hazard_id).misses == 2

    # Resolved to this hazard by memory, or same label within match radius.
    w.step(0.6, seen={cone.hazard_id})
    w.step(0.6, [det("reflective_cone", 4.3, 2.0)])
    assert w.hazard(cone.hazard_id).misses == 2
    # A different label at the same spot is not this hazard.
    w.step(0.6, [det("pole", 4.0, 0.0)])
    assert w.hazard(cone.hazard_id).misses == 3

    # Real sighting through memory: misses reset, belief + hit_gain.
    before = w.hazard(cone.hazard_id).belief
    x, y = at_polar(4.0, 0.0)
    seen = w.memory.remember("reflective_cone-5", "reflective_cone", x, y, 0.9, timestamp=w.clock.t)
    assert seen.hazard_id == cone.hazard_id
    assert seen.misses == 0
    assert seen.belief == pytest.approx(min(1.0, before + CFG.hit_gain))


def test_heading_is_used_for_visibility(tmp_path):
    w = World(tmp_path)
    pole = w.add("pole", 4.0, 0.0)  # world (0, 4)
    facing_right = Pose(x=0.0, y=0.0, heading_deg=90.0, timestamp=0.0)
    w.step(0.6, pose=facing_right)  # pole is now 90 deg to the left: not in view
    assert w.hazard(pole.hazard_id).misses == 0


def test_config_section_matches_dataclass():
    section = load_config()["memory_reasoning"]
    cfg = MemoryReasoningConfig.from_config(section)
    assert set(section) == set(MemoryReasoningConfig.__dataclass_fields__)
    assert cfg == MemoryReasoningConfig()  # repo config = documented defaults
    assert isinstance(cfg.forget_after_misses, int)
    with pytest.raises(ValueError, match="unknown"):
        MemoryReasoningConfig.from_config({"forget_after_mises": 3})
