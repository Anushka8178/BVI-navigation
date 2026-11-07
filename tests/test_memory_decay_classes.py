"""Class-aware decay + reinforcement across journeys (memory: categories)."""

import pytest

from navigation.config import load_config
from navigation.memory import HazardEntry, RouteMemory, effective_half_life
from navigation.models import Pose
from navigation.reasoning import AcousticAttention

DAY = 86_400.0
T0 = 1_750_000_000.0  # some epoch time


class Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def memory_from_repo_config(tmp_path, clock, session="s1", **overrides):
    cfg = dict(load_config()["memory"])
    cfg.update(overrides)
    return RouteMemory.from_config(
        cfg, db_path=str(tmp_path / "m.db"), now_fn=clock, session_id=session
    )


def labels_in(memory):
    return sorted(h.label for h in memory.get_hazards())


def test_effective_half_life_reinforcement_and_cap():
    assert effective_half_life(100.0, 1, 0.5, 3.0) == 100.0
    assert effective_half_life(100.0, 2, 0.5, 3.0) == 150.0
    assert effective_half_life(100.0, 3, 0.5, 3.0) == 200.0
    assert effective_half_life(100.0, 10, 0.5, 3.0) == 300.0  # capped at x3
    assert effective_half_life(100.0, 0, 0.5, 3.0) == 100.0  # never below base
    assert effective_half_life(100.0, 5, 0.0, 1.0) == 100.0  # reinforcement off


def test_repo_config_categories(tmp_path):
    memory = memory_from_repo_config(tmp_path, Clock(T0))
    assert memory.half_life_for("person") == 10.0
    assert memory.half_life_for("reflective_cone") == 7200.0
    assert memory.half_life_for("pothole") == 1_209_600.0
    assert memory.half_life_for("not_a_class") == 3600.0  # default_half_life_s
    assert memory.half_life_for("pothole", sessions_seen=3) == 2 * 1_209_600.0

    # Dynamic category and the matching rule's dynamic_labels agree.
    cfg = load_config()["memory"]
    assert set(cfg["categories"]["dynamic"]["labels"]) == set(cfg["dynamic_labels"])


def test_pothole_survives_two_days_person_gone_after_a_minute_cone_after_a_day(tmp_path):
    clock = Clock(T0)
    memory = memory_from_repo_config(tmp_path, clock)
    memory.remember("pothole-1", "pothole", 0.0, 5.0, 0.8, timestamp=T0)
    memory.remember("person-1", "person", 1.0, 3.0, 0.9, timestamp=T0)
    memory.remember("reflective_cone-1", "reflective_cone", -1.0, 4.0, 0.9, timestamp=T0)

    clock.t = T0 + 60.0
    assert labels_in(memory) == ["pothole", "reflective_cone"]  # person forgotten

    clock.t = T0 + DAY
    assert labels_in(memory) == ["pothole"]  # cone forgotten

    clock.t = T0 + 2 * DAY
    hazards = memory.get_hazards()
    assert [h.label for h in hazards] == ["pothole"]
    assert hazards[0].decay_weight == pytest.approx(0.5 ** (2 / 14))


def test_reinforcement_lengthens_life_of_hazard_seen_on_three_walks(tmp_path):
    clock = Clock(T0)
    # Two cones, 10 m apart: one seen on three walks (an hour apart), one once.
    for hour, session in enumerate(("walk-1", "walk-2", "walk-3")):
        clock.t = T0 + hour * 3600.0
        walk = memory_from_repo_config(tmp_path, clock, session=session)
        assert len(walk.get_hazards()) == min(hour, 1)  # still remembered at the next walk
        walk.remember("reflective_cone-1", "reflective_cone", 0.0, 5.0, 0.9, timestamp=clock.t)
    walk.remember("reflective_cone-2", "reflective_cone", 10.0, 5.0, 0.9, timestamp=clock.t)

    by_x = {round(h.world_x): h for h in walk.get_hazards()}
    assert by_x[0].sessions_seen == 3 and by_x[10].sessions_seen == 1

    # 12 h later: single-walk cone (half-life 2 h, weight 1/64) is forgotten,
    # the three-walk cone (half-life 4 h, weight 1/8) is still remembered.
    clock.t += 12 * 3600.0
    hazards = walk.get_hazards()
    assert [round(h.world_x) for h in hazards] == [0]
    assert hazards[0].decay_weight == pytest.approx(0.125)


def test_without_categories_falls_back_to_global_half_life(tmp_path):
    cfg = dict(load_config()["memory"])
    del cfg["categories"]
    memory = RouteMemory.from_config(cfg, db_path=str(tmp_path / "m.db"), now_fn=Clock(0.0))
    assert memory.half_life_for("pothole") == cfg["decay_half_life_s"] == 30.0
    assert memory.half_life_for("person", sessions_seen=5) == 30.0  # no reinforcement


def test_label_in_two_categories_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="pole"):
        RouteMemory(
            str(tmp_path / "m.db"),
            categories={"a": {"half_life_s": 1, "labels": ["pole"]},
                        "b": {"half_life_s": 2, "labels": ["pole"]}},
        )


def test_eviction_drops_least_trusted_first(tmp_path):
    clock = Clock(T0)
    memory = memory_from_repo_config(tmp_path, clock, max_entries=2)
    memory.remember("pothole-1", "pothole", 0.0, 5.0, 0.5, timestamp=T0)
    memory.remember("reflective_cone-1", "reflective_cone", 3.0, 5.0, 0.9, timestamp=T0)
    clock.t = T0 + 3600.0
    # Cone has decayed to 0.5 (pothole ~1.0); a new pole overflows the cap.
    memory.remember("pole-1", "pole", 6.0, 5.0, 0.9, timestamp=clock.t)
    assert labels_in(memory) == ["pole", "pothole"]


def test_attention_trust_term_uses_belief():
    pose = Pose(x=0.0, y=0.0, heading_deg=0.0, timestamp=0.0)
    sure = HazardEntry("a", "pothole", 0.0, 4.0, 0.9, 0.0, decay_weight=1.0, belief=1.0)
    doubtful = HazardEntry("b", "pothole", 0.0, 4.0, 0.9, 0.0, decay_weight=1.0, belief=0.2)
    ranked = AcousticAttention(top_k=2, weights=(0.0, 0.0, 1.0)).rank([doubtful, sure], pose)
    assert [h.object_id for h in ranked] == ["a", "b"]
    assert ranked[1].priority == pytest.approx(0.2)
