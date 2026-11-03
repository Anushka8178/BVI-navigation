from navigation.memory import HazardEntry
from navigation.models import Pose
from navigation.planner import LocalPathPlanner


def make_pose():
    return Pose(x=0.0, y=0.0, heading_deg=0.0, timestamp=0.0)


def confirm(planner, pose, hazard, count=3):
    decision = None
    for _ in range(count):
        decision = planner.plan(pose, [hazard])
    return decision


def test_blocked_center_recommends_right():
    pose = make_pose()
    hazard = HazardEntry(
        object_id="car-1", label="car", world_x=-0.8, world_y=2.0,
        confidence=0.9, timestamp=0.0,
    )

    planner = LocalPathPlanner()
    decision = confirm(planner, pose, hazard)

    assert decision.current_path_safe is False
    assert decision.selected_path in {"right", "slight-right"}
    assert decision.guidance_changed is True


def test_blocked_center_recommends_left():
    pose = make_pose()
    hazard = HazardEntry(
        object_id="car-2", label="car", world_x=0.8, world_y=2.0,
        confidence=0.9, timestamp=0.0,
    )

    planner = LocalPathPlanner()
    decision = confirm(planner, pose, hazard)

    assert decision.current_path_safe is False
    assert decision.selected_path in {"left", "slight-left"}


def test_clear_path_stays_ahead():
    decision = LocalPathPlanner().plan(make_pose(), [])

    assert decision.current_path_safe is True
    assert decision.selected_path == "ahead"


def test_far_static_object_does_not_trigger_guidance():
    pose = make_pose()
    hazard = HazardEntry(
        object_id="pole-2", label="pole", world_x=0.2, world_y=4.5,
        confidence=0.66, timestamp=0.0,
    )

    planner = LocalPathPlanner()
    decision = confirm(planner, pose, hazard, count=5)

    assert decision.current_path_safe is True
    assert decision.selected_path == "ahead"
    assert decision.blocking_hazards == []


def test_one_noisy_blocking_frame_does_not_trigger_guidance():
    pose = make_pose()
    hazard = HazardEntry(
        object_id="pole-1", label="pole", world_x=0.0, world_y=2.0,
        confidence=0.8, timestamp=0.0,
    )

    planner = LocalPathPlanner()
    decision = planner.plan(pose, [hazard])

    assert decision.current_path_safe is True
    assert decision.guidance_changed is False


def test_guidance_clears_after_consecutive_clear_frames():
    pose = make_pose()
    hazard = HazardEntry(
        object_id="pole-1", label="pole", world_x=0.0, world_y=2.0,
        confidence=0.8, timestamp=0.0,
    )

    planner = LocalPathPlanner()
    blocked = confirm(planner, pose, hazard)
    assert blocked.current_path_safe is False

    cleared = None
    for _ in range(3):
        cleared = planner.plan(pose, [])

    assert cleared.current_path_safe is True
    assert cleared.selected_path == "ahead"
