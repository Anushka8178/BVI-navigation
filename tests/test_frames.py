"""Frame-convention tests (CLAUDE.md "Conventions").

World: x = right of start heading, y = forward at start (m).
Heading: degrees clockwise from +y, so forward = (sin h, cos h).
"""

from math import isclose

import pytest

from navigation.memory import HazardEntry
from navigation.models import Detection, Pose
from navigation.planner import LocalPathPlanner
from navigation.reasoning import AcousticAttention
from navigation.sensing import local_to_world, simulated_pose, world_to_local


def detection_at(relative_x: float, relative_y: float) -> Detection:
    return Detection(
        object_id="pole-1",
        label="pole",
        relative_x=relative_x,
        relative_y=relative_y,
        confidence=0.9,
    )


def hazard_at(world_x: float, world_y: float) -> HazardEntry:
    return HazardEntry(
        object_id="pole-1",
        label="pole",
        world_x=world_x,
        world_y=world_y,
        confidence=0.9,
        timestamp=0.0,
    )


def test_heading_0_ahead_point_maps_to_plus_y() -> None:
    pose = Pose(x=1.0, y=2.0, heading_deg=0.0, timestamp=0.0)

    world = local_to_world(pose, detection_at(0.0, 3.0))

    assert isclose(world.x, 1.0, abs_tol=1e-9)
    assert isclose(world.y, 5.0, abs_tol=1e-9)


def test_heading_90_ahead_point_maps_to_plus_x() -> None:
    pose = Pose(x=1.0, y=2.0, heading_deg=90.0, timestamp=0.0)

    world = local_to_world(pose, detection_at(0.0, 3.0))

    assert isclose(world.x, 4.0, abs_tol=1e-9)
    assert isclose(world.y, 2.0, abs_tol=1e-9)


def test_hazard_ahead_is_on_the_left_after_turning_right_90() -> None:
    start = Pose(x=0.0, y=0.0, heading_deg=0.0, timestamp=0.0)
    world = local_to_world(start, detection_at(0.0, 4.0))

    turned = Pose(x=0.0, y=0.0, heading_deg=90.0, timestamp=1.0)
    relative_x, relative_y = world_to_local(turned, world.x, world.y)
    seen_again = detection_at(relative_x, relative_y)

    assert relative_x < 0.0
    assert seen_again.azimuth_deg < 0.0  # negative = LEFT
    assert isclose(seen_again.azimuth_deg, -90.0, abs_tol=1e-6)


@pytest.mark.parametrize("heading_deg", [0.0, 37.0, 90.0, 180.0, -135.0, 271.5])
def test_world_to_relative_to_world_round_trip(heading_deg: float) -> None:
    pose = Pose(x=-1.5, y=3.25, heading_deg=heading_deg, timestamp=0.0)
    world_x, world_y = 2.0, -0.75

    relative_x, relative_y = world_to_local(pose, world_x, world_y)
    world = local_to_world(pose, detection_at(relative_x, relative_y))

    assert isclose(world.x, world_x, abs_tol=1e-9)
    assert isclose(world.y, world_y, abs_tol=1e-9)


def test_attention_and_planner_agree_with_local_to_world() -> None:
    # Facing +x: a hazard localized straight ahead must be fully aligned for
    # AcousticAttention and on the planner's walking corridor.
    pose = Pose(x=0.0, y=0.0, heading_deg=90.0, timestamp=0.0)
    world = local_to_world(pose, detection_at(0.0, 2.0))
    ahead = hazard_at(world.x, world.y)
    behind = hazard_at(-world.x, -world.y)

    ranked = AcousticAttention(top_k=2, weights=(0.0, 1.0, 0.0)).rank([behind, ahead], pose)
    assert ranked[0] is ahead
    assert isclose(ahead.priority, 1.0)
    assert isclose(behind.priority, 0.0)

    planner = LocalPathPlanner()
    assert planner.time_to_collision(pose, ahead) is not None
    assert planner.time_to_collision(pose, behind) is None


def test_simulated_pose_walks_forward_along_heading() -> None:
    start = simulated_pose(0.0)
    assert (start.x, start.y, start.heading_deg) == (0.0, 0.0, 0.0)

    # Early on (heading ~0) the user moves mostly along +y.
    early = simulated_pose(1.0)
    assert early.y > 0.45
    assert 0.0 < early.x < 0.1

    # Small step displacement points along the current heading (+ small drift).
    t, dt = 60.0, 0.01
    a, b = simulated_pose(t), simulated_pose(t + dt)
    step_x, step_y = world_to_local(a, b.x, b.y)
    assert isclose(step_y / dt, 0.5, abs_tol=1e-3)   # forward speed
    assert isclose(step_x / dt, 0.05, abs_tol=1e-2)  # rightward drift
