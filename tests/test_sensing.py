from math import isclose

from navigation.models import Detection, Motion, Pose
from navigation.sensing import SimulatedSensor, local_to_world


def create_detection(relative_x: float, relative_y: float) -> Detection:
    return Detection(
        object_id="pole-1",
        label="pole",
        relative_x=relative_x,
        relative_y=relative_y,
        confidence=0.90,
        motion=Motion.STATIC,
    )


def test_simulated_sensor_produces_six_deterministic_poses() -> None:
    sensor = SimulatedSensor()

    poses = sensor.get_poses()

    assert len(poses) == 6
    assert all(isinstance(pose, Pose) for pose in poses)

    # Calling it again must return identical values - no randomness anywhere.
    assert sensor.get_poses() == poses


def test_local_to_world_at_zero_heading_is_a_plain_offset() -> None:
    pose = Pose(x=1.0, y=2.0, heading_deg=0.0, timestamp=0.0)
    detection = create_detection(relative_x=3.0, relative_y=4.0)

    world = local_to_world(pose, detection)

    # At heading 0 degrees there is no rotation, so the local offset is
    # simply added on top of the pose position.
    assert isclose(world.x, 1.0 + 3.0)
    assert isclose(world.y, 2.0 + 4.0)


def test_local_to_world_rotates_with_heading() -> None:
    pose = Pose(x=0.0, y=0.0, heading_deg=90.0, timestamp=0.0)
    detection = create_detection(relative_x=1.0, relative_y=0.0)

    world = local_to_world(pose, detection)

    # A 90 degree heading rotates the local +x axis onto the world +y axis.
    assert isclose(world.x, 0.0, abs_tol=1e-9)
    assert isclose(world.y, 1.0, abs_tol=1e-9)