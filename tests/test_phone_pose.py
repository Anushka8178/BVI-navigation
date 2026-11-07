"""PhonePose logic, fed with recorded SensorServer JSON (no network)."""

import json
from math import cos, isclose, radians, sin

import random

from navigation.config import load_config
from navigation.models import Detection
from navigation.sensing import (
    ACCELEROMETER,
    ROTATION_VECTOR,
    STEP_DETECTOR,
    PhonePose,
    heading_from_rotation_vector,
    local_to_world,
)

# Recorded from the phone (SensorServer, rotation_vector, lying on a desk).
RECORDED_ROTATION = [
    '{"values":[-0.011442615,-0.039927673,0.12131625,0.9917445,-1.0],'
    '"timestamp":2853188762769042,"accuracy":3}',
    '{"values":[-0.01143171,-0.03991745,0.121385254,0.99173665,-1.0],'
    '"timestamp":2853188963558417,"accuracy":3}',
]
STEP = '{"values":[1.0],"timestamp":2853190000000000,"accuracy":0}'


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def chest_message(compass_deg: float, pitch_deg: float = 0.0) -> str:
    """Rotation-vector message for an upright phone, back camera facing
    `compass_deg` (clockwise from north), tilted down by `pitch_deg`."""
    upright = (cos(radians(45)), sin(radians(45)), 0.0, 0.0)  # +90 deg about X
    tilt = radians(-pitch_deg) / 2
    upright = quat_mul(upright, (cos(tilt), sin(tilt), 0.0, 0.0))
    half = radians(-compass_deg) / 2  # clockwise = negative about +Z (up)
    w, x, y, z = quat_mul((cos(half), 0.0, 0.0, sin(half)), upright)
    return json.dumps({"values": [x, y, z, w, -1.0], "timestamp": 0, "accuracy": 3})


def values(message: str):
    return json.loads(message)["values"]


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def make_pose(stride_m: float = 0.7, **kwargs):
    clock = FakeClock()
    pose = PhonePose(
        "10.0.0.1", 8081, stride_m=stride_m, stale_after_s=2.0, clock=clock, **kwargs
    )
    return pose, clock


def accel_message(magnitude: float, t_s: float) -> str:
    # Phone upright: gravity is mostly along device +Y; only |a| matters.
    return json.dumps(
        {"values": [0.3, magnitude * 0.98, magnitude * 0.199], "timestamp": int(t_s * 1e9)}
    )


def feed_accel(pose, clock, start_s, seconds, signal, rate_hz=50.0):
    """Feed `signal(t)` (m/s^2) as accelerometer messages, advancing the clock."""
    n = int(seconds * rate_hz)
    for i in range(n):
        t = start_s + i / rate_hz
        clock.now = 100.0 + t
        pose.handle_message(ACCELEROMETER, accel_message(signal(t), t))


def walking(cadence_hz=1.8, amplitude=2.5, noise=0.3, seed=1):
    """|acceleration| of a chest-mounted phone while walking: gravity plus
    one bounce per step, plus sensor noise."""
    rng = random.Random(seed)
    return lambda t: (
        9.81 + amplitude * sin(2 * 3.141592653589793 * cadence_hz * t)
        + rng.uniform(-noise, noise)
    )


def test_heading_follows_camera_direction_clockwise():
    assert isclose(heading_from_rotation_vector(values(chest_message(0))), 0.0, abs_tol=1e-6)
    assert isclose(heading_from_rotation_vector(values(chest_message(90))), 90.0, abs_tol=1e-6)
    assert isclose(heading_from_rotation_vector(values(chest_message(-45))), -45.0, abs_tol=1e-6)
    # Tilting the phone down/up does not change where it faces.
    assert isclose(heading_from_rotation_vector(values(chest_message(30, 25))), 30.0, abs_tol=1e-6)


def test_flat_phone_uses_top_of_phone():
    # Identity quaternion: phone flat, top pointing north.
    assert isclose(heading_from_rotation_vector([0.0, 0.0, 0.0, 1.0]), 0.0, abs_tol=1e-9)
    # Rotated 90 deg counter-clockwise about up: top points west.
    s = sin(radians(45))
    assert isclose(heading_from_rotation_vector([0.0, 0.0, s, s]), -90.0, abs_tol=1e-6)
    # Three-value form (w derived) and the recorded messages work too.
    assert isclose(heading_from_rotation_vector([0.0, 0.0, s]), -90.0, abs_tol=1e-6)
    assert heading_from_rotation_vector(values(RECORDED_ROTATION[0])) is not None


def test_first_heading_is_zero_and_turns_are_clockwise():
    pose, _ = make_pose()
    pose.handle_message(ROTATION_VECTOR, chest_message(200))  # arbitrary compass
    assert isclose(pose.get_pose(1.0).heading_deg, 0.0, abs_tol=1e-6)

    pose.handle_message(ROTATION_VECTOR, chest_message(290))  # turn right 90
    assert isclose(pose.get_pose(2.0).heading_deg, 90.0, abs_tol=1e-6)

    pose.handle_message(ROTATION_VECTOR, chest_message(150))  # left 50 from start
    assert isclose(pose.get_pose(3.0).heading_deg, -50.0, abs_tol=1e-6)


def test_calibrate_rezeroes_current_heading():
    pose, _ = make_pose()
    pose.calibrate()  # before any data: applies to the first sample
    pose.handle_message(ROTATION_VECTOR, chest_message(10))
    pose.handle_message(ROTATION_VECTOR, chest_message(70))
    assert isclose(pose.get_pose(0.0).heading_deg, 60.0, abs_tol=1e-6)

    pose.calibrate()
    assert isclose(pose.get_pose(0.0).heading_deg, 0.0, abs_tol=1e-6)
    pose.handle_message(ROTATION_VECTOR, chest_message(80))
    assert isclose(pose.get_pose(0.0).heading_deg, 10.0, abs_tol=1e-6)


def test_steps_advance_along_heading():
    pose, _ = make_pose(stride_m=0.7)
    pose.handle_message(STEP_DETECTOR, STEP)  # no heading yet: ignored
    assert pose.step_count == 0

    pose.handle_message(ROTATION_VECTOR, chest_message(45))
    pose.handle_message(STEP_DETECTOR, STEP)
    pose.handle_message(STEP_DETECTOR, STEP)
    p = pose.get_pose(1.0)
    assert isclose(p.x, 0.0, abs_tol=1e-6) and isclose(p.y, 1.4, abs_tol=1e-6)

    pose.handle_message(ROTATION_VECTOR, chest_message(135))  # turned right 90
    pose.handle_message(STEP_DETECTOR, STEP)
    p = pose.get_pose(2.0)
    assert isclose(p.x, 0.7, abs_tol=1e-6) and isclose(p.y, 1.4, abs_tol=1e-6)
    assert pose.step_count == 3
    assert p.timestamp == 2.0


def test_step_direction_matches_local_to_world():
    # A step must move the user to where "straight ahead" maps in the world.
    pose, _ = make_pose(stride_m=1.0)
    pose.handle_message(ROTATION_VECTOR, chest_message(0))
    pose.handle_message(ROTATION_VECTOR, chest_message(-120))
    before = pose.get_pose(0.0)
    ahead = local_to_world(before, Detection("x-1", "pole", 0.0, 1.0, 0.9))
    pose.handle_message(STEP_DETECTOR, STEP)
    after = pose.get_pose(0.0)
    assert isclose(after.x, ahead.x, abs_tol=1e-6)
    assert isclose(after.y, ahead.y, abs_tol=1e-6)


def test_pose_invalid_without_data_and_after_2_s_silence():
    pose, clock = make_pose()
    assert not pose.get_pose(0.0).valid  # nothing received yet

    for message in RECORDED_ROTATION:
        pose.handle_message(ROTATION_VECTOR, message)
    assert pose.get_pose(0.0).valid

    clock.now += 1.9
    assert pose.get_pose(0.0).valid
    clock.now += 0.2  # 2.1 s since the last rotation message
    assert not pose.get_pose(0.0).valid

    pose.handle_message(STEP_DETECTOR, STEP)  # steps alone don't revive it
    assert not pose.get_pose(0.0).valid
    pose.handle_message(ROTATION_VECTOR, RECORDED_ROTATION[1])
    assert pose.get_pose(0.0).valid


def test_malformed_messages_are_ignored():
    pose, _ = make_pose()
    for bad in ["not json", '{"timestamp": 1}', '{"values": ["x"]}', '{"values": []}']:
        pose.handle_message(ROTATION_VECTOR, bad)
        pose.handle_message(STEP_DETECTOR, bad)
    assert not pose.get_pose(0.0).valid
    assert pose.step_count == 0


def test_urls_use_configured_ip_and_port():
    pose = PhonePose("10.47.242.217", 8081)
    assert pose.url(ROTATION_VECTOR) == (
        "ws://10.47.242.217:8081/sensor/connect?type=android.sensor.rotation_vector"
    )
    assert pose.url(STEP_DETECTOR).endswith("type=android.sensor.step_detector")


# -- Accelerometer fallback step counter ------------------------------------


def test_accelerometer_counts_synthetic_walk():
    pose, clock = make_pose()
    pose.handle_message(ROTATION_VECTOR, chest_message(0))
    # 10 s of walking at 1.8 steps/s, after step_detector has been silent 5 s.
    feed_accel(pose, clock, start_s=5.0, seconds=10.0, signal=walking())

    assert pose.step_source == "accelerometer"
    assert abs(pose.accel_steps - 18) <= 1
    assert pose.step_count == pose.accel_steps
    p = pose.get_pose(0.0)
    assert isclose(p.y, 0.7 * pose.step_count, abs_tol=1e-6)  # walked straight ahead
    assert isclose(p.x, 0.0, abs_tol=1e-6)
    pose.handle_message(ROTATION_VECTOR, chest_message(0))  # heading still streaming
    assert pose.get_pose(0.0).valid
    assert isclose(pose.accel_rate_hz, 50.0, rel_tol=0.01)


def test_standing_still_counts_no_steps():
    pose, clock = make_pose()
    pose.handle_message(ROTATION_VECTOR, chest_message(0))
    feed_accel(pose, clock, 5.0, 10.0, walking(amplitude=0.0, noise=0.4))
    assert pose.accel_steps == 0


def test_min_gap_between_steps():
    # A 4 Hz bounce would be 40 steps in 10 s; the 0.3 s gap keeps every
    # other peak (one per 0.5 s), so about 20.
    pose, clock = make_pose()
    feed_accel(pose, clock, 5.0, 10.0, walking(cadence_hz=4.0, amplitude=3.0, noise=0.0))
    assert 19 <= pose.accel_steps <= 21


def test_fallback_only_after_step_detector_silent_5_s():
    pose, clock = make_pose()
    pose.handle_message(ROTATION_VECTOR, chest_message(0))

    feed_accel(pose, clock, 0.0, 4.5, walking())  # first 4.5 s: not yet
    assert pose.step_source == "step_detector"
    assert pose.accel_steps >= 7 and pose.step_count == 0

    feed_accel(pose, clock, 4.5, 5.5, walking(seed=2))  # now the fallback is active
    assert pose.step_source == "accelerometer"
    applied = pose.step_count
    assert applied > 0

    # step_detector wakes up: it takes over, accelerometer peaks are ignored.
    pose.handle_message(STEP_DETECTOR, STEP)
    assert pose.step_source == "step_detector"
    assert pose.step_count == applied + 1
    feed_accel(pose, clock, 10.0, 3.0, walking(seed=3))
    assert pose.step_count == applied + 1

    clock.now += 5.0  # silent again for 5 s: back to the fallback
    assert pose.step_source == "accelerometer"


def test_fallback_pose_invalid_when_accelerometer_stops():
    pose, clock = make_pose()
    feed_accel(pose, clock, 5.0, 1.0, walking())
    pose.handle_message(ROTATION_VECTOR, chest_message(0))
    assert pose.step_source == "accelerometer" and pose.get_pose(0.0).valid

    clock.now += 1.0
    pose.handle_message(ROTATION_VECTOR, chest_message(0))  # heading still fine
    clock.now += 1.5  # accelerometer silent 2.5 s
    pose.handle_message(ROTATION_VECTOR, chest_message(0))
    assert not pose.get_pose(0.0).valid


def test_from_config_uses_repo_config():
    config = load_config()
    pose = PhonePose.from_config(config["network"], config["pose"])
    assert pose.step_threshold == 10.5
    assert pose.step_min_gap_s == 0.3
    assert pose.step_fallback_after_s == 0.5
    assert pose.url(ACCELEROMETER) == (
        "ws://10.47.242.217:8081/sensor/connect?type=android.sensor.accelerometer"
    )
