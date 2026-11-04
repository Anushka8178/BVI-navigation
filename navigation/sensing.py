"""
Localization: where the user is and which way they face.

Pose sources:

    PhonePose        live: phone rotation vector (heading) + step detector
                     (position), from the SensorServer app. Used for the
                     Android rig (--pose phone).
    simulated_pose   deterministic walk for recorded videos (--pose sim).
    SimulatedSensor  six fixed poses for the tests.

Frame convention (see CLAUDE.md):
    world x = right of the start heading, world y = forward at the start (m).
    heading = degrees CLOCKWISE from +y, so forward = (sin h, cos h) and
    right = (cos h, -sin h). Detections use relative_x = right,
    relative_y = forward.

`local_to_world()` is the other half of the localization contract: given the
current Pose and a Detection (which is expressed relative to the camera),
it produces the position that persistent memory should actually store.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, radians, sin, sqrt
from typing import Any, Callable, NamedTuple, Sequence

from .models import Detection, Pose


class SimulatedSensor:
    """
    Deterministic stand-in for real SLAM.

    Represents a person walking a few steps forward (+y at heading 0) with a
    slight rightward drift (+x) and a slight right (clockwise) turn. Call `get_poses()` to get all six poses at
    once, or `get_pose(frame_id)` to get just one (frame_id is 1-based, the
    same convention `FramePacket.frame_id` already uses elsewhere).
    """

    _POSES: tuple[Pose, ...] = (
        Pose(x=0.0, y=0.0, heading_deg=0.0, timestamp=0.0),
        Pose(x=0.0, y=0.5, heading_deg=0.0, timestamp=1.0),
        Pose(x=0.0, y=1.0, heading_deg=0.0, timestamp=2.0),
        Pose(x=0.1, y=1.5, heading_deg=5.0, timestamp=3.0),
        Pose(x=0.2, y=2.0, heading_deg=5.0, timestamp=4.0),
        Pose(x=0.3, y=2.5, heading_deg=10.0, timestamp=5.0),
    )

    def get_poses(self) -> list[Pose]:
        """Return all six poses, in walking order. Always the same values."""
        return list(self._POSES)

    def get_pose(self, frame_id: int) -> Pose:
        """Return the pose for one frame. frame_id is 1-based (1 through 6)."""
        return self._POSES[frame_id - 1]


class WorldPoint(NamedTuple):
    """
    A detection's position after converting it into world coordinates.

    This is the hand-off value: localization (this file) produces it, and
    persistent memory (navigation/memory.py, the other teammate's file) is
    what's expected to consume it - e.g. store (world_x, world_y) per hazard
    in SQLite.
    """

    x: float
    y: float


def local_to_world(pose: Pose, detection: Detection) -> WorldPoint:
    """
    Convert a detection from camera-relative coordinates into world coordinates.

    A Detection's (relative_x, relative_y) is only meaningful relative to
    wherever the camera currently is and which way it's currently facing.
    To remember a hazard across frames (memory's job), it first has to be
    expressed in one fixed world frame - that's what this function does,
    using the current Pose.
    """
    theta = radians(pose.heading_deg)
    rx, ry = detection.relative_x, detection.relative_y

    # Heading is clockwise from +y: forward = (sin h, cos h), right = (cos h, -sin h).
    world_x = pose.x + rx * cos(theta) + ry * sin(theta)
    world_y = pose.y - rx * sin(theta) + ry * cos(theta)

    return WorldPoint(world_x, world_y)


def world_to_local(pose: Pose, world_x: float, world_y: float) -> tuple[float, float]:
    """Inverse of `local_to_world()`: world point -> (relative_x, relative_y).

    relative_x = right of the user (m), relative_y = ahead of the user (m).
    """
    theta = radians(pose.heading_deg)
    dx, dy = world_x - pose.x, world_y - pose.y

    relative_x = dx * cos(theta) - dy * sin(theta)
    relative_y = dx * sin(theta) + dy * cos(theta)

    return relative_x, relative_y


def simulated_pose(
    timestamp: float,
    walking_speed_m_s: float = 0.5,
    lateral_drift_m_s: float = 0.05,
    turn_rate_deg_s: float = 1.0,
) -> Pose:
    """Deterministic simulated walking pose used instead of real SLAM.

    The user walks FORWARD along the current heading, drifts slightly to
    their right, and turns slowly clockwise (to the right). Position is the
    exact integral of that motion, so it always agrees with the heading.
    """
    heading_deg = turn_rate_deg_s * timestamp
    h = radians(heading_deg)
    omega = radians(turn_rate_deg_s)

    if abs(omega) < 1e-9:
        x = lateral_drift_m_s * timestamp
        y = walking_speed_m_s * timestamp
    else:
        # Integral of v*(sin wt, cos wt) + d*(cos wt, -sin wt) from 0 to t.
        x = (walking_speed_m_s * (1.0 - cos(h)) + lateral_drift_m_s * sin(h)) / omega
        y = (walking_speed_m_s * sin(h) + lateral_drift_m_s * (cos(h) - 1.0)) / omega

    return Pose(x=x, y=y, heading_deg=heading_deg, timestamp=timestamp)


@dataclass(frozen=True)
class LocalizedDetection:
    """
    The actual handoff object for persistent memory (navigation/memory.py).

    `WorldPoint` alone is just an (x, y) pair - memory also needs to know
    *which* hazard that point belongs to and how much to trust it. This
    bundles the `WorldPoint` produced by `local_to_world()` together with
    the rest of `Detection` that memory is expected to care about, so the
    handoff is one object instead of several loose arguments:

        memory.save(localize_detection(pose, detection))
    """

    object_id: str
    label: str
    world_x: float
    world_y: float
    confidence: float
    motion: str
    timestamp: float


def localize_detection(pose: Pose, detection: Detection) -> LocalizedDetection:
    """
    Full localization handoff for one detection: run `local_to_world()` and
    package the result with the detection metadata memory needs to store it.

    This is the one function the memory module actually needs to call.
    """
    world = local_to_world(pose, detection)

    return LocalizedDetection(
        object_id=detection.object_id,
        label=detection.label,
        world_x=world.x,
        world_y=world.y,
        confidence=detection.confidence,
        motion=detection.motion.value,
        timestamp=pose.timestamp,
    )


# ----------------------------------------------------------------------
# Real pose from the chest-mounted phone (SensorServer app)
# ----------------------------------------------------------------------

ROTATION_VECTOR = "android.sensor.rotation_vector"
STEP_DETECTOR = "android.sensor.step_detector"


def _wrap_deg(angle: float) -> float:
    """Wrap an angle to [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def heading_from_rotation_vector(values: Sequence[float]) -> float | None:
    """Compass heading (degrees clockwise from north) from a rotation vector.

    `values` is Android's rotation vector: quaternion (x, y, z[, w]).
    The phone is chest-mounted with the back camera facing forward, so the
    heading is where the camera looks: the device -Z axis projected onto
    the ground. If the phone is nearly flat (camera pointing at the floor)
    the device +Y axis (top of the phone) is used instead, which is
    Android's usual azimuth. Returns None for malformed input.
    """
    if len(values) < 3:
        return None

    qx, qy, qz = (float(v) for v in values[:3])
    if len(values) >= 4:
        qw = float(values[3])
    else:
        qw = sqrt(max(0.0, 1.0 - qx * qx - qy * qy - qz * qz))

    norm = sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-9:
        return None
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    # Columns of the device->world rotation matrix (world: x east,
    # y north, z up), as in SensorManager.getRotationMatrixFromVector().
    top_east = 2.0 * (qx * qy - qz * qw)  # device +Y
    top_north = 1.0 - 2.0 * (qx * qx + qz * qz)
    camera_east = -2.0 * (qx * qz + qy * qw)  # device -Z
    camera_north = -2.0 * (qy * qz - qx * qw)

    if hypot(camera_east, camera_north) >= 0.5:
        east, north = camera_east, camera_north
    else:
        east, north = top_east, top_north

    return _wrap_deg(degrees(atan2(east, north)))


class PhonePose:
    """Live pose from the phone's SensorServer app.

    - Heading: from android.sensor.rotation_vector, relative to the heading
      at calibration, so the start direction is world +y and heading is
      clockwise from +y (CLAUDE.md convention).
    - Position: each android.sensor.step_detector event moves the user
      `stride_m` along the current heading.
    - Validity: the pose is invalid before the first heading arrives, when
      no rotation data has arrived for `stale_after_s`, and while the step
      stream is disconnected (position would silently freeze).

    Each sensor has its own WebSocket in a background daemon thread that
    reconnects after a drop. All the logic is in `handle_message()`, which
    the tests feed with recorded JSON (no network).
    """

    def __init__(
        self,
        phone_ip: str,
        sensor_port: int,
        stride_m: float = 0.7,
        stale_after_s: float = 2.0,
        reconnect_delay_s: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] = print,
    ) -> None:
        self.phone_ip = phone_ip
        self.sensor_port = int(sensor_port)
        self.stride_m = float(stride_m)
        self.stale_after_s = float(stale_after_s)
        self.reconnect_delay_s = float(reconnect_delay_s)
        self._clock = clock
        self._log = log

        self._lock = threading.Lock()
        self._x = 0.0
        self._y = 0.0
        self._raw_heading: float | None = None
        self._heading_offset = 0.0
        self._calibrate_pending = True  # the first heading becomes 0
        self._last_rotation_time: float | None = None
        self.step_count = 0
        # No step socket until start(); tests feed messages directly.
        self._step_connected = True

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._sockets: dict[str, Any] = {}

    # -- Connection --------------------------------------------------------

    def url(self, sensor_type: str) -> str:
        return (
            f"ws://{self.phone_ip}:{self.sensor_port}"
            f"/sensor/connect?type={sensor_type}"
        )

    def start(self) -> "PhonePose":
        """Open both WebSockets in background threads. Returns self."""
        try:
            import websocket  # websocket-client
        except ImportError as error:
            raise RuntimeError(
                "Phone pose needs websocket-client: "
                "python -m pip install websocket-client (or use --pose sim)"
            ) from error

        with self._lock:
            self._step_connected = False

        for sensor_type in (ROTATION_VECTOR, STEP_DETECTOR):
            thread = threading.Thread(
                target=self._run_socket,
                args=(websocket, sensor_type),
                name=f"PhonePose-{sensor_type.rsplit('.', 1)[-1]}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
        return self

    def close(self) -> None:
        self._stop.set()
        for ws in list(self._sockets.values()):
            try:
                ws.close()
            except Exception:
                pass
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads.clear()

    def wait_for_heading(self, timeout_s: float) -> bool:
        """Block until the first heading arrives. True if it did in time."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._raw_heading is not None:
                    return True
            time.sleep(0.05)
        return False

    def _run_socket(self, websocket: Any, sensor_type: str) -> None:
        name = sensor_type.rsplit(".", 1)[-1]
        url = self.url(sensor_type)

        def on_open(_ws: Any) -> None:
            self._log(f"PhonePose: connected {name} ({url})")
            if sensor_type == STEP_DETECTOR:
                with self._lock:
                    self._step_connected = True

        def on_message(_ws: Any, message: Any) -> None:
            self.handle_message(sensor_type, message)

        def on_error(_ws: Any, error: Any) -> None:
            if not self._stop.is_set():
                self._log(f"PhonePose: {name} error: {error}")

        while not self._stop.is_set():
            ws = websocket.WebSocketApp(
                url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
            )
            self._sockets[sensor_type] = ws
            try:
                # Pings detect a dead link (e.g. phone left the hotspot).
                ws.run_forever(ping_interval=5, ping_timeout=4)
            except Exception as error:  # never let the thread die
                self._log(f"PhonePose: {name} crashed: {error}")

            if sensor_type == STEP_DETECTOR:
                with self._lock:
                    self._step_connected = False
            if self._stop.is_set():
                break
            self._log(
                f"PhonePose: {name} disconnected; "
                f"reconnecting in {self.reconnect_delay_s:.1f} s"
            )
            self._stop.wait(self.reconnect_delay_s)

    # -- Message handling (pure logic) ---------------------------------------

    def handle_message(self, sensor_type: str, message: str | bytes | dict) -> None:
        """Apply one SensorServer message: {"values": [...], "timestamp": ns}."""
        try:
            data = json.loads(message) if isinstance(message, (str, bytes)) else message
            values = [float(v) for v in data["values"]]
        except (ValueError, TypeError, KeyError):
            return  # ignore malformed messages

        if sensor_type == ROTATION_VECTOR:
            heading = heading_from_rotation_vector(values)
            if heading is None:
                return
            with self._lock:
                self._raw_heading = heading
                self._last_rotation_time = self._clock()
                if self._calibrate_pending:
                    self._heading_offset = heading
                    self._calibrate_pending = False

        elif sensor_type == STEP_DETECTOR:
            with self._lock:
                if self._raw_heading is None:
                    return  # direction unknown before the first heading
                h = radians(_wrap_deg(self._raw_heading - self._heading_offset))
                self._x += self.stride_m * sin(h)
                self._y += self.stride_m * cos(h)
                self.step_count += 1

    def calibrate(self) -> None:
        """Make the current heading 0 (world +y). If no heading has arrived
        yet, the next one is used."""
        with self._lock:
            if self._raw_heading is None:
                self._calibrate_pending = True
            else:
                self._heading_offset = self._raw_heading
                self._calibrate_pending = False

    def get_pose(self, t: float) -> Pose:
        """Latest pose, stamped with the pipeline time `t`."""
        with self._lock:
            fresh = (
                self._last_rotation_time is not None
                and self._clock() - self._last_rotation_time <= self.stale_after_s
            )
            heading = (
                0.0
                if self._raw_heading is None
                else _wrap_deg(self._raw_heading - self._heading_offset)
            )
            return Pose(
                x=self._x,
                y=self._y,
                heading_deg=heading,
                timestamp=t,
                valid=fresh and self._step_connected,
            )