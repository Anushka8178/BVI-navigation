from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass
class TrackState:
    x: float
    y: float
    timestamp: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0


class HazardTrajectoryTracker:
    """
    Estimates world-frame velocity for each YOLO tracking ID.

    The estimate is deliberately conservative because the current
    prototype uses monocular distance plus simulated localization.
    """

    def __init__(
        self,
        smoothing: float = 0.35,
        max_speed_m_s: float = 8.0,
    ) -> None:
        self.tracks: dict[str, TrackState] = {}
        self.smoothing = smoothing
        self.max_speed_m_s = max_speed_m_s

    def update(
        self,
        object_id: str,
        x: float,
        y: float,
        timestamp: float,
    ) -> TrackState:
        previous = self.tracks.get(object_id)

        if previous is None:
            state = TrackState(
                x=x,
                y=y,
                timestamp=timestamp,
            )
            self.tracks[object_id] = state
            return state

        dt = timestamp - previous.timestamp
        if dt <= 0.0:
            return previous

        raw_vx = (x - previous.x) / dt
        raw_vy = (y - previous.y) / dt

        speed = hypot(raw_vx, raw_vy)
        if speed > self.max_speed_m_s:
            raw_vx = 0.0
            raw_vy = 0.0

        vx = (
            self.smoothing * raw_vx
            + (1.0 - self.smoothing) * previous.velocity_x
        )
        vy = (
            self.smoothing * raw_vy
            + (1.0 - self.smoothing) * previous.velocity_y
        )

        state = TrackState(
            x=x,
            y=y,
            timestamp=timestamp,
            velocity_x=vx,
            velocity_y=vy,
        )
        self.tracks[object_id] = state
        return state


def predict_position(
    state: TrackState,
    seconds: float,
) -> tuple[float, float]:
    return (
        state.x + state.velocity_x * seconds,
        state.y + state.velocity_y * seconds,
    )
