"""Synthetic camera walk for testing the visual SLAM (simulation only).

Renders a chest-height camera walking inside a square room whose four
walls carry a unique random texture. The true camera pose is known, so
SLAM output can be checked against ground truth without a phone, a
camera or a dataset download. Also provides a fake phone helper with
realistic step-length and heading errors (the dead-reckoning baseline).

World frame (CLAUDE.md): x right, y forward, z up; heading clockwise
from +y. Camera: x right, y down, z forward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from navigation.models import Pose

PX_PER_M = 120


def make_texture(width_m: float, height_m: float, seed: int) -> NDArray[np.uint8]:
    """Unique, corner-rich grayscale texture (random shapes + fine noise)."""
    rng = np.random.default_rng(seed)
    w, h = int(width_m * PX_PER_M), int(height_m * PX_PER_M)
    img = np.full((h, w), 128, np.uint8)
    for _ in range(int(width_m * height_m * 25)):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        size = int(rng.integers(6, 60))
        color = int(rng.integers(0, 256))
        kind = rng.integers(0, 3)
        if kind == 0:
            cv2.rectangle(img, (x, y), (x + size, y + int(size * rng.uniform(0.3, 1.5))), color, -1)
        elif kind == 1:
            cv2.circle(img, (x, y), size // 2, color, -1)
        else:
            pts = rng.integers(-size, size, size=(3, 2)) + np.array([x, y])
            cv2.fillPoly(img, [pts.astype(np.int32)], color)
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


@dataclass
class Room:
    half_size_m: float = 8.0
    height_m: float = 3.0
    seed: int = 7
    walls: list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.uint8]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        s, hgt = self.half_size_m, self.height_m
        corners = [(-s, s), (s, s), (s, -s), (-s, -s)]
        for n in range(4):
            a = np.array([*corners[n], 0.0])
            b = np.array([*corners[(n + 1) % 4], 0.0])
            length = float(np.linalg.norm(b - a))
            self.walls.append((a, b, make_texture(length, hgt, self.seed * 10 + n)))


def camera_axes(heading_deg: float, pitch_deg: float) -> NDArray[np.float64]:
    """Rows: camera x (right), y (down), z (forward) in world coords."""
    h, p = math.radians(heading_deg), math.radians(pitch_deg)
    r = np.array([math.cos(h), -math.sin(h), 0.0])
    f0 = np.array([math.sin(h), math.cos(h), 0.0])
    d0 = np.array([0.0, 0.0, -1.0])
    f = math.cos(p) * f0 + math.sin(p) * d0
    d = math.cos(p) * d0 - math.sin(p) * f0
    return np.stack([r, d, f])


def render(
    room: Room,
    x: float,
    y: float,
    heading_deg: float,
    K: NDArray[np.float64],
    size: tuple[int, int] = (640, 360),
    cam_height_m: float = 1.3,
    pitch_deg: float = 0.0,
    strip_m: float = 0.5,
) -> NDArray[np.uint8]:
    w_img, h_img = size
    canvas = np.full((h_img, w_img), 90, np.uint8)
    canvas[h_img // 2 :, :] = 60  # flat floor tone below the horizon
    axes = camera_axes(heading_deg, pitch_deg)
    centre = np.array([x, y, cam_height_m])
    for a, b, tex in room.walls:
        length = float(np.linalg.norm(b - a))
        direction = (b - a) / length
        n_strips = int(math.ceil(length / strip_m))
        for k in range(n_strips):
            s0, s1 = k * strip_m, min((k + 1) * strip_m, length)
            p0, p1 = a + direction * s0, a + direction * s1
            corners3d = [
                p0 + [0, 0, room.height_m],
                p1 + [0, 0, room.height_m],
                p1,
                p0,
            ]
            cam = np.array([axes @ (c - centre) for c in corners3d])
            if np.any(cam[:, 2] < 0.3):
                continue
            proj = (K @ cam.T).T
            dst = (proj[:, :2] / proj[:, 2:3]).astype(np.float32)
            x0, y0 = np.floor(dst.min(axis=0)).astype(int)
            x1, y1 = np.ceil(dst.max(axis=0)).astype(int)
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, w_img), min(y1, h_img)
            if x1 <= x0 or y1 <= y0:
                continue
            u0, u1 = int(s0 * PX_PER_M), int(s1 * PX_PER_M)
            th = tex.shape[0]
            src = np.array([[u0, 0], [u1, 0], [u1, th - 1], [u0, th - 1]], np.float32)
            H = cv2.getPerspectiveTransform(src, dst - np.array([x0, y0], np.float32))
            roi_w, roi_h = x1 - x0, y1 - y0
            warped = cv2.warpPerspective(tex, H, (roi_w, roi_h), flags=cv2.INTER_LINEAR, borderValue=0)
            mask = cv2.warpPerspective(
                np.full_like(tex, 255), H, (roi_w, roi_h), flags=cv2.INTER_NEAREST, borderValue=0
            )
            roi = canvas[y0:y1, x0:x1]
            np.copyto(roi, warped, where=mask > 0)
    return canvas


def square_walk(
    side_m: float = 8.0,
    laps: int = 1,
    speed_m_s: float = 1.0,
    turn_deg_s: float = 60.0,
    fps: float = 10.0,
) -> list[tuple[float, float, float, float]]:
    """(t, x, y, heading) of a clockwise square walk that returns to start.
    Starts at (-side/2, -side/2) facing +y (heading 0)."""
    out = []
    t = 0.0
    x, y, h = -side_m / 2, -side_m / 2, 0.0
    dt = 1.0 / fps
    step = speed_m_s * dt
    for _lap in range(laps):
        for _edge in range(4):
            travelled = 0.0
            while travelled < side_m - 1e-9:
                out.append((t, x, y, h))
                d = min(step, side_m - travelled)
                x += d * math.sin(math.radians(h))
                y += d * math.cos(math.radians(h))
                travelled += d
                t += dt
            turned = 0.0
            while turned < 90.0 - 1e-9:
                out.append((t, x, y, h))
                d = min(turn_deg_s * dt, 90.0 - turned)
                h = (h + d) % 360.0
                turned += d
                t += dt
    out.append((t, x, y, h))
    return out


class FakePhone:
    """Dead-reckoning phone: steps of `true_stride_m` counted, but the
    pipeline assumes `stride_m`; heading drifts at `drift_deg_s`."""

    def __init__(
        self,
        stride_m: float = 0.7,
        true_stride_m: float = 0.75,
        drift_deg_s: float = 0.15,
        noise_deg: float = 1.0,
        seed: int = 3,
    ) -> None:
        self.stride_m = stride_m
        self.true_stride_m = true_stride_m
        self.drift_deg_s = drift_deg_s
        self.noise_deg = noise_deg
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self._walked = 0.0
        self._heading = 0.0
        self._x = 0.0
        self._y = 0.0
        self._t = 0.0
        self._last_xy: tuple[float, float] | None = None

    def update(self, t: float, x: float, y: float, heading: float) -> None:
        if self._last_xy is not None:
            self._walked += math.hypot(x - self._last_xy[0], y - self._last_xy[1])
        self._last_xy = (x, y)
        self._t = t
        self._heading = heading + self.drift_deg_s * t + float(self.rng.normal(0, self.noise_deg))
        while self._walked >= (self.step_count + 1) * self.true_stride_m:
            self.step_count += 1
            h = math.radians(self._heading)
            self._x += self.stride_m * math.sin(h)
            self._y += self.stride_m * math.cos(h)

    def get_pose(self, t: float) -> Pose:
        return Pose(x=self._x, y=self._y, heading_deg=((self._heading + 180) % 360) - 180, timestamp=t, valid=True)
