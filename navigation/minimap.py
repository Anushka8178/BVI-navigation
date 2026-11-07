"""Top-down mini-map overlay for the annotated video (SLAM demo view).

Draws, in a corner of the frame: the walked trajectory, SLAM keyframes,
loop-closure links, remembered hazards (from persistent memory), the
current position + heading, and the SLAM state. World frame (CLAUDE.md):
x right, y forward (drawn upwards), heading clockwise from +y.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import cv2
import numpy as np

STATE_COLORS = {
    "TRACKING": (80, 220, 80),
    "RELOCALIZING": (0, 200, 255),
    "LOST": (60, 60, 255),
    "INIT": (200, 200, 200),
}


def draw_minimap(
    image: np.ndarray,
    trajectory: Sequence[tuple[float, float]],
    keyframes: Iterable[tuple[float, float]],
    loop_links: Iterable[tuple[tuple[float, float], tuple[float, float]]],
    hazards: Iterable[tuple[float, float, str, float]],
    pose: tuple[float, float, float],
    state: str,
    info: str = "",
    size: int = 260,
    margin: int = 12,
) -> None:
    """Draw the mini-map in place in the bottom-right corner.

    hazards: (world_x, world_y, label, decay_weight).
    """
    h_img, w_img = image.shape[:2]
    size = min(size, int(0.45 * h_img), w_img - 2 * margin)
    if size < 80:
        return
    x0, y0 = w_img - size - margin, h_img - size - margin
    panel = image[y0 : y0 + size, x0 : x0 + size]
    overlay = np.zeros_like(panel)
    overlay[:] = (25, 25, 25)
    cv2.addWeighted(overlay, 0.75, panel, 0.25, 0, panel)

    keyframes = list(keyframes)
    hazards = list(hazards)
    loop_links = list(loop_links)
    pts = list(trajectory) + keyframes + [(hx, hy) for hx, hy, _l, _w in hazards] + [pose[:2]]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 6.0) * 1.15
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    inner = size - 36
    scale = inner / span

    def to_px(x: float, y: float) -> tuple[int, int]:
        return (
            int(size / 2 + (x - cx) * scale),
            int(size / 2 + 10 - (y - cy) * scale),
        )

    # 1 m scale bar
    bar = max(int(scale), 4)
    cv2.line(panel, (8, size - 10), (8 + bar, size - 10), (200, 200, 200), 2)
    cv2.putText(panel, "1 m", (12 + bar, size - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    for a, b in loop_links:
        cv2.line(panel, to_px(*a), to_px(*b), (255, 120, 255), 1, cv2.LINE_AA)
    if len(trajectory) >= 2:
        poly = np.array([to_px(x, y) for x, y in trajectory], np.int32)
        cv2.polylines(panel, [poly], False, (230, 200, 60), 1, cv2.LINE_AA)
    for kx, ky in keyframes:
        cv2.circle(panel, to_px(kx, ky), 2, (180, 180, 180), -1)
    for hx, hy, label, weight in hazards:
        shade = int(80 + 175 * max(0.0, min(weight, 1.0)))
        p = to_px(hx, hy)
        cv2.circle(panel, p, 4, (0, 0, shade), -1)
        cv2.putText(panel, label[:8], (p[0] + 5, p[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, shade), 1)

    px, py = to_px(pose[0], pose[1])
    h = math.radians(pose[2])
    tip = (int(px + 14 * math.sin(h)), int(py - 14 * math.cos(h)))
    color = STATE_COLORS.get(state, (255, 255, 255))
    cv2.arrowedLine(panel, (px, py), tip, color, 2, cv2.LINE_AA, tipLength=0.4)
    cv2.circle(panel, (px, py), 4, color, -1)

    cv2.putText(panel, f"SLAM {state}", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    if info:
        cv2.putText(panel, info, (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (210, 210, 210), 1, cv2.LINE_AA)
