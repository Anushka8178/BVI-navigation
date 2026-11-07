"""Draw the local path planner's walking lanes on the camera frame.

The planner (navigation/planner.py) checks five straight lanes in the
user's walking direction (lateral offsets -1.5, -0.75, 0, +0.75, +1.5 m,
`path_width_m` wide, out to `horizon_m`). This projects those lanes onto
the ground in the image with the SAME camera model perception uses to
place detections (navigation/yolo_perception.py): a level camera at
`camera_height_m`, horizon at the image centre row, vertical focal length
0.95 x image height, and a linear horizontal field of view.

Colours: the chosen lane is a green carpet with an arrow, other free lanes
have a thin green outline, blocked lanes are tinted red. The user's
current lane (offset 0) is outlined in white when it is not the chosen one.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import cv2
import numpy as np

GREEN = (80, 220, 80)
RED = (60, 60, 230)
WHITE = (255, 255, 255)


def ground_to_pixel(
    right_m: float,
    forward_m: float,
    image_w: int,
    image_h: int,
    camera_height_m: float,
    hfov_deg: float,
) -> tuple[float, float]:
    """Image (u, v) of a ground point `forward_m` ahead, `right_m` right."""
    focal_v = 0.95 * image_h
    v = image_h / 2.0 + camera_height_m * focal_v / forward_m
    angle = math.degrees(math.atan2(right_m, forward_m))
    u = image_w * (0.5 + angle / hfov_deg)
    return u, v


def nearest_visible_m(image_h: int, camera_height_m: float) -> float:
    """Distance of the ground point at the bottom row of the image."""
    return camera_height_m * 0.95 * image_h / (image_h / 2.0)


def lane_polygon(
    offset_m: float,
    width_m: float,
    near_m: float,
    far_m: float,
    image_w: int,
    image_h: int,
    camera_height_m: float,
    hfov_deg: float,
    steps: int = 12,
) -> np.ndarray:
    """Image polygon of a straight lane on the ground."""
    left, right = offset_m - width_m / 2.0, offset_m + width_m / 2.0
    dists = np.linspace(near_m, far_m, steps)
    edge_l = [ground_to_pixel(left, d, image_w, image_h, camera_height_m, hfov_deg) for d in dists]
    edge_r = [ground_to_pixel(right, d, image_w, image_h, camera_height_m, hfov_deg) for d in dists]
    return np.array(edge_l + edge_r[::-1], dtype=np.int32)


def draw_lanes(
    image: np.ndarray,
    decision: Any,
    path_width_m: float,
    horizon_m: float,
    camera_height_m: float,
    hfov_deg: float,
    draw_unsafe_only_when_blocked: bool = False,
) -> None:
    """Draw the planner's lanes for this frame onto `image` in place.

    `decision` is a planner PathDecision (candidates, selected_offset,
    current_path_safe).
    """
    h, w = image.shape[:2]
    near = max(nearest_visible_m(h, camera_height_m), 0.5)
    far = max(horizon_m, near + 0.5)
    candidates: Sequence[Any] = getattr(decision, "candidates", []) or []
    selected = float(getattr(decision, "selected_offset", 0.0))

    overlay = image.copy()
    outlines = []
    for cand in candidates:
        offset = float(cand.lateral_offset)
        poly = lane_polygon(offset, path_width_m, near, far, w, h, camera_height_m, hfov_deg)
        is_selected = abs(offset - selected) < 1e-6
        if is_selected:
            cv2.fillPoly(overlay, [poly], GREEN)
        elif not cand.safe:
            if not draw_unsafe_only_when_blocked or not decision.current_path_safe:
                cv2.fillPoly(overlay, [poly], RED)
        outlines.append((poly, offset, cand.safe, is_selected))
    cv2.addWeighted(overlay, 0.35, image, 0.65, 0, image)

    for poly, offset, safe, is_selected in outlines:
        if is_selected:
            cv2.polylines(image, [poly], True, GREEN, 2, cv2.LINE_AA)
        elif abs(offset) < 1e-6:
            cv2.polylines(image, [poly], True, WHITE, 1, cv2.LINE_AA)
        else:
            cv2.polylines(image, [poly], True, GREEN if safe else RED, 1, cv2.LINE_AA)

    # Arrow from the user (bottom centre) to the far end of the chosen lane.
    end = ground_to_pixel(selected, far, w, h, camera_height_m, hfov_deg)
    start_i = (w // 2, h - 6)
    end_i = (int(end[0]), int(end[1]))
    cv2.arrowedLine(image, start_i, end_i, WHITE, 3, cv2.LINE_AA, tipLength=0.25)
    name = next(
        (c.name for c in candidates if abs(float(c.lateral_offset) - selected) < 1e-6),
        "ahead",
    )
    label = name.upper() if decision.current_path_safe else f"GO {name.upper()}"
    cv2.putText(image, label, (end_i[0] - 30, end_i[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)


def lanes_for_minimap(decision: Any) -> list[tuple[float, bool, bool]]:
    """(offset_m, safe, selected) for each planner lane, for draw_minimap."""
    selected = float(getattr(decision, "selected_offset", 0.0))
    return [
        (float(c.lateral_offset), bool(c.safe), abs(float(c.lateral_offset) - selected) < 1e-6)
        for c in getattr(decision, "candidates", []) or []
    ]
