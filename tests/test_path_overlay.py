"""Lane overlay: projection geometry and drawing (no camera needed)."""

import numpy as np
import pytest

from navigation.memory import HazardEntry
from navigation.minimap import draw_minimap
from navigation.models import Pose
from navigation.path_overlay import (
    draw_lanes,
    ground_to_pixel,
    lane_polygon,
    lanes_for_minimap,
    nearest_visible_m,
)
from navigation.planner import LocalPathPlanner

W, H = 640, 360


def test_ground_point_straight_ahead_is_centred_and_below_horizon():
    u, v = ground_to_pixel(0.0, 3.0, W, H, 1.3, 70.0)
    assert u == pytest.approx(W / 2)
    assert v > H / 2
    # farther points are closer to the horizon row
    assert ground_to_pixel(0.0, 6.0, W, H, 1.3, 70.0)[1] < v


def test_right_of_path_is_right_in_the_image():
    assert ground_to_pixel(1.0, 3.0, W, H, 1.3, 70.0)[0] > W / 2
    assert ground_to_pixel(-1.0, 3.0, W, H, 1.3, 70.0)[0] < W / 2


def test_bottom_row_matches_nearest_visible_distance():
    d = nearest_visible_m(H, 1.3)
    assert ground_to_pixel(0.0, d, W, H, 1.3, 70.0)[1] == pytest.approx(H)


def test_centre_lane_polygon_is_symmetric():
    poly = lane_polygon(0.0, 0.65, 2.5, 5.0, W, H, 1.3, 70.0)
    assert np.mean(poly[:, 0]) == pytest.approx(W / 2, abs=1.0)


def blocked_decision():
    planner = LocalPathPlanner(confirm_frames=1)
    pole = HazardEntry(
        object_id="pole-1", label="pole", world_x=0.0, world_y=3.0, confidence=0.9,
        timestamp=0.0, decay_weight=1.0, velocity_x=0.0, velocity_y=0.0, hazard_id="a",
    )
    return planner.plan(Pose(0.0, 0.0, 0.0, 0.0), [pole])


def test_blocked_centre_lane_is_red_and_chosen_lane_green():
    decision = blocked_decision()
    assert not decision.current_path_safe and decision.selected_offset != 0.0
    image = np.zeros((H, W, 3), np.uint8)
    draw_lanes(image, decision, 0.65, 5.0, 1.3, 70.0)
    # sample inside the centre lane, away from the arrow
    u, v = ground_to_pixel(0.25, 4.0, W, H, 1.3, 70.0)
    b, g, r = image[int(v), int(u)]
    assert r > g and r > b
    u, v = ground_to_pixel(decision.selected_offset, 4.5, W, H, 1.3, 70.0)
    if 0 <= u < W:
        b, g, r = image[int(v), int(u)]
        assert g > r


def test_lanes_for_minimap_marks_one_selected_lane():
    lanes = lanes_for_minimap(blocked_decision())
    assert len(lanes) == 5
    assert sum(1 for _o, _s, sel in lanes if sel) == 1
    image = np.zeros((H, W, 3), np.uint8)
    draw_minimap(image, [(0, 0)], [], [], [], (0.0, 0.0, 0.0), "TRACKING", lanes=lanes)
    assert image.any()
