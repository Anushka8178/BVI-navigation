"""Ground-level classes: flat-ground distance and severity bonus.

Pure logic: no camera, GPU or YOLO model needed.
"""

import pytest

from navigation.config import load_config
from navigation.hazard import HazardSeverityClassifier
from navigation.memory import DEFAULT_DYNAMIC_LABELS
from navigation.models import Detection
from navigation.yolo_perception import YOLOPerception, ground_plane_distance

H = 720
FOCAL = H * 0.95


def test_lower_box_bottom_is_closer():
    far = ground_plane_distance(500, H, 1.3, FOCAL)
    near = ground_plane_distance(650, H, 1.3, FOCAL)
    assert near < far


def test_ground_distance_matches_flat_ground_formula():
    # 300 px below the centre row: 1.3 * 684 / 300 = 2.964 m
    assert ground_plane_distance(660, H, 1.3, FOCAL) == pytest.approx(1.3 * FOCAL / 300)


def test_ground_distance_is_clamped():
    assert ground_plane_distance(H, H, 1.3, FOCAL, min_m=0.5, max_m=8.0) >= 0.5
    assert ground_plane_distance(H / 2 + 1, H, 1.3, FOCAL) == 8.0
    # At or above the horizon -> far.
    assert ground_plane_distance(100, H, 1.3, FOCAL) == 8.0


def test_higher_camera_sees_same_row_further_away():
    assert ground_plane_distance(600, H, 1.5, FOCAL) > ground_plane_distance(600, H, 1.0, FOCAL)


def test_new_class_heights_present():
    heights = YOLOPerception.ASSUMED_HEIGHT_M
    assert heights["tree"] == 3.0
    assert heights["autorickshaw"] == 1.7
    assert heights["animal"] == 1.0
    assert heights["barrier"] == 1.0


def _det(label, ground_level=False, x=0.0, y=3.0):
    return Detection(
        object_id=f"{label}-1",
        label=label,
        relative_x=x,
        relative_y=y,
        confidence=0.9,
        ground_level=ground_level,
    )


def test_ground_hazard_gets_score_bonus():
    clf = HazardSeverityClassifier(ground_hazard_bonus=0.25)
    pothole = clf.classify(_det("pothole", ground_level=True))
    cone = clf.classify(_det("reflective_cone"))
    assert pothole.score == pytest.approx(cone.score + 0.25)


def test_ground_bonus_makes_ahead_pothole_urgent_sooner():
    # 3 m straight ahead: outside very_close_m (2.5), so only the score
    # trigger can make it urgent. score = 0.5/4 + 0.25 = 0.375 without bonus.
    clf = HazardSeverityClassifier(urgency_threshold=0.6, ground_hazard_bonus=0.25)
    assert clf.classify(_det("pothole", ground_level=True)).urgent
    assert not clf.classify(_det("reflective_cone")).urgent


def test_zero_bonus_disables_it():
    clf = HazardSeverityClassifier(ground_hazard_bonus=0.0)
    a = clf.classify(_det("curb", ground_level=True))
    b = clf.classify(_det("ashcan"))
    assert a.score == pytest.approx(b.score)


def test_config_has_ground_and_moving_settings():
    config = load_config()
    assert config["perception"]["camera_height_m"] == 1.3
    assert config["hazard"]["ground_hazard_bonus"] == 0.25
    assert set(config["hazard"]["ground_labels"]) == {"pothole", "stairs", "curb", "drain"}
    assert {"autorickshaw", "animal"} <= set(config["memory"]["dynamic_labels"])
    assert {"autorickshaw", "animal"} <= DEFAULT_DYNAMIC_LABELS
