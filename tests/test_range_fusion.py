"""Ultrasonic + camera fusion with fake readings (no ESP32 needed)."""

import math

import pytest

from navigation.config import load_config
from navigation.hazard import HazardSeverityClassifier
from navigation.range_fusion import (
    UNKNOWN_OBSTACLE,
    RangeFusionConfig,
    fuse_range,
)
from navigation.models import Detection

CFG = RangeFusionConfig()


def det(oid: str, distance: float, azimuth_deg: float, label: str = "pole") -> Detection:
    a = math.radians(azimuth_deg)
    return Detection(oid, label, relative_x=distance * math.sin(a),
                     relative_y=distance * math.cos(a), confidence=0.8)


def test_no_reading_changes_nothing():
    detections = [det("pole-1", 3.0, 0.0)]
    result = fuse_range(detections, None, CFG)
    assert result.detections == detections
    assert result.corrected_id is None and not result.added_unknown


def test_nearest_detection_in_cone_gets_measured_distance():
    near = det("pole-1", 3.0, 4.0)
    far = det("car-2", 3.5, -2.0, "car")
    side = det("person-3", 1.0, 30.0, "person")  # outside the 7.5 deg cone
    result = fuse_range([far, near, side], 2.4, CFG)

    assert result.corrected_id == "pole-1"
    fused = {d.object_id: d for d in result.detections}
    assert fused["pole-1"].distance == pytest.approx(2.4)
    assert fused["pole-1"].azimuth_deg == pytest.approx(4.0)  # bearing kept
    assert fused["car-2"] == far and fused["person-3"] == side
    assert not result.added_unknown


@pytest.mark.parametrize("reading", [0.1, 4.5])
def test_reading_outside_trusted_range_does_not_replace(reading):
    detections = [det("pole-1", reading * 1.2, 0.0)]
    result = fuse_range(detections, reading, CFG)
    assert result.corrected_id is None
    assert result.detections[0] == detections[0]


def test_close_reading_without_detection_adds_unknown_obstacle():
    result = fuse_range([det("person-3", 2.0, 25.0, "person")], 0.9, CFG)
    assert result.added_unknown
    unknown = result.detections[-1]
    assert unknown.label == UNKNOWN_OBSTACLE
    assert (unknown.relative_x, unknown.relative_y, unknown.confidence) == (0.0, 0.9, 1.0)
    # The hazard classifier treats it as an urgent obstacle straight ahead.
    severity = HazardSeverityClassifier(**load_config()["hazard"])
    assert severity.classify(unknown).urgent


def test_far_reading_without_detection_adds_nothing():
    result = fuse_range([], 2.0, CFG)
    assert result.detections == [] and not result.added_unknown


def test_disagreeing_reading_is_another_obstacle_not_the_detection():
    # Camera: pole 6 m ahead. Ultrasonic: 1.0 m (glass door in front of it).
    pole = det("pole-1", 6.0, 0.0)
    result = fuse_range([pole], 1.0, CFG)
    assert result.corrected_id is None
    assert result.detections[0] == pole  # not dragged to 1 m
    assert result.added_unknown and result.detections[1].relative_y == 1.0


def test_agreeing_close_reading_corrects_instead_of_adding():
    result = fuse_range([det("reflective_cone-4", 1.6, 2.0, "reflective_cone")], 1.0, CFG)
    assert result.corrected_id == "reflective_cone-4"
    assert not result.added_unknown
    assert result.detections[0].distance == pytest.approx(1.0)


def test_reading_much_farther_than_camera_is_ignored():
    # Camera: person 1.0 m ahead. Ultrasonic: 3.0 m (beam passes beside them).
    # Ratio 1/3 is outside agree_ratio, and 3.0 m is not an unknown obstacle.
    person = det("person-2", 1.0, 0.0, "person")
    result = fuse_range([person], 3.0, CFG)
    assert result.corrected_id is None and not result.added_unknown
    assert result.detections == [person]


def test_agree_ratio_boundary_still_corrects():
    result = fuse_range([det("pole-1", 2.0, 0.0)], 1.0, CFG)  # ratio exactly 2.0
    assert result.corrected_id == "pole-1"
    assert result.detections[0].distance == pytest.approx(1.0)
    result = fuse_range([det("pole-1", 2.01, 0.0)], 1.0, CFG)  # just outside
    assert result.corrected_id is None and result.added_unknown


def test_unknown_obstacle_threshold_is_strict():
    assert not fuse_range([], CFG.unknown_obstacle_m, CFG).added_unknown
    assert fuse_range([], CFG.unknown_obstacle_m - 0.01, CFG).added_unknown


def test_very_close_reading_alerts_even_below_min_m():
    # 0.1 m is below min_m, so it is not forced onto the cone detection,
    # but something is right in front of the user: always alert.
    pole = det("pole-1", 0.12, 0.0)
    result = fuse_range([pole], 0.1, CFG)
    assert result.corrected_id is None
    assert result.detections[0] == pole
    assert result.added_unknown and result.detections[-1].relative_y == 0.1


def test_describe_fusion_status_line():
    from navigation.range_fusion import describe_fusion

    assert describe_fusion(None, False) == "Range: off (--no-esp32)"
    assert describe_fusion(fuse_range([], None, CFG), False) == "Range: ESP32 not connected"
    assert describe_fusion(fuse_range([], None, CFG), True) == "Range: no reading"
    corrected = fuse_range([det("person-3", 1.4, 0.0, "person")], 0.98, CFG)
    assert describe_fusion(corrected, True) == "Range: 0.98 m -> person-3 (ultrasonic distance)"
    unknown = fuse_range([], 1.0, CFG)
    assert describe_fusion(unknown, True) == "Range: 1.00 m UNKNOWN OBSTACLE ahead"
    assert describe_fusion(fuse_range([], 3.0, CFG), True) == "Range: 3.00 m (no match)"


def test_config_section():
    section = load_config()["range_fusion"]
    assert RangeFusionConfig.from_config(section) == CFG
    with pytest.raises(ValueError, match="unknown"):
        RangeFusionConfig.from_config({"cone": 7.5})
