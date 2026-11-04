import inspect

import pytest

from navigation.config import (
    ConfigError,
    apply_overrides,
    load_config,
    parse_set_option,
)
from navigation.hazard import HazardSeverityClassifier
from navigation.planner import LocalPathPlanner
from navigation.trajectory import HazardTrajectoryTracker


def test_repo_config_loads_with_planner_branch_safety_margin():
    config = load_config()
    assert config["planner"]["safety_margin_m"] == 0.25
    assert config["camera"]["source"] == "http://10.47.242.217:8080/video"
    assert config["network"]["esp32_distance_port"] == 5005
    assert config["network"]["esp32_vibration_port"] == 5006
    assert config["memory"]["match_radius_m"] == 1.0
    assert config["network"]["phone_ip"] == "10.47.242.217"
    assert config["network"]["sensor_port"] == 8081
    assert config["pose"]["stride_m"] == 0.7
    assert config["pose"]["stale_after_s"] == 2.0
    assert config["perception"]["weights"] == "best.pt"


@pytest.mark.parametrize(
    "section, cls",
    [
        ("planner", LocalPathPlanner),
        ("hazard", HazardSeverityClassifier),
        ("trajectory", HazardTrajectoryTracker),
    ],
)
def test_sections_passed_as_kwargs_match_constructors(section, cls):
    # The main script does cls(**config[section]); every key must be a
    # real constructor argument, and every argument must be configured.
    config = load_config()
    params = set(inspect.signature(cls.__init__).parameters) - {"self"}
    assert set(config[section]) == params
    cls(**config[section])


def test_parse_set_option_types_values():
    assert parse_set_option("planner.safety_margin_m=0.3") == ("planner.safety_margin_m", 0.3)
    assert parse_set_option("planner.confirm_frames=5") == ("planner.confirm_frames", 5)
    assert parse_set_option("audio.device=null") == ("audio.device", None)
    assert parse_set_option("network.phone_ip=192.168.43.1") == (
        "network.phone_ip",
        "192.168.43.1",
    )
    assert parse_set_option("camera.source=http://1.2.3.4:8080/video") == (
        "camera.source",
        "http://1.2.3.4:8080/video",
    )
    with pytest.raises(ConfigError):
        parse_set_option("no-equals-sign")


def test_apply_overrides_changes_copy_and_skips_none():
    config = {"planner": {"safety_margin_m": 0.25, "horizon_m": 5.0}}
    result = apply_overrides(
        config,
        {"planner.safety_margin_m": 0.4, "planner.horizon_m": None},
    )
    assert result["planner"] == {"safety_margin_m": 0.4, "horizon_m": 5.0}
    assert config["planner"]["safety_margin_m"] == 0.25  # original untouched


def test_later_overrides_win():
    # Same order as the main script: --set first, then dedicated flags.
    config = {"perception": {"confidence": 0.35}}
    config = apply_overrides(config, [("perception.confidence", 0.5)])
    config = apply_overrides(config, {"perception.confidence": 0.6})
    assert config["perception"]["confidence"] == 0.6


def test_unknown_key_is_an_error():
    with pytest.raises(ConfigError, match="safety_margin"):
        apply_overrides({"planner": {"safety_margin_m": 0.25}}, {"planner.safety_margin": 0.3})
    with pytest.raises(ConfigError):
        apply_overrides({"planner": {}}, {"plannr.horizon_m": 1.0})


def test_missing_file_and_missing_section(tmp_path):
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(tmp_path / "nope.yaml")

    partial = tmp_path / "partial.yaml"
    partial.write_text("planner:\n  horizon_m: 5.0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="missing section"):
        load_config(partial)
