"""Load config.yaml once and apply command-line overrides.

config.yaml is the only place tunable numbers are defined. The main
script loads it with load_config(), applies --set overrides and
dedicated flags (e.g. --confidence) with apply_overrides(), and then
passes each section to the matching constructor.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# Sections the main script relies on. A missing one is a clear error
# at start-up instead of a KeyError in the middle of a run.
REQUIRED_SECTIONS = (
    "camera",
    "network",
    "esp32",
    "range_fusion",
    "pose",
    "perception",
    "hazard",
    "planner",
    "trajectory",
    "memory",
    "memory_reasoning",
    "attention",
    "audio",
)


class ConfigError(ValueError):
    pass


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read a YAML config file into a plain nested dict."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path.resolve()}")

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"Could not parse {config_path}: {error}") from error

    if not isinstance(data, dict):
        raise ConfigError(f"{config_path} must contain a mapping of sections.")

    missing = [name for name in REQUIRED_SECTIONS if not isinstance(data.get(name), dict)]
    if missing:
        raise ConfigError(f"{config_path} is missing section(s): {', '.join(missing)}")

    return data


def parse_set_option(text: str) -> tuple[str, Any]:
    """Parse one --set option "section.key=value".

    The value is read as YAML, so 0.3 -> float, 3 -> int, true -> bool,
    null -> None and [a, b] -> list. Anything else stays a string.
    """
    key, separator, raw_value = text.partition("=")
    key = key.strip()

    if not separator or not key:
        raise ConfigError(f"--set expects section.key=value, got '{text}'.")

    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError:
        value = raw_value

    return key, value


def apply_overrides(
    config: Mapping[str, Any],
    overrides: Mapping[str, Any] | Iterable[tuple[str, Any]],
) -> dict[str, Any]:
    """Return a copy of config with dotted-key overrides applied.

    Overrides whose value is None are skipped, so argparse flags that
    were not given (default None) leave the config value alone.
    Only existing keys can be overridden; a typo is an error, not a
    silently ignored setting.
    """
    result = copy.deepcopy(dict(config))
    items = overrides.items() if isinstance(overrides, Mapping) else overrides

    for dotted_key, value in items:
        if value is None:
            continue

        parts = dotted_key.split(".")
        node: Any = result

        for depth, part in enumerate(parts):
            if not isinstance(node, dict) or part not in node:
                known = ", ".join(sorted(node)) if isinstance(node, dict) else ""
                where = ".".join(parts[:depth]) or "top level"
                raise ConfigError(
                    f"Unknown config key '{dotted_key}' ('{part}' not in {where}; "
                    f"known: {known})."
                )
            if depth == len(parts) - 1:
                node[part] = value
            else:
                node = node[part]

    return result
