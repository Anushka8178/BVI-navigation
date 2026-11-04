"""Live check of PhonePose: prints heading and x, y while you turn and walk.

Needs the phone running the SensorServer app on the same network
(network.phone_ip / network.sensor_port in config.yaml).

    python -m scripts.manual.check_phone_pose --seconds 20

Expected: heading starts at 0, goes positive when you turn right
(clockwise) and negative when you turn left; each step moves x, y by
pose.stride_m along the heading. "valid" turns False if the phone stops
sending rotation data for pose.stale_after_s.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from navigation.config import ConfigError, apply_overrides, load_config, parse_set_option  # noqa: E402
from navigation.sensing import PhonePose  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=20.0, help="How long to run")
    parser.add_argument("--rate", type=float, default=5.0, help="Lines per second")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="SECTION.KEY=VALUE")
    args = parser.parse_args()

    try:
        config = apply_overrides(
            load_config(args.config), [parse_set_option(t) for t in args.overrides]
        )
    except ConfigError as error:
        raise SystemExit(f"Config error: {error}") from error

    network, pose_cfg = config["network"], config["pose"]
    phone = PhonePose(
        phone_ip=str(network["phone_ip"]),
        sensor_port=int(network["sensor_port"]),
        stride_m=float(pose_cfg["stride_m"]),
        stale_after_s=float(pose_cfg["stale_after_s"]),
        reconnect_delay_s=float(pose_cfg["reconnect_delay_s"]),
    )
    try:
        phone.start()
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    try:
        if not phone.wait_for_heading(float(pose_cfg["startup_wait_s"])):
            print(
                f"No rotation data from ws://{phone.phone_ip}:{phone.sensor_port} "
                "yet - is SensorServer running? Waiting anyway..."
            )
        phone.calibrate()
        print("Calibrated: current heading = 0. Turn and walk now.")
        print(f"{'t (s)':>6} {'heading':>8} {'x (m)':>7} {'y (m)':>7} {'steps':>5}  valid")

        start = time.monotonic()
        while (elapsed := time.monotonic() - start) < args.seconds:
            pose = phone.get_pose(elapsed)
            print(
                f"{elapsed:6.1f} {pose.heading_deg:+8.1f} {pose.x:+7.2f} "
                f"{pose.y:+7.2f} {phone.step_count:5d}  {pose.valid}",
                flush=True,
            )
            time.sleep(1.0 / args.rate)
    except KeyboardInterrupt:
        pass
    finally:
        phone.close()


if __name__ == "__main__":
    main()
