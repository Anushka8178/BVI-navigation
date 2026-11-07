"""Manual check: list COM ports and print live ESP32 range readings.

Usage (PowerShell, from the repo root):
    python scripts\\manual\\check_esp32.py              # port from config.yaml
    python scripts\\manual\\check_esp32.py --port COM7
    python scripts\\manual\\check_esp32.py --list       # only list ports

Ctrl+C to stop. Close the Arduino Serial Monitor first: only one program
can hold the COM port.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from navigation.config import load_config  # noqa: E402
from navigation.range_sensor import RangeSensor  # noqa: E402


def list_ports() -> None:
    try:
        from serial.tools import list_ports as serial_list_ports
    except ImportError:
        raise SystemExit("pyserial is not installed: pip install -r requirements.txt")

    ports = list(serial_list_ports.comports())
    if not ports:
        print("No serial ports found. Is the ESP32 plugged in (data cable, driver)?")
        return
    print("Available serial ports:")
    for port in ports:
        print(f"  {port.device:8s} {port.description}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", default=None, help="Serial port (default: esp32.serial_port)")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--list", action="store_true", help="Only list ports")
    args = parser.parse_args()

    list_ports()
    if args.list:
        return

    esp32_cfg = dict(load_config(args.config)["esp32"])
    if args.port:
        esp32_cfg["serial_port"] = args.port

    sensor = RangeSensor.from_config(esp32_cfg)
    print(f"\nReading {sensor.serial_port} at {sensor.baud} baud (Ctrl+C to stop)...\n")
    sensor.start()

    last_count = 0
    last_rate_time = time.monotonic()
    rate_hz = 0.0
    try:
        while True:
            time.sleep(0.1)
            now = time.monotonic()
            if now - last_rate_time >= 1.0:
                rate_hz = (sensor.lines_parsed - last_count) / (now - last_rate_time)
                last_count = sensor.lines_parsed
                last_rate_time = now

            reading = sensor.latest()
            distance_m = sensor.latest_m()
            if reading is None:
                status = "waiting for data..." if sensor.connected else "port not open"
            elif distance_m is None:
                status = "no echo" if reading.distance_cm < 0 else "STALE"
            else:
                status = f"{distance_m:5.2f} m"
            esp_ms = "-" if reading is None else str(reading.esp_millis)
            print(f"\r{status:20s} esp_millis={esp_ms:>10s}  {rate_hz:4.1f} Hz   ", end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sensor.close()


if __name__ == "__main__":
    main()
