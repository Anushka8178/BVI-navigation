"""HC-SR04 range readings from the ESP32 over USB serial.

The ESP32 (firmware/esp32_range/esp32_range.ino) prints one line per
reading at 115200 baud:

    D,<distance_cm>,<esp_millis>    e.g. "D,123.4,56789"
    D,-1,<esp_millis>               no echo / out of range

RangeSensor reads the port in a background thread and keeps the latest
reading. latest_m() returns the distance in metres, or None when there
is no fresh, valid reading. A missing port never crashes the pipeline:
one clear warning is printed, the thread keeps retrying in the
background, and latest_m() returns None meanwhile.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol


class SerialLike(Protocol):
    def readline(self) -> bytes: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class RangeReading:
    distance_cm: float     # -1 = no echo
    esp_millis: int
    received_at: float     # laptop monotonic time (s)


def parse_line(line: bytes | str) -> tuple[float, int] | None:
    """Parse one "D,<cm>,<millis>" line into (cm, millis).

    Returns None for anything else (boot messages, "# ..." comments,
    partial lines after the port opens mid-line, garbage).
    """
    if isinstance(line, bytes):
        line = line.decode("ascii", errors="ignore")
    parts = line.strip().split(",")
    if len(parts) != 3 or parts[0] != "D":
        return None
    try:
        cm = float(parts[1])
        millis = int(parts[2])
    except ValueError:
        return None
    if cm != cm or millis < 0:  # NaN or negative time
        return None
    return cm, millis


def default_serial_factory(port: str, baud: int, timeout_s: float) -> SerialLike:
    import serial  # pyserial; imported here so tests don't need it

    return serial.Serial(port, baud, timeout=timeout_s)


class RangeSensor:
    def __init__(
        self,
        serial_port: str,
        baud: int = 115200,
        stale_after_s: float = 0.3,
        read_timeout_s: float = 0.2,
        reconnect_delay_s: float = 1.0,
        serial_factory: Callable[[str, int, float], SerialLike] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.serial_port = serial_port
        self.baud = int(baud)
        self.stale_after_s = float(stale_after_s)
        self.read_timeout_s = float(read_timeout_s)
        self.reconnect_delay_s = float(reconnect_delay_s)
        self._serial_factory = serial_factory or default_serial_factory
        self._clock = clock

        self._lock = threading.Lock()
        self._latest: RangeReading | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._warned = False
        self.connected = False
        self.lines_parsed = 0

    @classmethod
    def from_config(cls, esp32_cfg: Mapping[str, Any], **kwargs: Any) -> "RangeSensor":
        return cls(
            serial_port=str(esp32_cfg["serial_port"]),
            baud=int(esp32_cfg["baud"]),
            stale_after_s=float(esp32_cfg["stale_after_s"]),
            read_timeout_s=float(esp32_cfg["read_timeout_s"]),
            reconnect_delay_s=float(esp32_cfg["reconnect_delay_s"]),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Readings
    # ------------------------------------------------------------------
    def feed_line(self, line: bytes | str) -> bool:
        """Store one line if it is a valid reading. Returns True if stored."""
        parsed = parse_line(line)
        if parsed is None:
            return False
        cm, millis = parsed
        with self._lock:
            self._latest = RangeReading(cm, millis, self._clock())
            self.lines_parsed += 1
        return True

    def latest(self) -> RangeReading | None:
        with self._lock:
            return self._latest

    def latest_m(self) -> float | None:
        """Latest distance in metres, or None if stale, no echo or no data."""
        reading = self.latest()
        if reading is None or reading.distance_cm < 0:
            return None
        if self._clock() - reading.received_at > self.stale_after_s:
            return None
        return reading.distance_cm / 100.0

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="RangeSensor", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _warn_once(self, message: str) -> None:
        if not self._warned:
            self._warned = True
            print(f"WARNING: {message}")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                port = self._serial_factory(
                    self.serial_port, self.baud, self.read_timeout_s
                )
            except ImportError:
                self._warn_once(
                    "pyserial is not installed (pip install -r requirements.txt); "
                    "range sensor disabled."
                )
                return
            except Exception as error:  # SerialException, OSError, ...
                self._warn_once(
                    f"ESP32 range sensor not available on {self.serial_port} "
                    f"({error}). Check the USB cable and esp32.serial_port "
                    "(scripts/manual/check_esp32.py lists ports), or use "
                    "--no-esp32. Continuing without it; retrying in the background."
                )
                self._stop.wait(self.reconnect_delay_s)
                continue

            self.connected = True
            if self._warned:
                print(f"ESP32 range sensor connected on {self.serial_port}.")
            try:
                while not self._stop.is_set():
                    self.feed_line(port.readline())
            except Exception as error:
                print(
                    f"WARNING: ESP32 serial read failed on {self.serial_port} "
                    f"({error}); reconnecting."
                )
                self._warned = True
            finally:
                self.connected = False
                try:
                    port.close()
                except Exception:
                    pass
            self._stop.wait(self.reconnect_delay_s)
