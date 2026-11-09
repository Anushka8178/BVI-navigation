import threading
import time

import pytest

from navigation.config import load_config
from navigation.range_sensor import RangeSensor, parse_line


class FakeClock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class FakeSerial:
    """Serves the given lines, then behaves like an idle port (timeouts)."""

    def __init__(self, lines: list[bytes]) -> None:
        self.lines = list(lines)
        self.closed = False
        self.drained = threading.Event()

    def readline(self) -> bytes:
        if self.lines:
            return self.lines.pop(0)
        self.drained.set()
        time.sleep(0.01)
        return b""

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    "line, expected",
    [
        (b"D,123.4,56789\r\n", (123.4, 56789)),
        (b"D,-1,1000\n", (-1.0, 1000)),
        ("D,50,7", (50.0, 7)),
        (b"# esp32_range ready\r\n", None),
        (b"", None),
        (b"4,56789\r\n", None),          # partial line after opening mid-line
        (b"D,abc,1\n", None),
        (b"D,1.0\n", None),
        (b"V,100\n", None),
        (b"D,nan,1\n", None),
        (b"\xff\xfeD,1,2\n", (1.0, 2)),   # non-ASCII boot noise is dropped
        (b"D,12.3,567D,45.6,890\n", None),  # two readings glued together
        (b"D,123.4,\n", None),            # truncated: no millis
        (b"D,,5\n", None),                # empty distance
        (b"D,-1,\n", None),               # truncated no-echo line
        (b"D,1,2,3\n", None),             # extra field
        (b"D,40.0,-3\n", None),           # negative millis
        (b"\r\n", None),
    ],
)
def test_parse_line(line, expected):
    assert parse_line(line) == expected


def make_sensor(clock, factory=None):
    return RangeSensor(
        "FAKE", stale_after_s=0.3, reconnect_delay_s=0.01,
        serial_factory=factory, clock=clock,
    )


def test_latest_m_converts_and_goes_stale():
    clock = FakeClock()
    sensor = make_sensor(clock)
    assert sensor.latest_m() is None  # no data yet

    assert sensor.feed_line(b"D,150.0,10\n")
    assert sensor.latest_m() == pytest.approx(1.5)

    clock.t += 0.29
    assert sensor.latest_m() == pytest.approx(1.5)
    clock.t += 0.02  # 310 ms old
    assert sensor.latest_m() is None

    sensor.feed_line(b"D,80,20\n")
    assert sensor.latest_m() == pytest.approx(0.8)


def test_no_echo_returns_none_and_garbage_keeps_last_reading():
    clock = FakeClock()
    sensor = make_sensor(clock)
    sensor.feed_line(b"D,200,1\n")
    assert not sensor.feed_line(b"garbage\n")
    assert sensor.latest_m() == pytest.approx(2.0)

    sensor.feed_line(b"D,-1,2\n")
    assert sensor.latest_m() is None
    assert sensor.latest().esp_millis == 2


def test_background_thread_reads_fake_serial_stream():
    fake = FakeSerial([b"# boot\r\n", b"D,300,1\r\n", b"xx\r\n", b"D,120.5,67\r\n"])
    sensor = RangeSensor(
        "FAKE", serial_factory=lambda port, baud, timeout: fake,
        reconnect_delay_s=0.01,
    )
    sensor.start()
    try:
        assert fake.drained.wait(2.0)
        assert sensor.lines_parsed == 2
        assert sensor.latest().esp_millis == 67
        assert sensor.latest_m() == pytest.approx(1.205)
    finally:
        sensor.close()
    assert fake.closed


def test_missing_port_warns_once_and_never_raises(capsys):
    attempts = []

    def failing_factory(port, baud, timeout):
        attempts.append(port)
        raise OSError("could not open port 'COM99'")

    sensor = RangeSensor("COM99", serial_factory=failing_factory, reconnect_delay_s=0.01)
    sensor.start()
    deadline = time.monotonic() + 2.0
    while len(attempts) < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    sensor.close()

    assert len(attempts) >= 3  # keeps retrying in the background
    assert sensor.latest_m() is None
    out = capsys.readouterr().out
    assert out.count("WARNING") == 1
    assert "COM99" in out and "--no-esp32" in out


class UnpluggedSerial(FakeSerial):
    """Serves its lines, then fails like a pulled USB cable."""

    def readline(self) -> bytes:
        if self.lines:
            return self.lines.pop(0)
        self.drained.set()
        raise OSError("ClearCommError failed (device disconnected)")


def test_unplug_warns_once_then_reconnects_and_reads_again(capsys):
    # Port 1: two readings, then the cable is pulled. Opening fails twice
    # (still unplugged), then port 2 (plugged back in) serves new readings.
    first = UnpluggedSerial([b"D,100,1\n", b"D,90,2\n"])
    second = FakeSerial([b"# esp32_range ready\r\n", b"D,55.5,10\n"])
    opens = []

    def factory(port, baud, timeout):
        opens.append(port)
        if len(opens) == 1:
            return first
        if len(opens) <= 3:
            raise OSError("could not open port 'COM5'")
        return second

    sensor = RangeSensor("COM5", serial_factory=factory, reconnect_delay_s=0.01)
    sensor.start()
    try:
        assert second.drained.wait(3.0)
        assert sensor.connected
        assert sensor.latest().esp_millis == 10
        assert sensor.latest_m() == pytest.approx(0.555)
        assert sensor.lines_parsed == 3
    finally:
        sensor.close()

    assert first.closed and second.closed
    out = capsys.readouterr().out
    # One warning for the unplug, none for the failed reopen attempts,
    # and one line when it is back.
    assert out.count("WARNING") == 1
    assert "reconnecting" in out
    assert out.count("ESP32 range sensor connected on COM5") == 1


def test_stale_or_no_echo_reading_leaves_fusion_unchanged():
    from navigation.range_fusion import RangeFusionConfig, fuse_range
    from navigation.models import Detection

    clock = FakeClock()
    sensor = make_sensor(clock)
    cfg = RangeFusionConfig()
    detections = [Detection("pole-1", "pole", 0.0, 3.0, 0.8)]

    sensor.feed_line(b"D,50,1\n")  # 0.5 m: would add an unknown obstacle
    assert fuse_range(detections, sensor.latest_m(), cfg).added_unknown

    clock.t += 0.31  # stale
    result = fuse_range(detections, sensor.latest_m(), cfg)
    assert result.detections == detections and not result.added_unknown

    sensor.feed_line(b"D,-1,2\n")  # no echo
    result = fuse_range(detections, sensor.latest_m(), cfg)
    assert result.detections == detections and result.corrected_id is None


def test_from_config_uses_esp32_section():
    sensor = RangeSensor.from_config(load_config()["esp32"])
    assert sensor.serial_port == "COM5"
    assert sensor.baud == 115200
    assert sensor.stale_after_s == 0.3
