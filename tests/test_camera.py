"""Frame-source tests with fake captures: no camera, phone or network needed."""

import threading

import cv2
import pytest

from navigation.camera import (
    FpsMeter,
    LatestFrameReader,
    VideoFileSource,
    is_url,
    open_source,
)


class FakeStream:
    """A capture whose read() blocks until the test releases the next frame."""

    def __init__(self):
        self.frames = []
        self.released = False
        self._ready = threading.Semaphore(0)
        self.closed_stream = False

    def push(self, frame):
        self.frames.append(frame)
        self._ready.release()

    def end(self):
        self.closed_stream = True
        self._ready.release()

    def isOpened(self):
        return True

    def get(self, prop):
        return 0.0

    def read(self):
        if not self._ready.acquire(timeout=2.0) or self.closed_stream:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


class FakeFile:
    def __init__(self, frames, fps=10.0):
        self.frames = list(frames)
        self.fps = fps

    def isOpened(self):
        return True

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else 0.0

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        pass


def wait_until(predicate, timeout=2.0):
    event = threading.Event()
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return True
        event.wait(0.01)
    return predicate()


def test_is_url():
    assert is_url("http://10.47.242.217:8080/video")
    assert is_url("rtsp://cam/stream")
    assert not is_url("videos/street6.mp4")
    assert not is_url(r"videos\street6.mp4")


def test_latest_frame_reader_keeps_only_newest_frame():
    stream = FakeStream()
    clock_values = iter([100.0, 100.1, 100.2, 100.3])
    reader = LatestFrameReader(
        "http://fake/video",
        capture_factory=lambda url: stream,
        read_timeout_s=1.0,
        clock=lambda: next(clock_values),
    )
    try:
        # Three frames arrive while the "pipeline" is busy.
        for name in ("a", "b", "c"):
            stream.push(name)
        assert wait_until(lambda: reader.frames_received == 3)

        frame, timestamp = reader.read()
        assert frame == "c"            # newest, not oldest
        assert timestamp == 100.2      # wall-clock time it arrived
        assert reader.frames_dropped == 2

        # Next frame arrives; read returns it without lag.
        stream.push("d")
        frame, timestamp = reader.read()
        assert (frame, timestamp) == ("d", 100.3)
    finally:
        stream.end()
        reader.close()


def test_latest_frame_reader_never_returns_same_frame_twice():
    stream = FakeStream()
    reader = LatestFrameReader(
        "http://fake/video",
        capture_factory=lambda url: stream,
        read_timeout_s=0.2,
        reconnect_delay_s=10.0,
    )
    try:
        stream.push("only")
        assert reader.read()[0] == "only"
        assert reader.read() == (None, None)  # timed out, no new frame
    finally:
        stream.end()
        reader.close()


def test_latest_frame_reader_unreachable_stream_has_clear_error():
    class Closed(FakeStream):
        def isOpened(self):
            return False

    with pytest.raises(RuntimeError, match="IP Webcam"):
        LatestFrameReader("http://fake/video", capture_factory=lambda url: Closed())


def test_video_file_source_uses_video_time(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"")
    source = VideoFileSource(path, capture_factory=lambda p: FakeFile(["f0", "f1", "f2"], fps=10.0))
    assert source.fps == 10.0
    assert source.read() == ("f0", 0.0)
    assert source.read() == ("f1", 0.1)
    assert source.read() == ("f2", 0.2)
    assert source.read() == (None, None)


def test_open_source_missing_file_has_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match="does not exist"):
        open_source(tmp_path / "missing.mp4")


def test_fps_meter_reports_once_per_interval():
    now = [0.0]
    meter = FpsMeter(interval_s=5.0, clock=lambda: now[0])
    results = []
    for frame in range(1, 51):  # 10 frames per second for 5 s
        now[0] = frame / 10
        results.append(meter.tick())
    reported = [value for value in results if value is not None]
    assert len(reported) == 1
    assert reported[0] == pytest.approx(10.0)
    assert results[-1] == reported[0]
