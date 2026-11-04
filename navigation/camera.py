"""Frame sources for the navigation demo: a recorded video file or a live URL.

Both sources have the same interface:

    frame, timestamp = source.read()   # frame is None at end / timeout
    source.close()

VideoFileSource reads every frame in order; the timestamp is the video
time (frame_index / fps), so recorded runs are repeatable.

LatestFrameReader is for live MJPEG streams such as the IP Webcam app
(http://<PHONE_IP>:8080/video). A background thread reads the stream as
fast as it arrives and keeps only the newest frame, so a slow pipeline
skips stale frames instead of falling further and further behind. The
timestamp is wall-clock time (time.time()) taken when the frame arrived.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2

CaptureFactory = Callable[[str], Any]


def is_url(source: str | Path) -> bool:
    """True for stream URLs (http://, https://, rtsp://, ...)."""
    return "://" in str(source)


def open_url_capture(url: str, timeout_s: float = 5.0) -> cv2.VideoCapture:
    """Open a network stream with open/read timeouts where OpenCV supports them,
    so an unreachable phone fails in seconds instead of hanging."""
    timeout_ms = int(timeout_s * 1000)
    open_prop = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
    read_prop = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)

    if open_prop is not None and read_prop is not None:
        try:
            return cv2.VideoCapture(
                url,
                cv2.CAP_FFMPEG,
                [open_prop, timeout_ms, read_prop, timeout_ms],
            )
        except (cv2.error, TypeError):
            pass

    return cv2.VideoCapture(url)


class VideoFileSource:
    """Every frame of a recorded video, timestamped with video time."""

    is_live = False

    def __init__(
        self,
        path: str | Path,
        capture_factory: CaptureFactory = cv2.VideoCapture,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise RuntimeError(f"Input video does not exist: {self.path.resolve()}")

        self._capture = capture_factory(str(self.path))
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open input video: {self.path.resolve()}")

        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        self._index = 0

    def read(self) -> tuple[Any | None, float | None]:
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None, None
        timestamp = self._index / self.fps
        self._index += 1
        return frame, timestamp

    def close(self) -> None:
        self._capture.release()


class LatestFrameReader:
    """Read a live stream in a background thread, keeping only the newest frame.

    read() waits up to read_timeout_s for a frame newer than the last one
    it returned. Frames that arrive while the pipeline is busy are
    overwritten and counted in frames_dropped. If the stream drops, the
    thread reopens it every reconnect_delay_s until close() is called.
    """

    is_live = True

    def __init__(
        self,
        url: str,
        capture_factory: CaptureFactory | None = None,
        read_timeout_s: float = 5.0,
        reconnect_delay_s: float = 1.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.url = url
        self.read_timeout_s = float(read_timeout_s)
        self.reconnect_delay_s = float(reconnect_delay_s)
        self._clock = clock
        self._capture_factory = capture_factory or (
            lambda target: open_url_capture(target, self.read_timeout_s)
        )

        self._condition = threading.Condition()
        self._frame: Any | None = None
        self._timestamp: float | None = None
        self._sequence = 0          # frames received so far
        self._returned_sequence = 0  # sequence of the last frame read() returned
        self._stopped = False

        self.frames_received = 0
        self.frames_dropped = 0
        self.fps: float | None = None

        self._capture = self._capture_factory(url)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(
                f"Could not open camera stream {url}. Check that the IP Webcam "
                "app is running, the phone IP is right and the laptop is on the "
                "phone's hotspot. Or pass --source <video file>."
            )

        reported_fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = reported_fps if 0.0 < reported_fps < 240.0 else None

        self._thread = threading.Thread(
            target=self._run,
            name="camera-reader",
            daemon=True,
        )
        self._thread.start()

    # Background thread ------------------------------------------------

    def _run(self) -> None:
        while not self._stopped:
            capture = self._capture
            ok, frame = (False, None) if capture is None else capture.read()

            if ok and frame is not None:
                timestamp = self._clock()
                with self._condition:
                    self._frame = frame
                    self._timestamp = timestamp
                    self._sequence += 1
                    self.frames_received += 1
                    self._condition.notify_all()
                continue

            # Stream dropped: reopen after a short wait.
            if capture is not None:
                capture.release()
            self._capture = None
            if self._wait_stopped(self.reconnect_delay_s):
                break
            print(f"Camera stream lost; reconnecting to {self.url} ...")
            try:
                capture = self._capture_factory(self.url)
            except Exception as error:  # keep retrying until close()
                print(f"Camera reconnect failed: {error}")
                continue
            if capture.isOpened():
                self._capture = capture
            else:
                capture.release()

    def _wait_stopped(self, seconds: float) -> bool:
        with self._condition:
            self._condition.wait_for(lambda: self._stopped, timeout=seconds)
            return self._stopped

    # Consumer side ----------------------------------------------------

    def read(self) -> tuple[Any | None, float | None]:
        with self._condition:
            got_new = self._condition.wait_for(
                lambda: self._stopped or self._sequence > self._returned_sequence,
                timeout=self.read_timeout_s,
            )
            if not got_new or self._sequence <= self._returned_sequence:
                return None, None

            skipped = self._sequence - self._returned_sequence - 1
            self.frames_dropped += max(skipped, 0)
            self._returned_sequence = self._sequence
            return self._frame, self._timestamp

    def close(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._thread.join(timeout=self.read_timeout_s + 1.0)
        if self._capture is not None and not self._thread.is_alive():
            self._capture.release()
            self._capture = None


def open_source(
    source: str | Path,
    read_timeout_s: float = 5.0,
    reconnect_delay_s: float = 1.0,
) -> VideoFileSource | LatestFrameReader:
    """A LatestFrameReader for URLs, otherwise a VideoFileSource."""
    if is_url(source):
        return LatestFrameReader(
            str(source),
            read_timeout_s=read_timeout_s,
            reconnect_delay_s=reconnect_delay_s,
        )
    return VideoFileSource(source)


class FpsMeter:
    """Count processed frames; tick() returns the FPS once per interval, else None."""

    def __init__(
        self,
        interval_s: float = 5.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.interval_s = float(interval_s)
        self._clock = clock
        self._window_start = clock()
        self._frames = 0

    def tick(self) -> float | None:
        self._frames += 1
        now = self._clock()
        elapsed = now - self._window_start
        if elapsed < self.interval_s:
            return None
        fps = self._frames / elapsed if elapsed > 0 else 0.0
        self._window_start = now
        self._frames = 0
        return fps
