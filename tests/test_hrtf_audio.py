"""Non-blocking HRTF playback (CLAUDE.md known bug #5).

Uses a fake output backend, so no audio device is needed.
"""

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from navigation.hrtf_renderer import HRTFRenderer

SOFA_PATH = Path(__file__).resolve().parents[1] / "hrtf" / "mit_kemar_normal_pinna.sofa"


class FakeBackend:
    """Blocks in play() until released or stopped, like a real device."""

    def __init__(self):
        self.played = []  # marker value of each sound played
        self.started = threading.Event()
        self.release = threading.Event()
        self.stop_calls = 0

    def play(self, audio, samplerate, device):
        self.played.append(float(audio.flat[0]))
        self.started.set()
        self.release.wait(timeout=5.0)
        self.release.clear()

    def stop(self):
        self.stop_calls += 1
        self.release.set()


@pytest.fixture
def renderer():
    backend = FakeBackend()
    r = HRTFRenderer(sofa_path=str(SOFA_PATH), backend=backend)
    yield r
    backend.release.set()
    r.close()


def tag_sounds(r):
    """Replace synthesis with tiny arrays marked by kind (1 = urgent, 2 = soft)."""
    r.create_warning_sound = lambda az, dur: np.full((4, 2), 1.0, dtype=np.float32)
    r.create_path_guidance_sound = lambda az, dur: np.full((4, 2), 2.0, dtype=np.float32)


def wait_until(condition, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True


def test_play_warning_returns_fast_while_audio_is_playing(renderer):
    backend = renderer.backend

    # First cue occupies the (blocking) backend.
    assert renderer.play_warning(-30.0)
    assert backend.started.wait(2.0)

    # The frame loop must not wait, even with real HRTF synthesis.
    start = time.perf_counter()
    queued = renderer.play_warning(30.0, force=True)
    elapsed = time.perf_counter() - start

    assert queued
    assert elapsed < 0.005, f"play_warning took {elapsed * 1000:.2f} ms"


def test_urgent_drops_queued_soft_cues_and_interrupts_playing_soft(renderer):
    tag_sounds(renderer)
    backend = renderer.backend

    assert renderer.play_path_guidance(-30.0)
    assert backend.started.wait(2.0)  # soft cue now playing
    assert renderer.play_path_guidance(30.0, force=True)  # queued soft

    assert renderer.play_warning(0.0)

    assert backend.stop_calls == 1  # playing soft cue was cut off
    assert wait_until(lambda: len(backend.played) == 2)
    assert backend.played == [2.0, 1.0]  # queued soft cue was dropped


def test_queue_is_bounded_and_soft_cannot_displace_urgent(renderer):
    tag_sounds(renderer)
    backend = renderer.backend

    assert renderer.play_warning(0.0)
    assert backend.started.wait(2.0)  # worker busy

    assert renderer.play_warning(10.0, force=True)
    assert renderer.play_warning(20.0, force=True)
    assert renderer.play_warning(30.0, force=True)  # oldest urgent dropped
    assert len(renderer._queue) == 2
    assert [c.azimuth_deg for c in renderer._queue] == [20.0, 30.0]

    assert not renderer.play_path_guidance(0.0, force=True)
    assert len(renderer._queue) == 2


def test_cooldowns_still_apply(renderer):
    tag_sounds(renderer)

    assert renderer.play_warning(0.0)
    assert not renderer.play_warning(0.0)
    assert renderer.play_path_guidance(0.0)
    assert not renderer.play_path_guidance(0.0)
