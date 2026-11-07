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
    """Replace synthesis with tiny arrays marked by kind (1 = urgent, 2 = soft, 3 = memory)."""
    r.create_warning_sound = lambda az, dur: np.full((4, 2), 1.0, dtype=np.float32)
    r.create_path_guidance_sound = lambda az, dur: np.full((4, 2), 2.0, dtype=np.float32)
    r.create_memory_cue_sound = lambda az, dist: np.full((4, 2), 3.0, dtype=np.float32)


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


# -- memory early-warning cue (lowest priority) ---------------------------------


def test_memory_cue_is_queued_and_played(renderer):
    tag_sounds(renderer)
    backend = renderer.backend
    assert renderer.play_memory_cue(-20.0, 5.0)
    assert backend.started.wait(2.0)
    backend.release.set()
    assert wait_until(lambda: backend.played == [3.0])


def test_urgent_drops_queued_memory_and_interrupts_playing_memory(renderer):
    tag_sounds(renderer)
    backend = renderer.backend

    assert renderer.play_memory_cue(0.0, 5.0)
    assert backend.started.wait(2.0)  # memory cue now playing
    assert renderer.play_memory_cue(10.0, 5.0, force=True)  # queued memory

    assert renderer.play_warning(0.0)
    assert backend.stop_calls == 1  # playing memory cue was cut off
    assert wait_until(lambda: len(backend.played) == 2)
    assert backend.played == [3.0, 1.0]  # queued memory cue was dropped


def test_memory_cue_never_displaces_urgent_or_path_cues(renderer):
    tag_sounds(renderer)
    backend = renderer.backend

    assert renderer.play_warning(0.0)
    assert backend.started.wait(2.0)  # worker busy
    assert renderer.play_path_guidance(30.0)
    assert renderer.play_path_guidance(-30.0, force=True)  # queue full of path cues
    assert not renderer.play_memory_cue(0.0, 5.0, force=True)
    assert [c.kind for c in renderer._queue] == ["soft", "soft"]

    assert renderer.play_warning(10.0, force=True)  # drops the path cues
    assert renderer.play_warning(20.0, force=True)  # queue full of urgent cues
    assert not renderer.play_memory_cue(0.0, 5.0, force=True)
    assert [c.kind for c in renderer._queue] == ["urgent", "urgent"]


def test_path_cue_replaces_queued_memory_cue_first(renderer):
    tag_sounds(renderer)
    backend = renderer.backend

    assert renderer.play_warning(0.0)
    assert backend.started.wait(2.0)  # worker busy
    assert renderer.play_path_guidance(-30.0)
    assert renderer.play_memory_cue(0.0, 5.0)  # queue full: soft + memory
    assert renderer.play_path_guidance(30.0, force=True)
    assert [(c.kind, c.azimuth_deg) for c in renderer._queue] == [("soft", -30.0), ("soft", 30.0)]


def test_memory_cue_cooldown(renderer):
    tag_sounds(renderer)
    assert renderer.play_memory_cue(0.0, 5.0)
    assert not renderer.play_memory_cue(0.0, 5.0)
    assert renderer.play_memory_cue(0.0, 5.0, force=True)


def test_memory_cue_sound_is_quieter_than_path_guidance_and_spatialized(renderer):
    memory = renderer.create_memory_cue_sound(-60.0, 4.0)
    path = renderer.create_path_guidance_sound(-60.0, 0.45)
    assert memory.ndim == 2 and memory.shape[1] == 2
    assert np.max(np.abs(memory)) < np.max(np.abs(path))
    # Physical -60 deg = LEFT: left channel louder.
    left, right = np.sum(memory[:, 0] ** 2), np.sum(memory[:, 1] ** 2)
    assert left > right
    # Two pips with a silent gap in between.
    envelope = np.abs(memory).max(axis=1)
    middle = envelope[int(0.10 * renderer.sample_rate): int(0.16 * renderer.sample_rate)]
    assert middle.max() < 0.25 * envelope.max()
    assert renderer.memory_cue_level(8.0) < renderer.memory_cue_level(3.0) < 0.5
