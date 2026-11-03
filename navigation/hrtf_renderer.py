from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import sofar
from scipy import signal

URGENT = "urgent"
SOFT = "soft"

# Name fragments that mark an output as Bluetooth. Bluetooth adds about
# 100-250 ms of latency, so it is only used when nothing else exists.
# Windows lists Bluetooth earbuds as e.g. "Headphones (OPPO Enco Buds2)"
# or "Headset (... Hands-Free AG Audio)", so these win over HEADPHONE_KEYWORDS.
BLUETOOTH_KEYWORDS = (
    "bluetooth",
    "hands-free",
    "handsfree",
    "a2dp",
    "airpods",
    "buds",
    "enco",
)

# Name fragments that suggest wired headphones / earphones.
HEADPHONE_KEYWORDS = (
    "headphone",
    "earphone",
    "headset",
)


def is_bluetooth_name(name: str) -> bool:
    lowered = name.lower()
    return any(keyword in lowered for keyword in BLUETOOTH_KEYWORDS)


def _is_output(device) -> bool:
    return int(device.get("max_output_channels", 0)) > 0


def _is_raw_host_api(device) -> bool:
    # "hostapi_name" is added by HRTFRenderer.find_output_device.
    return "wdm-ks" in str(device.get("hostapi_name", "")).lower()


def choose_output_device(devices, default_output=None, requested=None):
    """Pick an output device index from sounddevice-style device dicts.

    requested:
        None        -> automatic choice (see below)
        int or "3"  -> that device index (must be an output device)
        other str   -> first output device whose name contains it
                       (case-insensitive)

    Automatic choice, in order:
        1. wired headphones / earphones (headphone name, not Bluetooth)
        2. the system default output, if it is not Bluetooth
        3. any non-Bluetooth output device
        4. a Bluetooth device (with a latency warning)

    Returns (index, reason). Raises ValueError if a requested device
    does not exist or has no outputs, and RuntimeError if there are no
    output devices at all.
    """

    devices = list(devices)

    if requested is not None:
        text = str(requested).strip()

        try:
            index = int(text)
        except ValueError:
            index = None

        if index is not None:
            if not 0 <= index < len(devices):
                raise ValueError(
                    f"Audio device index {index} does not exist "
                    f"(valid: 0-{len(devices) - 1})."
                )
            if not _is_output(devices[index]):
                raise ValueError(
                    f"Audio device {index} ({devices[index]['name']}) "
                    "has no output channels."
                )
            return index, "requested by index"

        needle = text.lower()
        for index, device in enumerate(devices):
            if _is_output(device) and needle in device["name"].lower():
                return index, f"requested by name '{text}'"

        raise ValueError(
            f"No output audio device name contains '{text}'."
        )

    all_outputs = [
        (index, device)
        for index, device in enumerate(devices)
        if _is_output(device)
    ]

    if not all_outputs:
        raise RuntimeError("No audio output devices found.")

    # Raw WDM-KS endpoints often fail to open or stay silent; only
    # auto-pick one if nothing else exists (or if asked for by index).
    outputs = [
        (index, device)
        for index, device in all_outputs
        if not _is_raw_host_api(device)
    ] or all_outputs

    for index, device in outputs:
        name = device["name"].lower()
        if (
            any(keyword in name for keyword in HEADPHONE_KEYWORDS)
            and not is_bluetooth_name(name)
        ):
            return index, "wired headphones"

    if (
        default_output is not None
        and 0 <= int(default_output) < len(devices)
        and _is_output(devices[int(default_output)])
        and not is_bluetooth_name(devices[int(default_output)]["name"])
    ):
        return int(default_output), "system default output"

    for index, device in outputs:
        if not is_bluetooth_name(device["name"]):
            return index, "first non-Bluetooth output"

    return outputs[0][0], (
        "only Bluetooth outputs found - expect 100-250 ms extra latency; "
        "use wired earphones"
    )


class SoundDeviceBackend:
    """Default output backend: plays through sounddevice.

    play() blocks until the sound ends (or stop() is called), but it is
    only ever called from the renderer's worker thread.
    """

    def __init__(self):
        try:
            import sounddevice
        except Exception as error:  # PortAudio missing, no audio stack, ...
            raise RuntimeError(
                "sounddevice is not available "
                f"({error}). Run with --no-audio."
            ) from error

        self.sd = sounddevice

    def play(self, audio, samplerate, device):
        self.sd.play(
            audio,
            samplerate=samplerate,
            device=device,
            blocking=False,
        )
        self.sd.wait()

    def stop(self):
        self.sd.stop()


@dataclass
class _Cue:
    kind: str
    azimuth_deg: float
    cancelled: bool = False


class HRTFRenderer:
    """SOFA HRTF renderer for urgent hazards and soft path guidance.

    Coordinate conventions
    -----------------------
    The rest of the navigation system uses:

        negative azimuth = LEFT
        0 degrees        = AHEAD
        positive azimuth = RIGHT

    The MIT KEMAR SOFA file used by this project has the opposite
    horizontal azimuth convention:

        positive SOFA azimuth = LEFT
        negative / 360-SOFA  = RIGHT

    Therefore this renderer performs the conversion:

        physical azimuth -> SOFA azimuth

    exactly once, immediately before HRTF lookup.

    Playback is asynchronous: play_warning() and play_path_guidance()
    only put a cue on a small queue (max_queue items) and return at once.
    A background worker renders and plays the cues. Urgent cues drop any
    queued soft cues and interrupt a soft cue that is already playing.
    """

    def __init__(
        self,
        sofa_path="hrtf/mit_kemar_normal_pinna.sofa",
        output_device=None,
        warning_cooldown=0.8,
        backend=None,
        max_queue=2,
    ):
        print("Loading HRTF file...")

        self.sofa_path = sofa_path

        self.sofa = sofar.read_sofa_as_netcdf(
            sofa_path
        )

        self.sample_rate = int(
            np.asarray(
                self.sofa.Data_SamplingRate
            ).flat[0]
        )

        # --------------------------------------------------
        # Urgent warning cooldown
        # --------------------------------------------------

        self.warning_cooldown = float(
            warning_cooldown
        )

        self.last_warning_time = 0.0

        # --------------------------------------------------
        # Soft path-guidance cooldown
        # --------------------------------------------------

        self.path_guidance_cooldown = 1.0
        self.last_path_guidance_time = 0.0

        # --------------------------------------------------
        # Output backend
        # --------------------------------------------------

        use_sounddevice = backend is None

        self.backend = (
            SoundDeviceBackend()
            if use_sounddevice
            else backend
        )

        # --------------------------------------------------
        # Output device
        # --------------------------------------------------

        if use_sounddevice:
            # Resolves an index, a name or None (automatic) to an index.
            self.device = self.find_output_device(output_device)
        else:
            # Custom backends choose their own output.
            self.device = (
                output_device
                if output_device is not None
                else "default"
            )

        print(
            f"HRTF sample rate: {self.sample_rate}"
        )

        print(
            f"Audio output device: {self.device}"
        )

        if use_sounddevice and self.device is not None:
            try:
                device_info = self.backend.sd.query_devices(
                    self.device
                )

                print(
                    f"Using audio device: "
                    f"{device_info['name']}"
                )

            except Exception as error:
                print(
                    f"Could not query audio device: "
                    f"{error}"
                )

        # --------------------------------------------------
        # Playback worker
        # --------------------------------------------------

        self.max_queue = max(1, int(max_queue))
        self._queue: deque[_Cue] = deque()
        self._playing: _Cue | None = None
        self._closed = False
        self._lock = threading.Condition()

        self._worker = threading.Thread(
            target=self._worker_loop,
            name="hrtf-audio",
            daemon=True,
        )
        self._worker.start()

    # ======================================================
    # OUTPUT DEVICE
    # ======================================================

    def find_output_device(self, requested=None):
        """Choose the output device (see choose_output_device).

        requested may be a device index, a numeric string, a name
        fragment or None for automatic (wired headphones preferred,
        Bluetooth last). Raises RuntimeError with a clear message if
        the requested device cannot be used.
        """

        sd = self.backend.sd

        try:
            hostapis = sd.query_hostapis()
            devices = [
                dict(
                    device,
                    hostapi_name=hostapis[device["hostapi"]]["name"],
                )
                for device in sd.query_devices()
            ]
        except Exception as error:
            raise RuntimeError(
                f"Could not list audio devices ({error}). "
                "Run with --no-audio."
            ) from error

        print("\nAvailable audio devices:")

        for index, device in enumerate(devices):
            print(
                f"{index}: "
                f"{device['name']} ({device['hostapi_name']}) | "
                f"inputs="
                f"{device['max_input_channels']} | "
                f"outputs="
                f"{device['max_output_channels']}"
            )

        # sd.default.device is an (input, output) pair-like object.
        try:
            default_output = int(sd.default.device[1])
        except (TypeError, ValueError, IndexError):
            default_output = None

        if default_output is not None and default_output < 0:
            default_output = None

        try:
            index, reason = choose_output_device(
                devices,
                default_output,
                requested,
            )
        except (ValueError, RuntimeError) as error:
            raise RuntimeError(
                f"{error} Pick one from the list above with "
                "--audio-device <index|name>, or run with --no-audio."
            ) from error

        print(
            f"Selected output device {index} "
            f"({devices[index]['name']}): {reason}"
        )

        if is_bluetooth_name(devices[index]["name"]):
            print(
                "WARNING: Bluetooth audio adds about 100-250 ms of "
                "latency. Use wired earphones for the demo."
            )

        return index

    # ======================================================
    # COORDINATE CONVERSION
    # ======================================================

    @staticmethod
    def physical_to_sofa_azimuth(
        azimuth_deg: float,
    ) -> float:
        """Convert project physical azimuth to SOFA azimuth.

        Project convention:
            negative = LEFT
            0        = AHEAD
            positive = RIGHT

        MIT KEMAR SOFA convention observed in this file:
            positive = LEFT
            270/330  = RIGHT

        Examples:
            physical -30° -> SOFA 30°
            physical   0° -> SOFA 0°
            physical +30° -> SOFA 330°

        The result is normalized to [0, 360).
        """

        return (
            -float(azimuth_deg)
        ) % 360.0

    # ======================================================
    # HRTF LOOKUP
    # ======================================================

    def find_nearest_hrtf(
        self,
        sofa_azimuth_deg: float,
    ):
        """Find nearest horizontal-plane HRTF measurement.

        The argument MUST already be in SOFA azimuth convention.

        Elevation 0 degrees is preferred because the navigation
        cues represent horizontal left/right/ahead directions.
        """

        source_positions = np.asarray(
            self.sofa.SourcePosition,
            dtype=float,
        )

        if source_positions.ndim == 1:
            source_positions = (
                source_positions.reshape(
                    1,
                    -1,
                )
            )

        sofa_azimuths = (
            source_positions[:, 0]
        )

        sofa_elevations = (
            source_positions[:, 1]
        )

        requested = (
            float(sofa_azimuth_deg)
            % 360.0
        )

        # Circular angular difference.
        az_difference = np.abs(
            (
                sofa_azimuths
                - requested
                + 180.0
            )
            % 360.0
            - 180.0
        )

        # Prefer horizontal-plane measurements.
        elevation_penalty = (
            np.abs(sofa_elevations)
            * 0.01
        )

        score = (
            az_difference
            + elevation_penalty
        )

        index = int(
            np.argmin(score)
        )

        return (
            index,
            float(
                sofa_azimuths[index]
            ),
            float(
                sofa_elevations[index]
            ),
        )

    # ======================================================
    # SPATIALIZATION
    # ======================================================

    def _spatialize(
        self,
        tone: np.ndarray,
        physical_azimuth_deg: float,
        level: float,
    ):
        """Convolve a mono signal with the correct HRTF.

        physical_azimuth_deg is ALWAYS in project convention.

        Conversion to SOFA convention happens here and nowhere
        else in the application.
        """

        sofa_azimuth_deg = (
            self.physical_to_sofa_azimuth(
                physical_azimuth_deg
            )
        )

        (
            hrtf_index,
            used_sofa_azimuth,
            used_elevation,
        ) = self.find_nearest_hrtf(
            sofa_azimuth_deg
        )

        ir = np.asarray(
            self.sofa.Data_IR
        )

        left_ir = np.asarray(
            ir[
                hrtf_index,
                0,
            ],
            dtype=np.float64,
        )

        right_ir = np.asarray(
            ir[
                hrtf_index,
                1,
            ],
            dtype=np.float64,
        )

        # --------------------------------------------------
        # HRTF convolution
        # --------------------------------------------------

        left = signal.fftconvolve(
            tone,
            left_ir,
            mode="full",
        )

        right = signal.fftconvolve(
            tone,
            right_ir,
            mode="full",
        )

        length = min(
            len(left),
            len(right),
        )

        stereo = np.column_stack(
            (
                left[:length],
                right[:length],
            )
        )

        # --------------------------------------------------
        # Normalize
        # --------------------------------------------------

        maximum = float(
            np.max(
                np.abs(stereo)
            )
        )

        if maximum > 0.0:
            stereo /= maximum

        # --------------------------------------------------
        # Debug output
        # --------------------------------------------------

        print(
            f"HRTF physical azimuth="
            f"{physical_azimuth_deg:+.1f}°, "
            f"SOFA azimuth="
            f"{sofa_azimuth_deg:+.1f}°, "
            f"using SOFA measurement="
            f"{used_sofa_azimuth:+.1f}°, "
            f"elevation="
            f"{used_elevation:+.1f}°"
        )

        return (
            stereo * level
        ).astype(
            np.float32
        )

    # ======================================================
    # ENVELOPE
    # ======================================================

    @staticmethod
    def _envelope(
        sample_rate,
        sample_count,
        fade_s=0.025,
    ):
        fade = min(
            int(
                sample_rate
                * fade_s
            ),
            max(
                1,
                sample_count // 2,
            ),
        )

        envelope = np.ones(
            sample_count
        )

        envelope[:fade] = (
            np.linspace(
                0.0,
                1.0,
                fade,
            )
        )

        envelope[-fade:] = (
            np.linspace(
                1.0,
                0.0,
                fade,
            )
        )

        return envelope

    # ======================================================
    # URGENT WARNING SOUND
    # ======================================================

    def create_warning_sound(
        self,
        azimuth_deg=0.0,
        duration=0.5,
    ):
        """Create the loud urgent hazard cue.

        azimuth_deg uses the PROJECT physical convention.
        """

        try:
            sample_count = int(
                self.sample_rate
                * max(
                    float(duration),
                    0.1,
                )
            )

            t = (
                np.arange(
                    sample_count
                )
                / self.sample_rate
            )

            tone = (
                0.65
                * np.sin(
                    2
                    * np.pi
                    * 700
                    * t
                )
                + 0.35
                * np.sin(
                    2
                    * np.pi
                    * 1050
                    * t
                )
            )

            tone *= self._envelope(
                self.sample_rate,
                sample_count,
                0.03,
            )

            return self._spatialize(
                tone,
                azimuth_deg,
                0.85,
            )

        except Exception as error:

            print(
                f"Could not create HRTF "
                f"warning sound: {error}"
            )

            # Safe fallback: centered stereo.
            sample_count = int(
                self.sample_rate
                * 0.5
            )

            t = (
                np.arange(
                    sample_count
                )
                / self.sample_rate
            )

            fallback = (
                0.7
                * np.sin(
                    2
                    * np.pi
                    * 700
                    * t
                )
            ).astype(
                np.float32
            )

            return np.column_stack(
                (
                    fallback,
                    fallback,
                )
            )

    # ======================================================
    # SOFT PATH GUIDANCE SOUND
    # ======================================================

    def create_path_guidance_sound(
        self,
        azimuth_deg=0.0,
        duration=0.45,
    ):
        """Create a softer route-direction cue.

        azimuth_deg uses the PROJECT physical convention.

        Therefore:
            -30° = LEFT
             0° = AHEAD
            +30° = RIGHT
        """

        try:
            sample_count = int(
                self.sample_rate
                * max(
                    float(duration),
                    0.15,
                )
            )

            t = (
                np.arange(
                    sample_count
                )
                / self.sample_rate
            )

            tone = (
                0.60
                * np.sin(
                    2
                    * np.pi
                    * 420
                    * t
                )
                + 0.40
                * np.sin(
                    2
                    * np.pi
                    * 620
                    * t
                )
            )

            tone *= self._envelope(
                self.sample_rate,
                sample_count,
                0.03,
            )

            return self._spatialize(
                tone,
                azimuth_deg,
                0.50,
            )

        except Exception as error:

            print(
                f"Could not create path "
                f"guidance sound: {error}"
            )

            sample_count = int(
                self.sample_rate
                * 0.45
            )

            t = (
                np.arange(
                    sample_count
                )
                / self.sample_rate
            )

            fallback = (
                0.35
                * np.sin(
                    2
                    * np.pi
                    * 500
                    * t
                )
            ).astype(
                np.float32
            )

            return np.column_stack(
                (
                    fallback,
                    fallback,
                )
            )

    # ======================================================
    # PLAY URGENT WARNING
    # ======================================================

    def play_warning(
        self,
        azimuth_deg=0.0,
        force=False,
    ):
        """Queue an urgent warning from the obstacle's direction.

        azimuth_deg uses the PROJECT physical convention.

        Never blocks: returns True if the cue was queued, False if it
        was skipped (cooldown) or dropped.
        """

        now = time.monotonic()

        if (
            not force
            and (
                now
                - self.last_warning_time
            )
            < self.warning_cooldown
        ):
            return False

        if not self._enqueue(_Cue(URGENT, float(azimuth_deg))):
            return False

        self.last_warning_time = now

        return True

    # ======================================================
    # PLAY PATH GUIDANCE
    # ======================================================

    def play_path_guidance(
        self,
        azimuth_deg=0.0,
        force=False,
    ):
        """Queue a soft cue toward the recommended walking direction.

        azimuth_deg uses the PROJECT physical convention.

        Never blocks: returns True if the cue was queued, False if it
        was skipped (cooldown) or dropped.
        """

        now = time.monotonic()

        if (
            not force
            and (
                now
                - self.last_path_guidance_time
            )
            < self.path_guidance_cooldown
        ):
            return False

        if not self._enqueue(_Cue(SOFT, float(azimuth_deg))):
            return False

        self.last_path_guidance_time = now

        return True

    # ======================================================
    # QUEUE + WORKER
    # ======================================================

    def _enqueue(self, cue: _Cue) -> bool:
        """Add a cue to the bounded queue. Returns False if dropped."""

        interrupt = False

        with self._lock:
            if self._closed:
                return False

            if cue.kind == URGENT:
                # Urgent pre-empts: drop queued soft cues ...
                self._queue = deque(
                    queued
                    for queued in self._queue
                    if queued.kind == URGENT
                )

                # ... and cut off a soft cue that is playing now.
                if (
                    self._playing is not None
                    and self._playing.kind == SOFT
                ):
                    self._playing.cancelled = True
                    interrupt = True

                # Full of urgent cues: the newest direction wins.
                while len(self._queue) >= self.max_queue:
                    self._queue.popleft()

            else:
                if len(self._queue) >= self.max_queue:
                    oldest_soft = next(
                        (
                            queued
                            for queued in self._queue
                            if queued.kind == SOFT
                        ),
                        None,
                    )

                    if oldest_soft is None:
                        # Queue holds only urgent cues.
                        return False

                    self._queue.remove(oldest_soft)

            self._queue.append(cue)
            self._lock.notify()

        if interrupt:
            self._stop_backend()

        return True

    def _worker_loop(self):
        while True:
            with self._lock:
                while not self._queue and not self._closed:
                    self._lock.wait()

                if self._closed:
                    return

                cue = self._queue.popleft()
                self._playing = cue

            try:
                self._play_cue(cue)

            except Exception as error:
                print(
                    f"Could not play {cue.kind} "
                    f"sound: {error}"
                )

            finally:
                finished = time.monotonic()

                with self._lock:
                    self._playing = None

                    # Cooldown runs from the end of playback, as it
                    # did when playback was blocking.
                    if cue.kind == URGENT:
                        self.last_warning_time = finished
                    else:
                        self.last_path_guidance_time = finished

    def _play_cue(self, cue: _Cue):
        if cue.kind == URGENT:
            audio = self.create_warning_sound(
                cue.azimuth_deg,
                0.5,
            )

            label = (
                "Playing URGENT HRTF warning "
                f"at physical "
                f"{cue.azimuth_deg:+.1f}°"
            )

        else:
            audio = self.create_path_guidance_sound(
                cue.azimuth_deg,
                0.45,
            )

            label = (
                "Playing SOFT path-guidance "
                "cue toward physical "
                f"{cue.azimuth_deg:+.1f}°"
            )

        if cue.cancelled:
            return

        print(label)

        self.backend.play(
            audio,
            self.sample_rate,
            self.device,
        )

    # ======================================================
    # STOP / CLOSE
    # ======================================================

    def _stop_backend(self):
        try:
            self.backend.stop()

        except Exception as error:

            print(
                f"Could not stop audio: "
                f"{error}"
            )

    def stop(self):
        """Drop queued cues and stop currently playing audio."""

        with self._lock:
            self._queue.clear()

            if self._playing is not None:
                self._playing.cancelled = True

        self._stop_backend()

    def close(self, timeout=1.0):
        """Stop audio and shut down the worker thread."""

        with self._lock:
            self._closed = True
            self._lock.notify_all()

        self.stop()
        self._worker.join(timeout)
