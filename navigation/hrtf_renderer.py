import time
import numpy as np
import sounddevice as sd
import sofar
from scipy import signal


class HRTFRenderer:
    def __init__(
        self,
        sofa_path="hrtf/mit_kemar_normal_pinna.sofa",
        output_device=None,
        warning_cooldown=0.8,
    ):
        print("Loading HRTF file...")

        self.sofa_path = sofa_path

        # Use this loader because the file uses:
        # SimpleFreeFieldHRIR v0.3
        self.sofa = sofar.read_sofa_as_netcdf(sofa_path)

        # Correct field names for read_sofa_as_netcdf().
        self.sample_rate = int(np.asarray(self.sofa.Data_SamplingRate).flat[0])

        self.warning_cooldown = float(warning_cooldown)
        self.last_warning_time = 0.0
        self.path_guidance_cooldown = 1.2
        self.last_path_guidance_time = 0.0

        self.device = output_device

        if self.device is None:
            self.device = self.find_output_device()

        print(f"HRTF sample rate: {self.sample_rate}")
        print(f"Audio output device: {self.device}")

        if self.device is not None:
            try:
                device_info = sd.query_devices(self.device)
                print(f"Using audio device: {device_info['name']}")
            except Exception as error:
                print(f"Could not query audio device: {error}")

    def find_output_device(self):
        """
        Find a suitable audio output device.
        """
        try:
            devices = sd.query_devices()

            print("\nAvailable audio devices:")

            for index, device in enumerate(devices):
                print(
                    f"{index}: {device['name']} | "
                    f"inputs={device['max_input_channels']} | "
                    f"outputs={device['max_output_channels']}"
                )

            preferred_keywords = [
                "oppo",
                "enco",
                "bluetooth",
                "headphone",
                "headset",
                "earbud",
            ]

            for index, device in enumerate(devices):
                device_name = device["name"].lower()

                if device["max_output_channels"] > 0:
                    if any(
                        keyword in device_name
                        for keyword in preferred_keywords
                    ):
                        print(f"Automatically selected output device {index}")
                        return index

            default_device = sd.default.device

            if isinstance(default_device, (list, tuple)):
                default_output = default_device[1]
            else:
                default_output = default_device

            print(f"Using default output device: {default_output}")
            return default_output

        except Exception as error:
            print(f"Could not find output device: {error}")
            return None

    def find_nearest_hrtf(self, azimuth_deg):
        """
        Find the nearest HRTF measurement for the requested azimuth.
        """
        try:
            # Correct field name for the NetCDF-style SOFA object.
            source_positions = np.asarray(self.sofa.SourcePosition)

            if source_positions.ndim == 1:
                source_positions = source_positions.reshape(1, -1)

            sofa_azimuths = source_positions[:, 0]

            requested_azimuth = float(azimuth_deg) % 360.0

            angle_difference = np.abs(
                (sofa_azimuths - requested_azimuth + 180.0) % 360.0
                - 180.0
            )

            nearest_index = int(np.argmin(angle_difference))

            return nearest_index

        except Exception as error:
            print(f"Could not find nearest HRTF: {error}")
            return 0

    def create_warning_sound(self, azimuth_deg=0.0, duration=0.5):
        """
        Create a warning sound spatialized using the HRTF impulse response.
        """
        try:
            duration = float(duration)

            if duration <= 0:
                duration = 0.5

            frequency_1 = 700.0
            frequency_2 = 1050.0

            sample_count = int(self.sample_rate * duration)

            time_axis = np.arange(sample_count) / self.sample_rate

            # Warning tone.
            tone = (
                0.65 * np.sin(2.0 * np.pi * frequency_1 * time_axis)
                + 0.35 * np.sin(2.0 * np.pi * frequency_2 * time_axis)
            )

            # Fade-in and fade-out to avoid clicking sounds.
            fade_length = min(
                int(self.sample_rate * 0.03),
                max(1, sample_count // 2),
            )

            envelope = np.ones(sample_count)

            envelope[:fade_length] = np.linspace(
                0.0,
                1.0,
                fade_length,
            )

            envelope[-fade_length:] = np.linspace(
                1.0,
                0.0,
                fade_length,
            )

            tone = tone * envelope

            hrtf_index = self.find_nearest_hrtf(azimuth_deg)

            # Correct HRTF impulse-response field name.
            impulse_responses = np.asarray(self.sofa.Data_IR)

            print(
                f"HRTF IR shape: {impulse_responses.shape}"
                if hrtf_index == 0
                else "",
                end="",
            )

            # Expected shape is normally:
            # [measurement, receiver/ear, impulse-sample]
            left_ir = np.asarray(
                impulse_responses[hrtf_index, 0],
                dtype=np.float64,
            )

            right_ir = np.asarray(
                impulse_responses[hrtf_index, 1],
                dtype=np.float64,
            )

            # Apply the HRTF to both ears.
            left_audio = signal.fftconvolve(
                tone,
                left_ir,
                mode="full",
            )

            right_audio = signal.fftconvolve(
                tone,
                right_ir,
                mode="full",
            )

            output_length = min(
                len(left_audio),
                len(right_audio),
            )

            stereo_audio = np.column_stack(
                (
                    left_audio[:output_length],
                    right_audio[:output_length],
                )
            )

            # Normalize to prevent clipping.
            maximum = np.max(np.abs(stereo_audio))

            if maximum > 0:
                stereo_audio = stereo_audio / maximum

            # Convert to sounddevice-compatible format.
            stereo_audio = stereo_audio.astype(np.float32)

            # Increase the output level.
            stereo_audio = stereo_audio * 0.85

            return stereo_audio

        except Exception as error:
            print(f"Could not create HRTF warning sound: {error}")

            # Fallback stereo tone.
            sample_count = int(self.sample_rate * duration)
            time_axis = np.arange(sample_count) / self.sample_rate

            fallback_tone = (
                0.7
                * np.sin(2.0 * np.pi * 700.0 * time_axis)
            )

            fallback_tone = fallback_tone.astype(np.float32)

            return np.column_stack(
                (
                    fallback_tone,
                    fallback_tone,
                )
            )


    def create_path_guidance_sound(self, azimuth_deg=0.0, duration=0.22):
        """
        Create a softer, shorter HRTF cue for navigation guidance.

        This is intentionally different from the urgent hazard beep:
        - shorter duration
        - lower amplitude
        - lower, softer two-tone cue

        The azimuth is still spatialized through the same HRTF.
        """
        try:
            duration = max(float(duration), 0.08)

            frequency_1 = 420.0
            frequency_2 = 620.0
            sample_count = int(self.sample_rate * duration)
            time_axis = np.arange(sample_count) / self.sample_rate

            tone = (
                0.60 * np.sin(2.0 * np.pi * frequency_1 * time_axis)
                + 0.40 * np.sin(2.0 * np.pi * frequency_2 * time_axis)
            )

            fade_length = min(
                int(self.sample_rate * 0.025),
                max(1, sample_count // 2),
            )
            envelope = np.ones(sample_count)
            envelope[:fade_length] = np.linspace(0.0, 1.0, fade_length)
            envelope[-fade_length:] = np.linspace(1.0, 0.0, fade_length)
            tone *= envelope

            hrtf_index = self.find_nearest_hrtf(azimuth_deg)
            impulse_responses = np.asarray(self.sofa.Data_IR)

            left_ir = np.asarray(impulse_responses[hrtf_index, 0], dtype=np.float64)
            right_ir = np.asarray(impulse_responses[hrtf_index, 1], dtype=np.float64)

            left_audio = signal.fftconvolve(tone, left_ir, mode="full")
            right_audio = signal.fftconvolve(tone, right_ir, mode="full")

            output_length = min(len(left_audio), len(right_audio))
            stereo_audio = np.column_stack(
                (left_audio[:output_length], right_audio[:output_length])
            )

            maximum = np.max(np.abs(stereo_audio))
            if maximum > 0:
                stereo_audio /= maximum

            # Much quieter than the urgent warning.
            stereo_audio = (stereo_audio * 0.28).astype(np.float32)
            return stereo_audio

        except Exception as error:
            print(f"Could not create path guidance sound: {error}")
            sample_count = int(self.sample_rate * max(float(duration), 0.08))
            time_axis = np.arange(sample_count) / self.sample_rate
            fallback = (
                0.28
                * 0.5
                * np.sin(2.0 * np.pi * 500.0 * time_axis)
            ).astype(np.float32)
            return np.column_stack((fallback, fallback))

    def play_path_guidance(self, azimuth_deg=0.0, force=False):
        """
        Play the softer navigation/path-guidance cue.

        This has its own cooldown so path guidance does not interfere
        with the urgent-warning cooldown.
        """
        current_time = time.monotonic()

        if not force:
            elapsed_time = current_time - self.last_path_guidance_time
            if elapsed_time < self.path_guidance_cooldown:
                return False

        try:
            audio = self.create_path_guidance_sound(
                azimuth_deg=azimuth_deg,
                duration=0.22,
            )

            if audio is None or len(audio) == 0:
                return False

            print(
                f"Playing soft path-guidance cue at "
                f"{azimuth_deg:.1f} degrees"
            )

            sd.play(
                audio,
                samplerate=self.sample_rate,
                device=self.device,
                blocking=True,
            )

            self.last_path_guidance_time = time.monotonic()
            return True

        except Exception as error:
            print(f"Could not play path guidance sound: {error}")
            return False

    def play_warning(self, azimuth_deg=0.0, force=False):
        """
        Play a warning sound.

        Playback is blocking to ensure that the warning is not lost
        or interrupted by the video-processing loop.
        """
        current_time = time.monotonic()

        if not force:
            elapsed_time = current_time - self.last_warning_time

            if elapsed_time < self.warning_cooldown:
                return False

        try:
            audio = self.create_warning_sound(
                azimuth_deg=azimuth_deg,
                duration=0.5,
            )

            if audio is None or len(audio) == 0:
                print("Warning audio could not be generated.")
                return False

            print(
                f"Playing HRTF warning at "
                f"{azimuth_deg:.1f} degrees"
            )

            # Blocking playback guarantees that the warning is sent
            # completely to the selected output device.
            sd.play(
                audio,
                samplerate=self.sample_rate,
                device=self.device,
                blocking=True,
            )

            self.last_warning_time = time.monotonic()

            return True

        except Exception as error:
            print(f"Could not play warning sound: {error}")
            return False

    def stop(self):
        """
        Stop currently playing audio.
        """
        try:
            sd.stop()
        except Exception as error:
            print(f"Could not stop audio: {error}")

    def close(self):
        """
        Close the renderer and stop audio.
        """
        self.stop()