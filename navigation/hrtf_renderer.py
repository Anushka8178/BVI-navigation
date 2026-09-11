from __future__ import annotations

import numpy as np
import sounddevice as sd
import sofar

from scipy.signal import resample_poly, fftconvolve


class HRTFRenderer:
    """
    SOFA-based binaural HRTF renderer.

    Navigation convention:
        -90 degrees = LEFT
          0 degrees = AHEAD
        +90 degrees = RIGHT
    """

    def __init__(
        self,
        sofa_path: str = r"hrtf\mit_kemar_normal_pinna.sofa",
        sample_rate: int = 48000,
    ):
        self.requested_sample_rate = sample_rate

        # Find an available output device automatically.
        self.device = self.find_output_device()

        if self.device is None:
            raise RuntimeError(
                "No compatible audio output device found."
            )

        # Load the SOFA HRTF dataset.
        self.sofa = sofar.read_sofa_as_netcdf(sofa_path)

        self.hrtf_rate = int(
            float(
                np.asarray(
                    self.sofa.Data_SamplingRate
                ).flat[0]
            )
        )

        self.ir = np.asarray(
            self.sofa.Data_IR,
            dtype=np.float64,
        )

        self.positions = np.asarray(
            self.sofa.SourcePosition,
            dtype=np.float64,
        )

        # Use the HRTF's native sample rate by default.
        # The final audio will be resampled to the
        # requested output rate when necessary.
        self.sample_rate = self.requested_sample_rate

        print(
            f"Loaded SOFA HRTF: "
            f"{self.ir.shape[0]} positions, "
            f"{self.ir.shape[1]} ears, "
            f"{self.ir.shape[2]} samples, "
            f"{self.hrtf_rate} Hz"
        )

        print(
            f"Audio output device: "
            f"{self.device} - "
            f"{sd.query_devices(self.device)['name']}"
        )

    def find_output_device(self):
        """
        Find a suitable audio output device automatically.

        Prefer the user's OPPO earbuds when available.
        Otherwise fall back to the system default output.
        """

        devices = sd.query_devices()

        # Preferred device names.
        preferred_names = [
            "OPPO Enco Buds2",
            "Headphones",
        ]

        for preferred_name in preferred_names:
            for index, device in enumerate(devices):

                if device["max_output_channels"] <= 0:
                    continue

                device_name = device["name"].lower()

                if preferred_name.lower() in device_name:
                    return index

        # Fall back to Windows/system default output.
        default_device = sd.default.device

        if default_device is not None:
            try:
                default_output = int(default_device[1])

                if default_output >= 0:
                    device = devices[default_output]

                    if device["max_output_channels"] > 0:
                        return default_output

            except (TypeError, ValueError, IndexError):
                pass

        return None

    def find_nearest_hrtf(
        self,
        azimuth_deg: float,
    ):
        """
        Find the measured HRIR closest to the requested
        navigation azimuth.

        The SOFA dataset uses azimuth values from
        0 to 360 degrees, so negative navigation
        angles are converted automatically.
        """

        # Keep navigation angle within the useful range.
        azimuth_deg = max(
            -90.0,
            min(90.0, azimuth_deg),
        )

        # SOFA SourcePosition:
        # column 0 = azimuth
        # column 1 = elevation
        # column 2 = distance

        azimuths = self.positions[:, 0]
        elevations = self.positions[:, 1]

        # Circular angular difference.
        azimuth_error = np.abs(
            (
                (azimuths - azimuth_deg + 180.0)
                % 360.0
            )
            - 180.0
        )

        # Prefer measurements close to horizontal plane.
        elevation_error = np.abs(elevations)

        # Azimuth is the main criterion.
        # Elevation is only a small tie-breaker.
        score = (
            azimuth_error
            + 0.1 * elevation_error
        )

        index = int(np.argmin(score))

        return self.ir[index]

    def create_warning_sound(
        self,
        azimuth_deg: float,
        duration: float = 1.0,
        frequency: float = 700.0,
    ) -> np.ndarray:
        """
        Generate a binaural HRTF warning sound.
        """

        # Generate the warning tone at the HRTF's
        # native sample rate.
        samples = int(
            self.hrtf_rate * duration
        )

        t = np.linspace(
            0,
            duration,
            samples,
            endpoint=False,
        )

        mono = (
            0.5
            * np.sin(
                2 * np.pi * frequency * t
            )
        ).astype(np.float64)

        # Select the closest measured HRTF.
        hrtf = self.find_nearest_hrtf(
            azimuth_deg
        )

        # The tested SOFA file's ear channels need
        # to be swapped to match our navigation
        # convention:
        #
        # navigation LEFT  -> hrtf[1]
        # navigation RIGHT -> hrtf[0]

        left_ir = hrtf[1]
        right_ir = hrtf[0]

        # Apply the measured HRIR to each ear.
        left = fftconvolve(
            mono,
            left_ir,
        )

        right = fftconvolve(
            mono,
            right_ir,
        )

        stereo = np.column_stack(
            (left, right)
        )

        # Convert HRTF sample rate to the audio
        # output sample rate.
        if self.hrtf_rate != self.sample_rate:

            stereo = resample_poly(
                stereo,
                self.sample_rate,
                self.hrtf_rate,
                axis=0,
            )

        # Prevent clipping.
        peak = np.max(
            np.abs(stereo)
        )

        if peak > 0:
            stereo = (
                stereo / peak
            ) * 0.8

        return stereo.astype(
            np.float32
        )

    def play_warning(
        self,
        azimuth_deg: float,
    ):
        """
        Play a spatial warning through the
        automatically detected output device.
        """

        audio = self.create_warning_sound(
            azimuth_deg
        )

        device_name = sd.query_devices(
            self.device
        )["name"]

        print(
            f"Playing HRTF "
            f"{azimuth_deg:+.1f} degrees "
            f"on device {self.device} "
            f"({device_name})..."
        )

        sd.play(
            audio,
            samplerate=self.sample_rate,
            device=self.device,
        )