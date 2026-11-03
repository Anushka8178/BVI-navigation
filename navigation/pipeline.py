from __future__ import annotations

from dataclasses import dataclass, field

from .audio import make_warning
from .hazard import HazardSeverityClassifier
from .hrtf_renderer import HRTFRenderer
from .models import Detection, FramePacket, WarningEvent


@dataclass(frozen=True)
class CycleResult:
    frame_id: int
    detection_count: int
    warnings: list[WarningEvent]
    detections: list[Detection] = field(default_factory=list)


class NavigationPipeline:
    def __init__(
        self,
        perception,
        severity: HazardSeverityClassifier,
        audio_enabled: bool = True,
        audio_device=None,
        sofa_path: str = "hrtf/mit_kemar_normal_pinna.sofa",
        hrtf_renderer=None,
    ) -> None:
        """audio_enabled=False (the --no-audio path) skips HRTF entirely:
        warnings are still computed and returned, nothing is played.

        audio_device is an index, a numeric string or a name fragment
        (None = automatic, wired headphones preferred). hrtf_renderer
        injects a ready renderer (e.g. in tests) and wins over the rest.
        """

        self.perception = perception
        self.severity = severity

        if hrtf_renderer is not None:
            self.hrtf_renderer = hrtf_renderer
        elif audio_enabled:
            self.hrtf_renderer = HRTFRenderer(
                sofa_path=sofa_path,
                output_device=audio_device,
            )
        else:
            self.hrtf_renderer = None

    def process_cycle(
        self,
        frame: FramePacket,
    ) -> CycleResult:

        detections = self.perception.process(frame)

        warnings: list[WarningEvent] = []

        for detection in detections:

            result = self.severity.classify(detection)

            if result.urgent:

                warning = make_warning(
                    detection,
                    result.score,
                    result.reason,
                )

                warnings.append(warning)

                # Play spatial warning based on
                # the detected obstacle's azimuth.
                if self.hrtf_renderer is not None:
                    self.hrtf_renderer.play_warning(
                        warning.azimuth_deg
                    )

        return CycleResult(
            frame.frame_id,
            len(detections),
            warnings,
            detections,
        )

    def close(self) -> None:
        """Stop audio and its worker thread (no-op without audio)."""

        if self.hrtf_renderer is not None:
            self.hrtf_renderer.close()