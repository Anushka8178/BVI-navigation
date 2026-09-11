from __future__ import annotations

from dataclasses import dataclass

from .audio import make_warning
from .hazard import HazardSeverityClassifier
from .hrtf_renderer import HRTFRenderer
from .models import FramePacket, WarningEvent


@dataclass(frozen=True)
class CycleResult:
    frame_id: int
    detection_count: int
    warnings: list[WarningEvent]


class NavigationPipeline:
    def __init__(
        self,
        perception,
        severity: HazardSeverityClassifier,
    ) -> None:
        self.perception = perception
        self.severity = severity
        self.hrtf_renderer = HRTFRenderer()

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
                self.hrtf_renderer.play_warning(
                    warning.azimuth_deg
                )

        return CycleResult(
            frame.frame_id,
            len(detections),
            warnings,
        )