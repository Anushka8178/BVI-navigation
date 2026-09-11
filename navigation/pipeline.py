from __future__ import annotations

from dataclasses import dataclass, field

from .audio import make_warning
from .hazard import HazardSeverityClassifier
from .models import Detection, FramePacket, WarningEvent


@dataclass(frozen=True)
class CycleResult:
    frame_id: int
    detection_count: int
    warnings: list[WarningEvent]
    detections: list[Detection] = field(default_factory=list)


class NavigationPipeline:
    def __init__(self, perception, severity: HazardSeverityClassifier) -> None:
        self.perception = perception
        self.severity = severity

    def process_cycle(self, frame: FramePacket) -> CycleResult:
        detections = self.perception.process(frame)
        warnings: list[WarningEvent] = []
        for detection in detections:
            result = self.severity.classify(detection)
            if result.urgent:
                warnings.append(make_warning(detection, result.score, result.reason))
        return CycleResult(frame.frame_id, len(detections), warnings, detections)
