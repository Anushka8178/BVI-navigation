from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .audio import ConsoleSpatialAudio
from .hazard import HazardSeverityClassifier
from .memory import RouteMemory, SpatialMemory
from .models import (
    AudioEvent,
    Detection,
    FramePacket,
    PlanDecision,
    RouteCandidate,
)
from .reasoning import AcousticAttention, ExplainablePathPlanner


class PerceptionBackend(Protocol):
    """
    Common interface for every perception system.

    ScriptedPerception and YOLOPerception both contain
    a process() method that returns detections.
    """

    def process(self, frame: FramePacket) -> list[Detection]:
        ...


@dataclass(frozen=True)
class CycleResult:
    frame_id: int
    detections: int
    immediate_alerts: list[AudioEvent]
    awareness_alerts: list[AudioEvent]
    memory_size: int
    localization_valid: bool


class NavigationPipeline:
    """
    Connects perception, hazard detection, memory,
    attention, route reasoning and audio output.
    """

    def __init__(
        self,
        perception: PerceptionBackend,
        severity: HazardSeverityClassifier,
        spatial_memory: SpatialMemory,
        route_memory: RouteMemory,
        attention: AcousticAttention,
        planner: ExplainablePathPlanner,
        audio: ConsoleSpatialAudio,
        active_route: str = "route-a",
    ) -> None:
        self.perception = perception
        self.severity = severity
        self.spatial_memory = spatial_memory
        self.route_memory = route_memory
        self.attention = attention
        self.planner = planner
        self.audio = audio
        self.active_route = active_route

    def process_cycle(
        self,
        frame: FramePacket,
    ) -> CycleResult:
        # Detect objects in the current frame.
        detections = self.perception.process(frame)

        immediate_alerts: list[AudioEvent] = []

        # Check every detection for immediate danger.
        for detection in detections:
            severity_result = self.severity.classify(detection)

            if severity_result.urgent:
                audio_event = self.audio.immediate(
                    detection,
                    severity_result.score,
                )

                immediate_alerts.append(audio_event)

        # Add or update every detected object in spatial memory.
        for detection in detections:
            self.spatial_memory.update(
                detection,
                frame.pose,
                frame.timestamp,
            )

        # Remove old memory entries.
        self.spatial_memory.decay_and_prune(
            frame.timestamp
        )

        # Rank remembered hazards only when localization is valid.
        if frame.pose.localization_valid:
            ranked_hazards = self.attention.rank(
                self.spatial_memory.active(),
                frame.pose,
            )
        else:
            ranked_hazards = []

        # Convert ranked hazards into awareness audio events.
        awareness_alerts = [
            self.audio.awareness(entry, frame.pose)
            for entry in ranked_hazards
        ]

        # Store stable and highly confident hazards permanently.
        if frame.pose.localization_valid:
            for entry in ranked_hazards:
                is_static = entry.motion.value == "static"
                is_high_confidence = entry.confidence >= 0.95

                if is_static and is_high_confidence:
                    self.route_memory.remember_hazard(
                        self.active_route,
                        entry,
                    )

        return CycleResult(
            frame_id=frame.frame_id,
            detections=len(detections),
            immediate_alerts=immediate_alerts,
            awareness_alerts=awareness_alerts,
            memory_size=len(self.spatial_memory.entries),
            localization_valid=frame.pose.localization_valid,
        )

    def plan(
        self,
        routes: list[RouteCandidate],
    ) -> PlanDecision:
        """
        Compare candidate routes and return the safest
        explainable route decision.
        """

        return self.planner.choose(routes)