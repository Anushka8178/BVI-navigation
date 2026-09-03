from __future__ import annotations

from dataclasses import dataclass

from .audio import ConsoleSpatialAudio
from .hazard import HazardSeverityClassifier
from .memory import RouteMemory, SpatialMemory
from .models import AudioEvent, FramePacket, PlanDecision, RouteCandidate
from .perception import ScriptedPerception
from .reasoning import AcousticAttention, ExplainablePathPlanner


@dataclass(frozen=True)
class CycleResult:
    frame_id: int
    detections: int
    immediate_alerts: list[AudioEvent]
    awareness_alerts: list[AudioEvent]
    memory_size: int
    localization_valid: bool


class NavigationPipeline:
    """Orchestrates one processing cycle while keeping modules replaceable."""

    def __init__(
        self,
        perception: ScriptedPerception,
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

    def process_cycle(self, frame: FramePacket) -> CycleResult:
        detections = self.perception.process(frame)
        immediate: list[AudioEvent] = []

        # Safety-critical path: classify and emit without waiting for memory ranking.
        for detection in detections:
            result = self.severity.classify(detection)
            if result.urgent:
                immediate.append(self.audio.immediate(detection, result.score))

        # Always-executed memory path. Invalid localization safely prevents world anchoring.
        for detection in detections:
            self.spatial_memory.update(detection, frame.pose, frame.timestamp)
        self.spatial_memory.decay_and_prune(frame.timestamp)

        # Steady-state awareness path: limit output to top-k cached hazards.
        ranked = (
            self.attention.rank(self.spatial_memory.active(), frame.pose)
            if frame.pose.localization_valid
            else []
        )
        awareness = [self.audio.awareness(entry, frame.pose) for entry in ranked]

        # Only stable, high-confidence static hazards become cross-session route facts.
        if frame.pose.localization_valid:
            for entry in ranked:
                if entry.motion.value == "static" and entry.confidence >= 0.95:
                    self.route_memory.remember_hazard(self.active_route, entry)

        return CycleResult(
            frame_id=frame.frame_id,
            detections=len(detections),
            immediate_alerts=immediate,
            awareness_alerts=awareness,
            memory_size=len(self.spatial_memory.entries),
            localization_valid=frame.pose.localization_valid,
        )

    def plan(self, routes: list[RouteCandidate]) -> PlanDecision:
        return self.planner.choose(routes)
