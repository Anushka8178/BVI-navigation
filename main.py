from __future__ import annotations

import argparse
from pathlib import Path

from navigation.audio import ConsoleSpatialAudio
from navigation.hazard import HazardSeverityClassifier
from navigation.memory import RouteMemory, SpatialMemory
from navigation.models import RouteCandidate
from navigation.perception import ScriptedPerception
from navigation.pipeline import NavigationPipeline
from navigation.reasoning import AcousticAttention, ExplainablePathPlanner
from navigation.sensing import SimulatedSensor


def build_pipeline(database_path: Path) -> tuple[NavigationPipeline, RouteMemory]:
    route_memory = RouteMemory(database_path)
    pipeline = NavigationPipeline(
        perception=ScriptedPerception(),
        severity=HazardSeverityClassifier(),
        spatial_memory=SpatialMemory(),
        route_memory=route_memory,
        attention=AcousticAttention(top_k=3),
        planner=ExplainablePathPlanner(route_memory),
        audio=ConsoleSpatialAudio(),
    )
    return pipeline, route_memory


def print_event(prefix: str, event: object) -> None:
    print(f"  {prefix}: {event}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BVI navigation architecture prototype")
    parser.add_argument("--reset", action="store_true", help="clear the persistent demo database")
    args = parser.parse_args()

    database_path = Path(__file__).parent / "data" / "navigation.db"
    if args.reset and database_path.exists():
        database_path.unlink()

    pipeline, route_memory = build_pipeline(database_path)
    try:
        route_memory.record_visit("route-a")
        print("BVI NAVIGATION PROTOTYPE - six architecture tiers\n")
        for frame in SimulatedSensor():
            result = pipeline.process_cycle(frame)
            tracking = "OK" if result.localization_valid else "LOST: frame alerts only"
            print(
                f"FRAME {result.frame_id} | detections={result.detections} | "
                f"spatial-memory={result.memory_size} | localization={tracking}"
            )
            for event in result.immediate_alerts:
                print_event("URGENT BYPASS", event)
            for event in result.awareness_alerts:
                print_event("TOP-K MEMORY", event)
            print()

        routes = [
            RouteCandidate("route-a", "Main corridor", 82.0, 0.15),
            RouteCandidate("route-b", "Courtyard detour", 105.0, 0.05),
        ]
        decision = pipeline.plan(routes)
        print("EXPLAINABLE ROUTE DECISION")
        print(f"  {decision.rationale}")
        for name, score in decision.alternatives:
            print(f"  Alternative {name}: combined cost {score:.2f}")
    finally:
        route_memory.close()


if __name__ == "__main__":
    main()

