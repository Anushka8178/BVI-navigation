"""
End-to-end demo of the simulated localization pipeline:

    SimulatedSensor -> Pose -> Detection -> local_to_world() -> WorldPoint
                                                                     |
                                                    LocalizedDetection (handoff)
                                                                     |
                                                        RouteMemory.remember(...)

Run it with:

    python localization_demo.py
"""

from __future__ import annotations

import os

from navigation.memory import RouteMemory
from navigation.models import Detection, Motion
from navigation.sensing import SimulatedSensor, localize_detection

HAZARDS_BY_FRAME: dict[int, Detection] = {
    1: Detection("pole-1", "pole", relative_x=0.8, relative_y=0.3, confidence=0.91, motion=Motion.STATIC),
    2: Detection("pole-1", "pole", relative_x=0.6, relative_y=0.3, confidence=0.92, motion=Motion.STATIC),
    3: Detection("curb-2", "curb", relative_x=1.0, relative_y=0.5, confidence=0.88, motion=Motion.STATIC),
    4: Detection("person-3", "person", relative_x=1.2, relative_y=-0.4, confidence=0.95, motion=Motion.APPROACHING),
    5: Detection("person-3", "person", relative_x=0.7, relative_y=-0.4, confidence=0.96, motion=Motion.APPROACHING),
    6: Detection("bench-4", "bench", relative_x=1.5, relative_y=0.2, confidence=0.80, motion=Motion.STATIC),
}

DEMO_DB_PATH = "localization_demo_memory.db"


def main() -> None:
    sensor = SimulatedSensor()

    if os.path.exists(DEMO_DB_PATH):
        os.remove(DEMO_DB_PATH)
    memory = RouteMemory(DEMO_DB_PATH)

    print("Simulated localization + memory demo")
    print("=" * 60)

    for frame_id, pose in enumerate(sensor.get_poses(), start=1):
        detection = HAZARDS_BY_FRAME[frame_id]
        localized = localize_detection(pose, detection)

        memory.remember(
            object_id=localized.object_id,
            label=localized.label,
            world_x=localized.world_x,
            world_y=localized.world_y,
            confidence=localized.confidence,
            timestamp=localized.timestamp,
        )

        print(f"\nFrame {frame_id}")
        print(f"  Robot pose:   x={pose.x:.2f}, y={pose.y:.2f}, heading={pose.heading_deg:.1f} deg")
        print(f"  Detected {localized.label!r}: local position = "
              f"x={detection.relative_x:.2f}, y={detection.relative_y:.2f}")
        print(f"  World position:            x={localized.world_x:.2f}, y={localized.world_y:.2f}")
        print(f"  -> saved to memory: {localized}")

    print("\n" + "=" * 60)
    print(f"Processed {len(sensor.get_poses())} frames.")
    print(f"\nPersistent memory now contains {len(memory.get_hazards())} hazards "
          f"(saved to {DEMO_DB_PATH}):")
    for hazard in memory.get_hazards():
        print(f"  {hazard.label:8s} object_id={hazard.object_id:10s} "
              f"world=({hazard.world_x:.2f}, {hazard.world_y:.2f}) "
              f"confidence={hazard.confidence:.2f}")


if __name__ == "__main__":
    main()
