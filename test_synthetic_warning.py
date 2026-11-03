from __future__ import annotations

import time

from navigation.models import Detection, Motion
from navigation.hazard import HazardSeverityClassifier
from navigation.hrtf_renderer import HRTFRenderer


def create_detection(
    object_id: int,
    label: str,
    distance: float,
    azimuth_deg: float,
    motion: Motion,
) -> Detection:
    """
    Creates a synthetic detection.

    relative_x and relative_y are selected so that:
        distance = sqrt(relative_x**2 + relative_y**2)
        azimuth = atan2(relative_x, relative_y)
    """

    from math import radians, sin, cos

    angle_rad = radians(azimuth_deg)

    relative_x = distance * sin(angle_rad)
    relative_y = distance * cos(angle_rad)

    return Detection(
        object_id=object_id,
        label=label,
        relative_x=relative_x,
        relative_y=relative_y,
        confidence=0.95,
        motion=motion,
    )


def run_test(
    name: str,
    detection: Detection,
    classifier: HazardSeverityClassifier,
    renderer: HRTFRenderer,
) -> None:
    result = classifier.classify(detection)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print(f"Object       : {detection.label}")
    print(f"Distance     : {detection.distance:.2f} m")
    print(f"Azimuth      : {detection.azimuth_deg:+.1f} degrees")
    print(f"Motion       : {detection.motion.value}")
    print(f"Hazard score : {result.score:.2f}")
    print(f"Urgent       : {result.urgent}")
    print(f"Reason       : {result.reason}")

    if result.urgent:
        print("ACTION       : Playing HRTF warning...")
        renderer.play_warning(detection.azimuth_deg)
        time.sleep(3)
    else:
        print("ACTION       : No warning played.")


def main() -> None:
    classifier = HazardSeverityClassifier()
    renderer = HRTFRenderer()

    print("Synthetic YOLO-HRTF warning test")
    print("Use headphones for the best directional effect.")

    # Test 1: Very close obstacle directly ahead.
    center_detection = create_detection(
        object_id=1,
        label="person",
        distance=1.0,
        azimuth_deg=0.0,
        motion=Motion.APPROACHING,
    )

    # Test 2: Very close obstacle on the left.
    left_detection = create_detection(
        object_id=2,
        label="bicycle",
        distance=0.5,
        azimuth_deg=-30.0,
        motion=Motion.APPROACHING,
    )

    # Test 3: Very close obstacle on the right.
    right_detection = create_detection(
        object_id=3,
        label="traffic_cone",
        distance=0.5,
        azimuth_deg=30.0,
        motion=Motion.APPROACHING,
    )

    run_test(
        "TEST 1: CLOSE APPROACHING OBSTACLE AHEAD",
        center_detection,
        classifier,
        renderer,
    )

    run_test(
        "TEST 2: CLOSE APPROACHING OBSTACLE ON THE LEFT",
        left_detection,
        classifier,
        renderer,
    )

    run_test(
        "TEST 3: CLOSE APPROACHING OBSTACLE ON THE RIGHT",
        right_detection,
        classifier,
        renderer,
    )

    print("\n" + "=" * 60)
    print("Synthetic warning test completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()