import time

from navigation.pipeline import NavigationPipeline
from navigation.hazard import HazardSeverityClassifier
from navigation.models import Detection, FramePacket, Motion


class FakePerception:
    def __init__(self, relative_x):
        self.relative_x = relative_x
        self.last_result = None

    def process(self, frame):
        return [
            Detection(
                object_id="test-obstacle",
                label="person",
                relative_x=self.relative_x,
                relative_y=1.0,
                confidence=0.95,
                motion=Motion.APPROACHING,
            )
        ]


def test_direction(name, relative_x):
    print(f"\nTesting obstacle: {name}")

    pipeline = NavigationPipeline(
        perception=FakePerception(relative_x),
        severity=HazardSeverityClassifier(),
    )

    frame = FramePacket(
        frame_id=1,
        timestamp=0.0,
        image=None,
    )

    result = pipeline.process_cycle(frame)

    print("Warnings:", len(result.warnings))

    if result.warnings:
        warning = result.warnings[0]

        print("Direction:", warning.direction)
        print("Azimuth:", warning.azimuth_deg)
        print("Distance:", warning.distance_m)
        print("Reason:", warning.reason)


print("=== NAVIGATION + SPATIAL AUDIO TEST ===")

# LEFT
test_direction("LEFT", -0.5)
time.sleep(1)

# AHEAD
test_direction("AHEAD", 0.0)
time.sleep(1)

# RIGHT
test_direction("RIGHT", 0.5)
time.sleep(1)

print("\n=== TEST COMPLETE ===")