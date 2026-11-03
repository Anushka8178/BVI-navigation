import time

from navigation.pipeline import NavigationPipeline
from navigation.hazard import HazardSeverityClassifier
from navigation.models import Detection, FramePacket, Motion


class FakePerception:
    def __init__(self, relative_x):
        self.relative_x = relative_x

    def process(self, frame):
        return [
            Detection(
                object_id="video-obstacle",
                label="person",
                relative_x=self.relative_x,
                relative_y=1.0,
                confidence=0.95,
                motion=Motion.APPROACHING,
            )
        ]


def play_position(name, relative_x):
    print(f"\n🎧 Testing {name}")

    pipeline = NavigationPipeline(
        perception=FakePerception(relative_x),
        severity=HazardSeverityClassifier(),
    )

    frame = FramePacket(
        frame_id=1,
        timestamp=time.time(),
        image=None,
    )

    result = pipeline.process_cycle(frame)

    if result.warnings:
        warning = result.warnings[0]

        print(
            f"Direction: {warning.direction.upper()}"
        )

        print(
            f"Azimuth: "
            f"{warning.azimuth_deg:+.1f}°"
        )

        print(
            f"Distance: "
            f"{warning.distance_m:.2f} m"
        )

    time.sleep(1)


print("\n=== FAST VIDEO → HRTF TEST ===")

play_position("LEFT", -0.5)
play_position("AHEAD", 0.0)
play_position("RIGHT", 0.5)
play_position("AHEAD", 0.0)

print("\n=== TEST COMPLETE ===")