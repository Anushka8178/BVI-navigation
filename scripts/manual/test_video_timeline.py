import cv2
import time

from navigation.pipeline import NavigationPipeline
from navigation.hazard import HazardSeverityClassifier
from navigation.models import Detection, FramePacket, Motion


VIDEO_PATH = r"videos\hrtf_navigation_test_video_compatible.mp4"


class FakeVideoPerception:
    def __init__(self):
        self.position = -0.5

    def set_position(self, position):
        self.position = position

    def process(self, frame):
        return [
            Detection(
                object_id="video-obstacle",
                label="person",
                relative_x=self.position,
                relative_y=1.0,
                confidence=0.95,
                motion=Motion.APPROACHING,
            )
        ]


cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video")


fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

duration = frame_count / fps

print(f"Video FPS: {fps:.2f}")
print(f"Video duration: {duration:.2f} seconds")


perception = FakeVideoPerception()

pipeline = NavigationPipeline(
    perception=perception,
    severity=HazardSeverityClassifier(),
)


positions = [
    ("LEFT", -0.5),
    ("AHEAD", 0.0),
    ("RIGHT", 0.5),
    ("AHEAD", 0.0),
]


for name, position in positions:

    print(f"\n🎧 Video event: {name}")

    perception.set_position(position)

    success, frame = cap.read()

    if not success:
        break

    packet = FramePacket(
        frame_id=int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
        timestamp=time.time(),
        image=frame,
    )

    result = pipeline.process_cycle(packet)

    if result.warnings:

        warning = result.warnings[0]

        print(
            f"Direction: {warning.direction.upper()}"
        )

        print(
            f"Azimuth: "
            f"{warning.azimuth_deg:+.1f}°"
        )

    time.sleep(1)


cap.release()

print("\n=== VIDEO TIMELINE TEST COMPLETE ===")