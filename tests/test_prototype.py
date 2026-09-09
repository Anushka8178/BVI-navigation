import numpy as np

from navigation.hazard import (
    HazardSeverityClassifier,
)
from navigation.models import (
    Detection,
    FramePacket,
    Motion,
)
from navigation.pipeline import (
    NavigationPipeline,
)


def create_detection(
    distance: float,
    relative_x: float = 0.0,
    motion: Motion = Motion.STATIC,
) -> Detection:
    return Detection(
        object_id="person-1",
        label="person",
        relative_x=relative_x,
        relative_y=distance,
        confidence=0.90,
        motion=motion,
    )


def test_close_object_ahead_uses_safety_override() -> None:
    detection = create_detection(
        distance=2.0
    )

    classifier = HazardSeverityClassifier()

    result = classifier.classify(
        detection
    )

    assert result.urgent is True
    assert "override=True" in result.reason


def test_far_static_object_is_not_urgent() -> None:
    detection = create_detection(
        distance=10.0
    )

    classifier = HazardSeverityClassifier()

    result = classifier.classify(
        detection
    )

    assert result.urgent is False


def test_approaching_object_has_more_risk() -> None:
    classifier = HazardSeverityClassifier()

    static_detection = create_detection(
        distance=4.0,
        motion=Motion.STATIC,
    )

    approaching_detection = create_detection(
        distance=4.0,
        motion=Motion.APPROACHING,
    )

    static_result = classifier.classify(
        static_detection
    )

    approaching_result = classifier.classify(
        approaching_detection
    )

    assert (
        approaching_result.score
        > static_result.score
    )


class FakePerception:
    def process(
        self,
        frame: FramePacket,
    ) -> list[Detection]:
        return [
            create_detection(
                distance=2.0
            )
        ]


def test_pipeline_emits_warning() -> None:
    pipeline = NavigationPipeline(
        perception=FakePerception(),
        severity=HazardSeverityClassifier(),
    )

    # Tests must provide an actual image array,
    # not object().
    test_image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    frame = FramePacket(
        frame_id=1,
        timestamp=0.0,
        image=test_image,
    )

    result = pipeline.process_cycle(
        frame
    )

    assert result.detection_count == 1
    assert len(result.warnings) == 1
    assert result.warnings[0].label == "person"