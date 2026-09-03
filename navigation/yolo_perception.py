from __future__ import annotations

from collections import defaultdict
from math import tan, radians
from pathlib import Path

from .models import Detection, FramePacket, Motion


class YOLOPerception:
    """Real Ultralytics detector adapter with lightweight monocular ranging.

    The range is an approximation until RGB-D/SLAM is connected. The adapter's
    output contract is identical to ScriptedPerception, so downstream modules
    remain unchanged.
    """

    ASSUMED_HEIGHT_M = {
        "person": 1.70,
        "bicycle": 1.10,
        "bus": 3.20,
        "truck": 3.00,
        "car": 1.50,
        "motorbike": 1.10,
        "reflective_cone": 0.70,
        "ashcan": 0.90,
        "warning_column": 0.90,
        "spherical_roadblock": 0.50,
        "pole": 2.00,
        "dog": 0.60,
        "tricycle": 1.20,
        "fire_hydrant": 0.75,
        "stop_sign": 2.00,
    }

    def __init__(
        self,
        weights: str | Path,
        confidence: float = 0.35,
        horizontal_fov_deg: float = 70.0,
        device: int | str = 0,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run: pip install -r requirements-ml.txt"
            ) from exc
        self.model = YOLO(str(weights))
        self.confidence = confidence
        self.horizontal_fov_deg = horizontal_fov_deg
        self.device = device
        self.previous_distances: dict[str, float] = {}
        self.fallback_ids: defaultdict[str, int] = defaultdict(int)
        self.last_result = None

    def process(self, frame: FramePacket) -> list[Detection]:
        if frame.image is None:
            raise ValueError("YOLOPerception requires FramePacket.image")
        result = self.model.track(
            frame.image,
            persist=True,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )[0]
        self.last_result = result
        image_height, image_width = frame.image.shape[:2]
        detections: list[Detection] = []
        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_index = int(box.cls.item())
            label = str(result.names[class_index])
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            pixel_height = max(y2 - y1, 1.0)
            center_x = (x1 + x2) / 2.0
            track_id = int(box.id.item()) if box.id is not None else self._fallback_id(label)
            object_id = f"{label}-{track_id}"

            distance = self._estimate_distance(label, pixel_height, image_height)
            angle_deg = ((center_x / image_width) - 0.5) * self.horizontal_fov_deg
            relative_x = tan(radians(angle_deg)) * distance
            motion = self._motion(object_id, distance)
            detections.append(
                Detection(
                    object_id=object_id,
                    label=label,
                    relative_x=relative_x,
                    relative_y=distance,
                    confidence=confidence,
                    motion=motion,
                )
            )
        return detections

    def _estimate_distance(self, label: str, pixel_height: float, image_height: int) -> float:
        assumed_height = self.ASSUMED_HEIGHT_M.get(label, 1.0)
        focal_pixels = image_height * 0.95
        return min(max((focal_pixels * assumed_height) / pixel_height, 0.35), 20.0)

    def _motion(self, object_id: str, distance: float) -> Motion:
        previous = self.previous_distances.get(object_id)
        self.previous_distances[object_id] = distance
        if previous is not None and previous - distance > 0.25:
            return Motion.APPROACHING
        return Motion.STATIC

    def _fallback_id(self, label: str) -> int:
        self.fallback_ids[label] += 1
        return self.fallback_ids[label]
