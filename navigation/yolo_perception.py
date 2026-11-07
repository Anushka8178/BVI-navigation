from __future__ import annotations

from collections import defaultdict
from math import cos, radians, sin
from pathlib import Path
from typing import Any

from .models import Detection, FramePacket, Motion

DEFAULT_GROUND_LABELS = frozenset({"pothole", "stairs", "curb", "drain"})


def ground_plane_distance(
    bottom_y: float,
    image_height: int,
    camera_height_m: float,
    focal_pixels: float,
    min_m: float = 0.5,
    max_m: float = 8.0,
) -> float:
    """Distance to a point on flat ground from its image row.

    Assumes a level camera whose optical axis meets the horizon at the
    image centre row. A ground point imaged `dy` pixels below the centre
    is at distance camera_height * focal / dy. Rows at or above the
    horizon are treated as far away (max_m). Result clamped to
    [min_m, max_m].
    """
    dy = bottom_y - image_height / 2.0
    if dy <= 0.0:
        return max_m
    distance = camera_height_m * focal_pixels / dy
    return min(max(distance, min_m), max_m)


class YOLOPerception:
    """
    Runs YOLO object detection and tracking.

    It also estimates:

    - approximate object distance
    - horizontal direction
    - whether the object is approaching

    Distance is only a monocular approximation. It is not measured depth.
    """

    ASSUMED_HEIGHT_M = {
        "stop_sign": 2.00,
        "person": 1.70,
        "bicycle": 1.10,
        "bus": 3.20,
        "truck": 3.00,
        "car": 1.50,
        "motorbike": 1.10,
        "reflective_cone": 0.70,
        "ashcan": 0.90,
        "trashcan": 0.90,
        "warning_column": 0.90,
        "spherical_roadblock": 0.50,
        "pole": 2.00,
        "dog": 0.60,
        "tricycle": 1.20,
        "fire_hydrant": 0.75,
        "tree": 3.00,
        "autorickshaw": 1.70,
        "animal": 1.00,
        "barrier": 1.00,
    }

    def __init__(
        self,
        weights: str | Path,
        confidence: float = 0.35,
        horizontal_fov_deg: float = 70.0,
        device: str = "0",
        camera_height_m: float = 1.3,
        ground_labels: frozenset[str] | set[str] = DEFAULT_GROUND_LABELS,
        ground_min_m: float = 0.5,
        ground_max_m: float = 8.0,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is not installed. "
                "Run: pip install -r requirements.txt"
            ) from error

        weights_path = Path(weights)

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Trained model does not exist: {weights_path.resolve()}"
            )

        self.model = YOLO(str(weights_path))
        self.confidence = confidence
        self.horizontal_fov_deg = horizontal_fov_deg
        self.device = device
        self.camera_height_m = camera_height_m
        self.ground_labels = frozenset(ground_labels)
        self.ground_min_m = ground_min_m
        self.ground_max_m = ground_max_m

        # Stores the previous distance of each tracked object.
        self.previous_distances: dict[str, float] = {}

        # Used only when YOLO does not provide a tracking ID.
        self.fallback_ids: defaultdict[str, int] = defaultdict(int)

        # Stores the most recent Ultralytics result for drawing boxes.
        self.last_result: Any | None = None

    def process(self, frame: FramePacket) -> list[Detection]:
        results = self.model.track(
            source=frame.image,
            persist=True,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        if not results:
            self.last_result = None
            return []

        result = results[0]
        self.last_result = result

        image_height, image_width = frame.image.shape[:2]
        detections: list[Detection] = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_index = int(box.cls.item())
            label = str(result.names[class_index])
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = [
                float(value)
                for value in box.xyxy[0].tolist()
            ]

            pixel_height = max(y2 - y1, 1.0)
            center_x = (x1 + x2) / 2.0

            if box.id is not None:
                track_id = int(box.id.item())
            else:
                track_id = self._fallback_id(label)

            object_id = f"{label}-{track_id}"

            ground_level = label in self.ground_labels

            if ground_level:
                # Box height says little about a pothole or curb; use
                # where its bottom edge meets the ground instead.
                distance = ground_plane_distance(
                    bottom_y=y2,
                    image_height=image_height,
                    camera_height_m=self.camera_height_m,
                    focal_pixels=self._focal_pixels(image_height),
                    min_m=self.ground_min_m,
                    max_m=self.ground_max_m,
                )
            else:
                distance = self._estimate_distance(
                    label=label,
                    pixel_height=pixel_height,
                    image_height=image_height,
                )

            angle_deg = (
                (center_x / image_width) - 0.5
            ) * self.horizontal_fov_deg

            angle_rad = radians(angle_deg)

            relative_x = sin(angle_rad) * distance
            relative_y = cos(angle_rad) * distance

            motion = self._motion(
                object_id=object_id,
                distance=distance,
            )

            detection = Detection(
                object_id=object_id,
                label=label,
                relative_x=relative_x,
                relative_y=relative_y,
                confidence=confidence,
                motion=motion,
                ground_level=ground_level,
            )

            detections.append(detection)

        return detections

    def _estimate_distance(
        self,
        label: str,
        pixel_height: float,
        image_height: int,
    ) -> float:
        assumed_height = self.ASSUMED_HEIGHT_M.get(label, 1.0)

        focal_pixels = self._focal_pixels(image_height)

        estimated_distance = (
            focal_pixels * assumed_height
        ) / pixel_height

        # Prevent impossible or extreme values.
        return min(max(estimated_distance, 0.35), 20.0)

    @staticmethod
    def _focal_pixels(image_height: int) -> float:
        # Approximate vertical focal length in pixels. This must
        # eventually be replaced by proper camera calibration.
        return image_height * 0.95

    def _motion(
        self,
        object_id: str,
        distance: float,
    ) -> Motion:
        previous_distance = self.previous_distances.get(object_id)

        self.previous_distances[object_id] = distance

        if previous_distance is None:
            return Motion.STATIC

        distance_reduction = previous_distance - distance

        required_reduction = max(
            0.02,
            previous_distance * 0.01,
        )

        if distance_reduction > required_reduction:
            return Motion.APPROACHING

        return Motion.STATIC

    def _fallback_id(self, label: str) -> int:
        self.fallback_ids[label] += 1
        return self.fallback_ids[label]