from __future__ import annotations

import math

from .models import AudioEvent, Detection, HazardEntry, Pose


class ConsoleSpatialAudio:
    """Semantic audio adapter. Replace this class with a real HRTF renderer."""

    TIMBRES = {
        "person": "pulse",
        "pothole": "low-click",
        "stairs": "descending-tone",
        "door": "chime",
    }

    @staticmethod
    def _direction(azimuth_deg: float) -> str:
        if azimuth_deg < -15:
            return "left"
        if azimuth_deg > 15:
            return "right"
        return "ahead"

    @staticmethod
    def _distance_band(distance: float) -> str:
        if distance < 1.0:
            return "very-near"
        if distance < 2.5:
            return "near"
        return "far"

    def immediate(self, detection: Detection, score: float) -> AudioEvent:
        return AudioEvent(
            kind="immediate-bypass",
            label=detection.label,
            direction=self._direction(detection.azimuth_deg),
            distance_band=self._distance_band(detection.distance),
            urgency=f"critical:{score:.2f}",
            message=f"{self.TIMBRES.get(detection.label, 'beep')} fast repetition",
        )

    def awareness(self, entry: HazardEntry, pose: Pose) -> AudioEvent:
        dx, dy = entry.world_x - pose.x, entry.world_y - pose.y
        distance = math.hypot(dx, dy)
        world_bearing = math.degrees(math.atan2(dx, dy))
        azimuth = (world_bearing - pose.heading_deg + 180) % 360 - 180
        return AudioEvent(
            kind="ranked-awareness",
            label=entry.label,
            direction=self._direction(azimuth),
            distance_band=self._distance_band(distance),
            urgency=f"priority:{entry.priority:.2f}",
            message=self.TIMBRES.get(entry.label, "beep"),
        )
