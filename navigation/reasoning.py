from __future__ import annotations

import math
from .models import Pose
from .memory import HazardEntry


class AcousticAttention:
    def __init__(self, top_k: int = 3, weights: tuple[float, float, float] = (0.45, 0.25, 0.30)) -> None:
        self.top_k = top_k
        self.w_distance, self.w_alignment, self.w_recency = weights

    def rank(self, entries: list[HazardEntry], pose: Pose) -> list[HazardEntry]:
        heading = math.radians(pose.heading_deg)
        forward = (math.sin(heading), math.cos(heading))
        for entry in entries:
            dx, dy = entry.world_x - pose.x, entry.world_y - pose.y
            distance = math.hypot(dx, dy)
            alignment = 0.0 if distance == 0 else max((dx * forward[0] + dy * forward[1]) / distance, 0.0)
            entry.priority = (
                self.w_distance * (1.0 / (1.0 + distance))
                + self.w_alignment * alignment
                + self.w_recency * entry.decay_weight
            )
        return sorted(entries, key=lambda item: item.priority, reverse=True)[: self.top_k]

