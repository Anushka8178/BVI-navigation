from __future__ import annotations

import math

from .memory import RouteMemory
from .models import HazardEntry, PlanDecision, Pose, RouteCandidate


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


class ExplainablePathPlanner:
    def __init__(self, route_memory: RouteMemory) -> None:
        self.route_memory = route_memory

    def choose(self, routes: list[RouteCandidate]) -> PlanDecision:
        scored: list[tuple[RouteCandidate, float, int, float]] = []
        for route in routes:
            hazard_count, remembered_risk = self.route_memory.hazard_risk(route.route_id)
            score = route.length_m / 100.0 + route.base_risk + 0.35 * remembered_risk
            scored.append((route, score, hazard_count, remembered_risk))
        scored.sort(key=lambda item: item[1])
        selected, score, count, memory_risk = scored[0]
        rationale = (
            f"Selected {selected.name}: combined cost {score:.2f}. It is {selected.length_m:.0f}m "
            f"with base risk {selected.base_risk:.2f} and {count} remembered hazard(s) "
            f"contributing {memory_risk:.2f} risk."
        )
        alternatives = [(route.name, value) for route, value, _, _ in scored[1:]]
        return PlanDecision(selected, score, rationale, alternatives)
