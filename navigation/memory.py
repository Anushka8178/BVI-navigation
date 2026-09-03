from __future__ import annotations

import math
import sqlite3
from pathlib import Path

from .models import Detection, HazardEntry, Motion, Pose


class SpatialMemory:
    """Session cache: world-anchored hazards with class-dependent decay."""

    TTL = {
        Motion.STATIC: 12.0,
        Motion.CROSSING: 4.0,
        Motion.APPROACHING: 2.5,
    }

    def __init__(self, prune_threshold: float = 0.08) -> None:
        self.entries: dict[str, HazardEntry] = {}
        self.prune_threshold = prune_threshold

    @staticmethod
    def relative_to_world(detection: Detection, pose: Pose) -> tuple[float, float]:
        theta = math.radians(pose.heading_deg)
        world_x = pose.x + detection.relative_x * math.cos(theta) - detection.relative_y * math.sin(theta)
        world_y = pose.y + detection.relative_x * math.sin(theta) + detection.relative_y * math.cos(theta)
        return world_x, world_y

    def update(self, detection: Detection, pose: Pose, now: float) -> None:
        if not pose.localization_valid:
            return
        world_x, world_y = self.relative_to_world(detection, pose)
        self.entries[detection.object_id] = HazardEntry(
            key=detection.object_id,
            label=detection.label,
            world_x=world_x,
            world_y=world_y,
            confidence=detection.confidence,
            last_seen=now,
            motion=detection.motion,
            decay_weight=detection.confidence,
        )

    def decay_and_prune(self, now: float) -> None:
        expired: list[str] = []
        for key, entry in self.entries.items():
            age = max(0.0, now - entry.last_seen)
            tau = self.TTL[entry.motion]
            entry.decay_weight = entry.confidence * math.exp(-age / tau)
            if entry.decay_weight < self.prune_threshold:
                expired.append(key)
        for key in expired:
            del self.entries[key]

    def active(self) -> list[HazardEntry]:
        return list(self.entries.values())


class RouteMemory:
    """Cross-session route and confirmed-hazard persistence."""

    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database_path)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS route_visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                route_id TEXT NOT NULL,
                visited_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS route_hazards (
                route_id TEXT NOT NULL,
                hazard_key TEXT NOT NULL,
                label TEXT NOT NULL,
                world_x REAL NOT NULL,
                world_y REAL NOT NULL,
                confidence REAL NOT NULL,
                observations INTEGER NOT NULL DEFAULT 1,
                last_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (route_id, hazard_key)
            );
            """
        )
        self.connection.commit()

    def record_visit(self, route_id: str) -> None:
        self.connection.execute("INSERT INTO route_visits(route_id) VALUES (?)", (route_id,))
        self.connection.commit()

    def remember_hazard(self, route_id: str, entry: HazardEntry) -> None:
        self.connection.execute(
            """
            INSERT INTO route_hazards(route_id, hazard_key, label, world_x, world_y, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(route_id, hazard_key) DO UPDATE SET
                label=excluded.label,
                world_x=excluded.world_x,
                world_y=excluded.world_y,
                confidence=MAX(route_hazards.confidence, excluded.confidence),
                observations=route_hazards.observations + 1,
                last_seen=CURRENT_TIMESTAMP
            """,
            (route_id, entry.key, entry.label, entry.world_x, entry.world_y, entry.confidence),
        )
        self.connection.commit()

    def hazard_risk(self, route_id: str) -> tuple[int, float]:
        row = self.connection.execute(
            """
            SELECT COUNT(*) AS hazard_count,
                   COALESCE(SUM(confidence * MIN(observations, 3) / 3.0), 0) AS risk
            FROM route_hazards WHERE route_id = ?
            """,
            (route_id,),
        ).fetchone()
        return int(row["hazard_count"]), float(row["risk"])

    def close(self) -> None:
        self.connection.close()
