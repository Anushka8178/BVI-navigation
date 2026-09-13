from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass


@dataclass
class HazardEntry:
    """A hazard remembered in fixed world coordinates."""
    object_id: str
    label: str
    world_x: float
    world_y: float
    confidence: float
    timestamp: float
    decay_weight: float = 1.0
    priority: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0


class RouteMemory:
    """Persistent hazard memory backed by SQLite."""

    def __init__(self, db_path: str = "navigation_memory.db") -> None:
        self.db_path = db_path
        self._create_table()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _create_table(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS hazards (
                    object_id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    world_x REAL NOT NULL,
                    world_y REAL NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    decay_weight REAL NOT NULL DEFAULT 1.0,
                    velocity_x REAL NOT NULL DEFAULT 0.0,
                    velocity_y REAL NOT NULL DEFAULT 0.0
                )
                """
            )

            # Migrate databases created by the previous version.
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(hazards)"
                ).fetchall()
            }

            if "velocity_x" not in columns:
                connection.execute(
                    """
                    ALTER TABLE hazards
                    ADD COLUMN velocity_x REAL NOT NULL DEFAULT 0.0
                    """
                )

            if "velocity_y" not in columns:
                connection.execute(
                    """
                    ALTER TABLE hazards
                    ADD COLUMN velocity_y REAL NOT NULL DEFAULT 0.0
                    """
                )

    def remember(
        self,
        object_id: str,
        label: str,
        world_x: float,
        world_y: float,
        confidence: float,
        timestamp: float | None = None,
        velocity_x: float = 0.0,
        velocity_y: float = 0.0,
    ) -> HazardEntry:
        """Save/update a hazard and its current world-frame velocity."""

        if timestamp is None:
            timestamp = time.time()

        decay_weight = 1.0

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO hazards (
                    object_id,
                    label,
                    world_x,
                    world_y,
                    confidence,
                    timestamp,
                    decay_weight,
                    velocity_x,
                    velocity_y
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(object_id) DO UPDATE SET
                    label = excluded.label,
                    world_x = excluded.world_x,
                    world_y = excluded.world_y,
                    confidence = excluded.confidence,
                    timestamp = excluded.timestamp,
                    decay_weight = excluded.decay_weight,
                    velocity_x = excluded.velocity_x,
                    velocity_y = excluded.velocity_y
                """,
                (
                    object_id,
                    label,
                    world_x,
                    world_y,
                    confidence,
                    timestamp,
                    decay_weight,
                    velocity_x,
                    velocity_y,
                ),
            )

        return HazardEntry(
            object_id=object_id,
            label=label,
            world_x=world_x,
            world_y=world_y,
            confidence=confidence,
            timestamp=timestamp,
            decay_weight=decay_weight,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
        )

    def get_hazards(self) -> list[HazardEntry]:
        """Load all remembered hazards from SQLite."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    object_id,
                    label,
                    world_x,
                    world_y,
                    confidence,
                    timestamp,
                    decay_weight,
                    velocity_x,
                    velocity_y
                FROM hazards
                ORDER BY timestamp DESC
                """
            ).fetchall()

        return [
            HazardEntry(
                object_id=row[0],
                label=row[1],
                world_x=row[2],
                world_y=row[3],
                confidence=row[4],
                timestamp=row[5],
                decay_weight=row[6],
                velocity_x=row[7],
                velocity_y=row[8],
            )
            for row in rows
        ]

    def clear(self) -> None:
        """Delete all remembered hazards. Mainly useful for tests."""
        with self._connect() as connection:
            connection.execute("DELETE FROM hazards")
