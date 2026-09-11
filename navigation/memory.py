from __future__ import annotations

import contextlib
import sqlite3
import time
from dataclasses import dataclass


DEFAULT_HALF_LIFE_S = 30.0
DEFAULT_MIN_DECAY_WEIGHT = 0.05
DEFAULT_MAX_ENTRIES = 500


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


class RouteMemory:
    """Persistent hazard memory backed by SQLite.

    Two forgetting mechanisms keep this table from growing without bound
    and keep stale hazards from permanently outranking fresh ones:

    - Time decay: every time memory is touched, each hazard's
      `decay_weight` is recomputed as an exponential function of how long
      it has been since that hazard was last (re)observed. `decay_weight`
      halves every `decay_half_life_s` seconds of "silence" on that
      object, then feeds straight into `AcousticAttention.rank()`'s
      recency term (navigation/reasoning.py), so long-unseen hazards
      naturally sink in the ranking instead of sitting at a fixed 1.0
      forever.
    - Pruning: once a hazard's decay_weight drops below
      `min_decay_weight` it is treated as forgotten and deleted.
      Independently, if the table still holds more than `max_entries`
      rows (e.g. a long walk with many distinct tracked objects), the
      lowest-ranked surviving rows (decay_weight * confidence) are
      evicted until the cap is respected.

    "Now" for decay purposes is whatever clock the caller's timestamps
    already use -- wall-clock seconds (`remember()`'s default) or a
    video's relative playback seconds, as in the SLAM-memory demos. By
    default each decay pass uses the newest timestamp already stored in
    memory as "now", so both clocks work with no extra configuration.
    """

    def __init__(
        self,
        db_path: str = "navigation_memory.db",
        decay_half_life_s: float = DEFAULT_HALF_LIFE_S,
        min_decay_weight: float = DEFAULT_MIN_DECAY_WEIGHT,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        self.db_path = db_path
        self.decay_half_life_s = decay_half_life_s
        self.min_decay_weight = min_decay_weight
        self.max_entries = max_entries
        self._create_table()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _create_table(self) -> None:
        with contextlib.closing(self._connect()) as connection, connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS hazards (
                    object_id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    world_x REAL NOT NULL,
                    world_y REAL NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    decay_weight REAL NOT NULL DEFAULT 1.0
                )
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
    ) -> HazardEntry:
        """Save a hazard in persistent world coordinates.

        A freshly (re)observed hazard is written with a full decay_weight
        of 1.0 -- it was just seen, so nothing has faded yet. Every call
        also ages and prunes the rest of memory relative to this
        timestamp, so the table stays bounded even under continuous use.
        """

        if timestamp is None:
            timestamp = time.time()

        decay_weight = 1.0

        with contextlib.closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO hazards (
                    object_id,
                    label,
                    world_x,
                    world_y,
                    confidence,
                    timestamp,
                    decay_weight
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(object_id) DO UPDATE SET
                    label = excluded.label,
                    world_x = excluded.world_x,
                    world_y = excluded.world_y,
                    confidence = excluded.confidence,
                    timestamp = excluded.timestamp,
                    decay_weight = excluded.decay_weight
                """,
                (
                    object_id,
                    label,
                    world_x,
                    world_y,
                    confidence,
                    timestamp,
                    decay_weight,
                ),
            )

        self.decay_and_prune(now=timestamp)

        return HazardEntry(
            object_id=object_id,
            label=label,
            world_x=world_x,
            world_y=world_y,
            confidence=confidence,
            timestamp=timestamp,
            decay_weight=decay_weight,
        )

    def decay_and_prune(self, now: float | None = None) -> int:
        """Age every hazard's decay_weight, forget faded-out hazards, and
        cap total memory size.

        Safe to call as often as you like -- e.g. once per frame, or
        implicitly from `get_hazards()` -- since with no new detections
        it is a cheap no-op beyond the decay_weight refresh.

        Returns the number of rows removed (decayed-out + pruned).
        """
        with contextlib.closing(self._connect()) as connection, connection:
            if now is None:
                row = connection.execute("SELECT MAX(timestamp) FROM hazards").fetchone()
                now = row[0] if row and row[0] is not None else 0.0

            rows = connection.execute("SELECT object_id, timestamp FROM hazards").fetchall()
            for object_id, entry_timestamp in rows:
                elapsed = max(now - entry_timestamp, 0.0)
                if self.decay_half_life_s > 0:
                    decay_weight = 0.5 ** (elapsed / self.decay_half_life_s)
                else:
                    decay_weight = 1.0 if elapsed <= 0 else 0.0
                connection.execute(
                    "UPDATE hazards SET decay_weight = ? WHERE object_id = ?",
                    (decay_weight, object_id),
                )

            removed = connection.execute(
                "DELETE FROM hazards WHERE decay_weight < ?",
                (self.min_decay_weight,),
            ).rowcount
            removed = max(removed, 0)

            remaining = connection.execute("SELECT COUNT(*) FROM hazards").fetchone()[0]
            overflow = remaining - self.max_entries
            if overflow > 0:
                stale_rows = connection.execute(
                    """
                    SELECT object_id FROM hazards
                    ORDER BY (decay_weight * confidence) ASC, timestamp ASC
                    LIMIT ?
                    """,
                    (overflow,),
                ).fetchall()
                connection.executemany(
                    "DELETE FROM hazards WHERE object_id = ?", stale_rows
                )
                removed += len(stale_rows)

        return removed

    def get_hazards(self) -> list[HazardEntry]:
        """Load all remembered hazards from SQLite.

        Runs a decay/prune pass first (using the latest stored timestamp
        as "now"), so ranking always sees up-to-date decay_weight values
        even if nothing new has been remembered since the last read.
        """

        self.decay_and_prune()

        with contextlib.closing(self._connect()) as connection, connection:
            rows = connection.execute(
                """
                SELECT
                    object_id,
                    label,
                    world_x,
                    world_y,
                    confidence,
                    timestamp,
                    decay_weight
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
            )
            for row in rows
        ]

    def clear(self) -> None:
        """Delete all remembered hazards. Mainly useful for tests."""

        with contextlib.closing(self._connect()) as connection, connection:
            connection.execute("DELETE FROM hazards")
