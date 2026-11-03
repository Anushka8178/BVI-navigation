from __future__ import annotations

import contextlib
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass


DEFAULT_HALF_LIFE_S = 30.0
DEFAULT_MIN_DECAY_WEIGHT = 0.05
DEFAULT_MAX_ENTRIES = 500
DEFAULT_MATCH_RADIUS_M = 1.0
# Labels that can move on their own: matched against the position
# predicted from their stored velocity, not their last-seen position.
DEFAULT_DYNAMIC_LABELS = frozenset(
    {"person", "dog", "bicycle", "car", "bus", "truck", "motorbike", "tricycle"}
)


@dataclass
class HazardEntry:
    """A hazard remembered in fixed world coordinates.

    `hazard_id` is the stable key (uuid4) that survives across runs.
    `object_id` is only the most recent YOLO track id ("label-N") that
    observed it; track ids restart every run, so it is a hint, not a key.
    """

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
    hazard_id: str = ""


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

    Identity: each stored hazard has a uuid4 `hazard_id`. A new detection
    is resolved to a hazard in this order:

    1. Same YOLO track id seen earlier *in this run* -> same hazard.
    2. Nearest stored hazard with the same label within `match_radius_m`
       (dynamic labels use the position predicted from their velocity).
    3. Otherwise a new hazard.
    """

    _CREATE_SQL = """
        CREATE TABLE IF NOT EXISTS hazards (
            hazard_id TEXT PRIMARY KEY,
            object_id TEXT NOT NULL,
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

    def __init__(
        self,
        db_path: str = "navigation_memory.db",
        decay_half_life_s: float = DEFAULT_HALF_LIFE_S,
        min_decay_weight: float = DEFAULT_MIN_DECAY_WEIGHT,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        match_radius_m: float = DEFAULT_MATCH_RADIUS_M,
        dynamic_labels: frozenset[str] | set[str] = DEFAULT_DYNAMIC_LABELS,
    ) -> None:
        self.db_path = db_path
        self.decay_half_life_s = decay_half_life_s
        self.min_decay_weight = min_decay_weight
        self.max_entries = max_entries
        self.match_radius_m = match_radius_m
        self.dynamic_labels = frozenset(dynamic_labels)
        # Track id -> hazard_id, for this run only (YOLO ids restart per run).
        self._track_to_hazard: dict[str, str] = {}
        self._create_table()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _create_table(self) -> None:
        with contextlib.closing(self._connect()) as connection, connection:
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(hazards)"
                ).fetchall()
            }

            if columns and "hazard_id" not in columns:
                self._migrate_legacy_table(connection, columns)
            else:
                connection.execute(self._CREATE_SQL)

            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_hazards_label ON hazards(label)"
            )

    def _migrate_legacy_table(
        self, connection: sqlite3.Connection, columns: set[str]
    ) -> None:
        """Copy an old object_id-keyed table into the hazard_id schema.

        SQLite cannot change a primary key in place, so the old table is
        renamed, copied row by row (each row gets a fresh uuid4) and then
        dropped. This runs inside the caller's transaction, so a failure
        rolls back and leaves the old database untouched.
        """
        connection.execute("ALTER TABLE hazards RENAME TO hazards_legacy")
        connection.execute(self._CREATE_SQL)

        decay = "decay_weight" if "decay_weight" in columns else "1.0"
        velocity_x = "velocity_x" if "velocity_x" in columns else "0.0"
        velocity_y = "velocity_y" if "velocity_y" in columns else "0.0"
        rows = connection.execute(
            f"""
            SELECT object_id, label, world_x, world_y, confidence,
                   timestamp, {decay}, {velocity_x}, {velocity_y}
            FROM hazards_legacy
            """
        ).fetchall()
        connection.executemany(
            """
            INSERT INTO hazards (
                hazard_id, object_id, label, world_x, world_y, confidence,
                timestamp, decay_weight, velocity_x, velocity_y
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [(str(uuid.uuid4()), *row) for row in rows],
        )
        connection.execute("DROP TABLE hazards_legacy")

    def _match_hazard(
        self,
        connection: sqlite3.Connection,
        object_id: str,
        label: str,
        world_x: float,
        world_y: float,
        timestamp: float,
    ) -> str | None:
        """Return the hazard_id this detection belongs to, or None if new."""

        # 1. Track hint: same YOLO track earlier in this run.
        hinted = self._track_to_hazard.get(object_id)
        if hinted is not None:
            row = connection.execute(
                "SELECT label FROM hazards WHERE hazard_id = ?", (hinted,)
            ).fetchone()
            if row is not None and row[0] == label:
                return hinted

        # 2. Gated nearest neighbour among hazards with the same label.
        candidates = connection.execute(
            """
            SELECT hazard_id, object_id, world_x, world_y, timestamp,
                   velocity_x, velocity_y
            FROM hazards WHERE label = ?
            """,
            (label,),
        ).fetchall()

        best_id = None
        best_distance = self.match_radius_m
        for hazard_id, stored_object_id, x, y, seen_at, vx, vy in candidates:
            # Already updated this frame by another detection: one
            # detection per hazard per frame keeps nearby objects apart.
            if seen_at >= timestamp and stored_object_id != object_id:
                continue

            if label in self.dynamic_labels:
                dt = max(timestamp - seen_at, 0.0)
                x += vx * dt
                y += vy * dt

            distance = math.hypot(world_x - x, world_y - y)
            if distance <= best_distance:
                best_id = hazard_id
                best_distance = distance

        return best_id

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
        """Save a detection, merging it into an existing hazard if it matches.

        `object_id` is the YOLO track id ("label-N"). It is used as a hint
        so the same track keeps the same hazard, but never as the key. The
        returned entry carries the stable `hazard_id`, and its `object_id`
        is the one passed in, so callers can still link it to this frame's
        detection.

        A freshly (re)observed hazard is written with a full decay_weight
        of 1.0. Every call also ages and prunes the rest of memory relative
        to this timestamp, so the table stays bounded.
        """

        if timestamp is None:
            timestamp = time.time()

        decay_weight = 1.0

        with contextlib.closing(self._connect()) as connection, connection:
            hazard_id = self._match_hazard(
                connection, object_id, label, world_x, world_y, timestamp
            )
            if hazard_id is None:
                hazard_id = str(uuid.uuid4())

            connection.execute(
                """
                INSERT INTO hazards (
                    hazard_id,
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
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(hazard_id) DO UPDATE SET
                    object_id = excluded.object_id,
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
                    hazard_id,
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

        self._track_to_hazard[object_id] = hazard_id
        self.decay_and_prune(now=timestamp)

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
            hazard_id=hazard_id,
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

            rows = connection.execute("SELECT hazard_id, timestamp FROM hazards").fetchall()
            for hazard_id, entry_timestamp in rows:
                elapsed = max(now - entry_timestamp, 0.0)
                if self.decay_half_life_s > 0:
                    decay_weight = 0.5 ** (elapsed / self.decay_half_life_s)
                else:
                    decay_weight = 1.0 if elapsed <= 0 else 0.0
                connection.execute(
                    "UPDATE hazards SET decay_weight = ? WHERE hazard_id = ?",
                    (decay_weight, hazard_id),
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
                    SELECT hazard_id FROM hazards
                    ORDER BY (decay_weight * confidence) ASC, timestamp ASC
                    LIMIT ?
                    """,
                    (overflow,),
                ).fetchall()
                connection.executemany(
                    "DELETE FROM hazards WHERE hazard_id = ?", stale_rows
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
                    decay_weight,
                    velocity_x,
                    velocity_y,
                    hazard_id
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
                hazard_id=row[9],
            )
            for row in rows
        ]

    def clear(self) -> None:
        """Delete all remembered hazards. Mainly useful for tests."""

        with contextlib.closing(self._connect()) as connection, connection:
            connection.execute("DELETE FROM hazards")
        self._track_to_hazard.clear()
