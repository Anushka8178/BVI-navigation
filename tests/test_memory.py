import os
import sqlite3
import tempfile
import unittest

from navigation.memory import RouteMemory


class TestRouteMemoryDecay(unittest.TestCase):
    def setUp(self) -> None:
        handle, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        os.remove(self.db_path)

    def tearDown(self) -> None:
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_fresh_hazard_has_full_decay_weight(self) -> None:
        memory = RouteMemory(self.db_path, decay_half_life_s=10.0)

        memory.remember("pole-1", "pole", 1.0, 0.0, 0.9, timestamp=0.0)
        hazards = memory.get_hazards()

        self.assertEqual(len(hazards), 1)
        self.assertAlmostEqual(hazards[0].decay_weight, 1.0)

    def test_decay_weight_halves_after_one_half_life(self) -> None:
        memory = RouteMemory(self.db_path, decay_half_life_s=10.0, min_decay_weight=0.0)

        memory.remember("pole-1", "pole", 1.0, 0.0, 0.9, timestamp=0.0)
        # A second, unrelated hazard advances memory's notion of "now"
        # to t=10s (one half-life later) without touching pole-1.
        memory.remember("curb-2", "curb", 2.0, 0.0, 0.9, timestamp=10.0)

        hazards = {h.object_id: h for h in memory.get_hazards()}

        self.assertAlmostEqual(hazards["pole-1"].decay_weight, 0.5, places=3)
        self.assertAlmostEqual(hazards["curb-2"].decay_weight, 1.0, places=3)

    def test_reobserving_a_hazard_refreshes_its_decay_weight(self) -> None:
        memory = RouteMemory(self.db_path, decay_half_life_s=10.0, min_decay_weight=0.0)

        memory.remember("pole-1", "pole", 1.0, 0.0, 0.9, timestamp=0.0)
        memory.remember("pole-1", "pole", 1.0, 0.0, 0.9, timestamp=20.0)

        hazards = memory.get_hazards()

        self.assertEqual(len(hazards), 1)
        self.assertAlmostEqual(hazards[0].decay_weight, 1.0)
        self.assertAlmostEqual(hazards[0].timestamp, 20.0)

    def test_stale_hazard_is_forgotten_below_min_decay_weight(self) -> None:
        memory = RouteMemory(self.db_path, decay_half_life_s=10.0, min_decay_weight=0.05)

        memory.remember("pole-1", "pole", 1.0, 0.0, 0.9, timestamp=0.0)
        # ~5 half-lives later: 0.5**5 = 0.03125, below the 0.05 floor.
        memory.remember("curb-2", "curb", 2.0, 0.0, 0.9, timestamp=50.0)

        hazards = {h.object_id: h for h in memory.get_hazards()}

        self.assertNotIn("pole-1", hazards)
        self.assertIn("curb-2", hazards)

    def test_max_entries_prunes_lowest_ranked_hazards_first(self) -> None:
        memory = RouteMemory(
            self.db_path,
            decay_half_life_s=1_000_000.0,  # effectively no time-decay for this test
            min_decay_weight=0.0,
            max_entries=3,
        )

        # Same timestamp for all -> decay_weight ties at 1.0, so confidence
        # alone decides eviction order under the cap.
        memory.remember("a", "pole", 0.0, 0.0, 0.10, timestamp=0.0)
        memory.remember("b", "pole", 0.0, 0.0, 0.20, timestamp=0.0)
        memory.remember("c", "pole", 0.0, 0.0, 0.30, timestamp=0.0)
        memory.remember("d", "pole", 0.0, 0.0, 0.90, timestamp=0.0)

        hazards = memory.get_hazards()
        ids = {h.object_id for h in hazards}

        self.assertEqual(len(hazards), 3)
        self.assertNotIn("a", ids)  # lowest confidence, evicted first
        self.assertEqual(ids, {"b", "c", "d"})

    def test_decay_and_prune_returns_removed_count(self) -> None:
        memory = RouteMemory(self.db_path, decay_half_life_s=10.0, min_decay_weight=0.05)

        memory.remember("pole-1", "pole", 1.0, 0.0, 0.9, timestamp=0.0)

        removed = memory.decay_and_prune(now=50.0)

        self.assertEqual(removed, 1)
        self.assertEqual(memory.get_hazards(), [])

    def test_velocity_is_persisted_alongside_decay(self) -> None:
        memory = RouteMemory(self.db_path, decay_half_life_s=10.0, min_decay_weight=0.0)

        entry = memory.remember(
            "person-1", "person", 1.0, 2.0, 0.8, timestamp=0.0,
            velocity_x=0.3, velocity_y=-1.2,
        )
        memory.remember("pole-1", "pole", 0.0, 3.0, 0.9, timestamp=10.0)

        self.assertAlmostEqual(entry.velocity_x, 0.3)
        hazards = {hazard.object_id: hazard for hazard in memory.get_hazards()}
        self.assertAlmostEqual(hazards["person-1"].velocity_x, 0.3)
        self.assertAlmostEqual(hazards["person-1"].velocity_y, -1.2)
        self.assertAlmostEqual(hazards["person-1"].decay_weight, 0.5)
        self.assertAlmostEqual(hazards["pole-1"].velocity_x, 0.0)

    def test_legacy_database_without_velocity_columns_is_migrated(self) -> None:
        connection = sqlite3.connect(self.db_path)
        connection.execute(
            """
            CREATE TABLE hazards (
                object_id TEXT PRIMARY KEY, label TEXT NOT NULL,
                world_x REAL NOT NULL, world_y REAL NOT NULL,
                confidence REAL NOT NULL, timestamp REAL NOT NULL,
                decay_weight REAL NOT NULL DEFAULT 1.0
            )
            """
        )
        connection.execute(
            "INSERT INTO hazards VALUES ('pole-1', 'pole', 1.0, 0.0, 0.9, 0.0, 1.0)"
        )
        connection.commit()
        connection.close()

        hazards = RouteMemory(self.db_path).get_hazards()

        self.assertEqual(len(hazards), 1)
        self.assertAlmostEqual(hazards[0].velocity_x, 0.0)
        self.assertAlmostEqual(hazards[0].velocity_y, 0.0)
        self.assertEqual(hazards[0].object_id, "pole-1")
        self.assertEqual(len(hazards[0].hazard_id), 36)  # uuid4 string

    def test_legacy_object_id_keyed_database_is_migrated(self) -> None:
        connection = sqlite3.connect(self.db_path)
        connection.execute(
            """
            CREATE TABLE hazards (
                object_id TEXT PRIMARY KEY, label TEXT NOT NULL,
                world_x REAL NOT NULL, world_y REAL NOT NULL,
                confidence REAL NOT NULL, timestamp REAL NOT NULL,
                decay_weight REAL NOT NULL DEFAULT 1.0,
                velocity_x REAL NOT NULL DEFAULT 0.0,
                velocity_y REAL NOT NULL DEFAULT 0.0
            )
            """
        )
        connection.executemany(
            "INSERT INTO hazards VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("pole-1", "pole", 1.0, 0.0, 0.9, 0.0, 1.0, 0.0, 0.0),
                ("person-2", "person", 4.0, 4.0, 0.7, 0.0, 1.0, 0.2, 0.3),
            ],
        )
        connection.commit()
        connection.close()

        memory = RouteMemory(self.db_path, min_decay_weight=0.0)
        hazards = {h.object_id: h for h in memory.get_hazards()}

        self.assertEqual(set(hazards), {"pole-1", "person-2"})
        self.assertAlmostEqual(hazards["person-2"].velocity_y, 0.3)
        self.assertNotEqual(hazards["pole-1"].hazard_id, hazards["person-2"].hazard_id)

        # Reopening an already-migrated DB keeps the same ids.
        reopened = {h.object_id: h.hazard_id for h in RouteMemory(self.db_path).get_hazards()}
        self.assertEqual(reopened["pole-1"], hazards["pole-1"].hazard_id)


class TestRouteMemoryMatching(unittest.TestCase):
    def setUp(self) -> None:
        handle, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        os.remove(self.db_path)

    def tearDown(self) -> None:
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def _memory(self) -> RouteMemory:
        return RouteMemory(
            self.db_path, decay_half_life_s=1_000.0, min_decay_weight=0.0,
            match_radius_m=1.0,
        )

    def test_same_object_across_two_runs_is_matched_not_duplicated(self) -> None:
        run1 = self._memory()
        first = run1.remember("pole-1", "pole", 2.0, 5.0, 0.9, timestamp=0.0)

        # New run: YOLO track ids restart, so the same pole gets another id
        # and a slightly different position estimate.
        run2 = self._memory()
        second = run2.remember("pole-4", "pole", 2.3, 5.2, 0.8, timestamp=5.0)

        hazards = run2.get_hazards()
        self.assertEqual(len(hazards), 1)
        self.assertEqual(second.hazard_id, first.hazard_id)
        self.assertEqual(hazards[0].object_id, "pole-4")  # latest track hint
        self.assertAlmostEqual(hazards[0].world_x, 2.3)

    def test_reused_track_id_in_new_run_does_not_merge_distant_objects(self) -> None:
        run1 = self._memory()
        first = run1.remember("person-1", "person", 0.0, 2.0, 0.9, timestamp=0.0)

        run2 = self._memory()
        second = run2.remember("person-1", "person", 5.0, 8.0, 0.9, timestamp=5.0)

        self.assertNotEqual(second.hazard_id, first.hazard_id)
        self.assertEqual(len(run2.get_hazards()), 2)

    def test_two_people_three_metres_apart_stay_separate(self) -> None:
        memory = self._memory()

        a0 = memory.remember("person-1", "person", 0.0, 4.0, 0.9, timestamp=0.0)
        b0 = memory.remember("person-2", "person", 3.0, 4.0, 0.9, timestamp=0.0)
        a1 = memory.remember("person-1", "person", 0.1, 4.0, 0.9, timestamp=0.1)
        b1 = memory.remember("person-2", "person", 2.9, 4.0, 0.9, timestamp=0.1)

        self.assertNotEqual(a0.hazard_id, b0.hazard_id)
        self.assertEqual(a1.hazard_id, a0.hazard_id)
        self.assertEqual(b1.hazard_id, b0.hazard_id)
        self.assertEqual(len(memory.get_hazards()), 2)

    def test_same_track_keeps_hazard_even_beyond_radius(self) -> None:
        memory = self._memory()

        first = memory.remember("car-3", "car", 0.0, 10.0, 0.9, timestamp=0.0)
        second = memory.remember("car-3", "car", 0.0, 13.0, 0.9, timestamp=0.5)

        self.assertEqual(second.hazard_id, first.hazard_id)
        self.assertEqual(len(memory.get_hazards()), 1)

    def test_dynamic_label_matches_predicted_position(self) -> None:
        memory = self._memory()
        first = memory.remember(
            "person-1", "person", 0.0, 0.0, 0.9, timestamp=0.0,
            velocity_x=0.0, velocity_y=1.0,
        )

        # Track lost; new id 1.5 m further along the walking direction.
        # Raw distance 1.5 m > radius, predicted distance 0 m.
        matched = self._memory().remember(
            "person-9", "person", 0.0, 1.5, 0.9, timestamp=1.5,
        )
        self.assertEqual(matched.hazard_id, first.hazard_id)

    def test_static_label_does_not_use_velocity_prediction(self) -> None:
        memory = self._memory()
        first = memory.remember(
            "pole-1", "pole", 0.0, 0.0, 0.9, timestamp=0.0,
            velocity_x=0.0, velocity_y=1.0,  # e.g. tracker jitter
        )
        other = memory.remember("pole-2", "pole", 0.0, 1.5, 0.9, timestamp=1.5)

        self.assertNotEqual(other.hazard_id, first.hazard_id)

    def test_different_labels_never_match(self) -> None:
        memory = self._memory()
        pole = memory.remember("pole-1", "pole", 0.0, 2.0, 0.9, timestamp=0.0)
        cone = memory.remember("reflective_cone-1", "reflective_cone", 0.1, 2.0, 0.9, timestamp=1.0)

        self.assertNotEqual(pole.hazard_id, cone.hazard_id)

    def test_persist_false_writes_nothing_but_returns_entry(self) -> None:
        # Invalid pose: the planner still gets this frame's hazard, but the
        # untrusted world position must not reach persistent memory.
        memory = self._memory()
        stored = memory.remember("pole-1", "pole", 0.0, 2.0, 0.9, timestamp=0.0)

        entry = memory.remember(
            "pole-1", "pole", 5.0, 5.0, 0.8, timestamp=1.0, persist=False
        )
        unknown = memory.remember(
            "cone-3", "reflective_cone", 1.0, 1.0, 0.7, timestamp=1.0, persist=False
        )

        self.assertEqual((entry.world_x, entry.world_y), (5.0, 5.0))
        self.assertEqual(entry.hazard_id, stored.hazard_id)  # known track
        self.assertEqual(unknown.hazard_id, "")
        hazards = memory.get_hazards()
        self.assertEqual(len(hazards), 1)
        self.assertEqual((hazards[0].world_x, hazards[0].world_y), (0.0, 2.0))
        self.assertEqual(hazards[0].timestamp, 0.0)


if __name__ == "__main__":
    unittest.main()
