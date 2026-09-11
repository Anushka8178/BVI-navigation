import os
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


if __name__ == "__main__":
    unittest.main()
