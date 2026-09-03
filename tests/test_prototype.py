import tempfile
import unittest
from pathlib import Path

from navigation.hazard import HazardSeverityClassifier
from navigation.memory import RouteMemory, SpatialMemory
from navigation.models import Detection, Motion, Pose, RouteCandidate
from navigation.reasoning import AcousticAttention, ExplainablePathPlanner


class PrototypeTests(unittest.TestCase):
    def test_approaching_near_person_is_urgent(self) -> None:
        detection = Detection("p", "person", 0.1, 0.7, 0.95, Motion.APPROACHING)
        result = HazardSeverityClassifier().classify(detection)
        self.assertTrue(result.urgent)

    def test_static_memory_fades_but_persists_after_leaving_view(self) -> None:
        memory = SpatialMemory()
        memory.update(Detection("c", "chair", 0.0, 2.0, 0.9), Pose(0, 0, 0), now=1.0)
        memory.decay_and_prune(now=3.0)
        self.assertEqual(len(memory.active()), 1)
        self.assertTrue(0.0 < memory.active()[0].decay_weight < 0.9)

    def test_invalid_localization_does_not_write_world_memory(self) -> None:
        memory = SpatialMemory()
        memory.update(
            Detection("box", "box", 0.0, 1.0, 0.9),
            Pose(0, 0, 0, localization_valid=False),
            now=1.0,
        )
        self.assertFalse(memory.active())

    def test_attention_limits_output_to_top_k(self) -> None:
        memory = SpatialMemory()
        pose = Pose(0, 0, 0)
        for index in range(5):
            memory.update(Detection(str(index), "chair", 0.0, index + 1.0, 0.9), pose, 1.0)
        ranked = AcousticAttention(top_k=2).rank(memory.active(), pose)
        self.assertEqual(len(ranked), 2)
        self.assertGreaterEqual(ranked[0].priority, ranked[1].priority)

    def test_planner_compares_route_costs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            route_memory = RouteMemory(Path(temp_dir) / "test.db")
            try:
                planner = ExplainablePathPlanner(route_memory)
                decision = planner.choose(
                    [
                        RouteCandidate("a", "short", 80, 0.1),
                        RouteCandidate("b", "safe", 90, 0.2),
                    ]
                )
                self.assertEqual(decision.selected.route_id, "a")
                self.assertIn("Selected", decision.rationale)
            finally:
                route_memory.close()


if __name__ == "__main__":
    unittest.main()
