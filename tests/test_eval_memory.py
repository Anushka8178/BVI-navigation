"""tools/eval_memory.py: memory ablation on a simulated walk (tiny scenario)."""

import math
import time

from tools import eval_memory as em

TINY = em.Params(side_m=8.0, fps=5.0, sessions=(1, 2))


def test_tiny_scenario_runs_fast_and_variant_a_never_cues(tmp_path):
    started = time.perf_counter()
    results = em.run(TINY)
    elapsed = time.perf_counter() - started
    assert elapsed < 10.0, f"tiny scenario took {elapsed:.1f} s"

    assert {(r.variant, r.session) for r in results} == {
        (v, s) for v in em.VARIANTS for s in TINY.sessions
    }
    for r in results:
        if r.variant == "A":
            assert r.memory_cues == 0 and not r.cue_events
            assert r.memory_size == 0 and r.memory_at_start == 0
        if r.variant == "B":
            assert r.memory_cues == 0  # no cues before this work
        if r.variant in ("A", "B", "C"):
            assert r.forgotten == 0  # negative evidence only in D

    rows = em.pooled_rows(results)
    paths = em.write_outputs(rows, TINY, tmp_path)
    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
    assert "| Variant |" in paths["md"].read_text(encoding="utf-8")


def test_same_seed_gives_the_same_result():
    a = [r.row() for r in em.run(TINY, variants=("D",))]
    b = [r.row() for r in em.run(TINY, variants=("D",))]
    assert a == b


def test_memory_persists_across_walks_only_with_class_aware_decay():
    by = {(r.variant, r.session): r for r in em.run(TINY, variants=("B", "C"))}
    # Two days later the 30 s half-life has forgotten everything ...
    assert by[("B", 2)].memory_at_start == 0
    # ... class-aware decay still remembers the permanent hazards.
    assert by[("C", 2)].memory_at_start >= 3


def test_route_and_world_geometry():
    walk = em.route(TINY)
    t0, x0, y0, h0 = walk[0]
    assert (t0, x0, y0, h0) == (0.0, 0.0, 0.0, 0.0)  # start mark, facing +y
    assert math.hypot(walk[-1][1], walk[-1][2]) < 1e-6  # back at the start
    objects = {o.name: o for o in em.truth_objects(TINY)}
    assert 2 not in objects["cone-1"].sessions and 2 in objects["barrier-2"].sessions
    assert 3 not in objects["pothole-2"].sessions
    # pothole-1: 30 % along the first leg (+y), 0.3 m to the right (+x).
    assert (objects["pothole-1"].x, objects["pothole-1"].y) == (0.3, 0.3 * TINY.side_m)


def test_summarize_pools_runs():
    a = em.SessionResult("D", 2, 2.0, minutes=1.0, static_detected=4, warned=1,
                         lead_times=[1.0], memory_cues=2)
    b = em.SessionResult("D", 2, 2.0, minutes=1.0, static_detected=4, warned=3,
                         lead_times=[2.0, 3.0], memory_cues=4)
    row = em.summarize([a, b])
    assert row["warned_before_detection_pct"] == 50.0
    assert row["mean_lead_time_s"] == 2.0
    assert row["memory_cues"] == 3.0  # mean per run
    assert row["memory_cues_per_min"] == 3.0
    assert row["runs"] == 2
