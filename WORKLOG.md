# WORKLOG: memory-novelty branch

(The final summary goes at the top when all tasks are done.)

## 0. Setup

- Started on `slam-vo` (not on `nov8-integration` as the session snapshot said).
  `git status` showed two untracked files, `lanes.patch` and `slam-vo.patch`.
  Both were already applied: HEAD had `b8d7078` (SLAM) and `e958ac7` (lanes), and
  `navigation/path_overlay.py` exists. As instructed, they were committed on `slam-vo`
  as "WIP before memory-novelty" (`384e82d`) and nothing was deleted. They are
  redundant with those two commits and can be removed later.
- Created branch `memory-novelty` from `slam-vo`.
- Baseline: `python -m pytest -q` gave **147 passed in 28.9 s** (Python 3.14.7, `.venv`).

## Task 1: memory clock + journey bookkeeping

What changed
- `navigation/memory.py`
  - `SessionClock`: memory timestamps are epoch seconds. Live: `time.time()`.
    Recorded video: `run_start_epoch + elapsed_video_s`. `stamp(elapsed_s)` sets the
    time and calling the clock returns it, so it plugs into `RouteMemory(now_fn=...)`.
  - `RouteMemory(session_id=None, now_fn=time.time)`: `decay_and_prune(now=None)` and
    `get_hazards()` use `now_fn()` instead of the newest stored timestamp;
    `remember(timestamp=None)` stamps with `now_fn()`.
  - New columns (ALTER TABLE pattern, like `anchor`): `first_seen`, `times_seen`,
    `sessions_seen`, `last_session`, `belief` (default 1.0), `misses`. Every persisted
    sighting adds 1 to `times_seen`. The first sighting in a new session adds 1 to
    `sessions_seen` and sets `last_session`.
  - `HazardEntry` has the same fields, with defaults so existing callers keep working.
  - `remember()` now does the decay pass in the same transaction: one commit per sighting
    instead of two (measured 3.5 ms per `remember()` before; the eval tool calls it a lot).
- `video_navigation_demo_slam_memory.py`: memory gets `SessionClock(live)` and a uuid
  `session_id`. Memory timestamps are `memory_clock.stamp(elapsed_s)`. Tracker/TTC keep
  the pipeline timestamp. The JSON report has `memory_session_id`, `memory_clock` and
  the journey fields per remembered hazard.
- `localization_demo.py`: uses a `SessionClock` too. Otherwise its 0..5 s timestamps
  would look 56 years old and everything would be pruned at once.

Decisions
- **Old rows with non-epoch timestamps** (`< 1e9`, i.e. video time from before this change)
  are re-stamped to "now" when the journey columns are first added. Their real age is
  unknown; without this, every upgraded database would be wiped on first load (age ~56
  years). This happens only during that one-time migration. Rows that already have epoch
  timestamps keep them.
- A row migrated from an older schema counts as seen in 1 earlier session (`sessions_seen = 1`,
  `last_session = NULL`), so its next sighting makes `sessions_seen = 2`.
- Existing tests that relied on "now = newest stored timestamp" now inject a fake clock
  (`tests/test_memory.py`, one test in `tests/test_vslam.py`). Their expected values did not change.

Tests added (`tests/test_memory.py`)
- clock injection (decay advances with the clock even with no new sightings);
- times/sessions counting across two `RouteMemory` sessions (`sessions_seen == 2`);
- distinct default session ids;
- migration of an old table without the journey columns (old video-time row re-stamped,
  epoch row kept, defaults filled, reopening doesn't re-stamp);
- `SessionClock`: recorded-video stamps are increasing epoch seconds and a later rerun is
  later; live mode uses the wall clock; memory written with it decays correctly.

Result: `python -m pytest -q` gives **154 passed**.

## Task 2: class-aware decay with reinforcement across journeys

What changed
- `config.yaml` `memory:` now has `categories` (dynamic 10 s, temporary 7200 s,
  permanent 1209600 s = 14 days), `default_half_life_s: 3600`, `reinforce_per_session: 0.5`,
  `reinforce_max: 3.0`. `decay_half_life_s: 30` stays as the fallback when `categories`
  is removed.
- `navigation/memory.py`
  - `effective_half_life(base, sessions_seen, per_session, max)` and `decay_weight_for()`
    are pure functions.
  - `RouteMemory(categories=, default_half_life_s=, reinforce_per_session=0, reinforce_max=1)`.
    Constructor defaults give the old behaviour (one half-life, no reinforcement).
    `half_life_for(label, sessions_seen)`.
  - `RouteMemory.from_config(memory_cfg, **overrides)`: used by the demo (and the eval
    tool), so config to constructor wiring lives in one place.
  - Eviction above `max_entries`: lowest `decay_weight * belief` first, then lowest
    confidence, then oldest.
- `navigation/reasoning.py`: `AcousticAttention` recency term = `decay_weight * belief`.
- Demo: `RouteMemory.from_config(memory_cfg, db_path=..., session_id=..., now_fn=...)`.

Decisions
- Lifetime until pruning (`min_decay_weight 0.05`) = about 4.32 half-lives: person ~43 s,
  cone ~8.6 h (x2 after 3 walks), pothole ~60 days.
- Confidence is the tie-break in eviction (instead of a factor). Without it,
  `test_max_entries_prunes_lowest_ranked_hazards_first` would become order-dependent.
- A label listed in two categories is a `ValueError` (a clear config error).

Tests added
- `tests/test_memory_decay_classes.py`: reinforcement formula + cap; repo config half-lives;
  the dynamic category equals `dynamic_labels`; a pothole survives 2 simulated days, a person
  is gone after a minute, a cone after a day; a cone seen on 3 walks outlives one seen once
  (12 h later: 1/8 vs 1/64); fallback without categories; duplicate label error; eviction
  order; attention uses belief.
- `tests/test_config.py`: the memory section builds a `RouteMemory`.

Result: `python -m pytest -q` gives **163 passed**.

## Task 3: evidence-based forgetting (negative evidence)

(Done before Task 4: Task 4 builds on the same module.)

What changed
- New `navigation/memory_reasoning.py` (pure logic, used by the demo and the eval tool):
  `MemoryReasoningConfig.from_config()` (rejects unknown keys), `CameraModel` (hfov, camera
  height, image height, ground labels; nearest visible ground from
  `path_overlay.nearest_visible_m`), `pose_is_trusted(pose, slam_state)`, `expected_visible`,
  `is_occluded`, `is_seen` and `update_negative_evidence(...)` returning `ForgetEvent`s.
- `RouteMemory.record_miss(hazard_id, factor)` (misses += 1, belief *= factor),
  `RouteMemory.forget(hazard_id)`. `remember()` (a real sighting) sets `misses = 0` and
  `belief = min(1, belief + hit_gain)`, with `hit_gain` passed from config.
- `config.yaml` `memory_reasoning:` holds all the numbers from the task. It is a required
  config section now.
- Demo: after the frame's detections are remembered, runs `update_negative_evidence` with
  the trusted-pose rule (SLAM must be TRACKING) and the hazard_ids this frame matched. Logs
  `MEMORY | forgot <label> <id>: not seen in clear view N times`. The one `get_hazards()` read
  per frame is shared by attention and the mini-map (it used to be read twice). The mini-map
  shade uses decay x belief. The JSON report has a `negative_evidence` block.

Decisions
- "Matched by a current detection" = the detection was resolved to this hazard_id by
  `RouteMemory.remember()`, **or** a same-label detection lies within `memory.match_radius_m`
  (the same gate memory uses). The second part keeps a noisy duplicate from counting as a miss.
- The occlusion test compares the detection's distance with the hazard's range (not only
  its forward distance). The 0.5 m margin is `occlusion_margin_m` in config.
- The miss interval is checked as `now - last_miss >= miss_interval_s`.
- With `--pose sim` the pose is invented, so negative evidence (like memory itself) is
  only meaningful with `--pose slam` or `--pose phone`. The code does not special-case it.

Tests added (`tests/test_memory_reasoning.py`, 14): forgotten after 4 misses spaced
0.6 s (misses closer than 0.5 s don't count; log line checked); an occluded hazard gets no
miss but a detection 10 deg away doesn't hide it; FOV edge, too far, too close and
ground-level below the image bottom get no miss; a dynamic hazard is never touched; an
invalid pose or SLAM not TRACKING touches nothing; a matched detection is not a miss, a
different label is; a sighting resets misses and adds hit_gain; heading is used; config
section = dataclass.

Smoke run: `video_navigation_demo_slam_memory.py --source videos\street1.mp4 --no-audio
--no-esp32 --no-display` (temp db) processed 141 frames at 15 FPS on CPU with no errors.

Result: `python -m pytest -q` gives **177 passed**.
