# WORKLOG: memory-novelty branch

## Final summary (8 Nov 2026, for the 10 Nov evaluation)

**All 7 tasks are done and committed, one commit per task, on `memory-novelty` (from `slam-vo`).**
Tests: **217 passed** (baseline 147, so 70 new; all hardware-free), `python -m pytest -q`, about 25 s.

Done
1. Memory clock + journeys: epoch `SessionClock`, injectable `now_fn`, `session_id`,
   `first_seen/times_seen/sessions_seen/last_session/belief/misses` (old DBs migrate).
2. Class-aware decay (dynamic 10 s / temporary 2 h / permanent 14 days half-life) with
   reinforcement across walks (x1.5 per extra walk, capped at x3).
3. Evidence-based forgetting (`navigation/memory_reasoning.py`): a static hazard missed 4 times in
   clear view is forgotten; occlusion, FOV, range and ground-visibility guards.
4. Memory early warning: `select_memory_cues` + `MemoryEarlyWarning` + `HRTFRenderer.play_memory_cue`
   (own two-pip timbre, lowest priority, suppressed by urgent/path cues). Console, status panel,
   mini-map ring, `memory_cues` / `memory_summary` in the JSON report, lead time when confirmed.
5. ESP32 ultrasonic fusion (`navigation/range_fusion.py`): distance fix + `unknown_obstacle`.
   Ready, but not tried on hardware.
6. `tools/eval_memory.py`: 4-variant ablation over 3 simulated walks (CSV, markdown, PNG).
7. README novelty table (claim → code → test → evidence), two-walk demo instructions, CLAUDE.md
   updated, honest limitations in README and the JSON report.

Evaluation (simulation, SLAM pose model sigma 0.15 m, 5 seeded runs pooled; `runs/memory_eval/`)

| Variant | Walk | Remembered at start | Warned before camera alert % (lead s) | Warned before 1st detection % (lead s) | Stale / false cues | False hazards in memory | Removed still remembered | Memory cues/min | All cues/min | Pos. error m |
|---|---|---|---|---|---|---|---|---|---|---|
| A no memory | S2 / S3 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | - | - | 0 / 0 | 22.5 / 19.7 | - |
| B 30 s decay | S2 / S3 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 13.2 / 9.2 | 0 / 0 | 0 / 0 | 22.5 / 19.7 | 0.28 / 0.24 |
| C class decay + cues | S2 / S3 | 11.6 / 21.4 | 48.9 (3.65) / 33.3 (4.12) | 28.9 (0.32) / 27.5 (1.96) | 0 / 2.0 stale | 18.8 / 24.2 | 0 / 1.6 | 4.74 / 4.88 | 27.2 / 24.6 | 0.27 / 0.24 |
| D full | S2 / S3 | 9.0 / 12.2 | 48.9 (3.65) / 33.3 (3.42) | 28.9 (0.32) / 22.5 (1.42) | 0 / 2.0 stale | **7.6 / 6.8** | **0 / 0** | 4.74 / 4.47 | 27.2 / 24.1 | 0.27 / 0.25 |

(S1 is the first walk: no warnings from earlier walks are possible; 13.6 % of hazards still get a
within-walk memory cue before the alert, after being seen and then hidden. False cues are 0 in every
row after the `cue_min_sightings` fix. Objects lost from memory by negative evidence: 0 in every row.)
Phone dead reckoning: remembered positions 3.1-3.3 m off, cue direction error 13-18 deg
(SLAM model: 1.6-2.8 deg). Real VisualSLAM on rendered frames (1 run, new map each walk):
1.8-1.9 m, 9-12 deg; warned before the camera alert 37.5-55.6 % in walks 2-3 (single run, small counts).

How to read it honestly
- The memory cue mainly gives warning **before the camera's alert** (about 3.5-4 s, from about 8 m),
  not before the first detection (cue range = camera range = 8 m, so only 0.3-2 s there).
- Negative evidence is what keeps memory clean (false hazards 7 vs 24 rows, removed objects forgotten,
  no real object lost). It cannot prevent the first stale cue for an object removed round a corner.
- The SLAM noise model is optimistic; the rendered VisualSLAM check drifted 1.8-1.9 m over 80 m.
- Memory cues add about 2-5 cues/min to about 20/min of urgent+path cues: a real cognitive-load cost.

Still needs real-world testing
- Live two-walk test on campus with `--pose slam` (relocalization at the start mark, memory cues,
  forgetting a removed cone). Listen to `scripts/manual/test_memory_cue.py` with the wired earphones.
- Camera calibration (`scripts/manual/calibrate_camera.py`) for better SLAM and distances.
- ESP32 + HC-SR04: `scripts/manual/check_esp32.py`, then a live run; check the `range_fusion` block.

Risks
- With `--pose sim` (the default for video files), memory, negative evidence and memory cues are
  meaningless (the pose doesn't follow the video). Use `--pose slam` for any memory demo.
- Memory cue thresholds (2.5-8 m, 40 deg, 8 s / 2 s cooldowns, 3 sightings) were chosen once and
  not tuned with users.
- `navigation_memory.db` files from before this branch are migrated; rows with video-time
  timestamps are re-stamped as "seen now" (their real age is unknown).
- `git push` result: see the end of this file.

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

## Task 4: memory early warning (sonified top-k acoustic attention)

What changed
- `navigation/memory_reasoning.py`
  - `select_memory_cues(...)`. Candidates: remembered non-dynamic hazards not matched by
    this frame's detections, with a trusted pose, forward distance in [2.5, 8] m,
    |bearing| <= 40 deg and decay x belief >= 0.3. They are ranked by the existing
    `AcousticAttention` (top-k from config, on copies so priorities are not overwritten). The best
    one not on its 8 s cooldown wins, with a 2 s global cooldown and at most one cue per frame.
    `suppressed=True` (urgent or path cue due) returns nothing and books no cooldown.
  - `MemoryCue` (+ `describe()`: `pothole ahead-left 4.2 m | seen in 2 earlier walks, last
    2 days ago`), `direction_words`, `format_age`, `CueState`.
  - `CueLog`: lead time = first later frame in which the cued hazard_id is seen minus
    cue time; lead distance = distance at cue time; `summary()` for the report.
  - `MemoryEarlyWarning`: confirm, select, play (renderer optional), log, plus status-panel text.
    The demo and the eval tool both use it.
- `navigation/hrtf_renderer.py`: `play_memory_cue(azimuth_deg, distance_m)`, a new `MEMORY` kind.
  Two soft low pips (330 + 495 Hz, 90 ms each, 80 ms gap), level 0.32 to 0.22 by distance
  (path 0.50, urgent 0.85). Own cooldown `audio.memory_cue_cooldown_s: 2.0`. Lowest priority:
  urgent drops queued memory cues and interrupts a playing one; a path cue on a full queue
  replaces a memory cue first; a memory cue on a full queue is dropped. The physical to SOFA
  conversion stays in `_spatialize`.
- `navigation/models.py`: `expand_audio_azimuth()` (x2.5, +-90) shared by detections and memory cues.
- `navigation/minimap.py`: `highlight=(x, y)` draws an amber ring on the cued hazard.
- Demo: after urgent + path guidance, `early_warning.update(...)` with
  `suppressed = urgent warnings this frame or a path cue due` (computed whether audio is on
  or not). The planner is untouched. Status panel `Memory cue: ...` (amber, 3 s), mini-map ring,
  console `MEMORY CUE | ...`. JSON: `memory_cues` list + `memory_summary` (cues played, with
  audio, later confirmed, mean lead time/distance, hazards forgotten by negative evidence),
  plus a `memory_early_warning` entry in `audio_policy`. End-of-run console summary.
- `scripts/manual/test_memory_cue.py` (left/ahead/right + path/urgent for comparison), listed
  in `scripts/manual/README.md`.
- Robustness found while testing two walks back to back:
  - `RouteMemory._match_hazard`: the "already updated this frame" skip only applies to
    rows from this session. Before, a later-stamped row from an earlier run could block matching.
  - `SessionClock.start_after(memory.latest_timestamp())`: a recorded-video run's memory clock
    starts after everything already in memory, in case the earlier run replayed faster than
    real time (possible on the GPU).

Decisions
- **Added `cue_skip_seen_within_s: 1.0`** (config). A hazard the camera saw less than 1 s ago is
  still "in view" (a one-frame detector dropout), so it does not count as a remembered unseen
  hazard. Without this, the cue would fire on every dropout of a visible pothole. Chosen once
  for UX, before running any evaluation.
- "Earlier walks" = `sessions_seen` minus 1 if the hazard was also seen in this session.
- Memory-cue audio azimuth uses the same perceptual expansion as urgent cues (x2.5, clamp +-90),
  since both point at the hazard.
- Suppression uses "urgent warnings exist this frame" (even if the urgent sound is on
  cooldown) or "a path cue is due" (blocked lane, useful lateral escape, guidance changed).

Tests added
- `tests/test_memory_cues.py` (fake renderer): a pothole 5 m ahead, unseen, remembered from a
  walk 2 days ago gets a cue (exact console line); a currently detected one (by id or nearby same
  label) does not; behind / > 40 deg / too far / too close do not; facing away vs turning
  toward it; per-hazard and global cooldowns; suppression books no cooldown; untrusted pose and
  low trust; a one-frame dropout is not a memory; a dynamic hazard; "seen earlier in this walk";
  lead time 1.5 s / 5.0 m and the summary dict; formatting helpers; config.
- `tests/test_hrtf_audio.py`: memory cue played; urgent drops and interrupts memory; a memory cue
  cannot displace path or urgent cues; a path cue replaces a queued memory cue; memory cooldown;
  the memory sound is quieter than path guidance, louder on the left for -60 deg, two pips with a gap.
- `tests/test_memory.py`: `start_after`; matching across sessions with a clock that runs behind.

Two-walk smoke run (same video twice, same DB, `--pose sim`, no audio): walk 2 printed
`MEMORY CUE | pole ahead-right 3.4 m | seen in 1 earlier walk, last 7 s ago`. The report blocks
were filled. Memory doubled (10 to 20 rows) because `--pose sim` does not follow the video,
so a pole's world position drifts as the camera approaches. This is known bug #1 (use slam/phone).

Result: `python -m pytest -q` gives **202 passed**.

## Task 6: memory evaluation tool (`tools/eval_memory.py`)

(Done before Task 5, following the priority order; Task 7 docs last so they cover everything.)

What it does
- Hardware-free, deterministic per seed. Uses the real `RouteMemory` (temp SQLite DB per
  variant, fake clock), `AcousticAttention`, `memory_reasoning` (`update_negative_evidence`,
  `MemoryEarlyWarning`), `LocalPathPlanner`, `HazardSeverityClassifier` and
  `HazardTrajectoryTracker`, all built from `config.yaml`. Only the camera, the pose error and
  the audio timing are simulated. `SimAudio` mirrors `HRTFRenderer`: per-kind cooldown counted
  from the end of the sound.
- World: a 20 m square campus loop (80 m, clockwise, 1 m/s, 10 Hz, 60 deg/s turns in place) =
  `synthetic_scene.square_walk(20)` shifted to start at (0,0) facing +y. 10 static hazards
  (2 potholes, stairs, curb, 2 poles, 2 cones, 2 barriers) placed by fraction along each leg.
  Passing pedestrians come toward the user at 1.2 m/s, about 4 per minute, and occlude what is behind them.
- Sessions: S1 day 0; S2 day 2 (cone-1 removed, barrier-2 added); S3 day 9 (pothole-2 repaired).
- Variants: A no memory; B one 30 s half-life, no cues (system before this work, with the fixed
  epoch clock so only the decay policy differs); C class-aware decay + reinforcement + cues;
  D = C + negative evidence.
- Metrics per variant x session (pooled over `--runs` seeds, default 5): static hazards remembered
  at the walk's start; % warned before the first camera detection (and lead s / true distance m);
  % warned before the first camera *alert* (urgent/path cue) and that lead; stale cues; false
  cues; false hazards in memory at the end; removed objects still remembered; memory cues/min and
  urgent+path+memory cues/min; memory size; position error of remembered hazards; cue direction
  error; rows forgotten by negative evidence, how many belonged to objects still present, and
  objects lost from memory.
- Output: `runs/memory_eval/results.csv`, `results.md`, `results.png` (matplotlib; OpenCV fallback)
  and the table on the console. `--pose-model phone`, `--render-slam` (real `VisualSLAM` on
  `synthetic_scene` frames, room 4 m larger than the loop on each side, new map per session),
  `--runs`, `--seed`, `--side`, `--fps`, `--p-detect`, `--max-range`, `--out`.

Parameters (chosen once, before seeing results)
- Detector: the task's values: FOV 70 deg, range 1-8 m (ground classes from the nearest visible
  ground, 2.47 m at 1.3 m camera height), p_detect 0.85, distance noise 10 %, bearing noise 2 deg.
  False positives 0.01/frame (about one every 10 s), as transient single-frame boxes of a random static class
  at 2.5-8 m inside the view. People occlude objects behind them (0.3 m radius). That is physical,
  and it is the same situation the negative-evidence occlusion guard handles.
- SLAM pose model: error around the truth with sigma 0.15 m and 1 deg, correlated over 5 s (a
  Gauss-Markov process), a new random offset each walk (relocalization error). Phone model: dead reckoning from the start
  mark, stride error +-7 % and heading drift +-0.3 deg/s (random signs per walk) plus 1 deg noise.
  These are the same magnitudes as `tools/simulate_slam_walk.py`.
- The simulated walker follows the route and does not react to cues (exposure model).
- Walking 1 m/s, 10 Hz, 4 people/min: plausible campus values; not varied afterwards.

Bugs and design gaps found by the evaluation (fixed in the code, not in the results)
- **Unconfirmed one-frame detections were cued.** The first run (seed 7, before the fix) gave
  C/D 8 false memory cues in S1 and 11-16 per walk later (memory cues 7-14/min). Each transient false
  positive became "a remembered pole 6 m ahead" and was cued 1 s later. Fix:
  `memory_reasoning.cue_min_sightings: 3`, the same confirmation idea as `planner.confirm_frames: 3`.
  It applies to C and D alike. After it: 0 false cues.
- Truth mapping in the tool: a memory row's object is the most frequent object that updated it.
  With latest-wins, one false positive landing near a real pole would mark that row "false".
- The false-positive count of seed 7 looked high (17 in S2). I checked it over 30 seeds: mean 9.2 vs 8.6
  expected, so no bias. Because single seeds are noisy, the tool pools 5 seeded runs by default.

Tests (`tests/test_eval_memory.py`): a tiny scenario (8 m loop, 5 Hz, 2 walks) runs in < 10 s and
variant A never cues (B never cues; only D forgets); same seed gives the same result; B remembers
nothing 2 days later while C still remembers the permanent hazards; route/world geometry;
pooling over runs. `tests/test_memory_cues.py`: an unconfirmed one-frame hazard is not cued.

## Task 5: ESP32 ultrasonic fusion (hardware arrives later; ready and tested)

What changed
- New `navigation/range_fusion.py` (pure): `RangeFusionConfig.from_config()` (rejects unknown keys),
  `fuse_range(detections, range_m, cfg) -> FusionResult`, `with_distance()` (rescales
  relative x/y along the bearing with `dataclasses.replace`).
  - Distance fix: the nearest detection within +-`cone_deg` (7.5) of straight ahead takes the
    ultrasonic distance when the reading is in [`min_m` 0.2, `max_m` 4.0].
  - Unknown obstacle: a reading below `unknown_obstacle_m` (1.2) that no detection in the cone
    explains adds `Detection(label="unknown_obstacle", relative_x=0, relative_y=value,
    confidence=1.0)`. The hazard classifier rates it urgent; the planner sees it.
  - No reading (`None`: stale, no echo, sensor missing) changes nothing.
- `config.yaml` `range_fusion:` (required section). Demo: fusion right after perception when
  the range sensor runs; `unknown_obstacle` is remembered with `persist=False` (never stored);
  JSON report block `range_fusion` (frames with a reading, distances replaced, unknown-obstacle
  frames). `--no-esp32`: no sensor, so nothing changes.

Decision
- **Agreement gate `agree_ratio: 2.0`** (not in the task). The distance is replaced only if the
  camera's monocular estimate is within a factor 2 of the reading. Example: camera says a pole at 6 m,
  ultrasonic says 1.0 m. That is almost certainly something else in front (glass door, wall).
  Dragging the pole to 1 m would be wrong; instead the close reading becomes an `unknown_obstacle`
  (urgent) and the pole stays at 6 m. So "no detection ahead" in the task is read as "no
  detection ahead that agrees with the reading".

Tests (`tests/test_range_fusion.py`, 10): no reading -> unchanged; the nearest in-cone detection
gets the distance and keeps its bearing, others untouched; readings outside [0.2, 4] m don't
replace; a close reading with nothing ahead adds an urgent unknown obstacle; a far reading with
nothing ahead adds nothing; a disagreeing reading adds an obstacle and leaves the detection; an
agreeing close reading corrects instead of adding; config section.

Not tested on hardware (sensor not here yet): run `scripts/manual/check_esp32.py`, then a live
run without `--no-esp32` and check `range_fusion` in the JSON report.

### Task 6 results (final runs, from the committed code)

Commands: `python tools/eval_memory.py` (SLAM model, seeds 7-11, about 105 s),
`python tools/eval_memory.py --pose-model phone --out runs/memory_eval_phone`,
`python tools/eval_memory.py --render-slam --runs 1 --out runs/memory_eval_vslam` (about 5 min).
Full tables are in `runs/memory_eval*/results.md` (git-ignored). The summary table is at the top of this file.

Unexpected results, explained (not tuned away)
- **Short lead before the first detection (0.3-2 s).** The cue window (2.5-8 m) and the camera range
  (1-8 m) are the same, so memory can only be earlier when the camera has not had its first hit,
  is occluded, or the hazard is round a corner (cue at <= 40 deg, camera FOV +-35 deg). The useful
  lead is before the camera **alert** (urgent at about 2.5 m): 3.4-4.1 s.
- **Two stale cues in walk 3 in both C and D**: the repaired pothole sits 4.4 m round a corner. The
  first cue fires from about 9 m away at 30 deg, the second mid-turn, both before the camera has a clear
  view. D then forgets it (0 removed objects remembered); C keeps it.
- **D warns slightly less than C in walk 3 before the first detection (22.5 vs 27.5 %, i.e. 9 vs 11 of 40
  hazard-walks).** C keeps noisy duplicate rows (35 vs 16 rows); a duplicate placed a bit nearer
  sometimes fires the cue earlier. D forgot 1 present-object row per walk; 0 objects were lost.
- **Phone pose**: longer "lead" numbers (2.9-4.4 s before detection) are partly an artefact.
  Remembered positions are about 3 m off, so cues fire for the right object at the wrong place
  (cue direction error 13-18 deg). Negative evidence forgets misplaced rows (3.8 per walk in walk 3)
  and re-learns them.
- **Rendered VisualSLAM** (single run): 1.8-1.9 m position error, much worse than the 0.15 m noise
  model. The rendered room is 28 m wide (walls 4-24 m away), a single lap has no loop closure until
  the end, and each walk starts a new map (the simulation does not relocalize into the previous map
  or re-anchor hazards, unlike the demo). Treat it as a pessimistic check.

## Task 7: documentation and honesty pass

What changed
- `README.md`: new section **"Novelty → where it is in the code → how it is tested → evidence"**
  (7 claims), "Memory evaluation (simulation)" with commands and an honest reading of the numbers,
  "Two-walk memory demonstration (live)", SQL example with the journey columns, stale
  "decay not implemented" note fixed, limitations extended (monocular distance, ground visibility,
  simulation is not a user study, memory needs a good pose, first stale cue round a corner).
- `CLAUDE.md`: final evaluation date (10 Nov), modules list, config sections, memory /
  reasoning / evaluation summary, range fusion, conventions (memory clock, audio priority,
  memory cues never touch the planner). Known bugs 1-7 re-verified in the code and marked fixed,
  with what is still open.
- Demo JSON `limitations`: monocular distance (and whether the ultrasonic sensor was used),
  ground-level visibility, simulation-only evaluation, pose quality for memory.

Result: `python -m pytest -q` gives **217 passed**.

## Push

`git push -u origin memory-novelty` **succeeded**. Note that `origin` here is the fork
`https://github.com/Anushka8178/BVI-navigation` (not `ManishAdithya/BVI-navigation`). Nothing was
pushed to `main`. A PR can be opened from
`https://github.com/Anushka8178/BVI-navigation/pull/new/memory-novelty`.
