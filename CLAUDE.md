# CLAUDE.md — BVI Navigation (read this before every task)

## What this project is
Context-aware navigation assistant for blind and visually impaired (BVI) users.
YOLO detection → distance/direction → world-frame tracking → persistent hazard memory
(class-aware decay, reinforcement across walks, evidence-based forgetting) → local path planner
+ urgent bypass → top-k acoustic attention (sonified as memory early-warning cues) → HRTF audio.

**Final evaluation: Tue 10 Nov 2026** (the Android rig below was built for the 8 Nov demo).
Prefer small, working, tested changes over big refactors. Do not add heavy new dependencies.
Do not rewrite modules that already work.

## Target setup for 8 Nov (the "Android rig")
- **Camera:** Android phone, chest-mounted, running the *IP Webcam* app.
  The OpenCV source is `http://<PHONE_IP>:8080/video` (MJPEG).
- **Motion sensors:** the same phone runs the *SensorServer* app, which serves JSON over WebSocket:
  `ws://<PHONE_IP>:<PORT>/sensor/connect?type=android.sensor.rotation_vector`
  and `...type=android.sensor.step_detector`. Each message is JSON with `values` (floats) and
  `timestamp` (ns).
- **Range sensor:** an ESP32 with an HC-SR04 ultrasonic sensor (2–400 cm, 15° cone), mounted at chest height
  facing forward, next to the phone camera, connected to the laptop by **USB serial**. There is
  **no vibration motor**.
- **Compute:** a Windows laptop (RTX 3050) runs this Python pipeline. The laptop is connected to the phone's hotspot.
- **Audio:** HRTF cues from the laptop through **wired** earphones. Avoid Bluetooth, which adds about 100–250 ms.

### ESP32 → laptop protocol (USB serial, 115200 baud, plain ASCII, one line per reading)
- Firmware: `firmware/esp32_range/esp32_range.ino` (Arduino IDE, board "ESP32 Dev Module").
- ESP32 prints `D,<distance_cm>,<esp_millis>` about 15 times a second (median of 3 pings).
  If there's no echo, it prints `D,-1,<esp_millis>`. Other lines (boot messages) are ignored.
- One-way only: the laptop sends nothing to the ESP32.
- Wiring (sensor powered from 5 V because it was unreliable at 3.3 V; ESP32 GPIOs are not
  5 V tolerant, so Echo goes through a 1k/2k divider to about 3.3 V):
  HC-SR04 VCC → ESP32 VIN (5 V), GND → GND, Trig → GPIO26,
  Echo → R1 1kΩ → GPIO27, with R2 2kΩ (two 1kΩ in series) from GPIO27 → GND.
- Laptop side: `navigation/range_sensor.py` (`RangeSensor`, pyserial background thread);
  `latest_m()` returns None if the reading is older than 300 ms or is -1.
  `navigation/range_fusion.py` (config `range_fusion:`) fuses it with the detections each frame:
  the nearest detection within ±7.5° takes the measured distance (if the camera estimate agrees
  within ×2), an unexplained reading < 1.2 m adds an `unknown_obstacle` detection (urgent,
  never stored in memory). Written and unit-tested before the hardware arrived.
- The serial port and baud are configurable in `config.yaml` (`esp32:` section, default `COM5`;
  `--esp32-port` overrides), never hard-coded in logic. Check with `scripts/manual/check_esp32.py`.

## Repository facts
- Main repo: `ManishAdithya/BVI-navigation`, branch `main`.
- Work that is **not yet on main** lives on the fork `https://github.com/Anushka8178/BVI-navigation`:
  - `slam-memory-demo` → `RouteMemory.decay_and_prune()` (exponential decay, floor, max_entries)
    + `tests/test_memory.py` + the SQLite connection-closing fix (`contextlib.closing`).
  - `path-planner-hrtf-updated` → newest planner (TTC filtering, safety_margin 0.25) + HRTF
    physical→SOFA azimuth conversion, with path guidance played at the *walking* direction.
- Main entry point: `video_navigation_demo_slam_memory.py`. Branch `memory-novelty` (from
  `slam-vo`) adds the memory novelty; see `WORKLOG.md` for what changed and why.
- Modules: `navigation/` (models, camera, config, yolo_perception, hazard, sensing, memory,
  memory_reasoning, planner, trajectory, reasoning, audio, hrtf_renderer, range_sensor,
  range_fusion, vslam, minimap, path_overlay, slam_eval, synthetic_scene).
  HRTF file: `hrtf/mit_kemar_normal_pinna.sofa`.
- Config sections (`config.yaml`, loaded by `navigation/config.py`): camera, network, esp32,
  range_fusion, pose, perception, hazard, planner, trajectory, memory, memory_reasoning,
  attention, audio, slam.
- **Persistent memory** (`navigation/memory.py`, config `memory:`): SQLite, uuid `hazard_id`,
  matching by track hint then same-label nearest neighbour within `match_radius_m`.
  - Clock: epoch seconds (`SessionClock`: wall clock live, run start + video time for files),
    injectable `now_fn`. Each run has a `session_id`. Rows store `first_seen`, `times_seen`,
    `sessions_seen`, `last_session`, `belief`, `misses` (columns ALTERed into old databases).
  - Class-aware decay: `categories` (dynamic 10 s, temporary 2 h, permanent 14 days half-life),
    half-life × min(1 + 0.5 × (sessions_seen − 1), 3); pruned below `min_decay_weight`.
    `RouteMemory.from_config()` builds it.
- **Reasoning over memory** (`navigation/memory_reasoning.py`, config `memory_reasoning:`, pure):
  `update_negative_evidence` (a static hazard expected in clear view but not detected loses
  belief and is forgotten after 4 misses) and `MemoryEarlyWarning` / `select_memory_cues` (remembered,
  confirmed, currently unseen static hazard 2.5–8 m ahead within ±40° → one soft HRTF memory cue,
  ranked by `AcousticAttention`, cooldowns, suppressed by urgent/path cues; lead time logged).
  `HRTFRenderer.play_memory_cue` has its own timbre and the lowest queue priority.
- **Evaluation**: `tools/eval_memory.py` (memory ablation A/B/C/D over 3 simulated walks, output in
  `runs/memory_eval/`), `tools/simulate_slam_walk.py`, `tools/eval_walk_drift.py`,
  `tools/eval_slam_tum.py`.
- **Visual SLAM** (`--pose slam`, `navigation/vslam.py`, config `slam:`): keyframe monocular VO
  (ORB + essential matrix, dynamic YOLO boxes masked), scale from phone steps (or
  `slam.assumed_speed_m_s`), heading fused with the phone, loop closure (triangulation + PnP,
  2D pose graph with SciPy, consistency check), saved map `slam_map.npz` + relocalization so
  later runs share the world frame. Hazards store a keyframe anchor (`anchor`, `anchor_right`,
  `anchor_forward` columns) and `RouteMemory.reanchor()` moves them after loop closure.
  `navigation/synthetic_scene.py` renders a textured room for hardware-free SLAM tests.
- Detector: custom YOLO26 `best.pt`, 23 classes: stop_sign, person, bicycle, bus, truck, car,
  motorbike, reflective_cone, ashcan, warning_column, spherical_roadblock, pole, dog, tricycle,
  fire_hydrant, pothole, stairs, curb, drain, tree, autorickshaw, animal, barrier.
  Ground-level classes (pothole, stairs, curb, drain) get distance from the box bottom edge
  (flat ground, `perception.camera_height_m`) and a severity bonus (`hazard.ground_hazard_bonus`).
  Needs `ultralytics>=8.4`.

## Known bugs (status re-verified in the code on `memory-novelty`, 8 Nov 2026)
Fixed:
1. ~~Pose doesn't follow the video~~: `--pose phone` (dead reckoning) or `--pose slam` (visual SLAM).
   `--pose sim` (default for video files) still invents motion: memory, negative evidence and
   memory cues are only meaningful with `slam` or `phone`.
2. ~~Heading convention mismatch~~: `local_to_world`, `world_to_local`, `AcousticAttention`,
   `simulated_pose` (walks along +y at heading 0) and the planner all use heading clockwise from
   +y (`tests/test_frames.py`).
3. ~~Decay not on main~~: decay + pruning are on this branch, now class-aware with reinforcement
   and an epoch memory clock (`tests/test_memory.py`, `tests/test_memory_decay_classes.py`).
   (Still not on `main` until this branch is merged.)
4. ~~IDs collide~~: stable uuid `hazard_id`; the YOLO track id is only a per-run hint.
5. ~~Blocking audio~~: `HRTFRenderer` queues cues for a worker thread (`tests/test_hrtf_audio.py`).
6. ~~Inconsistent config~~: every tunable is in `config.yaml`; `tests/test_config.py` checks the
   sections match the constructors; safety margin is 0.25 everywhere.
7. ~~Tests split~~: all automated tests are pytest files in `tests/` (`pytest.ini`); manual
   hardware checks are in `scripts/manual/`.

Still open:
- Monocular distance is approximate (object height / box bottom edge) until the ultrasonic
  sensor is fitted; ground-level classes cannot be seen closer than ~2.5 m (camera model).
- The memory evaluation is a simulation, not a user study; live two-walk tests are still needed.
- `README.md` still describes some older demos (`main.py`, the six-tier console demo) that are
  not on this branch.

## Conventions (follow exactly)
- **World frame:** x = right of the start heading, y = forward at the start, metres.
  **Heading** = degrees **clockwise from +y** (compass-like), so forward = (sin h, cos h).
- **Camera-relative detection:** `relative_x` = right (m), `relative_y` = forward (m).
- **local_to_world (the only correct form):**
  `world_x = x + rx*cos(h) + ry*sin(h)` and `world_y = y - rx*sin(h) + ry*cos(h)` (h in radians).
  Check: h = 90° means facing +x, so a point straight ahead (0, d) maps to (x + d, y).
- **Azimuth for audio:** negative = left, 0 = ahead, positive = right (physical). Convert to SOFA's
  convention only inside `hrtf_renderer.py`.
- The planner gets **current-frame hazards only**. Memory is for context and attention, never to block the path.
  Memory cues are audio only and never change the planner.
- **Memory clock:** persistent memory uses epoch seconds (`SessionClock`); tracking and TTC keep
  the pipeline clock. Tests inject a fake clock (`now_fn`) — never rely on "newest stored timestamp".
- **Audio priority:** urgent (where the danger is) > path guidance (where to walk) > memory cue
  (remembered hazard ahead). A lower kind never displaces a higher one.
- Every tunable number goes in `config.yaml`, loaded once and passed to the constructors.

## Workflow rules for Claude
- Before editing, state a short plan: the files you'll touch and how you'll test it. Then do it.
- One task = one focused change. Don't fix unrelated things; list them at the end instead.
- After every change, run `python -m pytest -q` and show the result. Add or extend a test for any
  logic you change (pure-logic tests must not need a camera, phone, GPU or audio device).
- Code that needs hardware (phone, ESP32, audio) must fail gracefully, with a clear message and a
  `--no-audio` / `--no-esp32` / `--sim-pose` fallback, so the pipeline still runs on a recorded video.
- Never commit generated files: `videos/*_navigation.*`, `*.db`, `runs/`, `datasets/`, recordings.
- Windows + PowerShell is the main environment: use `pathlib`, no bash-only commands in docs.
- Keep `requirements.txt` the single source of dependency truth; add a dependency only if needed
  (OK: `websocket-client`, `pyyaml`).
- When finished, summarise: what changed, how it was tested, anything left over or risky.
