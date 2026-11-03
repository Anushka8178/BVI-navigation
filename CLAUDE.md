# CLAUDE.md — BVI Navigation (read this before every task)

## What this project is
Context-aware navigation assistant for blind and visually impaired (BVI) users.
YOLO detection → distance/direction → world-frame tracking → persistent hazard memory
(with decay) → local path planner + urgent bypass → top-k acoustic attention → HRTF audio.

**Hard deadline: 8 Nov 2026 (final review/demo).** Prefer small, working, tested changes over
big refactors. Do not add heavy new dependencies. Do not rewrite modules that already work.

## Target setup for 8 Nov (the "Android rig")
- **Camera:** Android phone, chest-mounted, running the *IP Webcam* app.
  The OpenCV source is `http://<PHONE_IP>:8080/video` (MJPEG).
- **Motion sensors:** the same phone runs the *SensorServer* app, which serves JSON over WebSocket:
  `ws://<PHONE_IP>:<PORT>/sensor/connect?type=android.sensor.rotation_vector`
  and `...type=android.sensor.step_detector`. Each message is JSON with `values` (floats) and
  `timestamp` (ns).
- **Range sensor:** an ESP32 with an HC-SR04 ultrasonic sensor (2–400 cm, 15° cone), mounted at chest height
  facing forward, next to the phone camera. The ESP32 also drives a vibration motor.
- **Compute:** a Windows laptop (RTX 3050) runs this Python pipeline. The laptop is connected to the phone's hotspot.
- **Audio:** HRTF cues from the laptop through **wired** earphones. Avoid Bluetooth, which adds about 100–250 ms.

### ESP32 ⇄ laptop protocol (UDP, plain ASCII, one message per datagram)
- ESP32 → laptop port **5005**: `D,<distance_cm>,<esp_millis>` about 15 times a second.
  If there's no echo, send `D,-1,<esp_millis>`.
- Laptop → ESP32 port **5006**: `V,<duration_ms>` vibrates the motor; `V,0` stops it.
- The ESP32 IP and the ports must be configurable (in `config.yaml`, never hard-coded in logic).

## Repository facts
- Main repo: `ManishAdithya/BVI-navigation`, branch `main`.
- Work that is **not yet on main** lives on the fork `https://github.com/Anushka8178/BVI-navigation`:
  - `slam-memory-demo` → `RouteMemory.decay_and_prune()` (exponential decay, floor, max_entries)
    + `tests/test_memory.py` + the SQLite connection-closing fix (`contextlib.closing`).
  - `path-planner-hrtf-updated` → newest planner (TTC filtering, safety_margin 0.25) + HRTF
    physical→SOFA azimuth conversion, with path guidance played at the *walking* direction.
- Main entry point: `video_navigation_demo_slam_memory.py`.
- Modules: `navigation/` (models, yolo_perception, hazard, sensing, memory, planner, trajectory,
  reasoning, audio, hrtf_renderer). HRTF file: `hrtf/mit_kemar_normal_pinna.sofa`.
- Detector: custom YOLO11n `best.pt`, 15 OD classes. These classes do NOT include stairs, curbs or potholes.

## Known bugs to fix (verified in the code)
1. **Pose doesn't follow the video:** `simulated_pose(t)` invents motion. Replace it with `PhonePose`.
2. **Heading convention mismatch:** `AcousticAttention` uses forward = (sin h, cos h), i.e. heading
   clockwise from +y, but `local_to_world` rotates counter-clockwise, and `simulated_pose` walks along +x
   at heading 0.
3. **Decay not on main:** `RouteMemory.remember()` hard-codes `decay_weight = 1.0`.
4. **IDs collide:** `object_id = f"{label}-{track_id}"`, and YOLO track IDs restart every run.
5. **Blocking audio:** `sounddevice` playback with `blocking=True` freezes the frame loop.
6. **Inconsistent config:** parameters are duplicated in code and the docs disagree (safety margin 0.50 vs 0.25).
7. **Tests split:** pytest-style functions are run with `unittest discover`, and test files are scattered
   at the repo root.

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
