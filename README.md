# Context-Aware BVI Navigation Prototype

This project contains both a deterministic architecture prototype and a real, dataset-backed YOLO training/inference path. The deterministic path makes the review reproducible; the ML path uses the submitted Obstacle Dataset and an NVIDIA GPU.

## What the prototype proves

- Six-tier modular architecture: sensing, perception, hazard triage, memory, reasoning, and audio output.
- Urgent hazards take a low-latency bypass directly to the audio layer.
- Every detection is still recorded in spatial memory, including urgent detections.
- Hazards remain known after leaving the current camera frame and fade using pose-aware time decay.
- Only the top-k hazards are sonified to reduce audio overload.
- Routes and confirmed hazards persist in SQLite across runs.
- The path planner returns both a route and a human-readable reason.
- If localization fails, current-frame detection and alerts continue without unsafe world-memory updates.

## Run on Windows or Linux

Python 3.10 or newer is sufficient. No GPU is required.

```bash
cd bvi_navigation_prototype
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Then run:

```bash
python main.py
```

> **Note (this branch):** `main.py` is not present on `slam-memory-demo`. The
> six-tier deterministic console demo described below is illustrative of the
> target architecture; on this branch, the equivalent working demos are
> `localization_demo.py` (simulated SLAM + persistent memory, no video) and
> `video_navigation_demo_slam_memory.py` (real YOLO detection + hazard
> scoring + simulated SLAM + persistent memory, on a recorded video — see
> below).

Run the tests (all automated tests live in `tests/`; no camera, phone, GPU or audio device needed):

```powershell
python -m pytest -q
```

Manual audio/device/video checks live in `scripts/manual/` (see its README) and are not run by pytest.

The program runs a deterministic six-frame walking scenario and prints detections, urgent bypass alerts, memory contents, top-k acoustic attention, and route explanations. Its database is created at `data/navigation.db`.

Reset the demo memory:

```bash
python main.py --reset
```

## Architecture mapping

| Report tier | Prototype implementation | Full-project replacement later |
| --- | --- | --- |
| Sensing | `SimulatedSensor` emits frame and pose packets | RGB-D/RealSense + IMU ROS2 node |
| Multimodal perception | `ScriptedPerception` emits labeled 3D detections | YOLO + segmentation + depth/SLAM |
| Hazard classifier | Rule-based distance, path alignment, motion scoring | Calibrated lightweight classifier |
| Persistent memory | In-memory spatial cache + SQLite route memory | Spatial index / optimized local DB |
| Context reasoning | Priority ranking + explainable two-route planner | Map-aware multipath planner |
| Spatial audio | Console renderer with left/right/distance/timbre semantics | True SOFA/HRTF binaural renderer |

The interfaces are intentional. A real detector can replace `ScriptedPerception` as long as it returns `Detection` objects; other modules do not need to change.

## Simulated localization demo

`navigation/sensing.py` is the localization stand-in used until real SLAM exists:

```text
SimulatedSensor -> Pose -> Detection -> local_to_world() -> WorldPoint
                                                                |
                                               LocalizedDetection (handoff)
                                                                |
                                                   memory.save(...)  [teammate's module]
```

- `SimulatedSensor` emits six fixed, deterministic `Pose` objects (no camera, no IMU) standing in for a future real SLAM system (RealSense RGB-D + IMU -> RTAB-Map/ORB-SLAM3 -> `Pose`). Nothing downstream needs to change when that swap happens.
- `local_to_world(pose, detection)` rotates and translates a detection's camera-relative `(relative_x, relative_y)` by the current pose's heading and position, producing a `WorldPoint(x, y)`. Heading is measured clockwise from world +y (heading 0° = forward, 90° = facing +x/right), so forward = (sin h, cos h).
- `localize_detection(pose, detection)` is the actual handoff function: it calls `local_to_world()` and bundles the result with the detection's id, label, confidence, motion, and the pose's timestamp into a `LocalizedDetection`, ready for `navigation/memory.py` (not yet implemented) to consume, e.g. `memory.save(localized)`.

Run the localization-only demo (no OpenCV/YOLO required):

```bash
python localization_demo.py
```

It prints, for each of the six simulated frames, the robot pose, the hand-picked hazard detection's local position, its resulting world position, and the `LocalizedDetection` that would be handed to memory.

## Real dataset and GPU path

Follow `TRAINING_GUIDE_WINDOWS.md`. The workflow is:

```text
Official OD VOC dataset
  -> tools/prepare_obstacle_dataset.py
  -> validated YOLO dataset + obstacle_dataset.yaml
  -> train_detector.py (YOLOv8n transfer learning)
  -> evaluate_detector.py (untouched test split)
  -> video_detector_demo.py (recorded video)
  -> YOLOPerception Detection objects
  -> existing hazard/memory/reasoning/audio architecture
```

The video adapter uses tracking IDs and approximate monocular ranging. RGB-D/SLAM can later replace the range and pose sources without changing hazard triage or reasoning.

Run detector-only inference on a recorded video:

```bash
python video_detector_demo.py --weights runs/detect/obstacle_yolov8n/weights/best.pt --video path/to/video.mp4
```

Run the full navigation pipeline on a recorded video:

```bash
python video_navigation_demo.py --weights runs/detect/obstacle_yolov8n/weights/best.pt --video path/to/video.mp4
```

`video_navigation_demo.py` covers detection and hazard scoring only — it does
not touch localization or memory.

### Full pipeline demo: detection + simulated SLAM + persistent memory

`video_navigation_demo_slam_memory.py` wires the remaining two tiers in:

```text
video frame
  -> YOLOPerception            (real YOLO detection + monocular distance/motion)
  -> HazardSeverityClassifier  (rule-based hazard scoring)
  -> simulated_pose()          (per-frame simulated localization, see navigation/sensing.py)
  -> local_to_world()          (navigation/sensing.py, unchanged)
  -> RouteMemory.remember()    (navigation/memory.py, SQLite persistence, unchanged)
  -> AcousticAttention.rank()  (navigation/reasoning.py, unchanged)
  -> annotated video + JSON report + navigation_memory.db
```

Every detected object is converted into world coordinates and written to
`navigation_memory.db` on every frame, whether or not it triggered a
warning. The video overlay shows the live simulated pose, active warnings,
and the top-k hazards currently ranked from persistent memory.

```bash
python video_navigation_demo_slam_memory.py --weights yolo11n.pt --source path/to/video.mp4 --output output_final.mp4 --device cpu
```

Add `--reset` to clear `navigation_memory.db` before the run instead of
appending to it. Add `--db path/to/file.db` to use a different database
file.

`simulated_pose()` is **not** real SLAM. It generates a deterministic
walking trajectory per frame the same way `navigation/sensing.py`'s
`SimulatedSensor` does for its six-frame demo, just extended to a video of
any length. It exists so the rest of the pipeline (hazard triage, memory,
reasoning) can be built and demonstrated against a stable interface before
a real SLAM system (RealSense RGB-D + IMU → RTAB-Map/ORB-SLAM3 → `Pose`)
is dropped in — see `navigation/sensing.py`'s own docstring.

## Visual SLAM (`--pose slam`)

`navigation/vslam.py` is a **real, lightweight keyframe monocular visual SLAM**
that runs on the laptop (OpenCV + NumPy + SciPy only, no phone app, no ROS).
It replaces the simulated pose and improves on phone dead reckoning, behind the
same `Pose` interface, so hazard triage, planner, memory, attention and HRTF are
unchanged.

| Part | What it does |
| --- | --- |
| Tracking | ORB features (grid-spread) matched to the current keyframe; essential matrix (RANSAC) gives rotation + translation direction. Features on dynamic YOLO boxes (people, vehicles, animals) are masked out. |
| Scale | Metric distance from the phone's step counter (steps x `pose.stride_m`); camera-only runs use `slam.assumed_speed_m_s`. |
| Heading | Visual yaw, pulled gently towards the phone's rotation-vector heading at each keyframe. |
| Map | Keyframes: pose + ORB features + a thumbnail for place recognition. |
| Loop closure | A revisited place is recognized, its metric relative pose measured (triangulation + PnP), and a 2D pose graph is optimized (SciPy least squares). A loop that would tear the odometry apart is rejected. |
| Hazard re-anchoring | Each remembered hazard stores the keyframe it was seen from; when loop closure moves keyframes, `RouteMemory.reanchor()` moves the hazards too. |
| Saved map + relocalization | The map is saved to `slam.map_path` (`slam_map.npz`). The next run recognizes where it is in that map and continues in **the same world frame**, so hazards in persistent memory line up across walks. Memory writing is paused until then. |
| Tracking loss | Falls back to phone dead reckoning (if connected) while trying to re-find a keyframe; without the phone the pose is invalid (alerts continue, memory pauses). |

Run it:

```powershell
# Live (IP Webcam + SensorServer). First run builds the map; later runs reuse it.
python video_navigation_demo_slam_memory.py --pose slam --trajectory-out runs\walk.csv

# Start fresh (new map + empty memory)
python video_navigation_demo_slam_memory.py --pose slam --slam-new-map --reset

# Recorded video, camera only
python video_navigation_demo_slam_memory.py --source videos\walk.mp4 --pose slam --no-phone --slam-new-map --reset
```

The annotated video has a mini-map (bottom right): keyframe path, loop-closure
links, remembered hazards, the planner's lanes and the SLAM state
(TRACKING / RELOCALIZING / LOST).

**Walking lanes** (`navigation/path_overlay.py`, every pose mode): the five
lanes the local planner checks each frame (-1.5, -0.75, 0, +0.75, +1.5 m,
`planner.path_width_m` wide, out to `planner.horizon_m`) are projected onto the
ground in the video with the same camera model perception uses. The chosen lane
is green with an arrow from the user ("AHEAD", or "GO LEFT" when the current
lane is blocked), blocked lanes are red, the current lane is outlined in white.
`--no-lanes` turns this off.
The JSON report has a `slam` block (keyframes, loop closures, relocalizations,
hazards moved, events, ms per frame).

For relocalization to work, **start each walk where an earlier walk passed**,
ideally at the same taped start mark facing the same way.

Calibrate the camera once (better accuracy than the `slam.hfov_deg` guess):
`python scripts/manual/calibrate_camera.py --source http://<PHONE_IP>:8080/video`,
then paste the printed `fx, fy, cx, cy, calib_width` into `config.yaml`.

Evaluation tools:

```powershell
# Simulation: SLAM vs phone dead reckoning on a known square path (no hardware)
python tools/simulate_slam_walk.py --laps 2 --video runs\sim_slam.mp4

# Our own walks: drift of a closed loop for each pose source
python tools/eval_walk_drift.py runs\loop_phone.csv runs\loop_slam.csv --png runs\loop.png

# TUM RGB-D benchmark (planar pioneer sequences fit the 2D model best)
python tools/eval_slam_tum.py C:\data\rgbd_dataset_freiburg2_pioneer_slam --camera fr2 --out runs\tum
```

Simulation result (`tools/simulate_slam_walk.py --laps 2`: 64 m, 7 % stride
error, phone heading drift 0.3 deg/s): end-point error 0.13 m with SLAM vs
1.58 m with phone dead reckoning (drift 0.2 % vs 2.5 %), 31 loop closures.

SLAM limitations (state them honestly):

- Monocular, 2D ground-plane pose (x, y, heading); assumes roughly flat ground.
  No bundle adjustment or 3D map, so it is far simpler than ORB-SLAM3.
- Metric scale is only as good as the step length (`pose.stride_m`) or the
  assumed speed; loop closure cannot fix a wrong stride.
- A pure sideways step in front of distant surfaces looks like a small turn
  to one camera; the walking-forward assumption covers normal walking.
- Relocalization works best within a few metres of where an earlier walk
  passed, facing a similar direction.
- Low texture (blank walls), low light, motion blur and very fast turns can
  lose tracking.
- Not yet evaluated on TUM / EuRoC or real campus walks: run the tools above.

## Verifying persistent memory with SQL

`video_navigation_demo_slam_memory.py` writes to `navigation_memory.db`, a
plain SQLite file, through `navigation/memory.py`'s `RouteMemory` class. You
can inspect it directly with SQL, independent of any of this project's
Python code — the most convincing way to demonstrate the memory is real and
actually persists.

**One-time setup (Windows):** install the SQLite command-line shell if you
don't already have it, then restart your terminal so `sqlite3` is on PATH:

```powershell
winget install -e --id SQLite.SQLite
```

**Inspect the database:**

```powershell
cd C:\path\to\BVI-navigation
sqlite3 navigation_memory.db
```

At the `sqlite>` prompt:

```sql
.headers on
.mode column
.schema hazards

SELECT COUNT(*) AS total_hazards FROM hazards;

SELECT object_id, label, world_x, world_y, confidence, timestamp
FROM hazards
ORDER BY timestamp DESC
LIMIT 15;

.quit
```

`.schema hazards` shows the table was created by `RouteMemory._create_table()`,
not hand-edited. The `SELECT` returns real rows written during the video
run: object ID, label, the world-frame coordinates from `local_to_world()`,
detection confidence, and the last-seen timestamp.

If `sqlite3` isn't installed, the same data is reachable from Python's
built-in `sqlite3` module without installing anything extra:

```powershell
python -c "import sqlite3; c = sqlite3.connect('navigation_memory.db'); c.row_factory = sqlite3.Row; [print(dict(r)) for r in c.execute('SELECT * FROM hazards ORDER BY timestamp DESC LIMIT 15')]"
```

## Code-review walkthrough

1. Start with `NavigationPipeline.process_cycle()` in `navigation/pipeline.py`. It is the architecture in executable form.
2. Point out that urgent detections call the audio renderer immediately, before memory ranking.
3. Point out that memory update is a separate unconditional step, so “bypass” means bypassing ranking latency, not skipping storage.
4. Show `RouteMemory` (`navigation/memory.py`) and `AcousticAttention.rank()` (`navigation/reasoning.py`) as the main research contribution. **Note:** the time-decay/pruning behavior (`SpatialMemory.decay_and_prune()`) described in earlier drafts of this architecture is not implemented on this branch — `RouteMemory` persists hazards but does not currently age or prune them.
5. Stop localization for one frame in the demo and show graceful degradation.
6. Restart the program without `--reset` and show that route hazards remain in SQLite.

## Honest prototype limitations

- The deterministic demo uses scripted perception; the optional ML path performs real YOLO inference.
- Positions are already expressed as relative 3D coordinates; no calibrated depth reconstruction is performed.
- Audio is represented as structured console events, not true HRTF waveforms.
- Route planning compares two predefined route candidates rather than a full semantic map.
- `video_navigation_demo_slam_memory.py`'s pose: `--pose slam` is real
  camera-based visual SLAM (see "Visual SLAM" above for its limits),
  `--pose phone` is phone dead reckoning, `--pose sim` is a deterministic
  simulated trajectory.
- YOLO's tracker can assign a new ID to the same physical object after
  occlusion or brief loss of track, so persistent-memory object counts can
  run higher than the number of distinct real-world objects actually
  present.

These are adapter-level limitations, not architectural shortcuts. The safety path, memory behavior, prioritization, persistence, explainability, and failure handling are executable.
