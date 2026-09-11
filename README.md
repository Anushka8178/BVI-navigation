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

Run the tests:

```bash
python -m unittest discover -s tests -v
```

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
- `local_to_world(pose, detection)` rotates and translates a detection's camera-relative `(relative_x, relative_y)` by the current pose's heading and position, producing a `WorldPoint(x, y)`. Heading 0° means the robot's +x direction.
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
- `video_navigation_demo_slam_memory.py`'s pose comes from a deterministic
  simulated trajectory (`simulated_pose()` / `navigation/sensing.py`), not
  real camera-based localization.
- YOLO's tracker can assign a new ID to the same physical object after
  occlusion or brief loss of track, so persistent-memory object counts can
  run higher than the number of distinct real-world objects actually
  present.

These are adapter-level limitations, not architectural shortcuts. The safety path, memory behavior, prioritization, persistence, explainability, and failure handling are executable.
