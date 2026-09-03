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

## Real dataset and GPU path

Follow `TRAINING_GUIDE_WINDOWS.md`. The workflow is:

```text
Official OD VOC dataset
  -> tools/prepare_obstacle_dataset.py
  -> validated YOLO dataset + obstacle_dataset.yaml
  -> train_detector.py (YOLOv8n transfer learning)
  -> evaluate_detector.py (untouched test split)
  -> live_detector_demo.py (webcam/video)
  -> YOLOPerception Detection objects
  -> existing hazard/memory/reasoning/audio architecture
```

The real adapter uses tracking IDs and approximate monocular ranging. RGB-D/SLAM can later replace the range and pose sources without changing hazard triage or reasoning.

## Code-review walkthrough

1. Start with `NavigationPipeline.process_cycle()` in `navigation/pipeline.py`. It is the architecture in executable form.
2. Point out that urgent detections call the audio renderer immediately, before memory ranking.
3. Point out that memory update is a separate unconditional step, so “bypass” means bypassing ranking latency, not skipping storage.
4. Show `SpatialMemory.decay_and_prune()` and `AcousticAttention.rank()` as the main research contribution.
5. Stop localization for one frame in the demo and show graceful degradation.
6. Restart the program without `--reset` and show that route hazards remain in SQLite.

## Honest prototype limitations

- The deterministic demo uses scripted perception; the optional ML path performs real YOLO inference.
- Positions are already expressed as relative 3D coordinates; no calibrated depth reconstruction is performed.
- Audio is represented as structured console events, not true HRTF waveforms.
- Route planning compares two predefined route candidates rather than a full semantic map.

These are adapter-level limitations, not architectural shortcuts. The safety path, memory behavior, prioritization, persistence, explainability, and failure handling are executable.
