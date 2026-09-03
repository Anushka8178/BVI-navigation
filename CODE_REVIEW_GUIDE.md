# Code Review Guide

## One-sentence project explanation

This prototype is an offline navigation decision pipeline for BVI users that detects and triages hazards, remembers them outside the camera view, limits audio overload, and explains why it chooses a route.

## What to demonstrate in five minutes

1. Run `python main.py --reset`.
2. In frame 2, show the approaching person marked `URGENT BYPASS`. Explain that this output is produced before cache ranking, reducing safety-critical latency.
3. In frame 3, note that the chair is still in the top-k output even though it is no longer detected in that frame. This proves spatial persistence.
4. In frame 4, show the pothole bypass and the three-item attention limit.
5. In frame 5, localization becomes invalid. New detections are still classified, but the system does not write guessed world coordinates or render cached spatial cues.
6. In frame 6, no objects are detected, but remembered static objects remain available and gradually fade.
7. At the end, show that the route planner selects the courtyard detour and prints its numerical rationale.

## The most important method

Open `navigation/pipeline.py` and explain `process_cycle()` in this order:

```text
sensor packet
  -> perception
  -> severity classifier
       -> urgent audio bypass
  -> spatial memory update and decay
  -> top-k acoustic attention
  -> awareness audio
  -> selected stable hazards persisted to route memory
```

There are two paths after classification:

- Safety path: urgent detection -> immediate output.
- Context path: all safely localizable detections -> memory -> ranking -> limited output.

The paths are not mutually exclusive. An urgent object is also remembered when localization is valid.

## Module-by-module explanation

### `models.py`

Defines typed data passed between modules. This prevents one module from depending on another module's internal implementation.

### `sensing.py`

Produces synchronized frame and pose packets. It is simulated now; the real version will read the RGB-D camera and IMU.

### `perception.py`

Returns labeled detections with relative 3D position, confidence, and motion. It represents the combined outputs of YOLO, semantic segmentation, depth, and SLAM.

### `hazard.py`

Uses understandable features: distance, direction alignment, motion, and hazardous class. A near approaching person or aligned pothole can cross the urgency threshold.

### `memory.py`

`SpatialMemory` transforms relative coordinates to world coordinates, then applies exponential decay. `RouteMemory` uses SQLite to retain confirmed route hazards across program restarts.

### `reasoning.py`

`AcousticAttention` computes the report's weighted priority equation and returns only top-k entries. `ExplainablePathPlanner` combines distance, base risk, and remembered risk and returns a rationale.

### `audio.py`

Maps class to timbre, distance to repetition band, and angle to left/ahead/right. It prints structured events now. A true HRTF renderer can replace this adapter.

## Questions the panel may ask

**Why not implement the full project now?**  
The full system requires calibrated hardware, GPU inference, training data, and user testing. This review prototype validates architecture, control flow, persistence, explainability, and failure behavior first.

**Is the perception real?**  
Not in this prototype. Inputs are deterministic so every reviewer sees the same safety and memory cases. The perception interface is ready for a YOLO/SLAM adapter.

**Is this really HRTF audio?**  
No. It implements the semantic contract for spatial audio but prints events. True HRTF requires a SOFA HRTF dataset and binaural convolution, planned for the next phase.

**Why SQLite?**  
It is local, offline, lightweight, inspectable, and already part of the submitted technology stack.

**How does it avoid duplicate hazards?**  
The prototype uses stable object IDs from perception. The next version should associate detections by class plus a spatial matching radius, as described in the report pseudocode.

**What if SLAM fails?**  
The system does not create guessed world-memory entries or play stale spatial cues. Frame-level hazard classification remains available, demonstrating graceful degradation.

**What is novel here?**  
The integration of field-of-view-independent egocentric hazard persistence, decay, top-k acoustic attention, and an urgent low-latency bypass within one assistive navigation pipeline.

## Next implementation milestones

1. Replace scripted perception with a webcam/YOLO CPU adapter.
2. Add monocular depth as a temporary substitute for RGB-D hardware.
3. Replace stable IDs with spatial association and deduplication.
4. Publish module messages as ROS2 topics after moving to Ubuntu/Jetson.
5. Add SOFA-based HRTF convolution and measure localization accuracy.
6. Integrate RGB-D SLAM and validate coordinate transforms.

Do not claim the prototype has real YOLO, SLAM, or HRTF. Say it validates the architecture and the project's core algorithms while keeping hardware-dependent modules replaceable.
