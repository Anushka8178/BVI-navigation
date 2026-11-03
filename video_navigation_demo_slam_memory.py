"""
video_navigation_demo_slam_memory.py

Full navigation pipeline demo on a recorded video:

    video frame
        -> YOLOPerception            (real YOLO detection + monocular distance/motion)
        -> HazardSeverityClassifier  (rule-based hazard scoring)
        -> simulated_pose()          (documented SLAM stand-in, see navigation/sensing.py)
        -> local_to_world()          (navigation/sensing.py, unchanged)
        -> RouteMemory.remember()    (navigation/memory.py, SQLite persistence, unchanged)
        -> AcousticAttention.rank()  (navigation/reasoning.py, unchanged)
        -> annotated video + JSON report + navigation_memory.db

This wires together three pieces that already existed in the repo as
separate, working-but-unconnected modules:

  - navigation/hazard.py    real, rule-based hazard detection
  - navigation/sensing.py   simulated localization -- a documented stand-in
                             for real SLAM, not real pose estimation from
                             the camera (see that file's own docstring)
  - navigation/memory.py    real SQLite persistent memory

Nothing in those three files is modified. The only new code here is the CLI
video loop and simulated_pose(), which generalizes sensing.SimulatedSensor's
six hardcoded poses to a video of any length, reusing the same Pose
dataclass and the same local_to_world() math.

Run it exactly like video_navigation_demo.py, e.g.:

    python video_navigation_demo_slam_memory.py --weights yolo11n.pt --source video_for_testing.mp4 --output output_slam_memory.mp4 --device cpu

To start persistent memory fresh instead of appending to the existing
navigation_memory.db, add --reset.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from navigation.audio import make_warning
from navigation.hazard import HazardSeverityClassifier
from navigation.memory import RouteMemory
from navigation.models import FramePacket, Pose
from navigation.reasoning import AcousticAttention
from navigation.sensing import localize_detection
from navigation.yolo_perception import YOLOPerception


ImageArray = NDArray[np.uint8]


def simulated_pose(timestamp: float) -> Pose:
    """
    One Pose per video frame.

    navigation/sensing.py's SimulatedSensor only ships six hand-picked poses,
    built for a six-frame console demo. This extends the exact same idea --
    a fixed, deterministic walking trajectory, no camera, no IMU -- to a
    video of any length, so the same Pose dataclass and the same
    local_to_world()/localize_detection() functions from sensing.py can be
    used completely unchanged.

    This is still the documented stand-in for real SLAM described in
    sensing.py's own docstring, not real pose estimation from the video.
    Swapping it for RealSense + ORB-SLAM3/RTAB-Map later requires no changes
    to anything downstream (hazard, memory, reasoning).
    """
    walking_speed_m_s = 0.5
    lateral_drift_m_s = 0.05
    turn_rate_deg_s = 1.0

    return Pose(
        x=walking_speed_m_s * timestamp,
        y=lateral_drift_m_s * timestamp,
        heading_deg=turn_rate_deg_s * timestamp,
        timestamp=timestamp,
    )


def create_writer(path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc("m", "p", "4", "v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {path.resolve()}")
    return writer


def draw_status(
    image: ImageArray,
    pose: Pose,
    detection_count: int,
    warnings: list,
    top_hazards: list,
) -> None:
    image_height, image_width = image.shape[:2]
    panel_height = min(185, image_height)

    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)

    lines = [
        f"SLAM(sim) pose: x={pose.x:.2f} y={pose.y:.2f} heading={pose.heading_deg:.1f} deg",
        f"Detections: {detection_count} | Warnings: {len(warnings)}",
    ]

    if warnings:
        for warning in warnings[:2]:
            lines.append(
                f"WARNING: {warning.label} | {warning.direction} | "
                f"{warning.distance_m:.1f} m | {warning.motion} | score={warning.score:.2f}"
            )
    else:
        lines.append("Status: no urgent hazard")

    if top_hazards:
        labels = ", ".join(f"{h.label}({h.priority:.2f})" for h in top_hazards)
        lines.append(f"Persistent memory top-k: {labels}")

    for index, line in enumerate(lines):
        colour = (0, 0, 255) if line.startswith("WARNING") else (255, 255, 255)
        cv2.putText(
            image,
            line,
            (15, 26 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Video BVI navigation demo: detection + hazard scoring + simulated SLAM + persistent memory"
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--db", type=Path, default=Path("navigation_memory.db"))
    parser.add_argument("--reset", action="store_true", help="Clear persistent memory before this run")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"Trained model does not exist: {args.weights.resolve()}")
    if not args.source.exists():
        raise SystemExit(f"Input video does not exist: {args.source.resolve()}")

    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise SystemExit(f"Could not open input video: {args.source.resolve()}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width <= 0 or frame_height <= 0:
        capture.release()
        raise SystemExit("Could not determine the video's frame size.")

    try:
        writer = create_writer(args.output, fps, (frame_width, frame_height))
    except RuntimeError as error:
        capture.release()
        raise SystemExit(str(error)) from error

    perception = YOLOPerception(weights=args.weights, confidence=args.confidence, device=args.device)
    severity = HazardSeverityClassifier()
    memory = RouteMemory(str(args.db))
    attention = AcousticAttention(top_k=3)

    if args.reset:
        memory.clear()
        print(f"Cleared persistent memory: {args.db.resolve()}")

    frame_id = 0
    warning_frames = 0
    unique_warning_ids: set[str] = set()
    warning_events: list[dict[str, Any]] = []

    processing_started = time.perf_counter()

    try:
        while True:
            success, captured_image = capture.read()
            if not success or captured_image is None:
                break

            image: ImageArray = np.asarray(captured_image, dtype=np.uint8)
            frame_id += 1
            timestamp = (frame_id - 1) / fps

            frame_packet = FramePacket(frame_id=frame_id, timestamp=timestamp, image=image)
            pose = simulated_pose(timestamp)

            detections = perception.process(frame_packet)

            warnings = []
            for detection in detections:
                result = severity.classify(detection)

                # Every detection is recorded in persistent memory, in world
                # coordinates, whether or not it is urgent -- matches the
                # architecture documented in README.md.
                localized = localize_detection(pose, detection)
                memory.remember(
                    object_id=localized.object_id,
                    label=localized.label,
                    world_x=localized.world_x,
                    world_y=localized.world_y,
                    confidence=localized.confidence,
                    timestamp=localized.timestamp,
                )

                if result.urgent:
                    warnings.append(make_warning(detection, result.score, result.reason))

            top_hazards = attention.rank(memory.get_hazards(), pose)

            if perception.last_result is not None:
                display: ImageArray = np.asarray(perception.last_result.plot(), dtype=np.uint8)
            else:
                display = image.copy()

            draw_status(display, pose, len(detections), warnings, top_hazards)
            writer.write(display)

            if warnings:
                warning_frames += 1
                for warning in warnings:
                    unique_warning_ids.add(warning.object_id)
                    warning_events.append(
                        {
                            "frame": frame_id,
                            "time_s": round(timestamp, 3),
                            "object_id": warning.object_id,
                            "label": warning.label,
                            "direction": warning.direction,
                            "distance_band": warning.distance_band,
                            "distance_m": round(warning.distance_m, 3),
                            "score": round(warning.score, 3),
                            "motion": warning.motion,
                            "reason": warning.reason,
                            "pose": {
                                "x": round(pose.x, 3),
                                "y": round(pose.y, 3),
                                "heading_deg": round(pose.heading_deg, 2),
                            },
                        }
                    )
                    print(
                        f"WARNING t={timestamp:.2f}s | {warning.label} | {warning.direction} | "
                        f"{warning.distance_m:.1f} m | {warning.motion} | score={warning.score:.2f} "
                        f"| pose=({pose.x:.2f},{pose.y:.2f})"
                    )

            if not args.no_display:
                cv2.imshow(
                    "BVI Navigation - detection + SLAM(sim) + memory - press Q to stop",
                    display,
                )
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    print("Processing stopped by user.")
                    break

    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    processing_seconds = time.perf_counter() - processing_started
    persisted_hazards = memory.get_hazards()

    report = {
        "source_video": str(args.source.resolve()),
        "model_weights": str(args.weights.resolve()),
        "annotated_video": str(args.output.resolve()),
        "memory_db": str(args.db.resolve()),
        "frames_processed": frame_id,
        "video_fps": fps,
        "warning_frames": warning_frames,
        "unique_warning_objects": len(unique_warning_ids),
        "processing_seconds": round(processing_seconds, 3),
        "warning_events": warning_events,
        "persistent_memory": [
            {
                "object_id": h.object_id,
                "label": h.label,
                "world_x": round(h.world_x, 3),
                "world_y": round(h.world_y, 3),
                "confidence": round(h.confidence, 3),
                "last_seen_timestamp": round(h.timestamp, 3),
            }
            for h in persisted_hazards
        ],
        "limitations": [
            "Distance is estimated from object height, not measured depth.",
            "Pose comes from a deterministic simulated trajectory (the documented "
            "SLAM stand-in in navigation/sensing.py), not real camera-based localization.",
            "Approaching motion is determined from frame-to-frame estimated distance.",
        ],
    }

    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print("PROCESSING COMPLETE")
    print(f"Frames processed: {frame_id}")
    print(f"Warning frames: {warning_frames}")
    print(f"Unique warning objects: {len(unique_warning_ids)}")
    print(f"Objects in persistent memory (SQLite): {len(persisted_hazards)}")
    print(f"Annotated video: {args.output.resolve()}")
    print(f"JSON report: {report_path.resolve()}")
    print(f"Memory DB: {args.db.resolve()}")


if __name__ == "__main__":
    main()
