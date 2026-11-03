"""Recorded-video BVI navigation demo.

Pipeline:
    video
      -> YOLO detection/tracking
      -> approximate monocular distance + physical direction
      -> localization + trajectory velocity
      -> current-frame local path planning + predicted collision/TTC
      -> hazard severity classification
      -> persistent SQLite memory + acoustic attention
      -> urgent HRTF or soft path-guidance HRTF
      -> annotated video + JSON report

Important:
    Pose/localization is simulated in this prototype. It is not real SLAM.

Example:
    python video_navigation_demo_slam_memory.py \
        --weights best.pt \
        --source videos\\street6.mp4 \
        --output videos\\street6_navigation.mp4 \
        --device cpu \
        --reset
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
from navigation.hrtf_renderer import HRTFRenderer
from navigation.memory import RouteMemory
from navigation.models import FramePacket, Pose
from navigation.planner import LocalPathPlanner
from navigation.reasoning import AcousticAttention
from navigation.sensing import localize_detection
from navigation.trajectory import HazardTrajectoryTracker
from navigation.yolo_perception import YOLOPerception


ImageArray = NDArray[np.uint8]


def simulated_pose(timestamp: float) -> Pose:
    """Deterministic simulated walking pose used instead of real SLAM."""
    walking_speed_m_s = 0.5
    lateral_drift_m_s = 0.05
    turn_rate_deg_s = 1.0

    return Pose(
        x=walking_speed_m_s * timestamp,
        y=lateral_drift_m_s * timestamp,
        heading_deg=turn_rate_deg_s * timestamp,
        timestamp=timestamp,
    )


def create_writer(
    path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc("m", "p", "4", "v")
    writer = cv2.VideoWriter(
        str(path),
        fourcc,
        fps,
        frame_size,
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create output video: {path.resolve()}"
        )

    return writer


def draw_status(
    image: ImageArray,
    pose: Pose,
    detection_count: int,
    warnings: list,
    top_hazards: list,
    path_decision,
    audio_enabled: bool,
) -> None:
    """Draw a compact navigation status panel on the frame."""
    image_height, image_width = image.shape[:2]
    panel_height = min(330, image_height)

    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (image_width, panel_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(
        overlay,
        0.65,
        image,
        0.35,
        0,
        image,
    )

    lines = [
        (
            f"Pose(sim): x={pose.x:.2f} y={pose.y:.2f} "
            f"heading={pose.heading_deg:.1f}°"
        ),
        f"Detections: {detection_count} | Urgent warnings: {len(warnings)}",
        f"Spatial audio: {'ON' if audio_enabled else 'OFF'}",
    ]

    if path_decision.current_path_safe:
        lines.append("PATH: SAFE | continue ahead")
    else:
        if path_decision.selected_path == "ahead":
            recommendation = "STOP / WAIT"
        else:
            recommendation = f"MOVE {path_decision.selected_path.upper()}"
        lines.append(f"PATH: UNSAFE | {recommendation}")

    if warnings:
        for warning in warnings[:2]:
            ttc = (
                "none"
                if warning.ttc_s is None
                else f"{warning.ttc_s:.2f}s"
            )
            lines.append(
                f"URGENT: {warning.label} | {warning.direction} | "
                f"{warning.distance_m:.1f}m | {warning.category} | TTC={ttc}"
            )
    else:
        lines.append("Status: no urgent hazard")

    if top_hazards:
        labels = ", ".join(
            f"{hazard.label}({hazard.priority:.2f})"
            for hazard in top_hazards
        )
        lines.append(f"Memory top-k: {labels}")

    if not path_decision.current_path_safe:
        blockers = ", ".join(
            path_decision.blocking_hazards[:2]
        ) or "predicted obstacle"
        ttc = (
            "none"
            if path_decision.current_path_ttc_s is None
            else f"{path_decision.current_path_ttc_s:.2f}s"
        )
        lines.append(f"Blocked by: {blockers} | TTC={ttc}")

    for index, line in enumerate(lines):
        if line.startswith("URGENT") or line.startswith("PATH: UNSAFE"):
            colour = (0, 0, 255)
        elif line.startswith("PATH: SAFE"):
            colour = (0, 180, 0)
        else:
            colour = (255, 255, 255)

        cv2.putText(
            image,
            line,
            (15, 26 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            colour,
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "BVI navigation demo: YOLO + path planning + "
            "trajectory/TTC + persistent memory + HRTF"
        )
    )

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="YOLO weights, e.g. best.pt",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Input recorded video",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Annotated output video",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="YOLO confidence threshold",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="YOLO device, e.g. cpu or 0",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("navigation_memory.db"),
        help="SQLite navigation memory database",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear persistent memory before this run",
    )
    parser.add_argument(
        "--hrtf",
        type=Path,
        default=Path("hrtf/mit_kemar_normal_pinna.sofa"),
        help="SOFA HRTF file",
    )
    parser.add_argument(
        "--audio-device",
        default=None,
        help="sounddevice output device index or name",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable HRTF audio",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not show the OpenCV window",
    )

    args = parser.parse_args()

    if not args.weights.exists():
        raise SystemExit(
            f"Trained model does not exist: {args.weights.resolve()}"
        )

    if not args.source.exists():
        raise SystemExit(
            f"Input video does not exist: {args.source.resolve()}"
        )

    if not args.no_audio and not args.hrtf.exists():
        raise SystemExit(
            f"HRTF SOFA file does not exist: {args.hrtf.resolve()}"
        )

    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise SystemExit(
            f"Could not open input video: {args.source.resolve()}"
        )

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_width <= 0 or frame_height <= 0:
        capture.release()
        raise SystemExit("Could not determine the video's frame size.")

    try:
        writer = create_writer(
            args.output,
            fps,
            (frame_width, frame_height),
        )
    except RuntimeError as error:
        capture.release()
        raise SystemExit(str(error)) from error

    print("\nInitializing navigation pipeline...\n")

    perception = YOLOPerception(
        weights=args.weights,
        confidence=args.confidence,
        device=args.device,
    )

    severity = HazardSeverityClassifier(
        urgency_threshold=0.72,
        directly_ahead_deg=20.0,
        very_close_m=2.5,
        immediate_close_m=1.5,
        critical_ttc_s=1.5,
        high_risk_ttc_s=2.5,
    )

    memory = RouteMemory(str(args.db))
    attention = AcousticAttention(top_k=3)

    planner = LocalPathPlanner(
        horizon_m=5.0,
        path_width_m=0.65,
        safety_margin_m=0.25,
        walking_speed_m_s=0.5,
        prediction_horizon_s=4.0,
        prediction_step_s=0.25,
        static_action_distance_m=2.75,
        dynamic_action_distance_m=6.0,
        dynamic_ttc_s=4.0,
        confirm_frames=3,
        clear_frames=3,
    )

    trajectory_tracker = HazardTrajectoryTracker(
        smoothing=0.35,
        max_speed_m_s=8.0,
    )

    hrtf_renderer: HRTFRenderer | None = None

    if not args.no_audio:
        print("Initializing HRTF spatial audio...")

        audio_device = args.audio_device
        if audio_device is not None:
            try:
                audio_device = int(audio_device)
            except ValueError:
                pass

        hrtf_renderer = HRTFRenderer(
            sofa_path=str(args.hrtf),
            output_device=audio_device,
        )
        print("HRTF spatial audio initialized.\n")
    else:
        print("HRTF spatial audio disabled.\n")

    if args.reset:
        memory.clear()
        print(
            f"Cleared persistent memory: {args.db.resolve()}"
        )

    frame_id = 0
    warning_frames = 0
    unique_warning_ids: set[str] = set()
    warning_events: list[dict[str, Any]] = []

    audio_warning_count = 0
    path_warning_audio_count = 0
    path_unsafe_frames = 0
    path_events: list[dict[str, Any]] = []

    processing_started = time.perf_counter()

    try:
        while True:
            success, captured_image = capture.read()
            if not success or captured_image is None:
                break

            image: ImageArray = np.asarray(
                captured_image,
                dtype=np.uint8,
            )

            frame_id += 1
            timestamp = (frame_id - 1) / fps

            frame_packet = FramePacket(
                frame_id=frame_id,
                timestamp=timestamp,
                image=image,
            )

            pose = simulated_pose(timestamp)
            detections = perception.process(frame_packet)

            # -------------------------------------------------
            # Current-frame hazards
            # -------------------------------------------------
            # Persistent memory is for contextual awareness. The planner must
            # only use hazards observed in the CURRENT frame, otherwise an old
            # object could incorrectly remain a physical obstacle.
            current_frame_hazards = []

            for detection in detections:
                localized = localize_detection(
                    pose,
                    detection,
                )

                track = trajectory_tracker.update(
                    object_id=localized.object_id,
                    x=localized.world_x,
                    y=localized.world_y,
                    timestamp=timestamp,
                )

                entry = memory.remember(
                    object_id=localized.object_id,
                    label=localized.label,
                    world_x=localized.world_x,
                    world_y=localized.world_y,
                    confidence=localized.confidence,
                    timestamp=localized.timestamp,
                    velocity_x=track.velocity_x,
                    velocity_y=track.velocity_y,
                )

                current_frame_hazards.append(entry)

            # -------------------------------------------------
            # Path planning first
            # -------------------------------------------------
            # The hazard classifier now receives path-blocking and TTC
            # information instead of judging every object in isolation.
            path_decision = planner.plan(
                pose,
                current_frame_hazards,
            )

            warnings = []
            audio_played_this_frame = False

            # -------------------------------------------------
            # Hazard classification + urgent audio
            # -------------------------------------------------
            for detection in detections:
                ttc_s = (
                    path_decision.collision_ttc_by_hazard or {}
                ).get(detection.object_id)

                result = severity.classify(
                    detection,
                    path_blocking=(
                        detection.object_id
                        in path_decision.blocking_hazards
                    ),
                    ttc_s=ttc_s,
                )

                if not result.urgent:
                    continue

                warning = make_warning(
                    detection,
                    result.score,
                    result.reason,
                    category=result.category,
                    ttc_s=result.ttc_s,
                )
                warnings.append(warning)

                # URGENT audio represents WHERE THE DANGER IS.
                if hrtf_renderer is not None:
                    audio_played = hrtf_renderer.play_warning(
                        detection.audio_azimuth_deg
                    )

                    if audio_played:
                        audio_warning_count += 1
                        audio_played_this_frame = True

            # -------------------------------------------------
            # Soft path guidance
            # -------------------------------------------------
            if not path_decision.current_path_safe:
                path_unsafe_frames += 1

                blocking_detection = next(
                    (
                        detection
                        for detection in detections
                        if detection.object_id
                        in path_decision.blocking_hazards
                    ),
                    None,
                )

                path_audio_played = False

                # If selected_path == ahead, the planner did not find a useful
                # lateral escape route. Do not play a misleading directional
                # cue. An immediate/TTC hazard can still use urgent audio.
                useful_directional_guidance = (
                    path_decision.selected_path != "ahead"
                )

                if (
                    hrtf_renderer is not None
                    and blocking_detection is not None
                    and useful_directional_guidance
                    and path_decision.guidance_changed
                    and not audio_played_this_frame
                ):
                    # SOFT audio represents WHERE TO WALK, not where the
                    # obstacle is. The planner maps left/right alternatives to
                    # -60/-30/0/+30/+60 degree HRTF cues.
                    path_audio_played = hrtf_renderer.play_path_guidance(
                        path_decision.guidance_azimuth_deg
                    )

                    if path_audio_played:
                        path_warning_audio_count += 1
                        audio_played_this_frame = True

                path_events.append(
                    {
                        "frame": frame_id,
                        "time_s": round(timestamp, 3),
                        "current_path_safe": False,
                        "recommended_path": path_decision.selected_path,
                        "recommended_offset_m": round(
                            path_decision.selected_offset,
                            3,
                        ),
                        "blocking_hazards": path_decision.blocking_hazards,
                        "current_path_ttc_s": path_decision.current_path_ttc_s,
                        "collision_ttc_by_hazard": path_decision.collision_ttc_by_hazard,
                        "reason": path_decision.reason,
                        "audio_played": path_audio_played,
                    }
                )

                print(f"PATH UNSAFE | {path_decision.reason}")

            # -------------------------------------------------
            # Persistent-memory attention
            # -------------------------------------------------
            top_hazards = attention.rank(
                memory.get_hazards(),
                pose,
            )

            # -------------------------------------------------
            # Annotated frame
            # -------------------------------------------------
            if perception.last_result is not None:
                display: ImageArray = np.asarray(
                    perception.last_result.plot(),
                    dtype=np.uint8,
                )
            else:
                display = image.copy()

            draw_status(
                display,
                pose,
                len(detections),
                warnings,
                top_hazards,
                path_decision,
                audio_enabled=hrtf_renderer is not None,
            )

            writer.write(display)

            # -------------------------------------------------
            # Save warning events
            # -------------------------------------------------
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
                            "azimuth_deg": round(
                                warning.azimuth_deg,
                                3,
                            ),
                            "distance_band": warning.distance_band,
                            "distance_m": round(
                                warning.distance_m,
                                3,
                            ),
                            "score": round(
                                warning.score,
                                3,
                            ),
                            "motion": warning.motion,
                            "category": warning.category,
                            "ttc_s": warning.ttc_s,
                            "reason": warning.reason,
                            "audio_enabled": hrtf_renderer is not None,
                            "path_safe": path_decision.current_path_safe,
                            "recommended_path": path_decision.selected_path,
                            "path_reason": path_decision.reason,
                            "pose": {
                                "x": round(pose.x, 3),
                                "y": round(pose.y, 3),
                                "heading_deg": round(
                                    pose.heading_deg,
                                    2,
                                ),
                            },
                        }
                    )

                    ttc_text = (
                        "none"
                        if warning.ttc_s is None
                        else f"{warning.ttc_s:.2f}s"
                    )

                    print(
                        f"WARNING | t={timestamp:.2f}s | "
                        f"{warning.label} | {warning.direction} | "
                        f"{warning.distance_m:.1f}m | "
                        f"{warning.category} | TTC={ttc_text} | "
                        f"score={warning.score:.2f} | "
                        f"physical_azimuth={warning.azimuth_deg:+.1f}° | "
                        f"audio_azimuth={warning.azimuth_deg * -2.5:+.1f}°"
                    )
                    

            if not args.no_display:
                cv2.imshow(
                    "BVI Navigation - YOLO + Path + TTC + HRTF - Q to stop",
                    display,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Processing stopped by user.")
                    break

    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

        if hrtf_renderer is not None:
            hrtf_renderer.close()

    processing_seconds = time.perf_counter() - processing_started
    persisted_hazards = memory.get_hazards()

    report = {
        "source_video": str(args.source.resolve()),
        "model_weights": str(args.weights.resolve()),
        "annotated_video": str(args.output.resolve()),
        "memory_db": str(args.db.resolve()),
        "hrtf_file": (
            str(args.hrtf.resolve())
            if not args.no_audio
            else None
        ),
        "audio_enabled": hrtf_renderer is not None,
        "audio_warnings_played": audio_warning_count,
        "path_warning_audio_played": path_warning_audio_count,
        "frames_processed": frame_id,
        "video_fps": fps,
        "warning_frames": warning_frames,
        "unique_warning_objects": len(unique_warning_ids),
        "path_unsafe_frames": path_unsafe_frames,
        "processing_seconds": round(processing_seconds, 3),
        "path_events": path_events,
        "warning_events": warning_events,
        "persistent_memory": [
            {
                "object_id": hazard.object_id,
                "label": hazard.label,
                "world_x": round(hazard.world_x, 3),
                "world_y": round(hazard.world_y, 3),
                "confidence": round(hazard.confidence, 3),
                "last_seen_timestamp": round(hazard.timestamp, 3),
                "velocity_x": round(hazard.velocity_x, 3),
                "velocity_y": round(hazard.velocity_y, 3),
            }
            for hazard in persisted_hazards
        ],
        "audio_policy": {
            "urgent": (
                "Immediate/close obstacle or predicted collision; "
                "HRTF direction represents the actual obstacle direction."
            ),
            "soft_path_guidance": (
                "Non-immediate path blockage with a useful alternative; "
                "HRTF direction represents the recommended walking direction."
            ),
            "silent": (
                "Distant/non-actionable objects do not produce audio."
            ),
        },
        "limitations": [
            "Distance is estimated from object height and bounding-box size, not measured depth.",
            "Pose comes from a deterministic simulated trajectory, not real SLAM/localization.",
            "Trajectory velocity is estimated from localized detections and can be noisy.",
            "TTC is a prototype prediction based on estimated relative motion and safety radii.",
            "HRTF uses the nearest horizontal-plane measurement available in the SOFA file.",
        ],
    }

    report_path = args.output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Frames processed: {frame_id}")
    print(f"Warning frames: {warning_frames}")
    print(f"Unique warning objects: {len(unique_warning_ids)}")
    print(f"Objects in persistent memory (SQLite): {len(persisted_hazards)}")
    print(f"HRTF audio enabled: {hrtf_renderer is not None}")
    print(f"HRTF warnings played: {audio_warning_count}")
    print(f"Path-unsafe frames: {path_unsafe_frames}")
    print(f"Path guidance HRTF warnings played: {path_warning_audio_count}")
    print(f"Annotated video: {args.output.resolve()}")
    print(f"JSON report: {report_path.resolve()}")
    print(f"Memory DB: {args.db.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
