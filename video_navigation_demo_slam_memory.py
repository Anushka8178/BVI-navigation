"""
video_navigation_demo_slam_memory.py

Full BVI navigation pipeline demo on a recorded video:

    video frame
        -> YOLOPerception
        -> HazardSeverityClassifier
        -> simulated_pose()
        -> localize_detection()
        -> RouteMemory (SQLite persistent memory)
        -> Trajectory estimation
        -> Local path safety / rerouting
        -> AcousticAttention
        -> Warning generation
        -> HRTF spatial audio
        -> annotated video + JSON report

Pipeline:

    Video
      |
      v
    YOLO detection + tracking
      |
      v
    Distance / direction / motion
      |
      v
    Hazard severity classification
      |
      +--------------------------+
      |                          |
      v                          v
    Localization            Urgent warning
      |                          |
      v                          v
    SQLite memory            HRTFRenderer
      |                          |
      v                          v
    AcousticAttention        Spatial audio
      |                          |
      +------------+-------------+
                   |
                   v
              Output video
              JSON report
              navigation_memory.db

IMPORTANT:
    The pose used here is a deterministic simulated trajectory.
    It is NOT real SLAM.

Run:

    python video_navigation_demo_slam_memory.py \
        --weights best.pt \
        --source videos\\street2.mp4 \
        --output videos\\street2_navigation.mp4 \
        --device cpu

Start with a fresh memory database:

    python video_navigation_demo_slam_memory.py \
        --weights best.pt \
        --source videos\\street2.mp4 \
        --output videos\\street2_navigation.mp4 \
        --device cpu \
        --reset

Disable audio if required:

    python video_navigation_demo_slam_memory.py \
        --weights best.pt \
        --source videos\\street2.mp4 \
        --output videos\\street2_navigation.mp4 \
        --device cpu \
        --no-audio
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
from navigation.hrtf_renderer import HRTFRenderer

from navigation.planner import LocalPathPlanner
from navigation.trajectory import HazardTrajectoryTracker


ImageArray = NDArray[np.uint8]

# Path-planning components are created inside main() so each video run
# starts with a fresh trajectory state.


def simulated_pose(timestamp: float) -> Pose:
    """
    Generate a deterministic simulated walking pose.

    This is a stand-in for real SLAM/localization.

    The pose changes continuously with video time:

        x          -> forward movement
        y          -> small lateral drift
        heading    -> small continuous turn

    This allows the rest of the navigation pipeline to work without
    requiring a real SLAM system, camera localization, IMU, or GPS.
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


def create_writer(
    path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:
    """
    Create the output video writer.
    """

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
    """
    Draw navigation status information on the video frame.
    """

    image_height, image_width = image.shape[:2]

    panel_height = min(290, image_height)

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
            f"SLAM(sim) pose: "
            f"x={pose.x:.2f} "
            f"y={pose.y:.2f} "
            f"heading={pose.heading_deg:.1f} deg"
        ),
        (
            f"Detections: {detection_count} | "
            f"Warnings: {len(warnings)}"
        ),
        (
            f"Spatial audio: "
            f"{'ON' if audio_enabled else 'OFF'}"
        ),
    ]

    if path_decision.current_path_safe:
        lines.append("PATH: SAFE | continue ahead")
    else:
        recommendation = (
            "STOP / WAIT"
            if path_decision.selected_path == "ahead"
            else f"MOVE {path_decision.selected_path.upper()}"
        )
        lines.append(f"PATH: UNSAFE | {recommendation}")

    if warnings:
        for warning in warnings[:2]:
            lines.append(
                f"WARNING: {warning.label} | "
                f"{warning.direction} | "
                f"{warning.distance_m:.1f} m | "
                f"{warning.motion} | "
                f"score={warning.score:.2f}"
            )
    else:
        lines.append("Status: no urgent hazard")

    if top_hazards:
        labels = ", ".join(
            f"{h.label}({h.priority:.2f})"
            for h in top_hazards
        )

        lines.append(
            f"Persistent memory top-k: {labels}"
        )

    if not path_decision.current_path_safe:
        blockers = ", ".join(path_decision.blocking_hazards[:2]) or "predicted obstacle"
        lines.append(f"Blocked by: {blockers}")

    for index, line in enumerate(lines):

        if line.startswith("WARNING") or line.startswith("PATH: UNSAFE"):
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
            0.55,
            colour,
            2,
            cv2.LINE_AA,
        )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Full BVI video navigation demo: "
            "YOLO + hazard scoring + simulated SLAM + "
            "persistent memory + HRTF spatial audio"
        )
    )

    # ---------------------------------------------------------
    # Required arguments
    # ---------------------------------------------------------

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to YOLO model weights, e.g. best.pt",
    )

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Input video path",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output annotated video path",
    )

    # ---------------------------------------------------------
    # YOLO arguments
    # ---------------------------------------------------------

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="YOLO confidence threshold (default: 0.35)",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="YOLO device, e.g. cpu or 0",
    )

    # ---------------------------------------------------------
    # Memory arguments
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # HRTF/audio arguments
    # ---------------------------------------------------------

    parser.add_argument(
        "--hrtf",
        type=Path,
        default=Path("hrtf/mit_kemar_normal_pinna.sofa"),
        help=(
            "SOFA HRTF file "
            "(default: hrtf/mit_kemar_normal_pinna.sofa)"
        ),
    )

    parser.add_argument(
        "--audio-device",
        default=None,
        help=(
            "Audio output device index. "
            "Leave unset for automatic/default device selection."
        ),
    )

    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable HRTF spatial warning audio",
    )

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not show the live OpenCV window",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Validate input files
    # ---------------------------------------------------------

    if not args.weights.exists():
        raise SystemExit(
            f"Trained model does not exist: "
            f"{args.weights.resolve()}"
        )

    if not args.source.exists():
        raise SystemExit(
            f"Input video does not exist: "
            f"{args.source.resolve()}"
        )

    if not args.hrtf.exists() and not args.no_audio:
        raise SystemExit(
            f"HRTF SOFA file does not exist: "
            f"{args.hrtf.resolve()}"
        )

    # ---------------------------------------------------------
    # Open video
    # ---------------------------------------------------------

    capture = cv2.VideoCapture(
        str(args.source)
    )

    if not capture.isOpened():
        raise SystemExit(
            f"Could not open input video: "
            f"{args.source.resolve()}"
        )

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    ) or 30.0

    frame_width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    frame_height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if frame_width <= 0 or frame_height <= 0:

        capture.release()

        raise SystemExit(
            "Could not determine the video's frame size."
        )

    # ---------------------------------------------------------
    # Create output video
    # ---------------------------------------------------------

    try:

        writer = create_writer(
            args.output,
            fps,
            (frame_width, frame_height),
        )

    except RuntimeError as error:

        capture.release()

        raise SystemExit(str(error)) from error

    # ---------------------------------------------------------
    # Initialize navigation components
    # ---------------------------------------------------------

    print()
    print("Initializing navigation pipeline...")
    print()

    perception = YOLOPerception(
        weights=args.weights,
        confidence=args.confidence,
        device=args.device,
    )

    severity = HazardSeverityClassifier()

    memory = RouteMemory(
        str(args.db)
    )

    attention = AcousticAttention(
        top_k=3
    )

    planner = LocalPathPlanner(
        horizon_m=5.0,
        path_width_m=0.65,
        safety_margin_m=0.50,
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

    # ---------------------------------------------------------
    # Initialize HRTF renderer
    # ---------------------------------------------------------

    hrtf_renderer: HRTFRenderer | None = None

    if not args.no_audio:

        print("Initializing HRTF spatial audio...")

        # Convert audio device from command-line string to int
        # when the user supplied a numeric device index.
        audio_device = args.audio_device

        if audio_device is not None:

            try:
                audio_device = int(audio_device)
            except ValueError:
                # Leave it as a string. sounddevice can also
                # accept device names in some configurations.
                pass

        hrtf_renderer = HRTFRenderer(
            sofa_path=str(args.hrtf),
            output_device=audio_device,
        )

        print("HRTF spatial audio initialized.")
        print()

    else:

        print("HRTF spatial audio disabled.")
        print()

    # ---------------------------------------------------------
    # Reset memory if requested
    # ---------------------------------------------------------

    if args.reset:

        memory.clear()

        print(
            f"Cleared persistent memory: "
            f"{args.db.resolve()}"
        )

    # ---------------------------------------------------------
    # Runtime counters
    # ---------------------------------------------------------

    frame_id = 0

    warning_frames = 0

    unique_warning_ids: set[str] = set()

    warning_events: list[
        dict[str, Any]
    ] = []

    audio_warning_count = 0
    path_warning_audio_count = 0
    path_unsafe_frames = 0
    path_events: list[dict[str, Any]] = []

    processing_started = time.perf_counter()

    # ---------------------------------------------------------
    # Main video-processing loop
    # ---------------------------------------------------------

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

            timestamp = (
                frame_id - 1
            ) / fps

            # -------------------------------------------------
            # Create frame packet
            # -------------------------------------------------

            frame_packet = FramePacket(
                frame_id=frame_id,
                timestamp=timestamp,
                image=image,
            )

            # -------------------------------------------------
            # Get simulated pose
            # -------------------------------------------------

            pose = simulated_pose(
                timestamp
            )

            # -------------------------------------------------
            # YOLO detection + tracking
            # -------------------------------------------------

            detections = perception.process(
                frame_packet
            )

            # -------------------------------------------------
            # Hazard processing
            # -------------------------------------------------

            warnings = []
            audio_played_this_frame = False
            current_frame_hazards = []

            for detection in detections:

                # ---------------------------------------------
                # Classify hazard
                # ---------------------------------------------

                result = severity.classify(
                    detection
                )

                # ---------------------------------------------
                # Convert local detection to world coordinates
                # ---------------------------------------------

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
                # Keep this frame's localized objects separate from
                # persistent memory. Path planning should react to what is
                # currently visible, not to stale objects from earlier in
                # the video or a previous run.
                current_frame_hazards.append(
                    memory.remember(
                        object_id=localized.object_id,
                        label=localized.label,
                        world_x=localized.world_x,
                        world_y=localized.world_y,
                        confidence=localized.confidence,
                        timestamp=localized.timestamp,
                        velocity_x=track.velocity_x,
                        velocity_y=track.velocity_y,
                    )
                )

                # ---------------------------------------------
                # Store EVERY detection in persistent memory
                # ---------------------------------------------

                # ---------------------------------------------
                # Process urgent hazards
                # ---------------------------------------------

                if result.urgent:

                    warning = make_warning(
                        detection,
                        result.score,
                        result.reason,
                    )

                    warnings.append(
                        warning
                    )

                    # -----------------------------------------
                    # HRTF SPATIAL AUDIO
                    # -----------------------------------------
                    #
                    # The warning contains the detected
                    # object's azimuth.
                    #
                    # Example:
                    #
                    #   -60° -> obstacle on the left
                    #     0° -> obstacle ahead
                    #   +60° -> obstacle on the right
                    #
                    # HRTFRenderer converts this azimuth into
                    # a stereo spatial audio signal.

                    if hrtf_renderer is not None:

                        audio_played = (
                            hrtf_renderer.play_warning(
                                warning.azimuth_deg
                            )
                        )

                        if audio_played:
                            audio_warning_count += 1
                            audio_played_this_frame = True

            # -------------------------------------------------
            # Evaluate the user's walking path
            # -------------------------------------------------
            path_decision = planner.plan(
                pose,
                current_frame_hazards,
            )

            if not path_decision.current_path_safe:
                path_unsafe_frames += 1

                blocking_detection = next(
                    (
                        detection
                        for detection in detections
                        if detection.object_id in path_decision.blocking_hazards
                    ),
                    None,
                )

                path_audio_played = False

                if (
                    hrtf_renderer is not None
                    and blocking_detection is not None
                    and path_decision.guidance_changed
                    and not audio_played_this_frame
                ):
                    path_audio_played = hrtf_renderer.play_path_guidance(
                        blocking_detection.azimuth_deg
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
                        "recommended_offset_m": round(path_decision.selected_offset, 3),
                        "blocking_hazards": path_decision.blocking_hazards,
                        "reason": path_decision.reason,
                        "audio_played": path_audio_played,
                    }
                )

                print(
                    f"PATH UNSAFE | {path_decision.reason}"
                )

            # -------------------------------------------------
            # Rank persistent hazards
            # -------------------------------------------------

            top_hazards = attention.rank(
                memory.get_hazards(),
                pose,
            )

            # -------------------------------------------------
            # Create annotated frame
            # -------------------------------------------------

            if perception.last_result is not None:

                display: ImageArray = np.asarray(
                    perception.last_result.plot(),
                    dtype=np.uint8,
                )

            else:

                display = image.copy()

            # -------------------------------------------------
            # Draw navigation status
            # -------------------------------------------------

            draw_status(
                display,
                pose,
                len(detections),
                warnings,
                top_hazards,
                path_decision,
                audio_enabled=(
                    hrtf_renderer is not None
                ),
            )

            # -------------------------------------------------
            # Write annotated frame
            # -------------------------------------------------

            writer.write(
                display
            )

            # -------------------------------------------------
            # Save warning events
            # -------------------------------------------------

            if warnings:

                warning_frames += 1

                for warning in warnings:

                    unique_warning_ids.add(
                        warning.object_id
                    )

                    warning_events.append(
                        {
                            "frame": frame_id,

                            "time_s": round(
                                timestamp,
                                3,
                            ),

                            "object_id":
                                warning.object_id,

                            "label":
                                warning.label,

                            "direction":
                                warning.direction,

                            "azimuth_deg":
                                round(
                                    warning.azimuth_deg,
                                    3,
                                ),

                            "distance_band":
                                warning.distance_band,

                            "distance_m":
                                round(
                                    warning.distance_m,
                                    3,
                                ),

                            "score":
                                round(
                                    warning.score,
                                    3,
                                ),

                            "motion":
                                warning.motion,

                            "reason":
                                warning.reason,

                            "audio_enabled":
                                hrtf_renderer is not None,

                            "path_safe":
                                path_decision.current_path_safe,

                            "recommended_path":
                                path_decision.selected_path,

                            "path_reason":
                                path_decision.reason,

                            "pose":
                                {
                                    "x":
                                        round(
                                            pose.x,
                                            3,
                                        ),

                                    "y":
                                        round(
                                            pose.y,
                                            3,
                                        ),

                                    "heading_deg":
                                        round(
                                            pose.heading_deg,
                                            2,
                                        ),
                                },
                        }
                    )

                    # -----------------------------------------
                    # Console warning
                    # -----------------------------------------

                    print(
                        f"WARNING "
                        f"t={timestamp:.2f}s | "
                        f"{warning.label} | "
                        f"{warning.direction} | "
                        f"{warning.distance_m:.1f} m | "
                        f"{warning.motion} | "
                        f"score={warning.score:.2f} | "
                        f"azimuth={warning.azimuth_deg:+.1f}° | "
                        f"pose=({pose.x:.2f},"
                        f"{pose.y:.2f})"
                    )

            # -------------------------------------------------
            # Display live video
            # -------------------------------------------------

            if not args.no_display:

                cv2.imshow(
                    (
                        "BVI Navigation - "
                        "YOLO + SLAM(sim) + Memory + HRTF "
                        "- press Q to stop"
                    ),
                    display,
                )

                if (
                    cv2.waitKey(1) & 0xFF
                ) == ord("q"):

                    print(
                        "Processing stopped by user."
                    )

                    break

    finally:

        # -----------------------------------------------------
        # Release video resources
        # -----------------------------------------------------

        capture.release()

        writer.release()

        cv2.destroyAllWindows()

        # -----------------------------------------------------
        # Stop and close HRTF renderer
        # -----------------------------------------------------

        if hrtf_renderer is not None:

            hrtf_renderer.close()

    # ---------------------------------------------------------
    # Final statistics
    # ---------------------------------------------------------

    processing_seconds = (
        time.perf_counter()
        - processing_started
    )

    persisted_hazards = (
        memory.get_hazards()
    )

    # ---------------------------------------------------------
    # Create JSON report
    # ---------------------------------------------------------

    report = {

        "source_video":
            str(args.source.resolve()),

        "model_weights":
            str(args.weights.resolve()),

        "annotated_video":
            str(args.output.resolve()),

        "memory_db":
            str(args.db.resolve()),

        "hrtf_file":
            (
                str(args.hrtf.resolve())
                if not args.no_audio
                else None
            ),

        "audio_enabled":
            hrtf_renderer is not None,

        "audio_warnings_played":
            audio_warning_count,

        "frames_processed":
            frame_id,

        "video_fps":
            fps,

        "warning_frames":
            warning_frames,

        "unique_warning_objects":
            len(unique_warning_ids),

        "processing_seconds":
            round(
                processing_seconds,
                3,
            ),

        "path_unsafe_frames":
            path_unsafe_frames,

        "path_warning_audio_played":
            path_warning_audio_count,

        "path_events":
            path_events,

        "warning_events":
            warning_events,

        "persistent_memory":
            [
                {
                    "object_id":
                        h.object_id,

                    "label":
                        h.label,

                    "world_x":
                        round(
                            h.world_x,
                            3,
                        ),

                    "world_y":
                        round(
                            h.world_y,
                            3,
                        ),

                    "confidence":
                        round(
                            h.confidence,
                            3,
                        ),

                    "last_seen_timestamp":
                        round(
                            h.timestamp,
                            3,
                        ),
                }

                for h in persisted_hazards
            ],

        "limitations":
            [
                (
                    "Distance is estimated from object "
                    "height, not measured depth."
                ),

                (
                    "Pose comes from a deterministic "
                    "simulated trajectory (the documented "
                    "SLAM stand-in in navigation/sensing.py), "
                    "not real camera-based localization."
                ),

                (
                    "Approaching motion is determined "
                    "from frame-to-frame estimated distance."
                ),

                (
                    "HRTF spatialization uses the nearest "
                    "available measurement from the SOFA "
                    "HRTF dataset."
                ),
            ],
    }

    # ---------------------------------------------------------
    # Save JSON report
    # ---------------------------------------------------------

    report_path = (
        args.output.with_suffix(".json")
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Final console output
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)

    print(
        f"Frames processed: "
        f"{frame_id}"
    )

    print(
        f"Warning frames: "
        f"{warning_frames}"
    )

    print(
        f"Unique warning objects: "
        f"{len(unique_warning_ids)}"
    )

    print(
        f"Objects in persistent memory (SQLite): "
        f"{len(persisted_hazards)}"
    )

    print(
        f"HRTF audio enabled: "
        f"{hrtf_renderer is not None}"
    )

    print(
        f"HRTF warnings played: "
        f"{audio_warning_count}"
    )

    print(
        f"Path-unsafe frames: "
        f"{path_unsafe_frames}"
    )

    print(
        f"Path guidance HRTF warnings played: "
        f"{path_warning_audio_count}"
    )

    print(
        f"Annotated video: "
        f"{args.output.resolve()}"
    )

    print(
        f"JSON report: "
        f"{report_path.resolve()}"
    )

    print(
        f"Memory DB: "
        f"{args.db.resolve()}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
