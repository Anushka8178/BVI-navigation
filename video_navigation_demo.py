from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from navigation.audio import ConsoleSpatialAudio
from navigation.hazard import HazardSeverityClassifier
from navigation.memory import RouteMemory, SpatialMemory
from navigation.models import FramePacket, Pose
from navigation.pipeline import NavigationPipeline
from navigation.reasoning import (
    AcousticAttention,
    ExplainablePathPlanner,
)
from navigation.yolo_perception import YOLOPerception


def play_warning() -> None:
    """Play a basic warning sound on Windows."""

    try:
        import winsound

        winsound.MessageBeep(
            winsound.MB_ICONEXCLAMATION
        )
    except (ImportError, RuntimeError):
        # Ignore the sound if Windows audio is unavailable.
        pass


def draw_status(
    image,
    result,
    video_time: float,
) -> None:
    """Draw pipeline information on the video."""

    height, width = image.shape[:2]
    panel_height = min(245, height)

    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (width, panel_height),
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
        f"Time: {video_time:.2f}s",
        (
            f"Detections: {result.detections} | "
            f"Spatial memory: {result.memory_size}"
        ),
        "Pipeline: YOLO -> Hazard -> Memory -> Top-K -> Warning",
    ]

    if result.immediate_alerts:
        for event in result.immediate_alerts[:2]:
            lines.append(
                f"URGENT: {event.label} | "
                f"{event.direction} | "
                f"{event.distance_band} | "
                f"{event.urgency}"
            )
    else:
        lines.append(
            "Status: No immediate danger"
        )

    for event in result.awareness_alerts[:3]:
        lines.append(
            f"TOP-K: {event.label} | "
            f"{event.direction} | "
            f"{event.distance_band} | "
            f"{event.urgency}"
        )

    for index, line in enumerate(lines):
        if line.startswith("URGENT"):
            color = (0, 0, 255)
        elif line.startswith("TOP-K"):
            color = (0, 255, 255)
        else:
            color = (255, 255, 255)

        y_position = 28 + index * 28

        cv2.putText(
            image,
            line,
            (15, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline video-based BVI navigation prototype"
        )
    )

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to the trained YOLO best.pt file",
    )

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the input video",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/annotated_video.mp4"
        ),
        help="Path for the annotated output video",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Minimum YOLO confidence",
    )

    args = parser.parse_args()

    # Verify that the model exists.
    if not args.weights.exists():
        raise SystemExit(
            f"Trained model does not exist: "
            f"{args.weights}"
        )

    # Verify that the input video exists.
    if not args.source.exists():
        raise SystemExit(
            f"Input video does not exist: "
            f"{args.source}"
        )

    # Create the output directory.
    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Create the event-log path.
    log_path = args.output.with_name(
        f"{args.output.stem}_events.txt"
    )

    # Open the input video.
    capture = cv2.VideoCapture(
        str(args.source)
    )

    if not capture.isOpened():
        raise SystemExit(
            f"Cannot open input video: "
            f"{args.source}"
        )

    # Read video information.
    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25.0

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # Create the output-video encoder.
    fourcc: int = cv2.VideoWriter_fourcc(  # type: ignore[attr-defined]
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(args.output),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()

        raise SystemExit(
            f"Cannot create output video: "
            f"{args.output}"
        )

    # Create the database directory.
    database_path = (
        Path(__file__).parent
        / "data"
        / "video_navigation.db"
    )

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Create all architecture modules.
    route_memory = RouteMemory(
        database_path
    )

    perception = YOLOPerception(
        weights=args.weights,
        confidence=args.confidence,
    )

    severity_classifier = (
        HazardSeverityClassifier()
    )

    spatial_memory = SpatialMemory()

    attention = AcousticAttention(
        top_k=3
    )

    planner = ExplainablePathPlanner(
        route_memory
    )

    audio = ConsoleSpatialAudio()

    # Connect the modules using the pipeline.
    pipeline = NavigationPipeline(
        perception=perception,
        severity=severity_classifier,
        spatial_memory=spatial_memory,
        route_memory=route_memory,
        attention=attention,
        planner=planner,
        audio=audio,
        active_route=args.source.stem,
    )

    frame_id = 0

    # Prevent repeated terminal warnings on every frame.
    last_urgent_time = -10.0
    last_awareness_time = -10.0

    print()
    print("BVI OFFLINE VIDEO NAVIGATION")
    print(
        f"Input video : "
        f"{args.source.resolve()}"
    )
    print(
        f"Model       : "
        f"{args.weights.resolve()}"
    )
    print(
        f"Output video: "
        f"{args.output.resolve()}"
    )
    print(
        f"Event log   : "
        f"{log_path.resolve()}"
    )
    print(
        f"Resolution  : {width}x{height}"
    )
    print(
        f"FPS         : {fps:.2f}"
    )
    print(
        f"Frames      : {total_frames}"
    )
    print()
    print("Processing started...")
    print()

    try:
        with log_path.open(
            mode="w",
            encoding="utf-8",
        ) as log_file:

            log_file.write(
                "BVI NAVIGATION VIDEO ANALYSIS\n"
            )

            log_file.write(
                f"Input video: "
                f"{args.source.resolve()}\n"
            )

            log_file.write(
                f"Model: "
                f"{args.weights.resolve()}\n"
            )

            log_file.write(
                f"Resolution: "
                f"{width}x{height}\n"
            )

            log_file.write(
                f"FPS: {fps:.2f}\n"
            )

            log_file.write(
                f"Total frames: "
                f"{total_frames}\n\n"
            )

            while True:
                ok, image = capture.read()

                if not ok:
                    break

                frame_id += 1

                # Calculate time using the video's FPS.
                video_time = frame_id / fps

                # Real SLAM is not implemented.
                # A fixed pose is used for the prototype.
                pose = Pose(
                    0.0,
                    0.0,
                    0.0,
                    localization_valid=True,
                )

                packet = FramePacket(
                    frame_id=frame_id,
                    timestamp=video_time,
                    pose=pose,
                    image=image,
                )

                # Run the complete navigation pipeline.
                result = pipeline.process_cycle(
                    packet
                )

                # Draw YOLO detection boxes.
                if perception.last_result is not None:
                    display = (
                        perception
                        .last_result
                        .plot()
                    )
                else:
                    display = image.copy()

                # Draw warnings and pipeline information.
                draw_status(
                    display,
                    result,
                    video_time,
                )

                # Process urgent warnings.
                urgent_warning_allowed = (
                    video_time
                    - last_urgent_time
                    >= 1.0
                )

                if (
                    result.immediate_alerts
                    and urgent_warning_allowed
                ):
                    last_urgent_time = video_time

                    play_warning()

                    for event in result.immediate_alerts:
                        message = (
                            f"TIME {video_time:.2f}s | "
                            f"URGENT BYPASS: {event}"
                        )

                        print(message)

                        log_file.write(
                            message + "\n"
                        )

                # Process normal Top-K awareness.
                awareness_allowed = (
                    video_time
                    - last_awareness_time
                    >= 1.0
                )

                if (
                    result.awareness_alerts
                    and awareness_allowed
                ):
                    last_awareness_time = video_time

                    for event in result.awareness_alerts:
                        message = (
                            f"TIME {video_time:.2f}s | "
                            f"TOP-K MEMORY: {event}"
                        )

                        print(message)

                        log_file.write(
                            message + "\n"
                        )

                # Save the annotated frame.
                writer.write(display)

                # Display the result while processing.
                cv2.imshow(
                    (
                        "BVI Offline Video Navigation "
                        "- press Q to stop"
                    ),
                    display,
                )

                # Press Q to stop.
                pressed_key = (
                    cv2.waitKey(1) & 0xFF
                )

                if pressed_key == ord("q"):
                    print()
                    print(
                        "Processing stopped by user."
                    )
                    break

                # Print processing progress.
                if frame_id % 100 == 0:
                    if total_frames > 0:
                        progress = (
                            frame_id
                            / total_frames
                            * 100
                        )
                    else:
                        progress = 0.0

                    print(
                        f"Processed "
                        f"{frame_id}/"
                        f"{total_frames} frames "
                        f"({progress:.1f}%)"
                    )

            log_file.write(
                "\nVIDEO PROCESSING COMPLETED\n"
            )

            log_file.write(
                f"Processed frames: "
                f"{frame_id}\n"
            )

    finally:
        capture.release()
        writer.release()
        route_memory.close()
        cv2.destroyAllWindows()

    print()
    print(
        "Processing completed successfully."
    )

    print(
        f"Annotated video saved to: "
        f"{args.output.resolve()}"
    )

    print(
        f"Event log saved to: "
        f"{log_path.resolve()}"
    )


if __name__ == "__main__":
    main()