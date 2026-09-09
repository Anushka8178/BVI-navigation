from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from navigation.hazard import HazardSeverityClassifier
from navigation.models import FramePacket
from navigation.pipeline import CycleResult, NavigationPipeline
from navigation.yolo_perception import YOLOPerception


ImageArray = NDArray[np.uint8]


def create_writer(
    path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:
    """
    Create the output video writer.

    frame_size must contain:
        (width, height)
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fourcc = cv2.VideoWriter.fourcc(
        "m",
        "p",
        "4",
        "v",
    )

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
    result: CycleResult,
) -> None:
    """
    Draw detection and warning information on the video frame.
    """

    image_height, image_width = image.shape[:2]
    panel_height = min(145, image_height)

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

    if result.warnings:
        lines = [
            (
                f"Detections: {result.detection_count} | "
                f"Warnings: {len(result.warnings)}"
            )
        ]

        for warning in result.warnings[:3]:
            lines.append(
                f"WARNING: {warning.label} | "
                f"{warning.direction} | "
                f"{warning.distance_m:.1f} m | "
                f"{warning.motion} | "
                f"score={warning.score:.2f}"
            )
    else:
        lines = [
            f"Detections: {result.detection_count}",
            "Status: no urgent hazard",
        ]

    for index, line in enumerate(lines):
        if line.startswith("WARNING"):
            colour = (0, 0, 255)
        else:
            colour = (255, 255, 255)

        cv2.putText(
            image,
            line,
            (15, 30 + index * 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            colour,
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Video-based BVI obstacle warning prototype"
    )

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to the trained best.pt file",
    )

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the input MP4 video",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the annotated MP4 video",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Minimum YOLO detection confidence",
    )

    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device number or cpu",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Process without opening a video window",
    )

    args = parser.parse_args()

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
    )

    if fps <= 0:
        fps = 30.0

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

    try:
        writer = create_writer(
            path=args.output,
            fps=fps,
            frame_size=(
                frame_width,
                frame_height,
            ),
        )
    except RuntimeError as error:
        capture.release()
        raise SystemExit(str(error)) from error

    perception = YOLOPerception(
        weights=args.weights,
        confidence=args.confidence,
        device=args.device,
    )

    pipeline = NavigationPipeline(
        perception=perception,
        severity=HazardSeverityClassifier(),
    )

    frame_id = 0
    warning_frames = 0

    unique_warning_ids: set[str] = set()

    warning_events: list[
        dict[str, Any]
    ] = []

    processing_started = time.perf_counter()

    try:
        while True:
            success, captured_image = capture.read()

            if not success or captured_image is None:
                break

            # Convert OpenCV MatLike into a proper NumPy image.
            image: ImageArray = np.asarray(
                captured_image,
                dtype=np.uint8,
            )

            frame_id += 1

            timestamp = (
                frame_id - 1
            ) / fps

            frame_packet = FramePacket(
                frame_id=frame_id,
                timestamp=timestamp,
                image=image,
            )

            result = pipeline.process_cycle(
                frame_packet
            )

            if perception.last_result is not None:
                plotted_image = (
                    perception.last_result.plot()
                )

                display: ImageArray = np.asarray(
                    plotted_image,
                    dtype=np.uint8,
                )
            else:
                display = image.copy()

            draw_status(
                image=display,
                result=result,
            )

            writer.write(display)

            if result.warnings:
                warning_frames += 1

                for warning in result.warnings:
                    unique_warning_ids.add(
                        warning.object_id
                    )

                    warning_record = {
                        "frame": frame_id,
                        "time_s": round(
                            timestamp,
                            3,
                        ),
                        "object_id": (
                            warning.object_id
                        ),
                        "label": warning.label,
                        "direction": (
                            warning.direction
                        ),
                        "distance_band": (
                            warning.distance_band
                        ),
                        "distance_m": round(
                            warning.distance_m,
                            3,
                        ),
                        "score": round(
                            warning.score,
                            3,
                        ),
                        "motion": warning.motion,
                        "reason": warning.reason,
                    }

                    warning_events.append(
                        warning_record
                    )

                    print(
                        f"WARNING "
                        f"t={timestamp:.2f}s | "
                        f"{warning.label} | "
                        f"{warning.direction} | "
                        f"{warning.distance_m:.1f} m | "
                        f"{warning.motion} | "
                        f"score={warning.score:.2f}"
                    )

            if not args.no_display:
                cv2.imshow(
                    "BVI Video Navigation - press Q to stop",
                    display,
                )

                pressed_key = (
                    cv2.waitKey(1) & 0xFF
                )

                if pressed_key == ord("q"):
                    print(
                        "Processing stopped by user."
                    )
                    break

    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    processing_seconds = (
        time.perf_counter()
        - processing_started
    )

    report = {
        "source_video": str(
            args.source.resolve()
        ),
        "model_weights": str(
            args.weights.resolve()
        ),
        "annotated_video": str(
            args.output.resolve()
        ),
        "frames_processed": frame_id,
        "video_fps": fps,
        "warning_frames": warning_frames,
        "unique_warning_objects": len(
            unique_warning_ids
        ),
        "processing_seconds": round(
            processing_seconds,
            3,
        ),
        "warning_events": warning_events,
        "limitations": [
            (
                "Distance is estimated from object "
                "height and is not measured depth."
            ),
            (
                "Approaching motion is determined "
                "from frame-to-frame estimated distance."
            ),
            (
                "This prototype performs obstacle "
                "awareness, not route planning."
            ),
        ],
    }

    report_path = args.output.with_suffix(
        ".json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("PROCESSING COMPLETE")
    print(
        f"Frames processed: {frame_id}"
    )
    print(
        f"Warning frames: {warning_frames}"
    )
    print(
        f"Unique warning objects: "
        f"{len(unique_warning_ids)}"
    )
    print(
        f"Annotated video: "
        f"{args.output.resolve()}"
    )
    print(
        f"JSON report: "
        f"{report_path.resolve()}"
    )


if __name__ == "__main__":
    main()