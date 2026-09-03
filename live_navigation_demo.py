from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from navigation.audio import ConsoleSpatialAudio
from navigation.hazard import HazardSeverityClassifier
from navigation.memory import RouteMemory, SpatialMemory
from navigation.models import FramePacket, Pose
from navigation.pipeline import NavigationPipeline
from navigation.reasoning import AcousticAttention, ExplainablePathPlanner
from navigation.yolo_perception import YOLOPerception


def play_warning() -> None:
    """Play a short Windows warning without adding another dependency."""
    try:
        import winsound

        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
    except (ImportError, RuntimeError):
        pass


def draw_status(image, result) -> None:
    height, width = image.shape[:2]
    panel_height = min(155, height)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)

    lines = [
        f"Detections: {result.detections} | Spatial memory: {result.memory_size}",
        "Pipeline: YOLO -> Triage -> Memory -> Top-K attention -> Audio",
    ]
    for event in result.immediate_alerts[:2]:
        lines.append(
            f"URGENT: {event.label} | {event.direction} | "
            f"{event.distance_band} | {event.urgency}"
        )
    if not result.immediate_alerts:
        lines.append("Status: no urgent hazard")

    for index, line in enumerate(lines):
        color = (0, 0, 255) if line.startswith("URGENT") else (255, 255, 255)
        cv2.putText(
            image,
            line,
            (15, 28 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified live BVI navigation prototype")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", default="0", help="webcam number or video path")
    parser.add_argument("--confidence", type=float, default=0.35)
    args = parser.parse_args()

    source: int | str = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Cannot open camera/video source: {source}")

    database = Path(__file__).parent / "data" / "live_navigation.db"
    route_memory = RouteMemory(database)
    perception = YOLOPerception(args.weights, confidence=args.confidence)
    pipeline = NavigationPipeline(
        perception=perception,
        severity=HazardSeverityClassifier(),
        spatial_memory=SpatialMemory(),
        route_memory=route_memory,
        attention=AcousticAttention(top_k=3),
        planner=ExplainablePathPlanner(route_memory),
        audio=ConsoleSpatialAudio(),
        active_route="live-demo-route",
    )

    frame_id = 0
    last_warning = 0.0
    started = time.monotonic()
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            frame_id += 1

            # Fixed origin is a webcam-demo pose, not SLAM localization. It lets
            # us demonstrate short-term egocentric memory while the camera stays put.
            packet = FramePacket(
                frame_id=frame_id,
                timestamp=time.monotonic() - started,
                pose=Pose(0.0, 0.0, 0.0, localization_valid=True),
                image=image,
            )
            result = pipeline.process_cycle(packet)
            display = perception.last_result.plot() if perception.last_result is not None else image
            draw_status(display, result)

            now = time.monotonic()
            if result.immediate_alerts and now - last_warning >= 1.0:
                play_warning()
                last_warning = now
                for event in result.immediate_alerts:
                    print(f"URGENT BYPASS: {event}")

            cv2.imshow("Unified BVI Navigation Prototype - Q to exit", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        route_memory.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
