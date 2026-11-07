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

Pose (--pose): "slam" runs the visual SLAM in navigation/vslam.py on the
camera frames (phone steps/heading help with scale and heading when the
SensorServer app is reachable; --no-phone for camera only); "phone" reads
heading + steps from the SensorServer app on the chest-mounted phone
(default for stream URLs); "sim" uses a simulated walk (default for video
files). While the pose is invalid (no sensor data, SLAM lost without the
phone, or SLAM still relocalizing in a saved map), nothing is written to
persistent memory.

All tunables come from config.yaml (see --config). Dedicated flags such as
--confidence, and any number of --set section.key=value, override it.

Examples (PowerShell):
    # Recorded video
    python video_navigation_demo_slam_memory.py `
        --source videos\\street6.mp4 --output videos\\street6_navigation.mp4 --reset

    # Live phone camera (IP Webcam app); default source is camera.source
    python video_navigation_demo_slam_memory.py --no-display

    # Live with visual SLAM (map saved to slam_map.npz, reused next run)
    python video_navigation_demo_slam_memory.py --pose slam

    # Recorded video with visual SLAM, camera only, fresh map
    python video_navigation_demo_slam_memory.py --source videos\\walk.mp4 `
        --pose slam --no-phone --slam-new-map --reset

Weights default to perception.weights in config.yaml (best.pt).
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from navigation.audio import make_warning
from navigation.camera import FpsMeter, is_url, open_source
from navigation.config import (
    ConfigError,
    apply_overrides,
    load_config,
    parse_set_option,
)
from navigation.hazard import HazardSeverityClassifier
from navigation.hrtf_renderer import HRTFRenderer
from navigation.memory import RouteMemory, SessionClock
from navigation.memory_reasoning import (
    CameraModel,
    MemoryEarlyWarning,
    MemoryReasoningConfig,
    NegativeEvidenceState,
    pose_is_trusted,
    update_negative_evidence,
)
from navigation.minimap import draw_minimap
from navigation.path_overlay import draw_lanes, lanes_for_minimap
from navigation.models import FramePacket, Pose
from navigation.planner import LocalPathPlanner
from navigation.range_fusion import UNKNOWN_OBSTACLE, RangeFusionConfig, fuse_range
from navigation.range_sensor import RangeSensor
from navigation.reasoning import AcousticAttention
from navigation.sensing import PhonePose, localize_detection, simulated_pose
from navigation.trajectory import HazardTrajectoryTracker
from navigation.vslam import (
    SlamConfig,
    VisualSLAM,
    boxes_from_yolo_result,
    relative,
)
from navigation.yolo_perception import YOLOPerception


ImageArray = NDArray[np.uint8]


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
    pose_mode: str = "sim",
    memory_line: str | None = None,
) -> None:
    """Draw a compact navigation status panel on the frame.

    memory_line: "Memory cue: ..." while a memory early-warning cue is recent.
    """
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
            f"Pose({pose_mode}{'' if pose.valid else ', INVALID'}): "
            f"x={pose.x:.2f} y={pose.y:.2f} heading={pose.heading_deg:.1f}°"
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

    if memory_line:
        lines.append(memory_line)

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
        elif line.startswith("Memory cue"):
            colour = (0, 200, 255)  # amber, as the mini-map ring
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
        default=None,
        help="YOLO weights (config: perception.weights, default best.pt)",
    )
    parser.add_argument(
        "--pose",
        choices=("sim", "phone", "slam"),
        default=None,
        help="Pose source: slam = visual SLAM on the camera frames (+ phone "
        "steps/heading if reachable), phone = SensorServer app (heading + "
        "steps), sim = simulated walk (default: phone for stream URLs, sim "
        "for files)",
    )
    parser.add_argument(
        "--slam-map",
        type=Path,
        default=None,
        help="SLAM map file (config: slam.map_path). Loaded at start to "
        "relocalize in the same world frame as earlier runs; saved at the end",
    )
    parser.add_argument(
        "--slam-new-map",
        action="store_true",
        help="Ignore an existing SLAM map and start a new one (it is "
        "overwritten at the end). Use --reset too: old hazards would be in "
        "the old map's frame",
    )
    parser.add_argument(
        "--no-phone",
        action="store_true",
        help="With --pose slam: do not use the phone's SensorServer data "
        "(scale then comes from slam.assumed_speed_m_s)",
    )
    parser.add_argument(
        "--no-lanes",
        action="store_true",
        help="Do not draw the path planner's walking lanes on the video",
    )
    parser.add_argument(
        "--trajectory-out",
        type=Path,
        default=None,
        help="Write the per-frame pose to this CSV (for tools/eval_walk_drift.py)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Config file (default: config.yaml in the repo root)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override a config value, e.g. --set planner.safety_margin_m=0.3 "
        "(repeatable)",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Recorded video file OR stream URL, e.g. "
        "http://<PHONE_IP>:8080/video (default: camera.source in config)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Annotated output video (default: <video>_navigation.mp4, or "
        "videos/live_<time>_navigation.mp4 for a URL)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="YOLO confidence threshold (config: perception.confidence)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="YOLO device, e.g. cpu or 0 (config: perception.device)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite navigation memory database (config: memory.db_path)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear persistent memory before this run",
    )
    parser.add_argument(
        "--hrtf",
        type=Path,
        default=None,
        help="SOFA HRTF file (config: audio.hrtf_path)",
    )
    parser.add_argument(
        "--audio-device",
        default=None,
        help="Audio output device index or name "
        "(config: audio.device; default: wired headphones, Bluetooth last)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable HRTF audio",
    )
    parser.add_argument(
        "--no-esp32",
        action="store_true",
        help="Do not use the ESP32 ultrasonic range sensor",
    )
    parser.add_argument(
        "--esp32-port",
        default=None,
        help="ESP32 serial port, e.g. COM5 (config: esp32.serial_port)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Stop cleanly after this many seconds of processing "
        "(useful for timed live runs)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not show the OpenCV window",
    )

    args = parser.parse_args()

    try:
        config = apply_overrides(
            load_config(args.config),
            [parse_set_option(text) for text in args.overrides],
        )
        # Dedicated flags win over both config.yaml and --set.
        config = apply_overrides(
            config,
            {
                "camera.source": args.source,
                "perception.weights": (
                    None if args.weights is None else str(args.weights)
                ),
                "perception.confidence": args.confidence,
                "perception.device": args.device,
                "memory.db_path": None if args.db is None else str(args.db),
                "audio.hrtf_path": None if args.hrtf is None else str(args.hrtf),
                "audio.device": args.audio_device,
                "esp32.serial_port": args.esp32_port,
                "slam.map_path": (
                    None if args.slam_map is None else str(args.slam_map)
                ),
            },
        )
    except ConfigError as error:
        raise SystemExit(f"Config error: {error}") from error

    camera_cfg = config["camera"]
    perception_cfg = config["perception"]
    memory_cfg = config["memory"]
    attention_cfg = config["attention"]
    audio_cfg = config["audio"]
    pose_cfg = config["pose"]
    network_cfg = config["network"]
    slam_cfg = config["slam"]

    source = str(camera_cfg["source"])
    live = is_url(source)
    weights_path = Path(perception_cfg["weights"])
    pose_mode = args.pose or ("phone" if live else "sim")
    db_path = Path(memory_cfg["db_path"])
    hrtf_path = Path(audio_cfg["hrtf_path"])

    if args.output is not None:
        output_path = args.output
    elif live:
        output_path = Path("videos") / (
            f"live_{time.strftime('%Y%m%d_%H%M%S')}_navigation.mp4"
        )
    else:
        source_path = Path(source)
        output_path = source_path.with_name(
            f"{source_path.stem}_navigation.mp4"
        )

    if not weights_path.exists():
        raise SystemExit(
            f"Trained model does not exist: {weights_path.resolve()}"
        )

    if not args.no_audio and not hrtf_path.exists():
        raise SystemExit(
            f"HRTF SOFA file does not exist: {hrtf_path.resolve()}"
        )

    print(f"Opening source: {source}")
    try:
        frame_source = open_source(
            source,
            read_timeout_s=camera_cfg["read_timeout_s"],
            reconnect_delay_s=camera_cfg["reconnect_delay_s"],
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    if live:
        fps = float(frame_source.fps or camera_cfg["live_output_fps"])
    else:
        fps = frame_source.fps

    # The writer is created on the first frame, because a stream's frame
    # size is only known once a frame has arrived.
    writer: cv2.VideoWriter | None = None

    print("\nInitializing navigation pipeline...\n")

    perception = YOLOPerception(
        weights=weights_path,
        confidence=float(perception_cfg["confidence"]),
        device=str(perception_cfg["device"]),
        camera_height_m=float(perception_cfg["camera_height_m"]),
        ground_labels=frozenset(perception_cfg["ground_labels"]),
        ground_min_m=float(perception_cfg["ground_min_m"]),
        ground_max_m=float(perception_cfg["ground_max_m"]),
    )

    severity = HazardSeverityClassifier(**config["hazard"])

    # Memory ages in epoch seconds across runs: wall clock for live
    # streams, run start + video time for recorded videos. Tracking and
    # TTC keep the pipeline clock.
    memory_clock = SessionClock(live=live)
    session_id = uuid.uuid4().hex
    try:
        reasoning_cfg = MemoryReasoningConfig.from_config(config["memory_reasoning"])
    except (TypeError, ValueError) as error:
        raise SystemExit(f"Config error: {error}") from error
    # Class-aware decay + reinforcement across journeys (config memory:).
    memory = RouteMemory.from_config(
        memory_cfg,
        db_path=str(db_path),
        session_id=session_id,
        now_fn=memory_clock,
        hit_gain=reasoning_cfg.hit_gain,
    )
    negative_state = NegativeEvidenceState()
    # Set on the first frame (needs the frame height).
    camera_model: CameraModel | None = None
    attention = AcousticAttention(
        top_k=int(attention_cfg["top_k"]),
        weights=tuple(float(w) for w in attention_cfg["weights"]),
    )

    planner = LocalPathPlanner(**config["planner"])

    trajectory_tracker = HazardTrajectoryTracker(**config["trajectory"])

    hrtf_renderer: HRTFRenderer | None = None

    if not args.no_audio:
        print("Initializing HRTF spatial audio...")

        try:
            hrtf_renderer = HRTFRenderer(
                sofa_path=str(hrtf_path),
                output_device=audio_cfg["device"],
                warning_cooldown=float(audio_cfg["warning_cooldown_s"]),
                path_guidance_cooldown=float(
                    audio_cfg["path_guidance_cooldown_s"]
                ),
                max_queue=int(audio_cfg["max_queue"]),
                bluetooth_keywords=audio_cfg["bluetooth_keywords"],
                headphone_keywords=audio_cfg["headphone_keywords"],
                memory_cue_cooldown=float(audio_cfg["memory_cue_cooldown_s"]),
            )
        except RuntimeError as error:
            frame_source.close()
            raise SystemExit(f"Audio setup failed: {error}")
        print("HRTF spatial audio initialized.\n")
    else:
        print("HRTF spatial audio disabled.\n")

    # Memory early warning: sonified top-k attention over remembered,
    # currently unseen hazards (navigation/memory_reasoning.py).
    early_warning = MemoryEarlyWarning(
        reasoning_cfg,
        attention,
        session_id=session_id,
        dynamic_labels=memory.dynamic_labels,
        match_radius_m=memory.match_radius_m,
        renderer=hrtf_renderer,
    )

    phone_pose: PhonePose | None = None

    use_phone = pose_mode == "phone" or (pose_mode == "slam" and not args.no_phone)
    if use_phone:
        phone_pose = PhonePose.from_config(network_cfg, pose_cfg)
        print(
            "Connecting to phone sensors at "
            f"ws://{phone_pose.phone_ip}:{phone_pose.sensor_port} ..."
        )
        try:
            phone_pose.start()
        except RuntimeError as error:
            if pose_mode == "slam":
                print(
                    f"WARNING: phone sensors unavailable ({error}); SLAM runs "
                    "camera-only, scale from slam.assumed_speed_m_s.\n"
                )
                phone_pose = None
            else:
                frame_source.close()
                if hrtf_renderer is not None:
                    hrtf_renderer.close()
                raise SystemExit(f"Phone pose setup failed: {error}") from error

    if phone_pose is not None:
        if phone_pose.wait_for_heading(float(pose_cfg["startup_wait_s"])):
            phone_pose.calibrate()
            print("Phone pose ready; current heading set as 0.\n")
        elif pose_mode == "slam":
            print(
                "WARNING: no rotation data from the phone yet. SLAM continues "
                "with the camera; phone steps/heading are used once they "
                "arrive.\n"
            )
        else:
            print(
                "WARNING: no rotation data from the phone yet. Check the "
                "SensorServer app and network.phone_ip/sensor_port, or use "
                "--pose sim. Continuing; memory is paused until the pose is "
                "valid, and the first heading becomes 0.\n"
            )

    # Visual SLAM is created on the first frame (needs the frame size).
    slam: VisualSLAM | None = None
    slam_config = SlamConfig.from_config(slam_cfg)
    slam_map_path = Path(slam_cfg["map_path"])
    slam_every_n = max(1, int(slam_cfg["process_every_n"]))
    slam_map_version = 0
    slam_seconds = 0.0
    slam_frames = 0
    reanchored_total = 0
    dynamic_labels = frozenset(memory_cfg["dynamic_labels"])
    if pose_mode == "slam":
        print(
            "Pose: visual SLAM"
            + (" + phone steps/heading" if phone_pose is not None else " (camera only)")
            + f"; map file {slam_map_path}"
            + (" (new map)" if args.slam_new_map else "")
            + "\n"
        )
        if args.slam_new_map and not args.reset:
            print(
                "NOTE: --slam-new-map without --reset: hazards already in "
                "memory are in the old map's frame.\n"
            )
    elif pose_mode == "sim":
        print("Pose: simulated walk (--pose sim).\n")

    trajectory_rows: list[tuple[float, float, float, float, int, str]] = []
    trajectory_xy: list[tuple[float, float]] = []

    # Range sensor: fused with the detections every frame
    # (navigation/range_fusion.py). A missing port prints one warning; the
    # run continues without it (latest_m() is None, nothing changes).
    range_sensor: RangeSensor | None = None
    try:
        fusion_cfg = RangeFusionConfig.from_config(config["range_fusion"])
    except (TypeError, ValueError) as error:
        raise SystemExit(f"Config error: {error}") from error
    range_frames = 0
    range_corrections = 0
    unknown_obstacle_frames = 0

    if not args.no_esp32:
        range_sensor = RangeSensor.from_config(config["esp32"])
        print(f"Starting ESP32 range sensor on {range_sensor.serial_port} ...\n")
        range_sensor.start()
    else:
        print("ESP32 range sensor disabled (--no-esp32).\n")

    if args.reset:
        memory.clear()
        print(
            f"Cleared persistent memory: {db_path.resolve()}"
        )
    # Recorded video: this run's memory clock starts after everything
    # already remembered (an earlier run may have replayed faster than
    # real time).
    memory_clock.start_after(memory.latest_timestamp())

    frame_id = 0
    warning_frames = 0
    unique_warning_ids: set[str] = set()
    warning_events: list[dict[str, Any]] = []

    audio_warning_count = 0
    path_warning_audio_count = 0
    path_unsafe_frames = 0
    path_events: list[dict[str, Any]] = []

    processing_started = time.perf_counter()
    fps_meter = FpsMeter(camera_cfg["fps_report_interval_s"])
    last_received = 0
    last_dropped = 0
    first_timestamp: float | None = None
    pose_was_valid: bool | None = None
    invalid_pose_frames = 0

    try:
        while True:
            if (
                args.max_seconds is not None
                and time.perf_counter() - processing_started >= args.max_seconds
            ):
                print(f"Reached --max-seconds {args.max_seconds}; stopping.")
                break

            captured_image, source_timestamp = frame_source.read()
            if captured_image is None or source_timestamp is None:
                if live:
                    print(
                        "No new camera frame for "
                        f"{camera_cfg['read_timeout_s']} s; stopping."
                    )
                break

            image: ImageArray = np.asarray(
                captured_image,
                dtype=np.uint8,
            )

            if writer is None:
                frame_height, frame_width = image.shape[:2]
                try:
                    writer = create_writer(
                        output_path,
                        fps,
                        (frame_width, frame_height),
                    )
                except RuntimeError as error:
                    raise SystemExit(str(error)) from error
                camera_model = CameraModel(
                    hfov_deg=float(perception.horizontal_fov_deg),
                    camera_height_m=float(perception_cfg["camera_height_m"]),
                    image_h=int(frame_height),
                    ground_labels=frozenset(perception_cfg["ground_labels"]),
                )

            if first_timestamp is None:
                first_timestamp = source_timestamp

            frame_id += 1
            # Video time for files, wall-clock time for live streams.
            # Tracking and TTC use this clock.
            timestamp = source_timestamp
            # Seconds since the first frame (simulated pose and report).
            elapsed_s = source_timestamp - first_timestamp
            # Persistent memory uses epoch seconds (see SessionClock).
            memory_time = memory_clock.stamp(elapsed_s)

            frame_packet = FramePacket(
                frame_id=frame_id,
                timestamp=timestamp,
                image=image,
            )

            detections = perception.process(frame_packet)

            # Ultrasonic fusion: measured distance for the detection straight
            # ahead, or an "unknown_obstacle" YOLO did not see.
            if range_sensor is not None:
                fusion = fuse_range(detections, range_sensor.latest_m(), fusion_cfg)
                detections = fusion.detections
                if fusion.range_m is not None:
                    range_frames += 1
                range_corrections += fusion.corrected_id is not None
                unknown_obstacle_frames += fusion.added_unknown

            if pose_mode == "slam":
                if slam is None:
                    frame_h, frame_w = image.shape[:2]
                    try:
                        slam = VisualSLAM(
                            slam_config,
                            (frame_w, frame_h),
                            helper=phone_pose,
                            map_path=slam_map_path,
                            new_map=args.slam_new_map,
                        )
                    except ValueError as error:
                        raise SystemExit(f"SLAM map error: {error}") from error
                if (frame_id - 1) % slam_every_n == 0:
                    slam_started = time.perf_counter()
                    pose = slam.process(
                        image,
                        timestamp,
                        boxes_from_yolo_result(perception.last_result, dynamic_labels),
                    )
                    slam_seconds += time.perf_counter() - slam_started
                    slam_frames += 1
                else:
                    pose = slam.get_pose(timestamp)
                if slam.map_version != slam_map_version:
                    # Loop closure moved keyframes: move anchored hazards too.
                    slam_map_version = slam.map_version
                    moved = memory.reanchor(slam.anchor_poses())
                    reanchored_total += moved
                    if moved:
                        print(f"SLAM | moved {moved} remembered hazard(s) with the corrected map")
            elif phone_pose is not None:
                pose = phone_pose.get_pose(timestamp)
            else:
                pose = simulated_pose(elapsed_s)

            if not pose.valid:
                invalid_pose_frames += 1
            if pose.valid != pose_was_valid:
                if pose_was_valid is not None or not pose.valid:
                    print(
                        "POSE | valid, memory writing resumed"
                        if pose.valid
                        else "POSE | INVALID ("
                        + (
                            f"SLAM {slam.state}"
                            if slam is not None
                            else "no phone sensor data"
                        )
                        + "), memory writing paused"
                    )
                pose_was_valid = pose.valid

            slam_state = slam.state if slam is not None else pose_mode.upper()
            trajectory_rows.append(
                (round(elapsed_s, 3), pose.x, pose.y, pose.heading_deg, int(pose.valid), slam_state)
            )
            if pose.valid:
                trajectory_xy.append((pose.x, pose.y))
            anchor_kf = slam.current_anchor() if slam is not None else None

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
                    timestamp=memory_time,
                    velocity_x=track.velocity_x,
                    velocity_y=track.velocity_y,
                    # Invalid pose: world position unknown, don't store it.
                    # An ultrasonic-only obstacle has no identity to remember.
                    persist=pose.valid and detection.label != UNKNOWN_OBSTACLE,
                    anchor=(
                        None
                        if anchor_kf is None
                        else (
                            anchor_kf[0],
                            *relative(
                                anchor_kf[1],
                                (localized.world_x, localized.world_y, 0.0),
                            )[:2],
                        )
                    ),
                )

                current_frame_hazards.append(entry)

            # -------------------------------------------------
            # Evidence-based forgetting (negative evidence)
            # -------------------------------------------------
            # A remembered static hazard that should be in clear view but
            # is not detected loses belief; after a few misses it is
            # forgotten (navigation/memory_reasoning.py). Needs a trusted
            # pose (SLAM: TRACKING).
            pose_trusted = pose_is_trusted(
                pose, slam.state if slam is not None else None
            )
            seen_hazard_ids = {
                entry.hazard_id for entry in current_frame_hazards if entry.hazard_id
            }
            remembered = memory.get_hazards()
            forget_events = update_negative_evidence(
                memory,
                remembered,
                pose,
                detections,
                seen_hazard_ids,
                memory_time,
                reasoning_cfg,
                camera_model,
                negative_state,
                pose_trusted=pose_trusted,
            )
            if forget_events:
                forgotten_ids = {event.hazard_id for event in forget_events}
                remembered = [
                    h for h in remembered if h.hazard_id not in forgotten_ids
                ]

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
            # A path-guidance cue is due this frame (audio on or not).
            path_cue_due = False

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
                path_cue_due = (
                    blocking_detection is not None
                    and useful_directional_guidance
                    and path_decision.guidance_changed
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
                        "time_s": round(elapsed_s, 3),
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
            # Memory early warning (lowest audio priority)
            # -------------------------------------------------
            # A remembered static hazard ahead that the camera does not
            # see now gets a soft "remembered hazard" cue. An urgent or
            # path cue this frame suppresses it; it never touches the
            # planner. Earlier cues are confirmed when the camera sees
            # the hazard (lead time).
            early_warning.update(
                remembered,
                pose,
                detections,
                seen_hazard_ids if pose.valid else set(),
                memory_time,
                pose_trusted=pose_trusted,
                suppressed=bool(warnings) or path_cue_due,
            )
            memory_line = early_warning.display_text(memory_time)

            # -------------------------------------------------
            # Persistent-memory attention
            # -------------------------------------------------
            top_hazards = attention.rank(
                list(remembered),
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
                pose_mode=pose_mode,
                memory_line=memory_line,
            )

            if not args.no_lanes:
                draw_lanes(
                    display,
                    path_decision,
                    path_width_m=float(config["planner"]["path_width_m"]),
                    horizon_m=float(config["planner"]["horizon_m"]),
                    camera_height_m=float(perception_cfg["camera_height_m"]),
                    hfov_deg=float(perception.horizontal_fov_deg),
                )

            draw_minimap(
                display,
                (
                    # SLAM: the keyframe chain is the corrected path.
                    [
                        (kf.pose[0], kf.pose[1])
                        for kf in sorted(slam.keyframes.values(), key=lambda k: k.kf_id)
                        if kf.kf_id >= slam.loaded_keyframes
                    ]
                    + [(pose.x, pose.y)]
                    if slam is not None
                    else trajectory_xy[-2000:]
                ),
                (
                    [(kf.pose[0], kf.pose[1]) for kf in slam.keyframes.values()]
                    if slam is not None
                    else []
                ),
                slam.loop_edges() if slam is not None else [],
                [
                    (h.world_x, h.world_y, h.label, h.decay_weight * h.belief)
                    for h in remembered[:40]
                ],
                (pose.x, pose.y, pose.heading_deg),
                slam_state,
                info=(
                    f"kf {len(slam.keyframes)}  loops {slam.loop_closures}  "
                    f"inl {slam.last_inliers}"
                    if slam is not None
                    else ""
                ),
                lanes=[] if args.no_lanes else lanes_for_minimap(path_decision),
                lane_width_m=float(config["planner"]["path_width_m"]),
                lane_length_m=float(config["planner"]["horizon_m"]),
                highlight=(
                    (early_warning.last_cue.world_x, early_warning.last_cue.world_y)
                    if memory_line and early_warning.last_cue is not None
                    else None
                ),
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
                            "time_s": round(elapsed_s, 3),
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
                        f"WARNING | t={elapsed_s:.2f}s | "
                        f"{warning.label} | {warning.direction} | "
                        f"{warning.distance_m:.1f}m | "
                        f"{warning.category} | TTC={ttc_text} | "
                        f"score={warning.score:.2f} | "
                        f"physical_azimuth={warning.azimuth_deg:+.1f}°"
                    )
                    

            measured_fps = fps_meter.tick()
            if measured_fps is not None:
                line = f"FPS | pipeline {measured_fps:5.1f}"
                if live:
                    interval = fps_meter.interval_s
                    received = frame_source.frames_received
                    dropped = frame_source.frames_dropped
                    line += (
                        f" | camera {(received - last_received) / interval:5.1f}"
                        f" | skipped stale frames {dropped - last_dropped}"
                    )
                    last_received, last_dropped = received, dropped
                print(line, flush=True)

            if not args.no_display:
                cv2.imshow(
                    "BVI Navigation - YOLO + Path + TTC + HRTF - Q to stop",
                    display,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Processing stopped by user.")
                    break

    except KeyboardInterrupt:
        print("Processing stopped by user (Ctrl+C).")

    finally:
        frame_source.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        if hrtf_renderer is not None:
            hrtf_renderer.close()

        if phone_pose is not None:
            phone_pose.close()

        if range_sensor is not None:
            range_sensor.close()

        if slam is not None and slam.keyframes:
            slam.save_map(slam_map_path)
            print(f"SLAM map saved: {slam_map_path.resolve()} ({len(slam.keyframes)} keyframes)")

    if args.trajectory_out is not None and trajectory_rows:
        args.trajectory_out.parent.mkdir(parents=True, exist_ok=True)
        lines = ["time_s,x,y,heading_deg,valid,state"]
        lines += [
            f"{t},{x:.4f},{y:.4f},{h:.3f},{v},{st}" for t, x, y, h, v, st in trajectory_rows
        ]
        args.trajectory_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Trajectory CSV: {args.trajectory_out.resolve()}")

    processing_seconds = time.perf_counter() - processing_started
    persisted_hazards = memory.get_hazards()

    report = {
        "source_video": source if live else str(Path(source).resolve()),
        "live_source": live,
        "model_weights": str(weights_path.resolve()),
        "pose_source": pose_mode,
        "annotated_video": str(output_path.resolve()),
        "memory_db": str(db_path.resolve()),
        "memory_session_id": session_id,
        "memory_clock": (
            "wall clock (live)"
            if live
            else f"run start epoch {memory_clock.run_start_epoch:.3f} + video time"
        ),
        "config": config,
        "hrtf_file": (
            str(hrtf_path.resolve())
            if not args.no_audio
            else None
        ),
        "audio_enabled": hrtf_renderer is not None,
        "audio_warnings_played": audio_warning_count,
        "path_warning_audio_played": path_warning_audio_count,
        "frames_processed": frame_id,
        "invalid_pose_frames": invalid_pose_frames,
        "slam": (
            None
            if slam is None
            else {
                "map_file": str(slam_map_path.resolve()),
                "map_id": slam.map_id,
                "keyframes": len(slam.keyframes),
                "keyframes_loaded_from_earlier_runs": slam.loaded_keyframes,
                "loop_closures": slam.loop_closures,
                "relocalizations": slam.relocalizations,
                "hazards_moved_by_map_corrections": reanchored_total,
                "final_state": slam.state,
                "frames_processed_by_slam": slam_frames,
                "mean_slam_ms_per_frame": (
                    round(1000 * slam_seconds / slam_frames, 1) if slam_frames else None
                ),
                "events": [
                    {"time": round(e.timestamp, 3), "kind": e.kind, "detail": e.detail}
                    for e in slam.events
                ],
            }
        ),
        "video_fps": fps,
        "warning_frames": warning_frames,
        "unique_warning_objects": len(unique_warning_ids),
        "path_unsafe_frames": path_unsafe_frames,
        "processing_seconds": round(processing_seconds, 3),
        "average_fps": (
            round(frame_id / processing_seconds, 2)
            if processing_seconds > 0
            else None
        ),
        "path_events": path_events,
        "warning_events": warning_events,
        "range_fusion": {
            "enabled": range_sensor is not None,
            "frames_with_reading": range_frames,
            "detection_distances_replaced": range_corrections,
            "unknown_obstacle_frames": unknown_obstacle_frames,
        },
        "memory_cues": [
            record.as_dict(memory_clock.run_start_epoch)
            for record in early_warning.cue_log.records
        ],
        "memory_summary": early_warning.cue_log.summary(
            hazards_forgotten=len(negative_state.forgotten)
        ),
        "negative_evidence": {
            "misses_recorded": negative_state.misses_recorded,
            "hazards_forgotten": len(negative_state.forgotten),
            "forgotten": [
                {
                    "hazard_id": event.hazard_id,
                    "label": event.label,
                    "misses": event.misses,
                    "belief": round(event.belief, 3),
                    "time_s": round(event.time - memory_clock.run_start_epoch, 3),
                }
                for event in negative_state.forgotten
            ],
        },
        "persistent_memory": [
            {
                "hazard_id": hazard.hazard_id,
                "object_id": hazard.object_id,
                "label": hazard.label,
                "world_x": round(hazard.world_x, 3),
                "world_y": round(hazard.world_y, 3),
                "confidence": round(hazard.confidence, 3),
                "last_seen_timestamp": round(hazard.timestamp, 3),
                "first_seen_timestamp": (
                    None if hazard.first_seen is None else round(hazard.first_seen, 3)
                ),
                "times_seen": hazard.times_seen,
                "sessions_seen": hazard.sessions_seen,
                "decay_weight": round(hazard.decay_weight, 4),
                "belief": round(hazard.belief, 3),
                "misses": hazard.misses,
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
            "memory_early_warning": (
                "Remembered static hazard ahead that the camera does not see "
                "now; two soft low pips from the hazard's direction; lowest "
                "priority (suppressed by urgent/path cues), never used by the "
                "planner."
            ),
            "silent": (
                "Distant/non-actionable objects do not produce audio."
            ),
        },
        "limitations": [
            (
                "Distance is monocular (object height, or box bottom edge for ground-level "
                "classes), not measured depth"
                + (
                    "; the ESP32 ultrasonic sensor corrects the detection straight ahead "
                    "when it has a fresh reading."
                    if range_sensor is not None
                    else "; the ESP32 ultrasonic sensor was not used (--no-esp32)."
                )
            ),
            "Ground-level hazards are only in view from about "
            f"{1.9 * float(perception_cfg['camera_height_m']):.1f} m ahead (chest camera model), "
            "so memory cannot confirm or forget them closer than that.",
            "Memory early-warning cues and evidence-based forgetting were evaluated in "
            "simulation (tools/eval_memory.py), not in a user study.",
            "Remembered positions are only as consistent as the pose: visual SLAM keeps them "
            "within a few tens of cm in simulation, phone dead reckoning drifts by metres per "
            "walk, and the simulated pose (--pose sim) does not follow the video at all.",
            (
                "Pose is phone heading + step counting (stride from config); "
                "position drifts over long walks."
                if pose_mode == "phone"
                else "Pose is lightweight monocular visual SLAM (2D ground-plane "
                "pose, ORB visual odometry, loop closure, saved map); metric "
                "scale comes from phone steps or an assumed walking speed, and "
                "tracking can fail in low texture, low light or fast turns."
                if pose_mode == "slam"
                else "Pose comes from a deterministic simulated trajectory, "
                "not real localization."
            ),
            "Trajectory velocity is estimated from localized detections and can be noisy.",
            "TTC is a prototype prediction based on estimated relative motion and safety radii.",
            "HRTF uses the nearest horizontal-plane measurement available in the SOFA file.",
        ],
    }

    report_path = output_path.with_suffix(".json")
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
    memory_summary = report["memory_summary"]
    print(
        f"Memory cues: {memory_summary['cues_played']} "
        f"(later confirmed by the camera: {memory_summary['cues_later_confirmed']}, "
        f"mean lead {memory_summary['mean_lead_time_s']} s / "
        f"{memory_summary['mean_lead_distance_m']} m)"
    )
    print(f"Hazards forgotten by negative evidence: {len(negative_state.forgotten)}")
    print(f"Average FPS: {report['average_fps']}")
    print(f"Annotated video: {output_path.resolve()}")
    print(f"JSON report: {report_path.resolve()}")
    print(f"Memory DB: {db_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
