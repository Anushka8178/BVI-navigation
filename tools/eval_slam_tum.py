"""Evaluate navigation/vslam.py on a TUM RGB-D sequence (RGB images only).

Download a sequence from https://cvg.cit.tum.de/data/datasets/rgbd-dataset
and unpack it. Planar sequences match our 2D ground-plane model best:
  - fr2/pioneer_360, fr2/pioneer_slam, fr2/pioneer_slam2/3 (camera on a robot)
  - fr3/walking_xyz, fr3/walking_halfsphere (people walking: shows masking)
The ground truth is projected onto its best-fit plane, then the SLAM path
is aligned with a similarity transform (Umeyama) and ATE RMSE is reported.

Scale: one camera cannot measure distance. By default the metric distance
between frames comes from the ground truth path length ("oracle step
counter", standing in for the phone's step counter), which isolates the
visual parts: rotation, direction, loop closure. --scale assumed uses
slam.assumed_speed_m_s instead (no ground truth used during the run).

    python tools/eval_slam_tum.py C:\\data\\rgbd_dataset_freiburg2_pioneer_slam --camera fr2
    python tools/eval_slam_tum.py ... --out runs\\tum_pioneer   # CSV + PNG
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from navigation.config import load_config  # noqa: E402
from navigation.models import Pose  # noqa: E402
from navigation.slam_eval import associate, ate_rmse, project_to_plane  # noqa: E402
from navigation.vslam import SlamConfig, VisualSLAM  # noqa: E402

# TUM RGB-D calibrated intrinsics (640 x 480).
TUM_CAMERAS = {
    "fr1": (517.3, 516.5, 318.6, 255.3),
    "fr2": (520.9, 521.0, 325.1, 249.7),
    "fr3": (535.4, 539.2, 320.1, 247.6),
}


def read_list(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            rows.append(line.split())
    return rows


class GtOdometer:
    """Metric distance walked, from the ground truth (no heading help)."""

    stride_m = 0.001

    def __init__(self, times: np.ndarray, xyz: np.ndarray) -> None:
        self.times = times
        self.cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
        self.step_count = 0

    def update(self, t: float) -> None:
        j = int(np.clip(np.searchsorted(self.times, t), 0, len(self.times) - 1))
        self.step_count = int(self.cum[j] / self.stride_m)

    def get_pose(self, t: float) -> Pose:
        return Pose(0.0, 0.0, 0.0, t, valid=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sequence", type=Path, help="Unpacked TUM sequence folder")
    parser.add_argument("--camera", choices=sorted(TUM_CAMERAS), default="fr2")
    parser.add_argument("--scale", choices=("gt", "assumed"), default="gt")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Folder for trajectory CSV + plot")
    args = parser.parse_args()

    rgb = read_list(args.sequence / "rgb.txt")
    gt_rows = read_list(args.sequence / "groundtruth.txt")
    gt_times = np.array([float(r[0]) for r in gt_rows])
    gt_xyz = np.array([[float(v) for v in r[1:4]] for r in gt_rows])

    fx, fy, cx, cy = TUM_CAMERAS[args.camera]
    section = dict(load_config()["slam"])
    section.update(fx=fx, fy=fy, cx=cx, cy=cy, calib_width=640, camera_pitch_deg=0.0)
    cfg = SlamConfig.from_config(section)

    helper = GtOdometer(gt_times, gt_xyz) if args.scale == "gt" else None
    slam: VisualSLAM | None = None
    times, est = [], []
    frames = rgb[: args.max_frames] if args.max_frames else rgb
    for n, (stamp, rel_path) in enumerate(frames):
        image = cv2.imread(str(args.sequence / rel_path))
        if image is None:
            continue
        t = float(stamp)
        if slam is None:
            slam = VisualSLAM(cfg, (image.shape[1], image.shape[0]), helper=helper, log=print)
        if helper is not None:
            helper.update(t)
        pose = slam.process(image, t)
        if pose.valid:
            times.append(t)
            est.append((pose.x, pose.y))
        if n % 200 == 0:
            print(f"frame {n}/{len(frames)} state={slam.state} kf={len(slam.keyframes)} loops={slam.loop_closures}")

    if slam is None or len(est) < 10:
        raise SystemExit("Too few tracked frames to evaluate.")

    # Keyframe poses after the final optimization are the map estimate;
    # per-frame poses are what the pipeline used live. Report both.
    pairs = associate(times, gt_times)
    est_xy = np.array([est[i] for i, _ in pairs])
    gt_plane = project_to_plane(gt_xyz)
    gt_xy = np.array([gt_plane[j] for _, j in pairs])
    ate_live, aligned = ate_rmse(est_xy, gt_xy)

    kf_times = [kf.timestamp for kf in sorted(slam.keyframes.values(), key=lambda k: k.kf_id)]
    kf_xy = np.array([kf.pose[:2] for kf in sorted(slam.keyframes.values(), key=lambda k: k.kf_id)])
    kf_pairs = associate(kf_times, gt_times)
    ate_kf = (
        ate_rmse(np.array([kf_xy[i] for i, _ in kf_pairs]), np.array([gt_plane[j] for _, j in kf_pairs]))[0]
        if len(kf_pairs) >= 3
        else float("nan")
    )
    length = float(np.sum(np.linalg.norm(np.diff(gt_xy, axis=0), axis=1)))

    print("\n" + "=" * 60)
    print(f"Sequence: {args.sequence.name}  (scale from {args.scale})")
    print(f"Frames tracked: {len(est)}/{len(frames)}  keyframes: {len(slam.keyframes)}  "
          f"loop closures: {slam.loop_closures}")
    print(f"Ground-truth path length: {length:.2f} m")
    print(f"ATE RMSE (live per-frame pose): {ate_live:.3f} m  ({100 * ate_live / max(length, 1e-9):.1f} % of path)")
    print(f"ATE RMSE (keyframes after loop closure): {ate_kf:.3f} m")
    print("=" * 60)

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        lines = ["time_s,est_x,est_y,gt_x,gt_y"]
        lines += [f"{times[i]:.4f},{a[0]:.4f},{a[1]:.4f},{g[0]:.4f},{g[1]:.4f}"
                  for (i, _), a, g in zip(pairs, aligned, gt_xy)]
        (args.out / "trajectory.csv").write_text("\n".join(lines) + "\n")
        draw_plot(args.out / "trajectory.png", gt_xy, aligned, f"{args.sequence.name}  ATE {ate_live:.3f} m")
        print(f"Saved {args.out / 'trajectory.csv'} and trajectory.png")


def draw_plot(path: Path, gt: np.ndarray, est: np.ndarray, title: str, size: int = 700) -> None:
    img = np.full((size, size, 3), 255, np.uint8)
    pts = np.vstack([gt, est])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = max(float(np.max(hi - lo)), 1e-6) * 1.1
    centre = (hi + lo) / 2

    def px(p):
        return (int(size / 2 + (p[0] - centre[0]) / span * (size - 80)),
                int(size / 2 + 20 - (p[1] - centre[1]) / span * (size - 80)))

    cv2.polylines(img, [np.array([px(p) for p in gt], np.int32)], False, (60, 60, 60), 2, cv2.LINE_AA)
    cv2.polylines(img, [np.array([px(p) for p in est], np.int32)], False, (40, 120, 255), 2, cv2.LINE_AA)
    cv2.putText(img, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "grey: ground truth   orange: SLAM", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(str(path), img)


if __name__ == "__main__":
    main()
