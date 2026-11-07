"""Simulation check: visual SLAM vs phone dead reckoning on a known path.

Renders a chest camera walking square laps inside a textured room
(navigation/synthetic_scene.py), with a simulated phone that miscounts
stride length and whose heading drifts. Prints drift and ATE for both
pose sources against ground truth, and can write a side-by-side video.
This validates the SLAM logic; it is not a substitute for real walks.

    python tools/simulate_slam_walk.py --laps 2 --video runs\\sim_slam.mp4
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
from navigation.minimap import draw_minimap  # noqa: E402
from navigation.slam_eval import ate_rmse  # noqa: E402
from navigation.synthetic_scene import FakePhone, Room, render, square_walk  # noqa: E402
from navigation.vslam import SlamConfig, VisualSLAM, camera_matrix  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--laps", type=int, default=2)
    parser.add_argument("--side", type=float, default=8.0, help="square side (m)")
    parser.add_argument("--stride-error", type=float, default=0.07, help="true stride = (1 + e) x configured")
    parser.add_argument("--heading-drift", type=float, default=0.3, help="phone heading drift (deg/s)")
    parser.add_argument("--video", type=Path, default=None)
    args = parser.parse_args()

    size = (640, 360)
    cfg = SlamConfig.from_config(load_config()["slam"])
    K = camera_matrix(cfg, *size)
    room = Room()
    walk = square_walk(side_m=args.side, laps=args.laps)
    phone = FakePhone(stride_m=0.7, true_stride_m=0.7 * (1 + args.stride_error), drift_deg_s=args.heading_drift)
    slam = VisualSLAM(cfg, size, helper=phone, log=print)
    x0, y0 = walk[0][1], walk[0][2]

    writer = None
    if args.video is not None:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter.fourcc(*"mp4v"), 10, (size[0] * 2, size[1]))

    gt, est_slam, est_phone = [], [], []
    for t, x, y, h in walk:
        phone.update(t, x, y, h)
        frame = render(room, x, y, h, K, size)
        pose = slam.process(frame, t)
        gt.append((x - x0, y - y0))
        est_slam.append((pose.x, pose.y))
        pp = phone.get_pose(t)
        est_phone.append((pp.x, pp.y))
        if writer is not None:
            left = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            right = np.zeros_like(left)
            kfs = [kf.pose[:2] for kf in sorted(slam.keyframes.values(), key=lambda k: k.kf_id)]
            draw_minimap(right, kfs + [(pose.x, pose.y)], kfs, slam.loop_edges(), [],
                         (pose.x, pose.y, pose.heading_deg), slam.state,
                         info=f"loops {slam.loop_closures}", size=340, margin=10)
            cv2.putText(right, "SLAM (keyframe map)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            writer.write(np.hstack([left, right]))
    if writer is not None:
        writer.release()

    gt_a, s_a, p_a = map(np.array, (gt, est_slam, est_phone))
    walked = float(np.sum(np.linalg.norm(np.diff(gt_a, axis=0), axis=1)))
    print("\n" + "=" * 64)
    print(f"Simulated walk: {args.laps} lap(s), {walked:.1f} m, stride error {100 * args.stride_error:.0f} %, "
          f"phone heading drift {args.heading_drift} deg/s")
    print(f"{'pose source':22s} {'end error m':>12s} {'drift %':>8s} {'ATE (Sim2) m':>13s}")
    for name, e in (("phone dead reckoning", p_a), ("visual SLAM", s_a)):
        end = math.dist(e[-1], gt_a[-1])
        ate, _ = ate_rmse(e, gt_a)
        print(f"{name:22s} {end:12.2f} {100 * end / walked:8.1f} {ate:13.3f}")
    print(f"SLAM: {len(slam.keyframes)} keyframes, {slam.loop_closures} loop closures")
    print("=" * 64)


if __name__ == "__main__":
    main()
