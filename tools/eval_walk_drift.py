"""Drift of real campus walks, from the demo's --trajectory-out CSV files.

Walk a closed loop that ends exactly at the taped start mark, once with
--pose phone and once with --pose slam (same route), each with
--trajectory-out. This prints distance walked, start-to-end error and
drift (% of distance) for each file, and saves a top-down PNG.

    python tools/eval_walk_drift.py runs\\loop_phone.csv runs\\loop_slam.csv --png runs\\loop.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from navigation.slam_eval import loop_drift  # noqa: E402

COLORS = [(40, 120, 255), (60, 170, 60), (200, 80, 40), (150, 60, 150)]


def load_xy(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("valid", "1") == "1"]
    return np.array([[float(r["x"]), float(r["y"])] for r in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, nargs="+")
    parser.add_argument("--png", type=Path, default=None)
    args = parser.parse_args()

    paths = []
    print(f"{'file':32s} {'distance m':>11s} {'end error m':>12s} {'drift %':>8s}")
    for path in args.csv:
        xy = load_xy(path)
        if len(xy) < 2:
            print(f"{path.name:32s} (no valid poses)")
            continue
        d = loop_drift(xy)
        print(f"{path.name:32s} {d['distance_m']:11.2f} {d['end_error_m']:12.2f} {d['drift_percent']:8.1f}")
        paths.append((path.stem, xy))

    if args.png is not None and paths:
        size = 700
        img = np.full((size, size, 3), 255, np.uint8)
        allp = np.vstack([xy for _n, xy in paths])
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        span = max(float(np.max(hi - lo)), 1.0) * 1.15
        centre = (hi + lo) / 2

        def px(p):
            return (int(size / 2 + (p[0] - centre[0]) / span * (size - 80)),
                    int(size / 2 + 20 - (p[1] - centre[1]) / span * (size - 80)))

        for n, (name, xy) in enumerate(paths):
            color = COLORS[n % len(COLORS)]
            cv2.polylines(img, [np.array([px(p) for p in xy], np.int32)], False, color, 2, cv2.LINE_AA)
            cv2.circle(img, px(xy[-1]), 6, color, -1)
            cv2.putText(img, name, (12, 24 + 22 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.circle(img, px(paths[0][1][0]), 8, (0, 0, 0), 2)
        cv2.putText(img, "o = start, dot = end", (12, size - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        args.png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.png), img)
        print(f"Saved {args.png}")


if __name__ == "__main__":
    main()
