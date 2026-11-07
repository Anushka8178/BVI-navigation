"""Calibrate the phone camera for the visual SLAM (manual, needs a camera).

Print a chessboard (e.g. 9 x 6 inner corners, search "opencv chessboard
pattern"), measure one square, and either:

  # live from the IP Webcam stream: press SPACE to grab a view, Q to finish
  python scripts/manual/calibrate_camera.py --source http://<PHONE_IP>:8080/video

  # or from a folder of photos taken at the SAME resolution as the stream
  python scripts/manual/calibrate_camera.py --images calib_photos

Grab 15-25 views with the board at different positions, distances and
tilts, covering the corners of the image. The script prints the `slam:`
values to paste into config.yaml (fx, fy, cx, cy, calib_width).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def find_corners(gray, pattern):
    found, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return None
    return cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=None, help="stream URL or camera index")
    parser.add_argument("--images", type=Path, default=None, help="folder of chessboard photos")
    parser.add_argument("--cols", type=int, default=9, help="inner corners per row")
    parser.add_argument("--rows", type=int, default=6, help="inner corners per column")
    parser.add_argument("--square-mm", type=float, default=25.0)
    args = parser.parse_args()
    if (args.source is None) == (args.images is None):
        raise SystemExit("Give exactly one of --source or --images.")

    pattern = (args.cols, args.rows)
    board = np.zeros((args.cols * args.rows, 3), np.float32)
    board[:, :2] = np.mgrid[0 : args.cols, 0 : args.rows].T.reshape(-1, 2) * args.square_mm / 1000.0
    obj_points, img_points, size = [], [], None

    if args.images is not None:
        for path in sorted(args.images.iterdir()):
            image = cv2.imread(str(path))
            if image is None:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            corners = find_corners(gray, pattern)
            print(f"{path.name}: {'ok' if corners is not None else 'board not found'}")
            if corners is not None:
                obj_points.append(board)
                img_points.append(corners)
                size = gray.shape[::-1]
    else:
        source = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise SystemExit(f"Could not open {args.source}")
        print("SPACE = grab view, Q = finish")
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = find_corners(gray, pattern)
            view = frame.copy()
            if corners is not None:
                cv2.drawChessboardCorners(view, pattern, corners, True)
            cv2.putText(view, f"views: {len(img_points)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("calibration", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and corners is not None:
                obj_points.append(board)
                img_points.append(corners)
                size = gray.shape[::-1]
            elif key == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    if len(img_points) < 8:
        raise SystemExit(f"Only {len(img_points)} usable views; take at least 8 (15+ is better).")
    rms, K, dist, _r, _t = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    print(f"\nRMS reprojection error: {rms:.3f} px (below ~0.5 is good)")
    print(f"Frame size: {size[0]} x {size[1]}\n")
    print("Paste into config.yaml under slam:")
    print(f"  fx: {K[0, 0]:.2f}")
    print(f"  fy: {K[1, 1]:.2f}")
    print(f"  cx: {K[0, 2]:.2f}")
    print(f"  cy: {K[1, 2]:.2f}")
    print(f"  calib_width: {size[0]}")
    hfov = 2 * np.degrees(np.arctan(size[0] / (2 * K[0, 0])))
    print(f"\n(horizontal FOV = {hfov:.1f} deg; distortion {np.round(dist.ravel()[:5], 3).tolist()})")


if __name__ == "__main__":
    main()
