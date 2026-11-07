"""
Monocular visual odometry for the BVI navigation demo.

Pipeline per frame:
    ORB keypoints -> match with previous frame -> essential matrix (RANSAC)
    -> recoverPose (R, t) -> integrate yaw + planar motion -> Pose

Convention: x forward (at start), y to the left, heading_deg counter-clockwise.

Limitations (state these honestly in the report):
  * Monocular: translation scale is unknown, so step length is approximated
    from an assumed walking speed and the frame time.
  * Camera intrinsics are approximated from image width (no calibration).
  * Odometry only: no loop closure / global map, so drift accumulates.
    This is the tracking front-end of a SLAM system, not full SLAM.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .models import Pose


class MonocularVisualOdometry:
    def __init__(
        self,
        focal_ratio: float = 0.9,
        walking_speed_mps: float = 1.0,
        min_flow_px: float = 1.5,
        n_features: int = 1500,
        work_width: int = 640,
    ) -> None:
        self.focal_ratio = focal_ratio
        self.walking_speed = walking_speed_mps
        self.min_flow_px = min_flow_px
        self.work_width = work_width
        self.orb = cv2.ORB_create(n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.K = None
        self.prev_kp = None
        self.prev_des = None
        self.prev_time = None

        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0  # radians, CCW
        self.trajectory: list[tuple[float, float]] = [(0.0, 0.0)]
        self.status = "init"
        self.inliers = 0

    # ------------------------------------------------------------------ #
    def _prepare(self, frame):
        h, w = frame.shape[:2]
        if w > self.work_width:
            scale = self.work_width / w
            frame = cv2.resize(frame, (self.work_width, int(h * scale)))
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.K is None:
            h, w = frame.shape[:2]
            f = self.focal_ratio * w
            self.K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
        return frame

    def _pose(self, timestamp: float) -> Pose:
        return Pose(
            x=self.x,
            y=self.y,
            heading_deg=math.degrees(self.heading),
            timestamp=timestamp,
        )

    def _store(self, kp, des, timestamp):
        self.prev_kp, self.prev_des, self.prev_time = kp, des, timestamp

    # ------------------------------------------------------------------ #
    def update(self, frame, timestamp: float) -> Pose:
        if frame is None:
            self.status = "no frame"
            return self._pose(timestamp)

        gray = self._prepare(frame)
        kp, des = self.orb.detectAndCompute(gray, None)

        if self.prev_des is None or des is None or len(kp) < 8:
            self.status = "init" if self.prev_des is None else "few features"
            self._store(kp, des, timestamp)
            return self._pose(timestamp)

        dt = timestamp - self.prev_time if self.prev_time is not None else 1 / 15
        if not (0 < dt < 0.5):
            dt = 1 / 15

        matches = self.matcher.match(self.prev_des, des)
        if len(matches) < 15:
            self.status = "lost (few matches)"
            self._store(kp, des, timestamp)
            return self._pose(timestamp)

        matches = sorted(matches, key=lambda m: m.distance)[:500]
        p1 = np.float64([self.prev_kp[m.queryIdx].pt for m in matches])
        p2 = np.float64([kp[m.trainIdx].pt for m in matches])

        flow = float(np.median(np.linalg.norm(p2 - p1, axis=1)))
        if flow < self.min_flow_px:
            # Camera is basically still: do not integrate noise.
            self.status = "still"
            self._store(kp, des, timestamp)
            return self._pose(timestamp)

        E, mask = cv2.findEssentialMat(p1, p2, self.K, cv2.RANSAC, 0.999, 1.0)
        if E is None or E.shape[0] < 3:
            self.status = "no essential matrix"
            self._store(kp, des, timestamp)
            return self._pose(timestamp)
        E = E[:3]

        n_inliers, R, t, _ = cv2.recoverPose(E, p1, p2, self.K, mask=mask)
        self.inliers = int(n_inliers)
        if n_inliers < 15:
            self.status = "low inliers"
            self._store(kp, des, timestamp)
            return self._pose(timestamp)

        # Camera motion expressed in the previous camera frame
        # (OpenCV camera axes: x right, y down, z forward).
        Rc = R.T
        turn_right = math.atan2(Rc[0, 2], Rc[2, 2])
        turn_right = max(-math.radians(30), min(math.radians(30), turn_right))

        d = (-Rc @ t).ravel()
        fwd, right = float(d[2]), float(d[0])
        norm = math.hypot(fwd, right)

        heading_before = self.heading
        self.heading -= turn_right  # right turn = clockwise = negative CCW

        # Only translate when motion is mostly forward (avoids fake steps
        # while turning on the spot). Scale = assumed walking speed * dt.
        if norm > 1e-6 and fwd / norm > 0.3:
            fwd, right = fwd / norm, right / norm
            step = self.walking_speed * dt
            h = heading_before
            self.x += step * (fwd * math.cos(h) + right * math.sin(h))
            self.y += step * (fwd * math.sin(h) - right * math.cos(h))
            self.trajectory.append((self.x, self.y))
            self.status = "tracking"
        else:
            self.status = "turning"

        self._store(kp, des, timestamp)
        return self._pose(timestamp)

    # ------------------------------------------------------------------ #
    def draw_minimap(self, image, size: int = 160, metres: float = 10.0):
        """Draw the walked path in the top-right corner of `image` (in place)."""
        h, w = image.shape[:2]
        x0, y0 = w - size - 10, 10
        cv2.rectangle(image, (x0, y0), (x0 + size, y0 + size), (0, 0, 0), -1)
        cv2.rectangle(image, (x0, y0), (x0 + size, y0 + size), (255, 255, 255), 1)
        scale = size / metres
        cx, cy = x0 + size // 2, y0 + size - 15

        def to_px(px, py):
            # forward = up on the map, left = left
            return int(cx - py * scale), int(cy - px * scale)

        pts = [to_px(px, py) for px, py in self.trajectory[-400:]]
        for a, b in zip(pts, pts[1:]):
            cv2.line(image, a, b, (0, 255, 0), 2)
        hx, hy = to_px(self.x, self.y)
        ex = int(hx - 12 * math.sin(self.heading))
        ey = int(hy - 12 * math.cos(self.heading))
        cv2.circle(image, (hx, hy), 4, (0, 0, 255), -1)
        cv2.line(image, (hx, hy), (ex, ey), (0, 0, 255), 2)
        cv2.putText(image, f"VO: {self.status}", (x0 + 4, y0 + size - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return image
