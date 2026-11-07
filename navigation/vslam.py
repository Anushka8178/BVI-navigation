"""Lightweight keyframe monocular visual SLAM with inertial-assisted scale.

What it is
----------
A real (not simulated) camera-based SLAM that runs on the laptop in plain
Python + OpenCV + NumPy + SciPy, behind the same `Pose` interface as
`PhonePose` and `simulated_pose()`:

- Tracking (visual odometry): ORB features on the chest-camera frame are
  matched against the current keyframe; the essential matrix (RANSAC)
  gives the camera rotation and translation direction. Features on
  dynamic YOLO boxes (people, vehicles, animals) are masked out first.
- Metric scale: one camera cannot measure distance, so the length of each
  movement comes from the phone's step counter (steps x stride) when the
  phone is connected, otherwise from an assumed walking speed.
- Heading fusion: the visual yaw is pulled gently towards the phone's
  rotation-vector heading at every keyframe (complementary filter), which
  limits long-term yaw drift.
- Mapping: keyframes (pose + ORB keypoints/descriptors + a tiny thumbnail
  for place recognition) form the map.
- Loop closure: when a new keyframe looks like an old one nearby, a loop
  edge is added and a 2D pose graph (x, y, heading) is optimized with
  scipy.optimize.least_squares (false loops are rejected by a consistency
  check). All keyframe poses
  move, and hazards anchored to keyframes move with them.
- Persistence + relocalization: the map is saved to disk. On the next run
  the first frames are matched against the saved keyframes; once one is
  recognized, this run continues in the SAME world frame as the previous
  run, so hazards in persistent memory line up across walks.
- Tracking loss: falls back to phone dead reckoning (if connected) while
  trying to re-find a known keyframe.

What it is not
--------------
Not ORB-SLAM3. The pose is 2D on a flat ground plane (x, y, heading),
loop edges constrain heading tightly but translation only loosely (they
assume the two views were taken close together), and there is no bundle
adjustment or 3D map. Low texture, low light and very fast turns can
lose tracking.

Conventions (CLAUDE.md)
-----------------------
World frame: x = right of the start heading, y = forward at the start.
Heading: degrees clockwise from +y. Camera (OpenCV): x right, y down,
z forward.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from navigation.models import Pose


# States reported by VisualSLAM.state
INIT = "INIT"  # no map yet, waiting for the first frame
RELOCALIZING = "RELOCALIZING"  # map loaded, not yet recognized where we are
TRACKING = "TRACKING"
LOST = "LOST"  # visual tracking lost; phone dead reckoning if available

EDGE_ODOM = 0
EDGE_LOOP = 1
EDGE_RELOC = 2

THUMB_SIZE = (32, 24)  # (w, h) for the place-recognition thumbnail


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlamConfig:
    """All tunables; built from config.yaml's `slam` section."""

    image_width: int = 640
    hfov_deg: float = 65.0
    # Calibrated intrinsics at the ORIGINAL frame size (null = use hfov_deg).
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    calib_width: int | None = None  # frame width the calibration was made at
    camera_pitch_deg: float = 0.0  # positive = camera tilted down
    n_features: int = 1500
    map_features: int = 800
    grid_cols: int = 8
    grid_rows: int = 6
    fast_threshold: int = 12
    ratio_test: float = 0.75
    ransac_px: float = 1.0
    min_inliers: int = 40
    lost_frames: int = 8
    kf_min_inliers: int = 120
    kf_parallax_px: float = 25.0
    kf_yaw_deg: float = 15.0
    kf_max_distance_m: float = 1.5
    min_trans_parallax_px: float = 2.0
    max_lateral_deg: float = 45.0
    dir_min_parallax_px: float = 8.0
    assumed_speed_m_s: float = 1.0
    heading_phone_weight: float = 0.05
    loop_min_gap: int = 30
    loop_radius_m: float = 3.0
    loop_radius_per_m: float = 0.05
    loop_candidates: int = 3
    loop_min_ncc: float = 0.55
    loop_min_inliers: int = 80
    reloc_candidates: int = 5
    reloc_min_inliers: int = 60
    reloc_every_n: int = 3
    odom_sigma_m: float = 0.05
    odom_sigma_per_m: float = 0.1
    odom_sigma_deg: float = 2.0
    loop_sigma_m: float = 0.2
    loop_sigma_unscaled_m: float = 1.0
    loop_sigma_deg: float = 3.0
    save_every_kf: int = 20
    graph_loss: str = "linear"
    graph_f_scale: float = 3.0
    loop_max_odom_sigma: float = 4.0

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> "SlamConfig":
        """Build from the `slam` dict; keys that are not fields are ignored
        (e.g. map_path, process_every_n, used by the demo script)."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in section.items() if k in names})


def camera_matrix(cfg: SlamConfig, frame_width: int, frame_height: int) -> NDArray[np.float64]:
    """Intrinsics at the SLAM working resolution (cfg.image_width)."""
    scale_to_work = cfg.image_width / float(frame_width)
    if cfg.fx is not None and cfg.fy is not None:
        calib_w = float(cfg.calib_width or frame_width)
        s = cfg.image_width / calib_w
        fx, fy = cfg.fx * s, cfg.fy * s
        cx = (cfg.cx if cfg.cx is not None else calib_w / 2.0) * s
        cy = (cfg.cy if cfg.cy is not None else frame_height * calib_w / frame_width / 2.0) * s
    else:
        work_w = cfg.image_width
        work_h = frame_height * scale_to_work
        fx = fy = (work_w / 2.0) / math.tan(math.radians(cfg.hfov_deg) / 2.0)
        cx, cy = work_w / 2.0, work_h / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# 2D pose maths (CLAUDE.md world frame)
# ---------------------------------------------------------------------------


def wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def compose(pose: Sequence[float], right: float, forward: float, dyaw_deg: float) -> tuple[float, float, float]:
    """Apply a motion given in the frame of `pose` (x, y, heading_deg)."""
    x, y, h = pose
    hr = math.radians(h)
    return (
        x + right * math.cos(hr) + forward * math.sin(hr),
        y - right * math.sin(hr) + forward * math.cos(hr),
        wrap_deg(h + dyaw_deg),
    )


def relative(pose_i: Sequence[float], pose_j: Sequence[float]) -> tuple[float, float, float]:
    """Motion (right, forward, dyaw) of pose_j expressed in pose_i's frame."""
    xi, yi, hi = pose_i
    xj, yj, hj = pose_j
    hr = math.radians(hi)
    dx, dy = xj - xi, yj - yi
    return (
        dx * math.cos(hr) - dy * math.sin(hr),
        dx * math.sin(hr) + dy * math.cos(hr),
        wrap_deg(hj - hi),
    )


def rot_x(angle_rad: float) -> NDArray[np.float64]:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def ground_motion_full(
    R: NDArray[np.float64], t: NDArray[np.float64], pitch_deg: float = 0.0
) -> tuple[float, float, float]:
    """Camera motion -> (dyaw_deg, right, forward) on the ground, with the
    translation length kept (metric if `t` is metric).

    `R, t` follow OpenCV `recoverPose(points_ref, points_cur)` /
    `solvePnP(points_in_ref_frame, pixels_cur)`: they map a point from the
    reference camera frame into the current camera frame
    (X_cur = R X_ref + t). The camera's own motion is the inverse. A camera
    tilted down by `pitch_deg` is first rotated back to a level frame.
    """
    Rp = rot_x(math.radians(pitch_deg))  # level -> camera
    Rg = Rp.T @ R @ Rp
    tg = Rp.T @ np.asarray(t, dtype=float).reshape(3)
    forward_axis = Rg.T @ np.array([0.0, 0.0, 1.0])  # current forward, in ref frame
    dyaw = math.degrees(math.atan2(forward_axis[0], forward_axis[2]))
    centre = -Rg.T @ tg  # current camera centre in the reference frame
    return dyaw, float(centre[0]), float(centre[2])


def ground_motion(
    R: NDArray[np.float64], t: NDArray[np.float64], pitch_deg: float = 0.0
) -> tuple[float, float, float]:
    """Like ground_motion_full, but the direction is a unit vector (or
    (0, 0) for no translation): monocular translation has no scale."""
    dyaw, right, fwd = ground_motion_full(R, t, pitch_deg)
    norm = math.hypot(right, fwd)
    if norm < 1e-9:
        return dyaw, 0.0, 0.0
    return dyaw, right / norm, fwd / norm


def kabsch_rotation(rays_ref: NDArray[np.float64], rays_cur: NDArray[np.float64]) -> NDArray[np.float64]:
    """Best rotation R with rays_cur ~ R @ rays_ref (rotation-only motion)."""
    H = rays_ref.T @ rays_cur
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


@dataclass
class FrameFeatures:
    keypoints: NDArray[np.float32]  # (N, 2) pixel coords at work resolution
    descriptors: NDArray[np.uint8]  # (N, 32)
    thumb: NDArray[np.float32]  # normalized thumbnail, flattened


def make_thumb(gray: NDArray[np.uint8]) -> NDArray[np.float32]:
    small = cv2.resize(gray, THUMB_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    small -= small.mean()
    std = float(small.std())
    if std > 1e-6:
        small /= std
    return small.reshape(-1)


def thumb_ncc(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    return float(np.dot(a, b) / a.size)


class FeatureExtractor:
    def __init__(self, cfg: SlamConfig) -> None:
        self.cfg = cfg
        self.orb = cv2.ORB_create(
            nfeatures=cfg.n_features * 2,
            scaleFactor=1.2,
            nlevels=8,
            fastThreshold=cfg.fast_threshold,
        )

    def extract(
        self,
        gray: NDArray[np.uint8],
        mask_boxes: Iterable[Sequence[float]] = (),
    ) -> FrameFeatures:
        h, w = gray.shape[:2]
        mask = np.full((h, w), 255, dtype=np.uint8)
        for x1, y1, x2, y2 in mask_boxes:
            cv2.rectangle(
                mask,
                (int(max(x1, 0)), int(max(y1, 0))),
                (int(min(x2, w - 1)), int(min(y2, h - 1))),
                0,
                -1,
            )
        keypoints = self.orb.detect(gray, mask)
        keypoints = self._bucket(keypoints, w, h)
        keypoints, descriptors = self.orb.compute(gray, keypoints)
        if descriptors is None or not keypoints:
            return FrameFeatures(
                np.zeros((0, 2), np.float32), np.zeros((0, 32), np.uint8), make_thumb(gray)
            )
        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        return FrameFeatures(pts, descriptors, make_thumb(gray))

    def _bucket(self, keypoints: Sequence[cv2.KeyPoint], w: int, h: int) -> list[cv2.KeyPoint]:
        """Keep the strongest corners per grid cell so features spread out."""
        cols, rows = self.cfg.grid_cols, self.cfg.grid_rows
        per_cell = max(1, self.cfg.n_features // (cols * rows))
        cells: dict[tuple[int, int], list[cv2.KeyPoint]] = {}
        for kp in keypoints:
            cx = min(int(kp.pt[0] * cols / w), cols - 1)
            cy = min(int(kp.pt[1] * rows / h), rows - 1)
            cells.setdefault((cx, cy), []).append(kp)
        kept: list[cv2.KeyPoint] = []
        for cell in cells.values():
            cell.sort(key=lambda k: k.response, reverse=True)
            kept.extend(cell[:per_cell])
        return kept


@dataclass
class MotionEstimate:
    dyaw_deg: float
    dir_right: float
    dir_forward: float
    inliers: int
    parallax_px: float  # median displacement of inliers
    trans_parallax_px: float  # displacement left after removing rotation
    rotation_only: bool


class MotionEstimator:
    def __init__(self, cfg: SlamConfig, K: NDArray[np.float64]) -> None:
        self.cfg = cfg
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def match(self, des_ref: NDArray[np.uint8], des_cur: NDArray[np.uint8]) -> list[tuple[int, int]]:
        if len(des_ref) < 2 or len(des_cur) < 2:
            return []
        pairs = self.matcher.knnMatch(des_ref, des_cur, k=2)
        good = []
        for pair in pairs:
            if len(pair) == 2 and pair[0].distance < self.cfg.ratio_test * pair[1].distance:
                good.append((pair[0].queryIdx, pair[0].trainIdx))
        return good

    def _rays(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        homog = np.hstack([pts, np.ones((len(pts), 1))])
        rays = (self.K_inv @ homog.T).T
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    def estimate(
        self, ref: FrameFeatures, cur: FrameFeatures, min_inliers: int
    ) -> MotionEstimate | None:
        """Relative motion of `cur` with respect to `ref`, or None."""
        good = self.match(ref.descriptors, cur.descriptors)
        if len(good) < max(min_inliers, 8):
            return None
        p_ref = ref.keypoints[[g[0] for g in good]].astype(np.float64)
        p_cur = cur.keypoints[[g[1] for g in good]].astype(np.float64)

        E, mask = cv2.findEssentialMat(
            p_ref, p_cur, self.K, method=cv2.RANSAC, prob=0.999, threshold=self.cfg.ransac_px
        )
        if E is None or mask is None:
            return None
        if E.shape[0] > 3:  # several solutions: keep the first
            E = E[:3]
        inl = mask.ravel().astype(bool)
        if int(inl.sum()) < min_inliers:
            return None
        p_ref, p_cur = p_ref[inl], p_cur[inl]
        parallax = float(np.median(np.linalg.norm(p_cur - p_ref, axis=1)))

        # Rotation-only model: how much displacement is left after
        # de-rotating? Small -> translation is not observable this frame.
        R_rot = kabsch_rotation(self._rays(p_ref), self._rays(p_cur))
        H = self.K @ R_rot @ self.K_inv
        warped = cv2.perspectiveTransform(p_ref.reshape(-1, 1, 2), H).reshape(-1, 2)
        trans_parallax = float(np.median(np.linalg.norm(p_cur - warped, axis=1)))

        if trans_parallax < self.cfg.min_trans_parallax_px:
            dyaw, _r, _f = ground_motion(R_rot, np.zeros(3), self.cfg.camera_pitch_deg)
            return MotionEstimate(dyaw, 0.0, 0.0, int(inl.sum()), parallax, trans_parallax, True)

        n_ok, R, t, pose_mask = cv2.recoverPose(E, p_ref, p_cur, self.K)
        if n_ok < min_inliers // 2:
            dyaw, _r, _f = ground_motion(R_rot, np.zeros(3), self.cfg.camera_pitch_deg)
            return MotionEstimate(dyaw, 0.0, 0.0, int(inl.sum()), parallax, trans_parallax, True)
        dyaw, dr, df = ground_motion(R, t, self.cfg.camera_pitch_deg)
        return MotionEstimate(dyaw, dr, df, int(inl.sum()), parallax, trans_parallax, False)


    def triangulate(
        self, a: FrameFeatures, b: FrameFeatures, baseline_m: float
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]] | None:
        """3D points (in a's camera frame, metres) seen by keyframes a and b.

        The two keyframes' metric distance `baseline_m` (from the pose
        graph) fixes the scale. Returns (indices into a's features, points).
        """
        good = self.match(a.descriptors, b.descriptors)
        if len(good) < 30:
            return None
        ia = np.array([g[0] for g in good])
        pa = a.keypoints[ia].astype(np.float64)
        pb = b.keypoints[[g[1] for g in good]].astype(np.float64)
        E, mask = cv2.findEssentialMat(pa, pb, self.K, method=cv2.RANSAC, prob=0.999, threshold=self.cfg.ransac_px)
        if E is None or mask is None:
            return None
        if E.shape[0] > 3:
            E = E[:3]
        _n, R, t, pose_mask = cv2.recoverPose(E, pa, pb, self.K, mask=mask.copy())
        keep = pose_mask.ravel() > 0
        if keep.sum() < 20:
            return None
        ia, pa, pb = ia[keep], pa[keep], pb[keep]
        t = t.reshape(3) * baseline_m
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t.reshape(3, 1)])
        X = cv2.triangulatePoints(P1, P2, pa.T, pb.T)
        X = (X[:3] / X[3]).T
        Xb = (R @ X.T).T + t
        ok = (X[:, 2] > 0.3) & (Xb[:, 2] > 0.3) & (X[:, 2] < 60.0)
        # Reject points with a tiny triangulation angle (unreliable depth).
        rays_a = X / np.linalg.norm(X, axis=1, keepdims=True)
        cb = -R.T @ t
        rays_b = (X - cb) / np.linalg.norm(X - cb, axis=1, keepdims=True)
        angle = np.degrees(np.arccos(np.clip(np.sum(rays_a * rays_b, axis=1), -1, 1)))
        ok &= angle > 1.0
        if ok.sum() < 20:
            return None
        return ia[ok], X[ok]

    def metric_relative(
        self,
        a: FrameFeatures,
        cloud: tuple[NDArray[np.int64], NDArray[np.float64]],
        cur: FrameFeatures,
        min_inliers: int,
    ) -> tuple[tuple[float, float, float], int] | None:
        """Metric (right, forward, dyaw) of `cur` in a's frame via PnP."""
        idx, pts3d = cloud
        lookup = {int(k): n for n, k in enumerate(idx)}
        good = [g for g in self.match(a.descriptors, cur.descriptors) if g[0] in lookup]
        if len(good) < max(min_inliers // 2, 12):
            return None
        obj = np.array([pts3d[lookup[g[0]]] for g in good], dtype=np.float64)
        img = np.array([cur.keypoints[g[1]] for g in good], dtype=np.float64)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, self.K, None, reprojectionError=2.0, iterationsCount=200, confidence=0.999
        )
        if not ok or inliers is None or len(inliers) < max(min_inliers // 2, 12):
            return None
        R, _ = cv2.Rodrigues(rvec)
        dyaw, right, fwd = ground_motion_full(R, tvec, self.cfg.camera_pitch_deg)
        return (right, fwd, dyaw), int(len(inliers))


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------


@dataclass
class Keyframe:
    kf_id: int
    timestamp: float
    pose: tuple[float, float, float]
    features: FrameFeatures
    odometer_m: float = 0.0  # helper odometer when created (this session)


@dataclass
class Edge:
    i: int  # kf_id
    j: int
    meas: tuple[float, float, float]  # (right, forward, dyaw) of j in i's frame
    kind: int = EDGE_ODOM
    sigma_m: float = 0.0  # 0 = use the default for this edge kind


def edge_residual(
    poses: dict[int, tuple[float, float, float]], e: Edge, sigma_t: float, sigma_h_deg: float
) -> float:
    """Largest whitened residual (in sigmas) of one edge."""
    r, f, dh = relative(poses[e.i], poses[e.j])
    return max(
        abs(r - e.meas[0]) / sigma_t,
        abs(f - e.meas[1]) / sigma_t,
        abs(wrap_deg(dh - e.meas[2])) / sigma_h_deg,
    )


def optimize_pose_graph(
    poses: dict[int, tuple[float, float, float]],
    edges: Sequence[Edge],
    fixed: int,
    sigmas: Callable[[Edge], tuple[float, float]],
    loss: str = "linear",
    f_scale: float = 3.0,
) -> dict[int, tuple[float, float, float]]:
    """2D pose-graph optimization (x, y, heading) by least squares.

    Plain least squares ("linear") spreads a genuine loop correction over
    the whole loop. Robust losses (cauchy, huber) treat a large real
    correction after long drift as an outlier, so false loops are instead
    rejected by `VisualSLAM` with a consistency check after optimizing."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    ids = [k for k in poses if k != fixed]
    if not ids or not edges:
        return dict(poses)
    index = {k: n for n, k in enumerate(ids)}
    x0 = np.array(
        [v for k in ids for v in (poses[k][0], poses[k][1], math.radians(poses[k][2]))]
    )
    fixed_pose = poses[fixed]
    used = [e for e in edges if e.i in poses and e.j in poses]
    weights = np.array([[1.0 / s_t, 1.0 / s_t, 1.0 / math.radians(s_h)] for s_t, s_h in map(sigmas, used)])
    meas = np.array([[e.meas[0], e.meas[1], math.radians(e.meas[2])] for e in used])

    def node(params: NDArray[np.float64], k: int) -> tuple[float, float, float]:
        if k == fixed:
            return fixed_pose[0], fixed_pose[1], math.radians(fixed_pose[2])
        n = index[k] * 3
        return params[n], params[n + 1], params[n + 2]

    def residuals(params: NDArray[np.float64]) -> NDArray[np.float64]:
        out = np.empty((len(used), 3))
        for row, e in enumerate(used):
            xi, yi, hi = node(params, e.i)
            xj, yj, hj = node(params, e.j)
            dx, dy = xj - xi, yj - yi
            r = dx * math.cos(hi) - dy * math.sin(hi)
            f = dx * math.sin(hi) + dy * math.cos(hi)
            dh = math.atan2(math.sin(hj - hi - meas[row, 2]), math.cos(hj - hi - meas[row, 2]))
            out[row] = ((r - meas[row, 0]), (f - meas[row, 1]), dh)
        return (out * weights).ravel()

    sparsity = lil_matrix((len(used) * 3, len(ids) * 3), dtype=int)
    for row, e in enumerate(used):
        for k in (e.i, e.j):
            if k != fixed:
                sparsity[row * 3 : row * 3 + 3, index[k] * 3 : index[k] * 3 + 3] = 1

    result = least_squares(  # noqa: E501
        residuals, x0, jac_sparsity=sparsity, loss=loss, f_scale=f_scale, max_nfev=200
    )
    out = {fixed: fixed_pose}
    for k in ids:
        n = index[k] * 3
        out[k] = (float(result.x[n]), float(result.x[n + 1]), wrap_deg(math.degrees(result.x[n + 2])))
    return out


# ---------------------------------------------------------------------------
# Inertial helper (PhonePose) interface
# ---------------------------------------------------------------------------


class InertialHelper(Protocol):
    step_count: int
    stride_m: float

    def get_pose(self, t: float) -> Pose: ...


# ---------------------------------------------------------------------------
# SLAM
# ---------------------------------------------------------------------------


@dataclass
class SlamEvent:
    timestamp: float
    kind: str  # "keyframe", "loop", "reloc", "lost", "tracking"
    detail: str


class VisualSLAM:
    """Keyframe visual SLAM. Call `process()` with each (or every Nth)
    frame; `get_pose()` returns the latest pose immediately."""

    def __init__(
        self,
        cfg: SlamConfig,
        frame_size: tuple[int, int],
        helper: InertialHelper | None = None,
        map_path: str | Path | None = None,
        new_map: bool = False,
        log: Callable[[str], None] = print,
    ) -> None:
        self.cfg = cfg
        self.frame_w, self.frame_h = frame_size
        self.scale = cfg.image_width / float(self.frame_w)
        self.work_size = (cfg.image_width, int(round(self.frame_h * self.scale)))
        self.K = camera_matrix(cfg, self.frame_w, self.frame_h)
        self.extractor = FeatureExtractor(cfg)
        self.estimator = MotionEstimator(cfg, self.K)
        self.helper = helper
        self.map_path = Path(map_path) if map_path is not None else None
        self._log = log

        self.map_id = uuid.uuid4().hex[:8]
        self.keyframes: dict[int, Keyframe] = {}
        self.edges: list[Edge] = []
        self.next_kf_id = 0
        self.first_kf_id: int | None = None

        self.state = INIT
        self.ref: Keyframe | None = None
        self.ref_rel = (0.0, 0.0, 0.0)  # current pose relative to ref
        self.pose_xyh = (0.0, 0.0, 0.0)
        self.timestamp = 0.0
        self.valid = False
        self.lost_count = 0
        self.frames_processed = 0
        self.dist_since_correction = 0.0
        self.loop_closures = 0
        self.relocalizations = 0
        self.map_version = 0  # bumped after every pose-graph optimization
        self.kf_since_save = 0
        self.events: list[SlamEvent] = []
        self.last_inliers = 0
        self.loaded_keyframes = 0
        self._clouds: dict[int, Any] = {}

        # phone frame -> map frame alignment (heading offset + anchor points)
        self._align: tuple[float, tuple[float, float], tuple[float, float, float]] | None = None
        self._last_odometer = 0.0
        self._last_time: float | None = None

        if self.map_path is not None and self.map_path.exists() and not new_map:
            self.load_map(self.map_path)
            if self.keyframes:
                self.state = RELOCALIZING
                log(
                    f"SLAM | loaded map {self.map_path} ({len(self.keyframes)} keyframes); "
                    "relocalizing - start where an earlier walk passed"
                )

    # -- helper (phone) access -----------------------------------------

    def _helper_pose(self, t: float) -> Pose | None:
        if self.helper is None:
            return None
        pose = self.helper.get_pose(t)
        return pose if pose.valid else None

    def _odometer(self, t: float, moving: bool) -> float:
        """Metres walked: phone steps, or assumed speed while moving."""
        if self.helper is not None:
            return self.helper.step_count * self.helper.stride_m
        if self._last_time is not None and moving:
            self._last_odometer += self.cfg.assumed_speed_m_s * max(t - self._last_time, 0.0)
        return self._last_odometer

    def _set_alignment(self, t: float) -> None:
        hp = self._helper_pose(t)
        if hp is None:
            return
        offset = wrap_deg(self.pose_xyh[2] - hp.heading_deg)
        self._align = (offset, (hp.x, hp.y), self.pose_xyh)

    def _phone_to_map(self, hp: Pose) -> tuple[float, float, float] | None:
        if self._align is None:
            return None
        offset, (px, py), (mx, my, _mh) = self._align
        o = math.radians(offset)
        dx, dy = hp.x - px, hp.y - py
        return (
            mx + dx * math.cos(o) + dy * math.sin(o),
            my - dx * math.sin(o) + dy * math.cos(o),
            wrap_deg(hp.heading_deg + offset),
        )

    def _fuse_heading(self, heading: float, t: float) -> float:
        hp = self._helper_pose(t)
        if hp is None or self._align is None:
            return heading
        phone_h = wrap_deg(hp.heading_deg + self._align[0])
        return wrap_deg(heading + self.cfg.heading_phone_weight * wrap_deg(phone_h - heading))

    # -- public API -----------------------------------------------------

    def get_pose(self, t: float) -> Pose:
        x, y, h = self.pose_xyh
        return Pose(x=x, y=y, heading_deg=h, timestamp=t, valid=self.valid)

    def process(
        self,
        image: NDArray[np.uint8],
        timestamp: float,
        mask_boxes: Iterable[Sequence[float]] = (),
    ) -> Pose:
        """Track one frame. `mask_boxes` are (x1, y1, x2, y2) in ORIGINAL
        frame pixels (dynamic YOLO boxes) whose features are ignored."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if gray.shape[1] != self.work_size[0]:
            gray = cv2.resize(gray, self.work_size, interpolation=cv2.INTER_AREA)
        boxes = [tuple(v * self.scale for v in box) for box in mask_boxes]
        feats = self.extractor.extract(gray, boxes)
        self.timestamp = timestamp
        self.frames_processed += 1

        if self.state == INIT:
            self._start_map(feats, timestamp)
        elif self.state == RELOCALIZING:
            relocalized = False
            if self.frames_processed % max(1, self.cfg.reloc_every_n) == 0:
                relocalized = self._try_relocalize(feats, timestamp, link_from=None)
            if not relocalized:
                self.valid = False
        elif self.state == LOST:
            self._process_lost(feats, timestamp)
        else:
            self._track(feats, timestamp)

        self._last_time = timestamp
        return self.get_pose(timestamp)

    # -- states -----------------------------------------------------------

    def _start_map(self, feats: FrameFeatures, t: float) -> None:
        if len(feats.keypoints) < self.cfg.min_inliers:
            self.valid = False
            return
        self.pose_xyh = (0.0, 0.0, 0.0)
        kf = self._add_keyframe(feats, t, self.pose_xyh, link=None)
        self.first_kf_id = kf.kf_id
        self.state = TRACKING
        self.valid = True
        self._set_alignment(t)
        self._event(t, "tracking", "new map started at (0, 0), heading 0")

    def _track(self, feats: FrameFeatures, t: float) -> None:
        assert self.ref is not None
        est = self.estimator.estimate(self.ref.features, feats, self.cfg.min_inliers)
        if est is None:
            self.lost_count += 1
            self.last_inliers = 0
            self._dead_reckon(t)
            if self.lost_count >= self.cfg.lost_frames:
                self.state = LOST
                self._event(t, "lost", "visual tracking lost; using phone dead reckoning"
                            if self.helper else "visual tracking lost; pose invalid")
            return
        self.lost_count = 0
        self.last_inliers = est.inliers

        moving = not est.rotation_only
        odo = self._odometer(t, moving)
        distance = max(odo - self.ref.odometer_m, 0.0)
        dr, df = self._direction(est)
        rel = (distance * dr, distance * df, est.dyaw_deg)
        raw = compose(self.ref.pose, *rel)
        heading = self._fuse_heading(raw[2], t)
        self.pose_xyh = (raw[0], raw[1], heading)
        self.ref_rel = relative(self.ref.pose, self.pose_xyh)
        self.valid = True

        if (
            est.inliers < self.cfg.kf_min_inliers
            or est.trans_parallax_px > self.cfg.kf_parallax_px
            or abs(est.dyaw_deg) > self.cfg.kf_yaw_deg
            or distance > self.cfg.kf_max_distance_m
        ):
            prev = self.ref
            kf = self._add_keyframe(feats, t, self.pose_xyh, link=prev)
            self.dist_since_correction += math.hypot(*relative(prev.pose, kf.pose)[:2])
            self._detect_loop(kf)
        self._set_alignment(t)

    def _direction(self, est: MotionEstimate) -> tuple[float, float]:
        """Walking direction in the reference frame. Falls back to the
        current camera forward when the visual direction is unreliable or
        implausibly sideways (people walk forward)."""
        fwd = (math.sin(math.radians(est.dyaw_deg)), math.cos(math.radians(est.dyaw_deg)))
        if (
            est.rotation_only
            or est.trans_parallax_px < self.cfg.dir_min_parallax_px
            or (est.dir_right == 0.0 and est.dir_forward == 0.0)
        ):
            return fwd
        angle = math.degrees(math.atan2(est.dir_right, est.dir_forward)) - est.dyaw_deg
        if abs(wrap_deg(angle)) > self.cfg.max_lateral_deg:
            return fwd
        return est.dir_right, est.dir_forward

    def _dead_reckon(self, t: float) -> None:
        hp = self._helper_pose(t)
        mapped = self._phone_to_map(hp) if hp is not None else None
        if mapped is None:
            self.valid = False
            return
        self.pose_xyh = mapped
        self.valid = True

    def _process_lost(self, feats: FrameFeatures, t: float) -> None:
        self._dead_reckon(t)
        assert self.ref is not None
        est = self.estimator.estimate(self.ref.features, feats, self.cfg.min_inliers)
        if est is not None:
            # Re-acquired the current keyframe: continue from it.
            self.state = TRACKING
            self.lost_count = 0
            self._event(t, "tracking", f"re-acquired keyframe {self.ref.kf_id}")
            self._track(feats, t)
            return
        if self.frames_processed % max(1, self.cfg.reloc_every_n) == 0:
            self._try_relocalize(feats, t, link_from=self.ref if self.valid else None)

    def _try_relocalize(self, feats: FrameFeatures, t: float, link_from: Keyframe | None) -> bool:
        candidates = sorted(
            self.keyframes.values(),
            key=lambda kf: thumb_ncc(kf.features.thumb, feats.thumb),
            reverse=True,
        )[: self.cfg.reloc_candidates]
        best: tuple[Keyframe, MotionEstimate] | None = None
        for kf in candidates:
            est = self.estimator.estimate(kf.features, feats, self.cfg.reloc_min_inliers)
            if est is not None and (best is None or est.inliers > best[1].inliers):
                best = (kf, est)
        if best is None:
            return False
        kf, est = best
        rel, sigma = self._loop_measurement(kf, feats, est, self.cfg.reloc_min_inliers)
        self.pose_xyh = compose(kf.pose, *rel)
        new = self._add_keyframe(feats, t, self.pose_xyh, link=None)
        self.edges.append(Edge(kf.kf_id, new.kf_id, rel, EDGE_RELOC, sigma))
        if link_from is not None and link_from.kf_id != kf.kf_id and self.helper is not None:
            # Weak dead-reckoning edge keeps the trajectory continuous.
            self.edges.append(Edge(link_from.kf_id, new.kf_id, relative(link_from.pose, new.pose), EDGE_LOOP))
        self.state = TRACKING
        self.valid = True
        self.lost_count = 0
        self.relocalizations += 1
        self._set_alignment(t)
        self._event(t, "reloc", f"relocalized to keyframe {kf.kf_id} ({est.inliers} inliers)")
        if link_from is not None:
            self._optimize()
        return True

    # -- map building -------------------------------------------------------

    def _add_keyframe(
        self,
        feats: FrameFeatures,
        t: float,
        pose: tuple[float, float, float],
        link: Keyframe | None,
    ) -> Keyframe:
        # Store the strongest features only (map size); bucketing already spread them.
        keep = min(len(feats.keypoints), self.cfg.map_features)
        stored = FrameFeatures(feats.keypoints[:keep].copy(), feats.descriptors[:keep].copy(), feats.thumb)
        moving_odo = self._odometer(t, False) if self.helper is not None else self._last_odometer
        kf = Keyframe(self.next_kf_id, t, pose, stored, moving_odo)
        self.next_kf_id += 1
        self.keyframes[kf.kf_id] = kf
        if link is not None:
            self.edges.append(Edge(link.kf_id, kf.kf_id, relative(link.pose, kf.pose), EDGE_ODOM))
        self.ref = kf
        self.ref_rel = (0.0, 0.0, 0.0)
        self.kf_since_save += 1
        if self.map_path is not None and self.kf_since_save >= self.cfg.save_every_kf:
            self.save_map(self.map_path)
        return kf

    def _detect_loop(self, kf: Keyframe) -> None:
        radius = self.cfg.loop_radius_m + self.cfg.loop_radius_per_m * self.dist_since_correction
        recent = {k for k in self.keyframes if k > kf.kf_id - self.cfg.loop_min_gap}
        candidates = []
        for other in self.keyframes.values():
            if other.kf_id in recent:
                continue
            if math.hypot(other.pose[0] - kf.pose[0], other.pose[1] - kf.pose[1]) > radius:
                continue
            ncc = thumb_ncc(other.features.thumb, kf.features.thumb)
            if ncc >= self.cfg.loop_min_ncc:
                candidates.append((ncc, other))
        candidates.sort(key=lambda c: c[0], reverse=True)
        for _ncc, other in candidates[: self.cfg.loop_candidates]:
            est = self.estimator.estimate(other.features, kf.features, self.cfg.loop_min_inliers)
            if est is None:
                continue
            rel, sigma = self._loop_measurement(other, kf.features, est, self.cfg.loop_min_inliers)
            # Loops need a metric measurement: a rotation-only loop edge
            # would wrongly pull two keyframes metres apart together.
            if sigma != self.cfg.loop_sigma_m or math.hypot(rel[0], rel[1]) > radius:
                continue
            edge = Edge(other.kf_id, kf.kf_id, rel, EDGE_LOOP, sigma)
            before = kf.pose
            if not self._optimize(new_edge=edge):
                self._event(kf.timestamp, "loop_rejected",
                            f"loop kf {kf.kf_id} -> kf {other.kf_id} rejected (inconsistent with odometry)")
                continue
            self.loop_closures += 1
            moved = math.hypot(kf.pose[0] - before[0], kf.pose[1] - before[1])
            self._event(
                kf.timestamp,
                "loop",
                f"loop closure kf {kf.kf_id} -> kf {other.kf_id} ({est.inliers} inliers, "
                f"{'metric' if sigma == self.cfg.loop_sigma_m else 'rotation-only'}), "
                f"map corrected by {moved:.2f} m",
            )
            self.dist_since_correction = 0.0
            return

    def _cloud(self, kf: Keyframe) -> Any:
        """Triangulated 3D points of `kf`, using its best odometry neighbour."""
        if kf.kf_id in self._clouds:
            return self._clouds[kf.kf_id]
        best = None
        for e in self.edges:
            if e.kind != EDGE_ODOM or kf.kf_id not in (e.i, e.j):
                continue
            other = self.keyframes.get(e.j if e.i == kf.kf_id else e.i)
            if other is None:
                continue
            baseline = math.hypot(other.pose[0] - kf.pose[0], other.pose[1] - kf.pose[1])
            if 0.25 <= baseline <= 3.0 and (best is None or baseline > best[0]):
                best = (baseline, other)
        cloud = None
        if best is not None:
            cloud = self.estimator.triangulate(kf.features, best[1].features, best[0])
        self._clouds[kf.kf_id] = cloud
        return cloud

    def _loop_measurement(
        self, kf: Keyframe, feats: FrameFeatures, est: MotionEstimate, min_inliers: int
    ) -> tuple[tuple[float, float, float], float]:
        """Relative pose of `feats` in kf's frame for a loop/reloc edge.

        Metric translation via PnP on kf's triangulated points when
        possible; otherwise rotation only, with a loose translation sigma.
        """
        cloud = self._cloud(kf)
        if cloud is not None:
            result = self.estimator.metric_relative(kf.features, cloud, feats, min_inliers)
            if result is not None:
                rel, _n = result
                return rel, self.cfg.loop_sigma_m
        return (0.0, 0.0, est.dyaw_deg), self.cfg.loop_sigma_unscaled_m

    def _sigmas(self, edge: Edge) -> tuple[float, float]:
        if edge.sigma_m > 0:
            return edge.sigma_m, self.cfg.loop_sigma_deg
        if edge.kind == EDGE_ODOM:
            dist = math.hypot(edge.meas[0], edge.meas[1])
            return self.cfg.odom_sigma_m + self.cfg.odom_sigma_per_m * dist, self.cfg.odom_sigma_deg
        return self.cfg.loop_sigma_m, self.cfg.loop_sigma_deg

    def _optimize(self, new_edge: Edge | None = None) -> bool:
        """Optimize the pose graph. With `new_edge` (a candidate loop), the
        edge is kept only if afterwards no odometry edge is stretched by
        more than `loop_max_odom_sigma` (a false loop would tear the
        odometry apart); otherwise it is dropped and nothing changes."""
        if self.first_kf_id is None:
            return False
        edges = self.edges + ([new_edge] if new_edge is not None else [])
        poses = {k: kf.pose for k, kf in self.keyframes.items()}
        new = optimize_pose_graph(
            poses, edges, self.first_kf_id, self._sigmas, self.cfg.graph_loss, self.cfg.graph_f_scale
        )
        if new_edge is not None:
            worst = max(
                (edge_residual(new, e, *self._sigmas(e)) for e in self.edges if e.kind == EDGE_ODOM),
                default=0.0,
            )
            if worst > self.cfg.loop_max_odom_sigma:
                return False
            self.edges.append(new_edge)
        for k, p in new.items():
            self.keyframes[k].pose = p
        if self.ref is not None:
            self.pose_xyh = compose(self.ref.pose, *self.ref_rel)
        self.map_version += 1
        self._clouds.clear()
        self._set_alignment(self.timestamp)
        return True

    # -- hazard anchors -------------------------------------------------------

    def anchor_id(self, kf_id: int) -> str:
        return f"{self.map_id}:{kf_id}"

    def current_anchor(self) -> tuple[str, tuple[float, float, float]] | None:
        """Anchor id + pose of the keyframe the camera is tracking now."""
        if self.ref is None or not self.valid or self.state != TRACKING:
            return None
        return self.anchor_id(self.ref.kf_id), self.ref.pose

    def anchor_poses(self) -> dict[str, tuple[float, float, float]]:
        return {self.anchor_id(k): kf.pose for k, kf in self.keyframes.items()}

    # -- persistence ------------------------------------------------------------

    def save_map(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        kfs = sorted(self.keyframes.values(), key=lambda kf: kf.kf_id)
        counts = np.array([len(kf.features.keypoints) for kf in kfs], dtype=np.int64)
        np.savez_compressed(
            path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz"),
            map_id=np.array(self.map_id),
            image_width=np.array(self.cfg.image_width),
            K=self.K,
            first_kf_id=np.array(-1 if self.first_kf_id is None else self.first_kf_id),
            next_kf_id=np.array(self.next_kf_id),
            kf_ids=np.array([kf.kf_id for kf in kfs], dtype=np.int64),
            kf_times=np.array([kf.timestamp for kf in kfs], dtype=np.float64),
            kf_poses=np.array([kf.pose for kf in kfs], dtype=np.float64).reshape(-1, 3),
            kf_thumbs=np.array([kf.features.thumb for kf in kfs], dtype=np.float32).reshape(len(kfs), -1),
            kf_counts=counts,
            kp=np.concatenate([kf.features.keypoints for kf in kfs]) if kfs else np.zeros((0, 2), np.float32),
            des=np.concatenate([kf.features.descriptors for kf in kfs]) if kfs else np.zeros((0, 32), np.uint8),
            edges=np.array([(e.i, e.j, e.kind) for e in self.edges], dtype=np.int64).reshape(-1, 3),
            edge_meas=np.array([e.meas for e in self.edges], dtype=np.float64).reshape(-1, 3),
            edge_sigma=np.array([e.sigma_m for e in self.edges], dtype=np.float64),
        )
        self.kf_since_save = 0

    def load_map(self, path: str | Path) -> None:
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            if int(data["image_width"]) != self.cfg.image_width:
                raise ValueError(
                    f"Map {path} was built at slam.image_width={int(data['image_width'])}, "
                    f"config has {self.cfg.image_width}. Use the same value or --slam-new-map."
                )
            self.map_id = str(data["map_id"])
            first = int(data["first_kf_id"])
            self.first_kf_id = None if first < 0 else first
            self.next_kf_id = int(data["next_kf_id"])
            offsets = np.concatenate([[0], np.cumsum(data["kf_counts"])])
            for n, kf_id in enumerate(data["kf_ids"]):
                a, b = int(offsets[n]), int(offsets[n + 1])
                feats = FrameFeatures(
                    data["kp"][a:b].astype(np.float32),
                    data["des"][a:b].astype(np.uint8),
                    data["kf_thumbs"][n].astype(np.float32),
                )
                pose = tuple(float(v) for v in data["kf_poses"][n])
                self.keyframes[int(kf_id)] = Keyframe(int(kf_id), float(data["kf_times"][n]), pose, feats)  # type: ignore[arg-type]
            sig = data["edge_sigma"] if "edge_sigma" in data else np.zeros(len(data["edges"]))
            self.edges = [
                Edge(int(i), int(j), tuple(float(v) for v in m), int(kind), float(sg))  # type: ignore[arg-type]
                for (i, j, kind), m, sg in zip(data["edges"], data["edge_meas"], sig)
            ]
        self.loaded_keyframes = len(self.keyframes)

    # -- misc ---------------------------------------------------------------------

    def _event(self, t: float, kind: str, detail: str) -> None:
        self.events.append(SlamEvent(t, kind, detail))
        self._log(f"SLAM | {detail}")

    def loop_edges(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        out = []
        for e in self.edges:
            if e.kind != EDGE_ODOM and e.i in self.keyframes and e.j in self.keyframes:
                a, b = self.keyframes[e.i].pose, self.keyframes[e.j].pose
                out.append(((a[0], a[1]), (b[0], b[1])))
        return out


def boxes_from_yolo_result(result: Any, labels: Iterable[str]) -> list[tuple[float, float, float, float]]:
    """(x1, y1, x2, y2) of detections whose class name is in `labels`."""
    wanted = set(labels)
    if result is None or getattr(result, "boxes", None) is None:
        return []
    names = getattr(result, "names", {}) or {}
    boxes = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        if names.get(cls_id, str(cls_id)) in wanted:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2))
    return boxes
