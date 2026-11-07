"""Trajectory evaluation helpers (no external `evo` dependency).

- associate(): match estimated and ground-truth timestamps (TUM style).
- umeyama(): best similarity transform (rotation, translation, scale)
  between two point sets (Umeyama 1991), any dimension.
- ate_rmse(): absolute trajectory error after alignment.
- project_to_plane(): fit the dominant plane of a 3D ground-truth path
  and express it in 2D, so it can be compared with our 2D ground-plane
  SLAM pose.
- loop_drift(): start-to-end error of a closed-loop walk.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def associate(
    est_times: Sequence[float],
    gt_times: Sequence[float],
    max_dt: float = 0.02,
) -> list[tuple[int, int]]:
    """Pairs (i_est, i_gt) of nearest timestamps within `max_dt` seconds."""
    gt = np.asarray(gt_times, dtype=float)
    pairs = []
    for i, t in enumerate(est_times):
        j = int(np.argmin(np.abs(gt - t)))
        if abs(gt[j] - t) <= max_dt:
            pairs.append((i, j))
    return pairs


def umeyama(
    src: NDArray[np.float64], dst: NDArray[np.float64], with_scale: bool = True
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """R, t, s minimizing sum ||dst - (s R src + t)||^2. Rows are points."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    n, dim = src.shape
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_s = (xs**2).sum() / n
    s = float(np.trace(np.diag(D) @ S) / var_s) if with_scale and var_s > 0 else 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def ate_rmse(
    est: NDArray[np.float64], gt: NDArray[np.float64], with_scale: bool = True
) -> tuple[float, NDArray[np.float64]]:
    """RMSE of position error after Umeyama alignment; also the aligned est."""
    R, t, s = umeyama(est, gt, with_scale)
    aligned = (s * (R @ np.asarray(est, float).T)).T + t
    err = np.linalg.norm(aligned - gt, axis=1)
    return float(np.sqrt(np.mean(err**2))), aligned


def project_to_plane(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """2D coordinates of 3D points in their best-fit plane (PCA)."""
    pts = np.asarray(points, dtype=float)
    centred = pts - pts.mean(axis=0)
    _U, _S, Vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ Vt[:2].T


def path_length(xy: NDArray[np.float64]) -> float:
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def loop_drift(xy: NDArray[np.float64]) -> dict[str, float]:
    """For a walk that ends where it started: distance, end error, drift %."""
    xy = np.asarray(xy, dtype=float)
    length = path_length(xy)
    end_error = float(math.hypot(*(xy[-1] - xy[0]))) if len(xy) else 0.0
    return {
        "distance_m": length,
        "end_error_m": end_error,
        "drift_percent": 100.0 * end_error / length if length > 0 else 0.0,
    }
