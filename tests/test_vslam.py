"""Visual SLAM tests: conventions, pose graph, map persistence,
relocalization, hazard re-anchoring and an end-to-end synthetic loop.
No camera, phone, GPU or audio device needed (synthetic rendered frames).
"""

import math
import sqlite3

import numpy as np
import pytest

from navigation.config import load_config
from navigation.memory import RouteMemory
from navigation.models import Pose
from navigation.slam_eval import ate_rmse, loop_drift, project_to_plane, umeyama
from navigation.synthetic_scene import FakePhone, Room, render, square_walk
from navigation.vslam import (
    EDGE_LOOP,
    EDGE_ODOM,
    LOST,
    RELOCALIZING,
    TRACKING,
    Edge,
    SlamConfig,
    VisualSLAM,
    boxes_from_yolo_result,
    camera_matrix,
    compose,
    ground_motion,
    optimize_pose_graph,
    relative,
)

SIZE = (640, 360)


@pytest.fixture(scope="module")
def room():
    return Room()


@pytest.fixture(scope="module")
def K():
    return camera_matrix(SlamConfig(), *SIZE)


def ry(theta_deg):
    c, s = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def opencv_motion(yaw_deg, centre):
    """(R, t) as recoverPose returns them for a camera that turned right by
    yaw_deg and moved to `centre` (in the reference camera frame)."""
    Rc = ry(yaw_deg)
    return Rc.T, -Rc.T @ np.asarray(centre, float)


# -- conventions ---------------------------------------------------------


def test_compose_follows_claude_md_heading_convention():
    # heading 90 = facing +x: one metre forward goes to +x
    x, y, h = compose((0.0, 0.0, 90.0), 0.0, 1.0, 0.0)
    assert x == pytest.approx(1.0) and y == pytest.approx(0.0, abs=1e-9)
    # heading 0: forward is +y, right is +x
    x, y, _ = compose((0.0, 0.0, 0.0), 1.0, 2.0, 0.0)
    assert (x, y) == pytest.approx((1.0, 2.0))


def test_relative_inverts_compose():
    base = (2.0, -1.0, 37.0)
    rel = (0.4, 1.3, -12.0)
    assert relative(base, compose(base, *rel)) == pytest.approx(rel)


def test_ground_motion_turn_right_increases_heading():
    dyaw, _r, _f = ground_motion(*opencv_motion(10.0, [0, 0, 0]))
    assert dyaw == pytest.approx(10.0)


def test_ground_motion_forward_and_right_translation():
    dyaw, r, f = ground_motion(*opencv_motion(0.0, [0, 0, 2.0]))
    assert (dyaw, r, f) == pytest.approx((0.0, 0.0, 1.0), abs=1e-9)
    _d, r, f = ground_motion(*opencv_motion(0.0, [1.0, 0, 0]))
    assert (r, f) == pytest.approx((1.0, 0.0), abs=1e-9)


def test_ground_motion_removes_camera_pitch():
    # Camera tilted down 20 deg walking level-forward: in its own frame the
    # motion has a "up" (negative y) component; ground motion is forward.
    pitch = 20.0
    p = math.radians(pitch)
    centre_cam = np.array([0.0, -math.sin(p), math.cos(p)])
    _d, r, f = ground_motion(np.eye(3), -centre_cam, pitch_deg=pitch)
    assert (r, f) == pytest.approx((0.0, 1.0), abs=1e-9)


# -- tracking on rendered frames ---------------------------------------------


def run(slam, room, K, poses, t0=0.0, dt=0.1, phone=None):
    out = []
    for n, (x, y, h) in enumerate(poses):
        t = t0 + n * dt
        if phone is not None:
            phone.update(t, x, y, h)
        out.append(slam.process(render(room, x, y, h, K, SIZE), t))
    return out


def test_walking_forward_increases_y_and_turning_right_increases_heading(room, K):
    slam = VisualSLAM(SlamConfig(), SIZE, log=lambda _m: None)
    forward = [(-4.0, -4.0 + 0.1 * n, 0.0) for n in range(25)]
    turn = [(-4.0, -1.6, 3.0 * n) for n in range(1, 16)]
    poses = run(slam, room, K, forward + turn)
    assert slam.state == TRACKING
    assert poses[24].y > 1.0 and abs(poses[24].x) < 0.5
    assert poses[-1].heading_deg == pytest.approx(45.0, abs=5.0)


def test_scale_comes_from_phone_steps(room, K):
    phone = FakePhone(stride_m=0.7, true_stride_m=0.7, drift_deg_s=0.0, noise_deg=0.0)
    slam = VisualSLAM(SlamConfig(), SIZE, helper=phone, log=lambda _m: None)
    poses = run(slam, room, K, [(-4.0, -4.0 + 0.1 * n, 0.0) for n in range(36)], phone=phone)
    # walked 3.5 m = 5 steps of 0.7 m exactly
    assert phone.step_count == 5
    assert poses[-1].y == pytest.approx(3.5, abs=0.15)


def test_dynamic_boxes_mask_features(room, K):
    slam = VisualSLAM(SlamConfig(), SIZE, log=lambda _m: None)
    image = render(room, -4.0, -4.0, 0.0, K, SIZE)
    full = slam.extractor.extract(image)
    masked = slam.extractor.extract(image, [(0, 0, 320, 360)])
    assert len(masked.keypoints) > 0
    assert np.all(masked.keypoints[:, 0] > 315)
    assert len(full.keypoints) > len(masked.keypoints)


def test_lost_tracking_falls_back_to_phone_or_goes_invalid(room, K):
    blank = np.full((SIZE[1], SIZE[0]), 90, np.uint8)
    cfg = SlamConfig(lost_frames=2)
    for helper in (None, FakePhone()):
        slam = VisualSLAM(cfg, SIZE, helper=helper, log=lambda _m: None)
        slam.process(render(room, -4.0, -4.0, 0.0, K, SIZE), 0.0)
        for n in range(4):
            pose = slam.process(blank, 0.1 * (n + 1))
        assert slam.state == LOST
        assert pose.valid is (helper is not None)


# -- pose graph -----------------------------------------------------------------


def test_pose_graph_closes_a_drifted_square_loop():
    true = [(0, 0, 0), (0, 4, 0), (0, 4, 90), (4, 4, 90), (4, 4, 180), (4, 0, 180), (4, 0, 270), (0, 0, 270)]
    true = [(x, y, ((h + 180) % 360) - 180) for x, y, h in true]
    edges = []
    drifted = {0: true[0]}
    for k in range(1, len(true)):
        r, f, dh = relative(true[k - 1], true[k])
        meas = (r, f * 0.95, dh + 2.0)  # 5 % scale error, 2 deg yaw error per step
        edges.append(Edge(k - 1, k, meas, EDGE_ODOM))
        drifted[k] = compose(drifted[k - 1], *meas)
    loop = Edge(0, len(true) - 1, relative(true[0], true[-1]), EDGE_LOOP)
    before = math.dist(drifted[7][:2], true[7][:2])
    out = optimize_pose_graph(
        drifted, edges + [loop], 0, lambda e: (0.2, 2.0) if e.kind == EDGE_ODOM else (0.1, 1.0)
    )
    after = math.dist(out[7][:2], true[7][:2])
    assert before > 0.5 and after < before / 4


def test_inconsistent_loop_is_rejected(room, K):
    slam = VisualSLAM(SlamConfig(), SIZE, log=lambda _m: None)
    run(slam, room, K, [(-4.0, -4.0 + 0.1 * n, 0.0) for n in range(40)])
    last = max(slam.keyframes)
    assert last >= 2
    poses_before = {k: kf.pose for k, kf in slam.keyframes.items()}
    # Claim the last keyframe is 10 m to the side of the first: impossible.
    bad = Edge(0, last, (10.0, 0.0, 0.0), EDGE_LOOP, 0.2)
    assert slam._optimize(new_edge=bad) is False
    assert bad not in slam.edges
    assert {k: kf.pose for k, kf in slam.keyframes.items()} == poses_before


# -- persistence + relocalization ---------------------------------------------------


def test_map_round_trip_and_relocalization(room, K, tmp_path):
    path = tmp_path / "map.npz"
    first = VisualSLAM(SlamConfig(), SIZE, map_path=path, log=lambda _m: None)
    run(first, room, K, [(-4.0, -4.0 + 0.1 * n, 0.0) for n in range(30)])
    first.save_map(path)

    second = VisualSLAM(SlamConfig(reloc_every_n=1), SIZE, map_path=path, log=lambda _m: None)
    assert second.state == RELOCALIZING
    assert len(second.keyframes) == len(first.keyframes)
    assert second.map_id == first.map_id
    pose = second.process(render(room, -4.0, -4.0 + 1.0, 0.0, K, SIZE), 100.0)
    assert second.state == TRACKING and pose.valid
    # Same world frame as the first run: about 1 m forward of its start.
    assert pose.y == pytest.approx(1.0, abs=0.4)
    assert abs(pose.x) < 0.4
    assert second.relocalizations == 1


def test_relocalizing_pose_is_invalid_until_recognized(room, K, tmp_path):
    path = tmp_path / "map.npz"
    first = VisualSLAM(SlamConfig(), SIZE, map_path=path, log=lambda _m: None)
    run(first, room, K, [(-4.0, -4.0 + 0.1 * n, 0.0) for n in range(10)])
    first.save_map(path)
    second = VisualSLAM(SlamConfig(reloc_every_n=1), SIZE, map_path=path, log=lambda _m: None)
    blank = np.full((SIZE[1], SIZE[0]), 90, np.uint8)
    assert second.process(blank, 0.0).valid is False
    assert second.state == RELOCALIZING


def test_new_map_flag_ignores_saved_map(room, K, tmp_path):
    path = tmp_path / "map.npz"
    first = VisualSLAM(SlamConfig(), SIZE, map_path=path, log=lambda _m: None)
    run(first, room, K, [(-4.0, -4.0, 0.0)])
    first.save_map(path)
    fresh = VisualSLAM(SlamConfig(), SIZE, map_path=path, new_map=True, log=lambda _m: None)
    assert fresh.state != RELOCALIZING and not fresh.keyframes


# -- end to end ----------------------------------------------------------------------


def test_square_loop_closure_removes_drift(room, K):
    phone = FakePhone()  # 7 % stride error + heading drift
    slam = VisualSLAM(SlamConfig(), SIZE, helper=phone, log=lambda _m: None)
    walk = square_walk()
    x0, y0 = walk[0][1], walk[0][2]
    scale = phone.stride_m / phone.true_stride_m  # the map's metric scale
    for t, x, y, h in walk:
        phone.update(t, x, y, h)
        pose = slam.process(render(room, x, y, h, K, SIZE), t)
    assert slam.loop_closures >= 1
    gt_end = ((walk[-1][1] - x0) * scale, (walk[-1][2] - y0) * scale)
    assert math.dist((pose.x, pose.y), gt_end) < 0.3


# -- hazard memory anchoring --------------------------------------------------------


def test_anchored_hazard_moves_with_its_keyframe(tmp_path):
    memory = RouteMemory(str(tmp_path / "m.db"), now_fn=lambda: 10.0)
    kf_pose = (1.0, 2.0, 90.0)
    right, forward = relative(kf_pose, (3.0, 2.0, 0.0))[:2]  # 2 m ahead of the keyframe
    memory.remember("pole-1", "pole", 3.0, 2.0, 0.9, timestamp=10.0, anchor=("m:5", right, forward))
    memory.remember("pole-2", "pole", 9.0, 9.0, 0.9, timestamp=10.0)  # unanchored
    moved = memory.reanchor({"m:5": (1.0, 3.0, 0.0)})  # keyframe moved 1 m and turned
    assert moved == 1
    by_id = {h.object_id: h for h in memory.get_hazards()}
    assert (by_id["pole-1"].world_x, by_id["pole-1"].world_y) == pytest.approx((1.0, 5.0))
    assert (by_id["pole-2"].world_x, by_id["pole-2"].world_y) == (9.0, 9.0)


def test_old_database_without_anchor_columns_still_loads(tmp_path):
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TABLE hazards (hazard_id TEXT PRIMARY KEY, object_id TEXT NOT NULL,
            label TEXT NOT NULL, world_x REAL NOT NULL, world_y REAL NOT NULL,
            confidence REAL NOT NULL, timestamp REAL NOT NULL,
            decay_weight REAL NOT NULL DEFAULT 1.0, velocity_x REAL NOT NULL DEFAULT 0.0,
            velocity_y REAL NOT NULL DEFAULT 0.0)"""
        )
        conn.execute("INSERT INTO hazards VALUES ('h1','pole-1','pole',1,2,0.9,5,1,0,0)")
    memory = RouteMemory(str(db))
    assert [h.hazard_id for h in memory.get_hazards()] == ["h1"]
    assert memory.reanchor({"x:1": (0.0, 0.0, 0.0)}) == 0
    memory.remember("pole-9", "pole", 5.0, 5.0, 0.9, timestamp=6.0, anchor=("x:1", 0.0, 1.0))
    assert memory.reanchor({"x:1": (0.0, 0.0, 0.0)}) == 1


# -- helpers ---------------------------------------------------------------------------


class _T:
    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v

    def tolist(self):
        return self.v


class _Box:
    def __init__(self, xyxy, cls):
        self.xyxy = [_T(list(xyxy))]
        self.cls = _T(cls)


class _Result:
    names = {0: "person", 1: "pole"}

    def __init__(self, boxes):
        self.boxes = boxes


def test_boxes_from_yolo_result_keeps_only_dynamic_labels():
    result = _Result([_Box((1, 2, 3, 4), 0), _Box((5, 6, 7, 8), 1)])
    assert boxes_from_yolo_result(result, {"person"}) == [(1.0, 2.0, 3.0, 4.0)]
    assert boxes_from_yolo_result(None, {"person"}) == []


def test_umeyama_recovers_similarity_and_ate_is_zero():
    rng = np.random.default_rng(0)
    gt = rng.normal(size=(50, 2))
    th = math.radians(30)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    est = (0.5 * (R @ gt.T)).T + [3.0, -1.0]
    err, _aligned = ate_rmse(est, gt)
    assert err == pytest.approx(0.0, abs=1e-9)
    _R, _t, s = umeyama(est, gt)
    assert s == pytest.approx(2.0)


def test_project_to_plane_and_loop_drift():
    square = np.array([[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0], [0.3, 0.4, 0]], float)
    tilted = square @ np.array([[1, 0, 0], [0, 0.8, 0.6], [0, -0.6, 0.8]])
    flat = project_to_plane(tilted)
    assert flat.shape == (5, 2)
    drift = loop_drift(square[:, :2])
    assert drift["end_error_m"] == pytest.approx(0.5)
    assert drift["drift_percent"] == pytest.approx(100 * 0.5 / drift["distance_m"])


def test_slam_config_section_matches_dataclass():
    import dataclasses

    section = load_config()["slam"]
    names = {f.name for f in dataclasses.fields(SlamConfig)}
    assert set(section) - names == {"map_path", "process_every_n"}
    assert names - set(section) == set()
    assert SlamConfig.from_config(section).image_width == 640
