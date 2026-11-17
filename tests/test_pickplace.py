import mujoco
import numpy as np
import pytest

from pickplace.controller import READY_Q, PickPlace, run_trial, wrap_quarter
from pickplace.evaluate import summarise, wilson
from pickplace.ik import IK, down_quat
from pickplace.perception import CubeEstimate, OverheadCamera, locate_cube, square_yaw
from pickplace.scene import HOME_Q, TrialConfig, build, reset, sample_trial


@pytest.fixture(scope="module")
def model():
    return build(TrialConfig())


def test_scene_has_robot_cube_and_cameras(model):
    assert model.nu == 8
    for name in ("cube", "hand", "left_finger"):
        model.body(name)
    model.camera("overhead")
    model.site("tcp")


def test_down_quat_points_gripper_down(model):
    ik = IK(model)
    for yaw in (-0.7, 0.0, 0.5):
        q, err = ik.solve(HOME_Q, np.array([0.5, 0.1, 0.2]), down_quat(yaw), iters=200)
        R = ik.data.site_xmat[ik.site].reshape(3, 3)
        assert np.allclose(R[:, 2], [0, 0, -1], atol=1e-3)
        assert np.allclose(R[:2, 1], [np.cos(yaw), np.sin(yaw)], atol=1e-3)


@pytest.mark.parametrize("target", [(0.4, 0.0, 0.05), (0.6, 0.25, 0.1), (0.45, -0.35, 0.03), (0.35, 0.2, 0.3)])
def test_ik_reaches_workspace_points(model, target):
    q, err = IK(model).solve(HOME_Q, np.array(target), down_quat(0.3), iters=200)
    assert np.linalg.norm(err[:3]) < 1e-3
    assert np.linalg.norm(err[3:]) < 1e-2
    assert np.all(q >= model.jnt_range[:7, 0]) and np.all(q <= model.jnt_range[:7, 1])


def test_square_yaw_recovers_rotation():
    g = np.linspace(-0.02, 0.02, 25)
    pts = np.array(np.meshgrid(g, g)).reshape(2, -1).T
    for a in (-0.6, -0.2, 0.0, 0.35, 0.7):
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        yaw, side = square_yaw(pts @ rot.T)
        assert abs(wrap_quarter(yaw - a)) < np.deg2rad(1)
        assert side == pytest.approx(0.04, abs=0.002)


def test_camera_finds_cube_within_2mm():
    rng = np.random.default_rng(42)
    for i in range(10):
        cfg = sample_trial(rng, i)
        m = build(cfg)
        d = mujoco.MjData(m)
        reset(m, d)
        d.qpos[:7] = READY_Q
        mujoco.mj_forward(m, d)
        cam = OverheadCamera(m)
        est = locate_cube(cam, d)
        cam.close()
        assert est is not None
        assert np.hypot(est.x - cfg.cube_pos[0], est.y - cfg.cube_pos[1]) < 0.002
        assert abs(np.degrees(wrap_quarter(est.yaw - cfg.cube_yaw))) < 3
        assert est.half == pytest.approx(cfg.cube_half, abs=0.001)


def test_camera_reports_nothing_when_cube_is_gone():
    cfg = TrialConfig(cube_pos=(3.0, 3.0))
    m = build(cfg)
    d = mujoco.MjData(m)
    reset(m, d)
    cam = OverheadCamera(m)
    assert locate_cube(cam, d) is None
    cam.close()


def test_pick_and_place_default_trial():
    r = run_trial(TrialConfig())
    assert r.success, r
    assert r.place_error_mm < 5
    assert r.phases["grip_force_n"] > 10


@pytest.mark.parametrize("seed", [1003, 1004, 1017])
def test_pick_and_place_random_trials(seed):
    r = run_trial(sample_trial(np.random.default_rng(seed), seed))
    assert r.success, r
    assert r.place_error_mm < 10


def test_empty_grasp_is_detected(monkeypatch):
    from pickplace import controller
    wrong = CubeEstimate(x=0.45, y=0.25, half=0.025, yaw=0.0, pixels=500)
    monkeypatch.setattr(controller, "locate_cube", lambda *a, **k: wrong)
    r = PickPlace(TrialConfig(cube_pos=(0.5, 0.0))).run()
    assert not r.success
    assert r.failure == "grasp_missed"


def test_unreachable_cube_is_rejected():
    r = run_trial(TrialConfig(cube_pos=(1.05, 0.0)))
    assert r.failure in ("unreachable", "not_detected")
    assert not r.success


def test_wilson_interval():
    lo, hi = wilson(200, 200)
    assert hi == pytest.approx(1.0) and 0.98 < lo < 0.99
    lo, hi = wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi < 0.35


def test_summary_counts_failures():
    rows = [
        {"success": 1, "failure": "", "place_error_mm": 1.0, "perception_error_mm": 0.5, "yaw_error_deg": 0.1,
         "cycle_time_s": 7.6, "wall_time_s": 1.0},
        {"success": 0, "failure": "grasp_slipped", "place_error_mm": 300.0, "perception_error_mm": 0.4,
         "yaw_error_deg": 0.2, "cycle_time_s": 3.9, "wall_time_s": 0.5},
    ]
    s = summarise(rows)
    assert s["success_rate"] == 0.5
    assert s["failures"] == {"grasp_slipped": 1}
    assert s["place_error_mm"]["median"] == 1.0
