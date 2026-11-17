import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .ik import ARM_DOF, IK, down_quat
from .perception import OverheadCamera, locate_cube
from .scene import GRIPPER_CLOSED, GRIPPER_OPEN, HOME_Q, TrialConfig, build, reset

READY_Q = HOME_Q.copy()
READY_Q[0] = -np.pi / 2

HOVER = 0.12
LIFT = 0.15


@dataclass
class Timing:
    approach: float = 1.6
    descend: float = 0.8
    close: float = 0.5
    lift: float = 0.8
    transport: float = 1.6
    place: float = 0.8
    release: float = 0.3
    retreat: float = 0.5
    settle: float = 0.5

    def scaled(self, f: float) -> "Timing":
        return Timing(**{k: v * f for k, v in self.__dict__.items() if k != "settle"}, settle=self.settle)


@dataclass
class Result:
    success: bool
    failure: str
    place_error_mm: float
    perception_error_mm: float
    yaw_error_deg: float
    cycle_time_s: float
    wall_time_s: float
    phases: dict = field(default_factory=dict)


def min_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def wrap_quarter(a):
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


class PickPlace:
    def __init__(self, cfg: TrialConfig, timing: Timing | None = None, depth_noise: float = 0.0,
                 stock_gripper: bool = False, gravity_ff: bool = True, soft_contact: bool = False,
                 recorder=None):
        self.cfg = cfg
        self.timing = timing or Timing()
        self.depth_noise = depth_noise
        self.model = build(cfg, stock_gripper, soft_contact)
        self.data = mujoco.MjData(self.model)
        self.ik = IK(self.model)
        self.recorder = recorder
        self.q_cmd = READY_Q.copy()
        self.grip_cmd = GRIPPER_OPEN
        self.cube = self.model.body("cube").id
        self.gravity_ff = gravity_ff
        self.kp = self.model.actuator_gainprm[:ARM_DOF, 0].copy()
        reset(self.model, self.data)
        self.data.qpos[:ARM_DOF] = READY_Q
        self.data.ctrl[:ARM_DOF] = READY_Q
        mujoco.mj_forward(self.model, self.data)

    @property
    def cube_pos(self):
        return self.data.xpos[self.cube].copy()

    def finger_gap(self):
        return float(self.data.qpos[7] + self.data.qpos[8])

    def step(self, n=1):
        for _ in range(n):
            ff = self.data.qfrc_bias[:ARM_DOF] / self.kp if self.gravity_ff else 0.0
            self.data.ctrl[:ARM_DOF] = self.q_cmd + ff
            self.data.ctrl[7] = self.grip_cmd
            mujoco.mj_step(self.model, self.data)
            if self.recorder:
                self.recorder(self.model, self.data)

    def steps(self, seconds):
        return max(1, int(round(seconds / self.model.opt.timestep)))

    def move_joints(self, q_goal, duration):
        q_start = self.q_cmd.copy()
        n = self.steps(duration)
        for i in range(1, n + 1):
            self.q_cmd = q_start + min_jerk(i / n) * (q_goal - q_start)
            self.step()

    def move_line(self, p_goal, quat, duration):
        p_start = self.ik.forward(self.q_cmd)
        n = self.steps(duration)
        for i in range(1, n + 1):
            p = p_start + min_jerk(i / n) * (p_goal - p_start)
            self.q_cmd, _ = self.ik.solve(self.q_cmd, p, quat, iters=4)
            self.step()

    def gripper(self, value, duration):
        self.grip_cmd = value
        self.step(self.steps(duration))

    def run(self) -> Result:
        t0 = time.perf_counter()
        cfg, tm = self.cfg, self.timing
        phases = {}
        rng = np.random.default_rng(cfg.seed)

        self.step(self.steps(0.2))
        camera = OverheadCamera(self.model)
        est = locate_cube(camera, self.data, self.depth_noise, rng)
        camera.close()
        if est is None:
            return self._result(False, "not_detected", t0, phases, None)

        yaw = wrap_quarter(est.yaw)
        quat = down_quat(yaw)
        grasp_z = max(est.half, 0.015)
        above_cube = np.array([est.x, est.y, grasp_z + HOVER])
        q_pre, err = self.ik.solve(HOME_Q, above_cube, quat, iters=200)
        if np.linalg.norm(err[:3]) > 0.005:
            return self._result(False, "unreachable", t0, phases, est)

        self.move_joints(q_pre, tm.approach)
        self.move_line(np.array([est.x, est.y, grasp_z]), quat, tm.descend)
        self.gripper(GRIPPER_CLOSED, tm.close)
        phases["gap_after_close_mm"] = round(self.finger_gap() * 1000, 1)
        phases["grip_force_n"] = round(abs(float(self.data.actuator_force[7])), 2)
        if self.finger_gap() < est.half:
            return self._result(False, "grasp_missed", t0, phases, est)

        self.move_line(np.array([est.x, est.y, LIFT]), quat, tm.lift)
        if self.cube_pos[2] < cfg.cube_half + 0.08:
            return self._result(False, "grasp_slipped", t0, phases, est)

        tx, ty = cfg.target_pos
        self.move_line(np.array([tx, ty, LIFT]), quat, tm.transport)
        if self.cube_pos[2] < cfg.cube_half + 0.08:
            return self._result(False, "dropped_in_transport", t0, phases, est)

        self.move_line(np.array([tx, ty, grasp_z + 0.005]), quat, tm.place)
        self.gripper(GRIPPER_OPEN, tm.release)
        self.move_line(np.array([tx, ty, grasp_z + HOVER]), quat, tm.retreat)
        self.step(self.steps(tm.settle))

        z_axis = self.data.xmat[self.cube].reshape(3, 3)[:, 2]
        tilt = np.degrees(np.arccos(np.clip(abs(z_axis[2]), -1, 1)))
        err = np.hypot(*(self.cube_pos[:2] - np.array(cfg.target_pos)))
        if tilt > 10:
            return self._result(False, "tipped_over", t0, phases, est)
        if err > 0.02:
            return self._result(False, "off_target", t0, phases, est)
        return self._result(True, "", t0, phases, est)

    def _result(self, ok, failure, t0, phases, est):
        cfg = self.cfg
        place_err = np.hypot(*(self.cube_pos[:2] - np.array(cfg.target_pos))) * 1000
        if est is not None:
            perc = np.hypot(est.x - cfg.cube_pos[0], est.y - cfg.cube_pos[1]) * 1000
            yaw_err = abs(np.degrees(wrap_quarter(est.yaw - cfg.cube_yaw)))
        else:
            perc = yaw_err = float("nan")
        return Result(ok, failure, round(float(place_err), 2), round(float(perc), 2), round(float(yaw_err), 2),
                      round(float(self.data.time), 3), round(time.perf_counter() - t0, 3), phases)


def run_trial(cfg: TrialConfig, **kwargs) -> Result:
    return PickPlace(cfg, **kwargs).run()
