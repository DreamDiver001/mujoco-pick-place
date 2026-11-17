import mujoco
import numpy as np

from .scene import HOME_Q

ARM_DOF = 7


def down_quat(yaw: float) -> np.ndarray:
    """Gripper pointing straight down, fingers closing along world (cos yaw, sin yaw)."""
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.column_stack([[-s, c, 0.0], [c, s, 0.0], [0.0, 0.0, -1.0]])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, rot.flatten())
    return q


class IK:
    def __init__(self, model: mujoco.MjModel, site: str = "tcp", damping: float = 1e-3, posture_gain: float = 0.05):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site = model.site(site).id
        self.damping = damping
        self.posture_gain = posture_gain
        self.lo = model.jnt_range[:ARM_DOF, 0]
        self.hi = model.jnt_range[:ARM_DOF, 1]
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def forward(self, q):
        self.data.qpos[:ARM_DOF] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.site].copy()

    def error(self, q, pos, quat):
        d = self.data
        d.qpos[:ARM_DOF] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        site_quat = np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, d.site_xmat[self.site])
        rot_err = np.zeros(3)
        mujoco.mju_subQuat(rot_err, quat, site_quat)
        rot_err = d.site_xmat[self.site].reshape(3, 3) @ rot_err
        return np.concatenate([pos - d.site_xpos[self.site], rot_err])

    def solve(self, q0, pos, quat, iters: int = 50, tol: float = 1e-4):
        q = np.array(q0, dtype=float)
        for _ in range(iters):
            err = self.error(q, pos, quat)
            if np.linalg.norm(err[:3]) < tol and np.linalg.norm(err[3:]) < 10 * tol:
                break
            mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, self.site)
            J = np.vstack([self._jp, self._jr])[:, :ARM_DOF]
            JJt = J @ J.T + self.damping * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            null = np.eye(ARM_DOF) - np.linalg.pinv(J) @ J
            dq += null @ (self.posture_gain * (HOME_Q - q))
            q = np.clip(q + dq, self.lo, self.hi)
        return q, self.error(q, pos, quat)
