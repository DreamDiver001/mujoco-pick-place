from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class CubeEstimate:
    x: float
    y: float
    half: float
    yaw: float
    pixels: int


class OverheadCamera:
    def __init__(self, model: mujoco.MjModel, width: int = 640, height: int = 480, camera: str = "overhead"):
        self.model = model
        self.cam = model.camera(camera).id
        self.w, self.h = width, height
        self.renderer = mujoco.Renderer(model, height, width)
        fovy = np.deg2rad(model.cam_fovy[self.cam])
        self.f = 0.5 * height / np.tan(fovy / 2)

    def capture(self, data: mujoco.MjData):
        r = self.renderer
        r.update_scene(data, camera=self.cam)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(data, camera=self.cam)
        depth = r.render().copy()
        r.disable_depth_rendering()
        return rgb, depth

    def deproject(self, data: mujoco.MjData, us, vs, depth):
        x = (us + 0.5 - self.w / 2) * depth / self.f
        y = -(vs + 0.5 - self.h / 2) * depth / self.f
        pts_cam = np.stack([x, y, -depth], axis=1)
        rot = data.cam_xmat[self.cam].reshape(3, 3)
        return pts_cam @ rot.T + data.cam_xpos[self.cam]

    def close(self):
        self.renderer.close()


def red_mask(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[..., i].astype(float) for i in range(3))
    total = r + g + b + 1e-6
    return (r / total > 0.45) & (r - np.maximum(g, b) > 50)


def square_yaw(xy: np.ndarray) -> tuple[float, float]:
    """Angle of the tightest square around the points, in [-pi/4, pi/4), and its side."""
    centred = xy - xy.mean(axis=0)
    best = (np.inf, 0.0, 0.0)
    for a in np.deg2rad(np.arange(-45, 45, 0.5)):
        c, s = np.cos(a), np.sin(a)
        p = centred @ np.array([[c, -s], [s, c]])
        side = max(np.ptp(p[:, 0]), np.ptp(p[:, 1]))
        if side < best[0]:
            best = (side, a, side)
    return best[1], best[2]


def locate_cube(camera: OverheadCamera, data: mujoco.MjData, depth_noise: float = 0.0,
                rng: np.random.Generator | None = None) -> CubeEstimate | None:
    rgb, depth = camera.capture(data)
    if depth_noise and rng is not None:
        depth = depth + rng.normal(0, depth_noise, depth.shape)
    mask = red_mask(rgb)
    vs, us = np.nonzero(mask)
    if len(us) < 20:
        return None
    pts = camera.deproject(data, us, vs, depth[vs, us])
    top_z = np.median(pts[:, 2])
    top = pts[np.abs(pts[:, 2] - top_z) < 0.006]
    if len(top) < 20:
        top = pts
    centre = top[:, :2].mean(axis=0)
    yaw, side = square_yaw(top[:, :2])
    half = float(np.clip(top_z / 2, 0.01, 0.04))
    return CubeEstimate(float(centre[0]), float(centre[1]), half, float(yaw), int(len(top)))
