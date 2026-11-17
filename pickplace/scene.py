from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PANDA_XML = ROOT / "third_party" / "franka_emika_panda" / "panda.xml"

HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
GRIPPER_OPEN, GRIPPER_CLOSED = 255.0, 0.0

GRIP_KP, GRIP_KV, GRIP_FORCE_LIMIT = 1000.0, 30.0, 40.0
CUBE_SOLREF = 0.005

CAMERA_POS = (0.5, 0.0, 1.3)
CAMERA_FOVY = 60.0


@dataclass
class TrialConfig:
    cube_pos: tuple = (0.5, 0.1)
    cube_yaw: float = 0.0
    cube_half: float = 0.025
    cube_mass: float = 0.1
    friction: float = 1.0
    target_pos: tuple = (0.45, -0.3)
    seed: int = 0
    extra: dict = field(default_factory=dict)


def sample_trial(rng: np.random.Generator, seed: int = 0) -> TrialConfig:
    return TrialConfig(
        cube_pos=(rng.uniform(0.38, 0.65), rng.uniform(-0.05, 0.30)),
        cube_yaw=rng.uniform(-np.pi, np.pi),
        cube_half=rng.uniform(0.018, 0.03),
        cube_mass=rng.uniform(0.05, 0.25),
        friction=rng.uniform(0.6, 1.2),
        target_pos=(rng.uniform(0.35, 0.6), rng.uniform(-0.40, -0.20)),
        seed=seed,
    )


def build(cfg: TrialConfig, stock_gripper: bool = False, soft_contact: bool = False) -> mujoco.MjModel:
    if not PANDA_XML.exists():
        raise FileNotFoundError(f"{PANDA_XML} not found, run: python scripts/fetch_panda.py")
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    for key in list(spec.keys):
        spec.delete(key)

    spec.option.timestep = 0.002
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]

    tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[0.22, 0.26, 0.3], rgb2=[0.18, 0.21, 0.25], width=300, height=300)
    floor_mat = spec.add_material(name="floor", texrepeat=[6, 6], texuniform=True)
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name

    world = spec.worldbody
    world.add_light(pos=[0, 0, 2.5], dir=[0, 0, -1], diffuse=[0.5, 0.5, 0.5])
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.05], material="floor")

    world.add_camera(name="overhead", pos=list(CAMERA_POS), fovy=CAMERA_FOVY, xyaxes=[1, 0, 0, 0, 1, 0])
    world.add_camera(name="side", pos=[1.26, -0.94, 0.85], xyaxes=[0.707, 0.707, 0, -0.336, 0.336, 0.879], fovy=50)

    world.add_geom(name="target", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.0005],
                   pos=[*cfg.target_pos, 0.0005], rgba=[0.2, 0.8, 0.3, 1], contype=0, conaffinity=0)

    h = cfg.cube_half
    cube = world.add_body(name="cube", pos=[*cfg.cube_pos, h],
                          quat=[np.cos(cfg.cube_yaw / 2), 0, 0, np.sin(cfg.cube_yaw / 2)])
    cube.add_freejoint(name="cube")
    cube.add_geom(name="cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[h, h, h], mass=cfg.cube_mass,
                  rgba=[0.85, 0.12, 0.1, 1], friction=[cfg.friction, 0.01, 0.001], condim=4,
                  priority=1, solref=[0.02 if soft_contact else CUBE_SOLREF, 1])

    if not stock_gripper:
        grip = spec.actuator("actuator8")
        grip.gainprm[0] = GRIP_KP * 0.04 / 255
        grip.biasprm[1:3] = [-GRIP_KP, -GRIP_KV]
        grip.forcerange = [-GRIP_FORCE_LIMIT, GRIP_FORCE_LIMIT]

    hand = spec.body("hand")
    hand.add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.005, 0, 0], rgba=[1, 1, 0, 0.5], group=4)

    return spec.compile()


def reset(model: mujoco.MjModel, data: mujoco.MjData):
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = HOME_Q
    data.qpos[7:9] = 0.04
    data.ctrl[:7] = HOME_Q
    data.ctrl[7] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)
