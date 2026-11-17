import argparse
import time

import mujoco
import numpy as np

from .controller import PickPlace
from .scene import sample_trial


def viewer_recorder(model, data, speed):
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.lookat[:] = [0.45, -0.05, 0.1]
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.6, 140, -25
    start = time.perf_counter()

    def rec(m, d):
        if int(round(d.time / m.opt.timestep)) % 8:
            return
        viewer.sync()
        lag = d.time / speed - (time.perf_counter() - start)
        if lag > 0:
            time.sleep(lag)
    return viewer, rec


def gif_recorder(model, frames, fps=20, width=400, height=300):
    renderer = mujoco.Renderer(model, height, width)
    every = int(round(1 / fps / model.opt.timestep))

    def rec(m, d):
        if int(round(d.time / m.opt.timestep)) % every == 0:
            renderer.update_scene(d, camera="side")
            frames.append(renderer.render().copy())
    return rec


def main():
    ap = argparse.ArgumentParser(description="Watch one pick-and-place trial")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed in the viewer")
    ap.add_argument("--gif", help="save a GIF from the side camera instead of opening the viewer")
    args = ap.parse_args()

    cfg = sample_trial(np.random.default_rng(args.seed), args.seed)
    print(f"cube at ({cfg.cube_pos[0]:.3f}, {cfg.cube_pos[1]:.3f}), {cfg.cube_half * 2000:.0f} mm, "
          f"{cfg.cube_mass * 1000:.0f} g, yaw {np.degrees(cfg.cube_yaw):.0f} deg")
    task = PickPlace(cfg)

    if args.gif:
        from PIL import Image
        frames = []
        task.recorder = gif_recorder(task.model, frames)
        result = task.run()
        sample = Image.fromarray(np.concatenate(frames[:: max(1, len(frames) // 8)], axis=0))
        palette = sample.quantize(160, dither=Image.Dither.NONE)
        images = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
        images[0].save(args.gif, save_all=True, append_images=images[1:], duration=50, loop=0, optimize=True)
        print(f"{len(frames)} frames saved to {args.gif}")
    else:
        viewer, task.recorder = viewer_recorder(task.model, task.data, args.speed)
        result = task.run()
        print(result)
        print("close the viewer window to exit")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
        return
    print(result)


if __name__ == "__main__":
    main()
