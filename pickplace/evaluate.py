import argparse
import csv
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .controller import Timing, run_trial
from .scene import ROOT, sample_trial

FIELDS = ["seed", "success", "failure", "cube_x", "cube_y", "cube_yaw_deg", "cube_size_mm", "cube_mass_kg",
          "friction", "target_x", "target_y", "place_error_mm", "perception_error_mm", "yaw_error_deg",
          "grip_force_n", "cycle_time_s", "wall_time_s"]


def one(args):
    seed, kwargs = args
    kwargs = dict(kwargs)
    kwargs["timing"] = Timing().scaled(kwargs.pop("time_scale", 1.0))
    cfg = sample_trial(np.random.default_rng(seed), seed)
    r = run_trial(cfg, **kwargs)
    return {
        "seed": seed, "success": int(r.success), "failure": r.failure,
        "cube_x": round(cfg.cube_pos[0], 4), "cube_y": round(cfg.cube_pos[1], 4),
        "cube_yaw_deg": round(float(np.degrees(cfg.cube_yaw)), 1), "cube_size_mm": round(cfg.cube_half * 2000, 1),
        "cube_mass_kg": round(cfg.cube_mass, 3), "friction": round(cfg.friction, 3),
        "target_x": round(cfg.target_pos[0], 4), "target_y": round(cfg.target_pos[1], 4),
        "place_error_mm": r.place_error_mm, "perception_error_mm": r.perception_error_mm,
        "yaw_error_deg": r.yaw_error_deg, "grip_force_n": r.phases.get("grip_force_n", ""),
        "cycle_time_s": r.cycle_time_s, "wall_time_s": r.wall_time_s,
    }


def pct(x, q):
    return round(float(np.percentile(x, q)), 2) if len(x) else None


def summarise(rows):
    n = len(rows)
    ok = [r for r in rows if r["success"]]
    lo, hi = wilson(len(ok), n)
    place = [r["place_error_mm"] for r in ok]
    perc = [r["perception_error_mm"] for r in rows if r["perception_error_mm"] == r["perception_error_mm"]]
    yaw = [r["yaw_error_deg"] for r in rows if r["yaw_error_deg"] == r["yaw_error_deg"]]
    return {
        "trials": n,
        "successes": len(ok),
        "success_rate": round(len(ok) / n, 4),
        "success_rate_95ci": [round(lo, 4), round(hi, 4)],
        "failures": dict(Counter(r["failure"] for r in rows if not r["success"]).most_common()),
        "place_error_mm": {"median": pct(place, 50), "p95": pct(place, 95), "max": pct(place, 100)},
        "perception_error_mm": {"median": pct(perc, 50), "p95": pct(perc, 95), "max": pct(perc, 100)},
        "yaw_error_deg": {"median": pct(yaw, 50), "p95": pct(yaw, 95), "max": pct(yaw, 100)},
        "cycle_time_s": pct([r["cycle_time_s"] for r in ok], 50),
        "wall_time_per_trial_s": round(float(np.mean([r["wall_time_s"] for r in rows])), 3),
    }


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def evaluate(trials, seed=1000, workers=None, **kwargs):
    jobs = [(seed + i, kwargs) for i in range(trials)]
    workers = workers or max(1, min(8, os.cpu_count() or 1))
    if workers == 1:
        rows = [one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers) as pool:
            rows = list(pool.map(one, jobs))
    return rows, summarise(rows)


def main():
    ap = argparse.ArgumentParser(description="Run randomised pick-and-place trials and report the results")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--depth-noise", type=float, default=0.0, help="std of depth noise in metres")
    ap.add_argument("--stock-gripper", action="store_true", help="use the unmodified Menagerie gripper actuator")
    ap.add_argument("--no-gravity-ff", action="store_true", help="turn off gravity feed-forward on the arm joints")
    ap.add_argument("--time-scale", type=float, default=1.0, help="scale every motion duration (0.5 = twice as fast)")
    ap.add_argument("--soft-contact", action="store_true", help="MuJoCo's default contact stiffness on the cube")
    ap.add_argument("--name", default="latest")
    args = ap.parse_args()

    kwargs = {"depth_noise": args.depth_noise, "stock_gripper": args.stock_gripper,
              "gravity_ff": not args.no_gravity_ff, "soft_contact": args.soft_contact,
              "time_scale": args.time_scale}
    rows, summary = evaluate(args.trials, args.seed, args.workers, **kwargs)
    summary["settings"] = {"seed": args.seed, **kwargs}

    out = ROOT / "reports" / args.name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    s = summary
    print(f"{s['successes']}/{s['trials']} succeeded ({s['success_rate']:.1%}, "
          f"95% CI {s['success_rate_95ci'][0]:.1%}-{s['success_rate_95ci'][1]:.1%})")
    for k, v in s["failures"].items():
        print(f"  {k:<22} {v}")
    print(f"place error mm       {s['place_error_mm']}")
    print(f"perception error mm  {s['perception_error_mm']}")
    print(f"yaw error deg        {s['yaw_error_deg']}")
    print(f"cycle time {s['cycle_time_s']} s (sim), {s['wall_time_per_trial_s']} s wall per trial")
    print(f"written to {out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
