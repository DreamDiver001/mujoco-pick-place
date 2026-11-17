import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
IMG = ROOT / "docs" / "img"

STAGES = [("1_baseline", "Baseline"), ("2_grip_force", "+ grip force"),
          ("3_gravity_ff", "+ gravity\nfeed-forward"), ("4_stiff_contact", "+ stiffer\ncube contact")]
SPEED = [("4_stiff_contact", 1.0), ("speed_0.75", 0.75), ("speed_0.5", 0.5), ("speed_0.35", 0.35),
         ("speed_0.25", 0.25)]

INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e4e4e4"
BLUE, ORANGE = "#2f6db5", "#d9822b"


def load(name):
    folder = REPORTS / name
    if not (folder / "summary.json").exists():
        return None, None
    return json.loads((folder / "summary.json").read_text()), pd.read_csv(folder / "trials.csv")


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def stages_chart():
    rows = [(label, *load(name)) for name, label in STAGES]
    rows = [r for r in rows if r[1] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    labels = [r[0] for r in rows]
    rate = [r[1]["success_rate"] * 100 for r in rows]
    bars = axes[0].bar(labels, rate, color=BLUE, width=0.55)
    axes[0].bar_label(bars, labels=[f"{v:.0f}%" for v in rate], fontsize=9, color=INK, padding=3)
    axes[0].set_ylim(0, 112)
    style(axes[0], "Success rate, 200 randomised trials", "% of trials")

    err = [r[2].loc[r[2].success == 1, "place_error_mm"] for r in rows]
    med = [e.median() for e in err]
    p95 = [e.quantile(0.95) for e in err]
    x = range(len(rows))
    axes[1].bar([i - 0.15 for i in x], med, width=0.3, color=BLUE, label="median")
    axes[1].bar([i + 0.15 for i in x], p95, width=0.3, color=ORANGE, label="95th percentile")
    for i in x:
        axes[1].text(i - 0.15, med[i] + 0.3, f"{med[i]:.1f}", ha="center", fontsize=8, color=INK)
        axes[1].text(i + 0.15, p95[i] + 0.3, f"{p95[i]:.1f}", ha="center", fontsize=8, color=INK)
    axes[1].set_xticks(list(x), labels)
    axes[1].legend(frameon=False, fontsize=8.5, labelcolor=INK)
    style(axes[1], "Placement error of successful trials", "mm from target centre")
    fig.tight_layout()
    fig.savefig(IMG / "stages.png", dpi=150)
    plt.close(fig)


def slip_chart():
    _, df = load("1_baseline")
    if df is None:
        return
    df["hold_ratio"] = df.friction * df.grip_force_n / (df.cube_mass_kg * 9.81)
    bins = [0, 0.75, 1.0, 1.25, 1.5, 2, 100]
    names = ["<0.75", "0.75-1", "1-1.25", "1.25-1.5", "1.5-2", ">2"]
    df["bin"] = pd.cut(df.hold_ratio, bins, labels=names)
    g = df.groupby("bin", observed=False).success.agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bars = ax.bar(names, g["mean"] * 100, color=BLUE, width=0.55)
    ax.bar_label(bars, labels=[f"{m:.0%}\nn={c}" for m, c in zip(g["mean"], g["count"])], fontsize=8,
                 color=INK, padding=3)
    ax.set_ylim(0, 125)
    ax.set_xlabel("friction x grip force / cube weight", color=MUTED, fontsize=9)
    style(ax, "Baseline: success rate by how hard the grip can hold", "% of trials")
    fig.tight_layout()
    fig.savefig(IMG / "baseline_slip.png", dpi=150)
    plt.close(fig)


def speed_chart():
    from pickplace.controller import Timing
    pts = []
    for name, scale in SPEED:
        s, df = load(name)
        if s is None:
            continue
        cycle = 0.2 + sum(Timing().scaled(scale).__dict__.values())
        ok = df[df.success == 1]
        pts.append((cycle, s["success_rate"] * 100, ok.place_error_mm.median() if len(ok) else None))
    if len(pts) < 2:
        return
    pts.sort()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    t = [p[0] for p in pts]
    axes[0].plot(t, [p[1] for p in pts], color=BLUE, linewidth=2, marker="o", markersize=6)
    for p in pts:
        axes[0].annotate(f"{p[1]:.0f}%", (p[0], p[1]), textcoords="offset points", xytext=(0, 8), ha="center",
                         fontsize=8.5, color=INK)
    axes[0].set_ylim(-5, 115)
    style(axes[0], "Success rate vs cycle time", "% of trials")
    axes[0].set_xlabel("cycle time per pick (s, simulated)", color=MUTED, fontsize=9)
    e = [p for p in pts if p[2] is not None]
    axes[1].plot([p[0] for p in e], [p[2] for p in e], color=BLUE, linewidth=2, marker="o", markersize=6)
    for p in e:
        axes[1].annotate(f"{p[2]:.1f}", (p[0], p[2]), textcoords="offset points", xytext=(0, 8), ha="center",
                         fontsize=8.5, color=INK)
    axes[1].set_xlim(axes[0].get_xlim())
    axes[1].set_ylim(0, max(p[2] for p in e) * 1.35)
    style(axes[1], "Median placement error vs cycle time (successful trials)", "mm")
    axes[1].set_xlabel("cycle time per pick (s, simulated)", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(IMG / "speed.png", dpi=150)
    plt.close(fig)


def table():
    print("| Run | Success | Failures | Place error median / p95 (mm) | Cycle time (s) |")
    print("|---|---|---|---|---|")
    names = [n for n, _ in STAGES] + [n for n, _ in SPEED[1:]] + ["noise_2mm", "noise_5mm"]
    for name in names:
        s, df = load(name)
        if s is None:
            continue
        fails = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in s["failures"].items()) or "none"
        lo, hi = s["success_rate_95ci"]
        print(f"| {name} | {s['successes']}/{s['trials']} ({s['success_rate']:.1%}, CI {lo:.1%}-{hi:.1%}) | "
              f"{fails} | {s['place_error_mm']['median']} / {s['place_error_mm']['p95']} | {s['cycle_time_s']} |")


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    stages_chart()
    slip_chart()
    speed_chart()
    table()
