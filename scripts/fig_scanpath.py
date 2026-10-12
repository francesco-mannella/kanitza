"""Scanpath figure for one trained run: executed fixations over the object, one panel per object and rotation.

Usage:
    python scripts/fig_scanpath.py RUN_DIR OUT.pdf [--exclude-fixation] [--worlds W ...]
"""
import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from exp_common import Run, set_device  # noqa: E402

ROTS = [round(0.2 * k, 1) for k in range(9)]
WORLDS = ["triangle", "square"]


def trace(run, world, rot):
    gaze, scene = [], {}

    def on_step(d):
        gaze.append(np.array(d["run"].env.retina_sim_pos, float))

    def on_decision(d):
        if d["t"] == 0:
            scene["obs"] = np.array(d["obs"]["RETINA"])
            scene["gaze"] = np.array(d["run"].env.retina_sim_pos, float)

    goals = run.test(world, pos=[40.0, 40.0], rot=rot, on_step=on_step, on_decision=on_decision)
    return np.array(gaze), scene, goals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--exclude-fixation", action="store_true")
    ap.add_argument("--worlds", nargs="+", default=WORLDS)
    a = ap.parse_args()
    set_device()
    run = Run(a.run, exclude_fixation=a.exclude_fixation)
    period = run.params.saccade_period
    fig, axs = plt.subplots(len(a.worlds), len(ROTS), figsize=(14, 1.85 * len(a.worlds)))
    for ax, (world, rot) in zip(axs.ravel(), [(w, r) for w in a.worlds for r in ROTS]):
        gaze, scene, goals = trace(run, world, rot)
        img = scene["obs"].astype(float)
        img = img / max(img.max(), 1.0)
        gx, gy = scene["gaze"]
        ax.imshow(img, extent=(gx - 40, gx + 40, gy - 40, gy + 40), origin="upper")
        fix = gaze[::period]
        ax.plot(fix[:, 0], fix[:, 1], "-", color="k", lw=0.8, alpha=0.7)
        for i, (p, q) in enumerate(zip(fix[:-1], fix[1:])):
            if np.linalg.norm(q - p) > 1:
                ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="->", color="k", lw=0.8))
        ax.scatter(fix[:, 0], fix[:, 1], c=np.arange(len(fix)), cmap="viridis", s=16, zorder=3)
        ax.scatter(*fix[0], s=50, facecolor="none", edgecolor="k", zorder=4)
        ax.set_xlim(12, 68)
        ax.set_ylim(12, 68)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{world}, {rot:.1f} rad", fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out)
    fig.savefig(os.path.splitext(a.out)[0] + ".svg")


if __name__ == "__main__":
    main()
