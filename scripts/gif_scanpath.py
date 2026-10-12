"""Scanpath GIFs for one trained run: one GIF per object, nine rotations side by side, one frame per fixation.

Usage:
    python scripts/gif_scanpath.py RUN_DIR OUT_DIR [--exclude-fixation] [--worlds W ...]
"""
import argparse
import io
import os
import sys

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from exp_common import Run, set_device  # noqa: E402
from fig_scanpath import ROTS, WORLDS, trace  # noqa: E402


def frame(world, traces, k):
    fig, axs = plt.subplots(3, 3, figsize=(7, 7))
    for ax, rot, (fix, img, gaze) in zip(axs.ravel(), ROTS, traces):
        gx, gy = gaze
        ax.imshow(img, extent=(gx - 40, gx + 40, gy - 40, gy + 40), origin="upper")
        p = fix[: k + 1]
        ax.plot(p[:, 0], p[:, 1], "-", color="k", lw=0.8)
        ax.scatter(p[:, 0], p[:, 1], c=np.arange(len(p)), cmap="viridis", vmin=0, vmax=len(fix), s=14, zorder=3)
        ax.scatter(*p[-1], s=60, facecolor="none", edgecolor="k", zorder=4)
        ax.set_xlim(12, 68)
        ax.set_ylim(12, 68)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{rot:.1f} rad", fontsize=8)
    fig.suptitle(f"{world}, fixation {k}", fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out_dir")
    ap.add_argument("--exclude-fixation", action="store_true")
    ap.add_argument("--worlds", nargs="+", default=WORLDS)
    a = ap.parse_args()
    set_device()
    os.makedirs(a.out_dir, exist_ok=True)
    run = Run(a.run, exclude_fixation=a.exclude_fixation)
    period = run.params.saccade_period
    for world in a.worlds:
        traces = []
        for rot in ROTS:
            gaze, scene, _ = trace(run, world, rot)
            img = scene["obs"].astype(float)
            traces.append((gaze[::period], img / max(img.max(), 1.0), scene["gaze"]))
        n = min(len(t[0]) for t in traces)
        traces = [(f[:n], i, g) for f, i, g in traces]
        frames = [frame(world, traces, k) for k in range(n)]
        out = os.path.join(a.out_dir, f"scanpath_{world}.gif")
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=600, loop=0)
        print(out, n, "frames", flush=True)


if __name__ == "__main__":
    main()
