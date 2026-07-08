#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chain_quality import collapse_consecutive, load_goal_sequence
from tests import ROTATIONS, SHAPES, goals_filename


def find_chains(seq):
    """Greedily extract every distinct repeating chain (a block that repeats
    consecutively at least twice) from the collapsed sequence, strongest
    (length * repeats) first. Each extraction consumes its occurrences so
    later searches operate on what's left, yielding one entry per unique
    chain rather than only the single best one."""
    collapsed = collapse_consecutive(seq)
    length = len(collapsed)
    covered = [False] * length
    chains = []

    while True:
        best = None
        for k in range(2, length // 2 + 1):
            for start in range(length - k + 1):
                if any(covered[start:start + k]):
                    continue
                block = collapsed[start:start + k]
                reps = 1
                pos = start + k
                occurrence_starts = [start]
                while (
                    pos + k <= length
                    and not any(covered[pos:pos + k])
                    and collapsed[pos:pos + k] == block
                ):
                    reps += 1
                    occurrence_starts.append(pos)
                    pos += k
                if reps >= 2:
                    score = k * reps
                    if best is None or score > best[0]:
                        best = (score, k, occurrence_starts)
        if best is None:
            break
        _, k, occurrence_starts = best
        chains.append([collapsed[s:s + k] for s in occurrence_starts])
        for s in occurrence_starts:
            for i in range(s, s + k):
                covered[i] = True
    return chains


def chain_mean_std(occurrences):
    arr = np.array(occurrences, dtype=float)
    return arr.mean(axis=0), arr.std(axis=0)


def plot_chain(ax, mean, std, color):
    xs, ys = mean[:, 0], mean[:, 1]
    ax.plot(list(xs) + [xs[0]], list(ys) + [ys[0]], "-o", color=color, linewidth=1.5, markersize=4)
    for (x, y), (sx, sy) in zip(mean, std):
        ax.add_patch(
            Ellipse((x, y), width=max(2 * sx, 0.05), height=max(2 * sy, 0.05), color=color, alpha=0.25, linewidth=0)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("simulation_dir", help="a single simulation folder containing goals-*.npy files")
    args = parser.parse_args()

    sim_dir = Path(args.simulation_dir).resolve()
    if not sim_dir.is_dir():
        raise SystemExit(f"not a directory: {sim_dir}")

    fig, axes = plt.subplots(len(ROTATIONS), len(SHAPES), figsize=(3 * len(SHAPES), 3 * len(ROTATIONS)))
    cmap = plt.get_cmap("tab10")

    for row, rotation in enumerate(ROTATIONS):
        for col, shape in enumerate(SHAPES):
            ax = axes[row][col]
            ax.set_xlim(-0.5, 9.5)
            ax.set_ylim(-0.5, 9.5)
            ax.set_xticks(range(10))
            ax.set_yticks(range(10))
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title(shape, fontsize=8)
            if col == 0:
                ax.set_ylabel(f"rot={rotation}", fontsize=8)

            f = sim_dir / goals_filename(shape, rotation)
            if not f.exists():
                ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center", fontsize=7)
                continue

            seq = load_goal_sequence(f)
            chains = find_chains(seq)
            summary = ", ".join(f"len={len(c[0])} x{len(c)}" for c in chains)
            print(f"{shape} rot={rotation}: {len(chains)} unique chain(s)" + (f" - {summary}" if chains else ""))

            for i, occurrences in enumerate(chains):
                mean, std = chain_mean_std(occurrences)
                plot_chain(ax, mean, std, cmap(i % 10))

    fig.tight_layout()
    out_path = sim_dir.parent / f"{sim_dir.name}_chains.png"
    fig.savefig(out_path, dpi=120)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
