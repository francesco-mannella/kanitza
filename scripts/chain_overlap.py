#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chain_paths import find_chains
from chain_quality import load_goal_sequence
from tests import ROTATIONS, SHAPES, goals_filename


def combo_chains(sim_dir, shape, rotation):
    """Each chain's occurrences are byte-identical by construction (that's
    the repeat condition in find_chains), so the first occurrence's point
    set already represents the whole chain."""
    f = sim_dir / goals_filename(shape, rotation)
    if not f.exists():
        return []
    seq = load_goal_sequence(f)
    return [frozenset(occurrences[0]) for occurrences in find_chains(seq)]


def jaccard(a, b):
    """0.0 when either side has no chain at all -- an absence of a chain
    isn't a shared chain, so two stalled combos shouldn't read as
    "100% overlap" just because both are empty."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def plot_matrix(combos, matrix, out_path):
    labels = [f"{c['shape'][:1]}{c['rotation']}" for c in combos]
    n = len(combos)

    fig, ax = plt.subplots(figsize=(3.5, 4))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("(shape, rotation) combo", fontsize=8)
    ax.set_ylabel("(shape, rotation) combo", fontsize=8)

    # Separate shape blocks (rotations grouped within each shape) with a
    # visible line so the two shapes' sub-matrices are easy to pick out.
    boundary = len(ROTATIONS)
    if 0 < boundary < n:
        ax.axhline(boundary - 0.5, color="black", linewidth=1)
        ax.axvline(boundary - 0.5, color="black", linewidth=1)

    ax.set_title("Chain overlap (Jaccard similarity)", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, label="Jaccard similarity", shrink=0.8)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("Jaccard similarity", fontsize=8)

    caption = (
        "Each row/column is a (shape, rotation) combo, labeled by shape initial "
        "(t = triangle, s = square) and rotation. Cell (i, j) is the Jaccard "
        "similarity -- intersection over union -- between the sets of goal "
        "points visited by combo i's and combo j's unique repeating chains; "
        "0 = no shared points, 1 = identical chain point sets. The diagonal "
        "is always 1 (a combo compared with itself); black lines separate "
        "the two shape blocks."
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=5.5, wrap=True)
    fig.savefig(out_path, dpi=300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("simulation_dir", help="a single simulation folder containing goals-*.npy files")
    args = parser.parse_args()

    sim_dir = Path(args.simulation_dir).resolve()
    if not sim_dir.is_dir():
        raise SystemExit(f"not a directory: {sim_dir}")

    combos = []
    for shape in SHAPES:
        for rotation in ROTATIONS:
            chains = combo_chains(sim_dir, shape, rotation)
            union = frozenset().union(*chains) if chains else frozenset()
            combos.append({"shape": shape, "rotation": rotation, "chains": chains, "union": union})

    print(f"{'shape':<10}{'rot':>6}  {'#chains':>8}  {'#points':>8}")
    for c in combos:
        print(f"{c['shape']:<10}{c['rotation']:>6}  {len(c['chains']):>8}  {len(c['union']):>8}")

    def label(c):
        return f"{c['shape'][:1]}{c['rotation']}"

    matrix = np.array([[jaccard(ci["union"], cj["union"]) for cj in combos] for ci in combos])

    print()
    print("Pairwise overlap (Jaccard similarity of each combo's unioned chain points):")
    print(f"{'':<8}" + "".join(f"{label(c):>6}" for c in combos))
    for ci, row in zip(combos, matrix):
        print(f"{label(ci):<8}" + "".join(f"{v:>6.2f}" for v in row))

    out_path = sim_dir.parent / f"{sim_dir.name}_overlap.png"
    plot_matrix(combos, matrix, out_path)
    print(f"saved {out_path}")

    print()
    print("Exact chain duplicates across different (shape, rotation) combos:")
    exact_pairs = []
    for i, ci in enumerate(combos):
        for cj in combos[i + 1:]:
            shared = [ch for ch in ci["chains"] if ch in cj["chains"]]
            if shared:
                exact_pairs.append((ci, cj, shared))
    if exact_pairs:
        for ci, cj, shared in exact_pairs:
            print(f"  {label(ci):<6} <-> {label(cj):<6}: {len(shared)} identical chain(s)")
    else:
        print("  none -- every (shape, rotation) uses a distinct chain")

    within_shape = []
    cross_shape = []
    for i, ci in enumerate(combos):
        for cj in combos[i + 1:]:
            score = jaccard(ci["union"], cj["union"])
            (within_shape if ci["shape"] == cj["shape"] else cross_shape).append(score)

    def avg(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    total_pairs = len(combos) * (len(combos) - 1) // 2
    print()
    print(f"Average within-shape rotation overlap: {avg(within_shape):.3f}")
    print(f"Average cross-shape overlap:            {avg(cross_shape):.3f}")
    print(f"Exact-duplicate pairs:                  {len(exact_pairs)} / {total_pairs}")


if __name__ == "__main__":
    main()
