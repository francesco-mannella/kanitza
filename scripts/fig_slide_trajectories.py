"""Test trajectories of a trained run and the trajectory plots of slides 27-32.

Usage:
    python scripts/fig_slide_trajectories.py RUN_DIR OUT_DIR [--seed 0] [--rots R R R] [--objects triangle square]

Replays the test of both objects at rotations 0.0 to 1.6 rad (40 decisions, object at the center, model policy)
and writes OUT_DIR/trajectories.csv: one row per decision with the goal unit on the 10x10 map (goal.x, goal.y),
the executed retina displacement and the trial. The first 3 decisions of each test are dropped from the plots,
as in scripts/paths.py. Three rotations per object are chosen by the scanpath criteria: no goal shared between
the scanpaths of different rotations, no goal shared with the other object, no equivalent rotations (square 90 deg,
triangle 120 deg, within 0.1 rad), executed saccades and many distinct goals. One plot per chosen trial, all trials
faint and the chosen one drawn thick, goes to OUT_DIR/trajectories_<object>_<rot>.png.
"""
import argparse
import itertools
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from exp_common import Run, set_device  # noqa: E402

ROTS = [round(0.2 * k, 1) for k in range(9)]
SYMMETRY = {"square": np.pi / 2, "triangle": 2 * np.pi / 3}
DROP = 3


def reseed(run, seed):
    """Make a test independent of the tests run before it."""
    torch.manual_seed(seed)
    run.agent.rng = np.random.RandomState(seed)
    run.off.rng = np.random.RandomState(seed)


def replay(run, world, rot, seed):
    reseed(run, seed)
    pre, post = [], []

    def on_decision(d):
        pre.append(np.array(d["run"].env.retina_sim_pos, float))

    def on_step(d):
        if d["t"] % d["run"].params.saccade_period == 0:
            post.append(np.array(d["run"].env.retina_sim_pos, float))

    goals = run.test(world, pos=[40.0, 40.0], rot=rot, on_decision=on_decision, on_step=on_step)
    goal = np.stack([np.asarray(g).reshape(-1)[:2] for g in goals["goal"]])
    moved = np.linalg.norm(np.array(post) - np.array(pre), axis=1)
    return pd.DataFrame(
        {
            "seed": seed,
            "object": world,
            "rot": rot,
            "decision": np.arange(len(goal)),
            "goal.y": goal[:, 0],
            "goal.x": goal[:, 1],
            "moved": moved,
        }
    )


def equivalent(world, a, b):
    d = (a - b) % SYMMETRY[world]
    return min(d, SYMMETRY[world] - d) < 0.1


def cells_of(g):
    g = g[g.decision >= DROP]
    return set(zip(g["goal.x"].astype(int), g["goal.y"].astype(int)))


def choose(df, world, other):
    """Rank the rotation triples by the mean, over seeds, of the goals shared between rotations or with the other object."""
    stats = {}
    for combo in itertools.combinations(ROTS, 3):
        if any(equivalent(world, a, b) for a, b in itertools.combinations(combo, 2)):
            continue
        if min(abs(a - b) for a, b in itertools.combinations(combo, 2)) < 0.4 - 1e-9:
            continue
        stats[combo] = [[], [], []]
    for seed, d in df.groupby("seed"):
        mine = d[d.object == world]
        foreign = cells_of(d[d.object == other])
        cells = {rot: cells_of(g) for rot, g in mine.groupby("rot")}
        for combo, acc in stats.items():
            acc[0].append(sum(len(cells[a] & cells[b]) for a, b in itertools.combinations(combo, 2)) + sum(len(cells[r] & foreign) for r in combo))
            acc[1].append(sum(len(cells[r]) for r in combo))
            acc[2].append(float(np.mean([(mine[mine.rot == r].moved < 1.0).mean() for r in combo])))
    ranked = [(np.mean(v[0]), -np.mean(v[1]), np.mean(v[2]), combo) for combo, v in stats.items()]
    ranked.sort()
    return ranked


def draw(df, world, rot, path):
    ink, muted, faint = "#111827", "#6b7280", "#d7dbe3"
    colour = {"triangle": "#e5484d", "square": "#3b6fd4"}
    dark = {"triangle": "#b3262c", "square": "#1f4aa8"}
    with plt.rc_context(
        {"font.size": 10, "axes.edgecolor": "#9ca3af", "axes.linewidth": 0.8, "xtick.color": muted, "ytick.color": muted}
    ):
        fig = plt.figure(figsize=(6.4, 4.8), dpi=200)
        ax = fig.add_axes([0.085, 0.115, 0.62, 0.84])
        side = np.arange(10)
        ax.scatter(*np.meshgrid(side, side), s=9, color=faint, zorder=1, linewidths=0)
        for (obj, r), g in df.groupby(["object", "rot"]):
            if obj == world and r == rot:
                continue
            g = g[g.decision >= DROP]
            ax.plot(g["goal.x"], g["goal.y"], "-", color=colour[obj], alpha=0.16, lw=1.0, zorder=2, solid_capstyle="round")
        chosen = df[(df.object == world) & (df.rot == rot) & (df.decision >= DROP)]
        xy = chosen[["goal.x", "goal.y"]].to_numpy(float)
        visits = pd.Series(map(tuple, xy)).value_counts()
        for (x, y), n in visits.items():
            ax.scatter([x], [y], s=26 + 20 * n, color="white", edgecolor=dark[world], linewidth=1.6, zorder=4)
        moves = dict.fromkeys((tuple(p0), tuple(p1)) for p0, p1 in zip(xy[:-1], xy[1:]) if not np.allclose(p0, p1))
        for p0, p1 in map(np.array, moves):
            step = p1 - p0
            normal = np.array([-step[1], step[0]]) / np.linalg.norm(step)
            ax.add_patch(
                FancyArrowPatch(
                    p0 + 0.08 * normal, p1 + 0.08 * normal, arrowstyle="-|>", mutation_scale=11, lw=1.9,
                    color=colour[world], shrinkA=6, shrinkB=6, zorder=3, alpha=0.95,
                )
            )
        ax.scatter(*xy[-1], s=95, marker="s", color=dark[world], zorder=5, edgecolor="white", linewidth=1.2)
        ax.scatter(*xy[0], s=48, marker="o", color=dark[world], zorder=6, edgecolor="white", linewidth=1.2)
        ax.set_xlim(-0.6, 9.6)
        ax.set_ylim(-0.6, 9.6)
        ax.set_aspect("equal")
        ax.set_xticks(side)
        ax.set_yticks(side)
        ax.tick_params(length=2.5, labelsize=8)
        ax.set_xlabel("map unit x", color=muted, labelpad=3)
        ax.set_ylabel("map unit y", color=muted, labelpad=3)
        for side_name in ("top", "right"):
            ax.spines[side_name].set_visible(False)
        handles = [
            Line2D([], [], color=colour[o], lw=2, alpha=0.5, label=o) for o in ("triangle", "square")
        ] + [
            Line2D([], [], color=colour[world], lw=2.2, label=f"{world}, {rot:.1f} rad\n({np.degrees(rot):.0f}\u00b0)"),
            Line2D([], [], marker="o", color=dark[world], lw=0, ms=7, label="first goal"),
            Line2D([], [], marker="s", color=dark[world], lw=0, ms=7, label="last goal"),
        ]
        fig.text(0.735, 0.88, "all trials", fontsize=9, color=muted, weight="bold")
        fig.text(0.735, 0.665, "this trial", fontsize=9, color=muted, weight="bold")
        for text, y in (("triangle", 0.82), ("square", 0.76)):
            fig.add_artist(Line2D([0.735, 0.775], [y + 0.012] * 2, color=colour[text], lw=2, alpha=0.55))
            fig.text(0.79, y, text, fontsize=9.5, color=ink, va="bottom")
        fig.add_artist(Line2D([0.735, 0.775], [0.617] * 2, color=colour[world], lw=2.4))
        fig.text(0.79, 0.585, f"{world}\n{rot:.1f} rad ({np.degrees(rot):.0f}\u00b0)", fontsize=9.5, color=ink, va="bottom", linespacing=1.3)
        fig.add_artist(Line2D([0.755], [0.51], marker="o", color=dark[world], ms=7, lw=0))
        fig.text(0.79, 0.498, "first goal", fontsize=9.5, color=ink, va="bottom")
        fig.add_artist(Line2D([0.755], [0.44], marker="s", color=dark[world], ms=7, lw=0))
        fig.text(0.79, 0.428, "last goal", fontsize=9.5, color=ink, va="bottom")
        fig.text(0.735, 0.345, "decisions 4 to 40\nbigger ring: goal\nvisited more often", fontsize=8, color=muted, va="top", linespacing=1.35)
        fig.savefig(path, facecolor="white")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0, help="first seed; the plots use this one")
    ap.add_argument("--n-seeds", type=int, default=5, help="seeds used to choose the rotations")
    ap.add_argument("--rots", type=float, nargs=3, help="override the chosen rotations (both objects)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    set_device()
    run = Run(a.run, seed=a.seed)
    frames = []
    for seed in range(a.seed, a.seed + a.n_seeds):
        for world in ("triangle", "square"):
            for rot in ROTS:
                frames.append(replay(run, world, rot, seed))
        print("seed", seed, "done", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(a.out, "trajectories.csv"), index=False)
    print("executed saccades: decisions with a retina move below 1 px:", round(float((df.moved < 1).mean()), 3))

    chosen = {}
    for world, other in (("triangle", "square"), ("square", "triangle")):
        ranked = choose(df, world, other)
        print(f"{world}: top triples (mean shared+foreign goals over {a.n_seeds} seeds, -mean distinct goals, blocked fraction)")
        for item in ranked[:5]:
            print("   ", tuple(round(float(x), 2) for x in item[:3]), item[3])
        chosen[world] = list(a.rots) if a.rots else list(ranked[0][3])
    df = df[df.seed == a.seed]
    for world, rots in chosen.items():
        for rot in rots:
            draw(df, world, rot, os.path.join(a.out, f"trajectories_{world}_{rot}.png"))
    print("chosen", chosen)


if __name__ == "__main__":
    main()
