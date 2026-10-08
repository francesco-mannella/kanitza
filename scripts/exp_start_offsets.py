"""E1a: do the scanpaths of a trained model depend on where the retina starts?"""
import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from exp_common import Run, find_runs, set_device  # noqa: E402

WORLDS = ["triangle", "square"]


def goal_cells(goals, burn_in, side):
    g = np.array([np.asarray(x).reshape(-1) for x in goals["goal"]])[burn_in:]
    cells = np.zeros(side * side, bool)
    cells[(g[:, 0] * side + g[:, 1]).astype(int)] = True
    return cells


def jaccard(cells):
    a = cells.astype(float)
    inter = a @ a.T
    return inter / (a.sum(1)[:, None] + a.sum(1)[None] - inter)


def measures(cells, labels):
    j = jaccard(cells)
    n = len(cells)
    pose = np.array([l[0] for l in labels])
    world = np.array([l[1] for l in labels])
    same = (pose[:, None] == pose[None]) & ~np.eye(n, dtype=bool)
    other = pose[:, None] != pose[None]
    nn, hit = [], []
    for i in range(n):
        cand = np.where(other[i])[0]
        nn.append(world[cand[np.argmax(j[i, cand])]] == world[i])
        rest = np.where(np.arange(n) != i)[0]
        hit.append(pose[rest[np.argmax(j[i, rest])]] == pose[i])
    return dict(same_pose=j[same].mean(), other_pose=j[other].mean(),
                invariance=j[same].mean() - j[other].mean(), acc_world=float(np.mean(nn)),
                acc_pose=float(np.mean(hit)))


def main():
    parser = argparse.ArgumentParser(description=(
        "Scanpaths from a 3x3 grid of retina start offsets, both objects and several "
        "rotations, for the model, a salience-only control (the maps read the view but the "
        "attention mask is uniform) and a random-goal control. A scanpath is the set of "
        "lattice cells visited after a burn-in. same_pose is the mean Jaccard overlap between "
        "tests of one pose at different offsets, other_pose between different poses, "
        "invariance their difference, acc_world the leave-one-pose-out 1-NN accuracy at "
        "recovering the object (degenerate when different poses share no cell), acc_pose the "
        "1-NN accuracy at recovering the pose among all other tests (added after a first "
        "run showed acc_world degenerate). Rule fixed before running: the model supports "
        "stability if its invariance exceeds that of both controls by more than the largest "
        "between-run SD of the model's invariance, and its mean acc_world is at least 0.85. "
        "A second verdict, labelled post hoc, uses acc_pose in place of acc_world."))
    parser.add_argument("paths", nargs="+", help="Run folders, or folders containing them.")
    parser.add_argument("--half-width", type=float, default=12.0,
                        help="Start offsets are {-h, 0, h} on each axis, in task units.")
    parser.add_argument("--rotations", type=float, nargs="+", default=[0.0, 0.6, 1.2])
    parser.add_argument("--burn-in", type=int, default=3)
    parser.add_argument("--policies", nargs="+", default=["model", "salience", "random"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", help="Write per-run measures to this CSV file.")
    args = parser.parse_args()

    set_device()
    runs = find_runs(args.paths)
    if not runs:
        sys.exit("No run folders found.")
    h = args.half_width
    offsets = list(itertools.product([-h, 0.0, h], repeat=2))
    rows = []
    for path in runs:
        run = Run(path, seed=args.seed)
        side = int(round(np.sqrt(run.params.maps_output_size)))
        for policy in args.policies:
            rng = np.random.RandomState(args.seed)
            cells, labels = [], []
            for w, world in enumerate(WORLDS):
                for r, rot in enumerate(args.rotations):
                    for offset in offsets:
                        goals = run.test(world, [40.0, 40.0], rot, start=offset, policy=policy, rng=rng)
                        cells.append(goal_cells(goals, args.burn_in, side))
                        labels.append((w * len(args.rotations) + r, w))
            m = measures(np.array(cells), labels)
            rows.append(dict(run=os.path.basename(path), policy=policy, **m))
            print(f"{rows[-1]['run'][-30:]:30s} {policy:9s} same {m['same_pose']:.2f} other "
                  f"{m['other_pose']:.2f} inv {m['invariance']:.2f} acc {m['acc_world']:.2f}",
                  file=sys.stderr, flush=True)

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("run,policy,same_pose,other_pose,invariance,acc_world,acc_pose\n")
            for r in rows:
                f.write(",".join(str(r[k]) for k in ["run", "policy", "same_pose", "other_pose",
                                                       "invariance", "acc_world", "acc_pose"]) + "\n")
    print(f"{len(runs)} run(s), {len(offsets)} offsets, rotations {args.rotations}")
    print(f"{'policy':9s} {'same_pose':>10s} {'other_pose':>11s} {'invariance':>16s} "
          f"{'acc_world':>10s} {'acc_pose':>9s}")
    summary = {}
    for policy in args.policies:
        sel = [r for r in rows if r["policy"] == policy]
        inv = np.array([r["invariance"] for r in sel])
        summary[policy] = dict(inv=inv.mean(), sd=inv.std(), acc=np.mean([r["acc_world"] for r in sel]),
                               accp=np.mean([r["acc_pose"] for r in sel]))
        print(f"{policy:9s} {np.mean([r['same_pose'] for r in sel]):10.2f} "
              f"{np.mean([r['other_pose'] for r in sel]):11.2f} "
              f"{inv.mean():9.2f}+-{inv.std():4.2f} {summary[policy]['acc']:10.2f} "
              f"{summary[policy]['accp']:9.2f}")
    if "model" in summary and len(summary) > 1:
        m = summary["model"]
        beats = all(m["inv"] - summary[p]["inv"] > m["sd"] for p in summary if p != "model")
        ok = beats and m["acc"] >= 0.85
        print(f"Rule: model invariance beats the controls by more than its SD ({m['sd']:.2f}): "
              f"{beats}; acc_world >= 0.85: {m['acc'] >= 0.85}; {'PASS' if ok else 'FAIL'}")
        ok2 = beats and m["accp"] >= 0.85
        print(f"Post hoc rule with acc_pose >= 0.85: {m['accp'] >= 0.85}; {'PASS' if ok2 else 'FAIL'}")


if __name__ == "__main__":
    main()
