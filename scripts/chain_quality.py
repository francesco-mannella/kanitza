#!/usr/bin/env python3
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED_SUFFIX_RE = re.compile(r"_(\d+)$")


def load_goal_sequence(npy_path):
    data = np.load(npy_path, allow_pickle=True)[0]
    return [tuple(np.asarray(g).ravel().tolist()) for g in data["goal"]]


def unique_points(seq):
    return len(set(seq))


def dedup_consecutive_len(seq):
    return sum(1 for i, g in enumerate(seq) if i == 0 or g != seq[i - 1])


def score_dir(experiment_dir):
    unique, dedup = [], []
    for f in sorted(experiment_dir.glob("goals-*.npy")):
        seq = load_goal_sequence(f)
        unique.append(unique_points(seq))
        dedup.append(dedup_consecutive_len(seq))
    return unique, dedup


def combo_key(dir_name):
    return SEED_SUFFIX_RE.sub("", dir_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".", help="folder containing a 'simulations' subfolder")
    args = parser.parse_args()

    simulations = Path(args.root).resolve() / "simulations"
    if not simulations.is_dir():
        raise SystemExit(f"no simulations dir found in {simulations.parent}")

    combo_unique = defaultdict(list)
    combo_dedup = defaultdict(list)

    print(f"{'directory':<75} {'avg_unique':>10} {'avg_dedup':>10}")
    for experiment_dir in sorted(simulations.iterdir()):
        if not experiment_dir.is_dir():
            continue
        unique, dedup = score_dir(experiment_dir)
        if not unique:
            continue
        avg_unique = sum(unique) / len(unique)
        avg_dedup = sum(dedup) / len(dedup)
        print(f"{experiment_dir.name:<75} {avg_unique:>10.2f} {avg_dedup:>10.2f}")

        key = combo_key(experiment_dir.name)
        combo_unique[key].extend(unique)
        combo_dedup[key].extend(dedup)

    print()
    print(f"{'combination':<75} {'avg_unique':>10} {'avg_dedup':>10} {'n_files':>8}")
    for key in sorted(combo_unique):
        u = combo_unique[key]
        d = combo_dedup[key]
        print(f"{key:<75} {sum(u) / len(u):>10.2f} {sum(d) / len(d):>10.2f} {len(u):>8}")


if __name__ == "__main__":
    main()
