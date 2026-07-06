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


def collapse_consecutive(seq):
    return [g for i, g in enumerate(seq) if i == 0 or g != seq[i - 1]]


def dedup_consecutive_len(seq):
    return len(collapse_consecutive(seq))


def cycle_length(seq):
    """Smallest period p such that the consecutive-collapsed sequence is
    exactly `p` values repeating. Equals the collapsed length when there's
    no shorter repeating pattern (e.g. [A,B,C,A,B,C] -> 3, not 6)."""
    collapsed = collapse_consecutive(seq)
    n = len(collapsed)
    for p in range(1, n + 1):
        if all(collapsed[i] == collapsed[i % p] for i in range(n)):
            return p
    return n


def score_dir(experiment_dir):
    unique, dedup, cycle = [], [], []
    for f in sorted(experiment_dir.glob("goals-*.npy")):
        seq = load_goal_sequence(f)
        unique.append(unique_points(seq))
        dedup.append(dedup_consecutive_len(seq))
        cycle.append(cycle_length(seq))
    return unique, dedup, cycle


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
    combo_cycle = defaultdict(list)

    print(f"{'directory':<75} {'avg_unique':>10} {'avg_dedup':>10} {'avg_cycle':>10}")
    for experiment_dir in sorted(simulations.iterdir()):
        if not experiment_dir.is_dir():
            continue
        unique, dedup, cycle = score_dir(experiment_dir)
        if not unique:
            continue
        avg_unique = sum(unique) / len(unique)
        avg_dedup = sum(dedup) / len(dedup)
        avg_cycle = sum(cycle) / len(cycle)
        print(f"{experiment_dir.name:<75} {avg_unique:>10.2f} {avg_dedup:>10.2f} {avg_cycle:>10.2f}")

        key = combo_key(experiment_dir.name)
        combo_unique[key].extend(unique)
        combo_dedup[key].extend(dedup)
        combo_cycle[key].extend(cycle)

    print()
    print(f"{'combination':<75} {'avg_unique':>10} {'avg_dedup':>10} {'avg_cycle':>10} {'n_files':>8}")
    for key in sorted(combo_unique):
        u = combo_unique[key]
        d = combo_dedup[key]
        c = combo_cycle[key]
        print(f"{key:<75} {sum(u) / len(u):>10.2f} {sum(d) / len(d):>10.2f} {sum(c) / len(c):>10.2f} {len(u):>8}")


if __name__ == "__main__":
    main()
