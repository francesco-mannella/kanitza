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


def cycle_score(seq):
    """Looks for a contiguous block of length k (2 <= k <= number of unique
    points) that repeats consecutively (non-overlapping) somewhere in the
    sequence, and scores it as k * repeats. Takes the best (highest-scoring)
    such block across all k and starting positions; 0 if nothing repeats.

    Maximal when the longest allowed block (k = unique_points) repeats the
    most times; minimal-but-nonzero when only the shortest block (k=2)
    repeats the minimum qualifying number of times (2, i.e. score 4).

    Runs on the consecutive-collapsed sequence, not the raw one, so that
    plain frame-holding (already measured by dedup_consecutive_len) doesn't
    mask an underlying cycle (e.g. [A,A,B,B,A,A,B,B] collapses to
    [A,B,A,B], a clean 2-cycle).
    """
    collapsed = collapse_consecutive(seq)
    n = len(set(collapsed))
    length = len(collapsed)
    best = 0
    for k in range(2, min(n, length // 2) + 1):
        best_reps = 1
        for start in range(length - k + 1):
            block = collapsed[start:start + k]
            reps = 1
            pos = start + k
            while pos + k <= length and collapsed[pos:pos + k] == block:
                reps += 1
                pos += k
            best_reps = max(best_reps, reps)
        if best_reps >= 2:
            best = max(best, k * best_reps)
    return best


def cycle_score_normalized(seq):
    """cycle_score scaled by the collapsed sequence length, so it's
    comparable across episodes regardless of dedup_consecutive_len (0 = no
    cycling, 1 = the entire collapsed sequence is one repeating block)."""
    return cycle_score(seq) / dedup_consecutive_len(seq)


def score_dir(experiment_dir):
    unique, dedup, cycle, cycle_norm = [], [], [], []
    for f in sorted(experiment_dir.glob("goals-*.npy")):
        seq = load_goal_sequence(f)
        unique.append(unique_points(seq))
        dedup.append(dedup_consecutive_len(seq))
        cycle.append(cycle_score(seq))
        cycle_norm.append(cycle_score_normalized(seq))
    return unique, dedup, cycle, cycle_norm


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
    combo_cycle_norm = defaultdict(list)

    print(f"{'directory':<75} {'avg_unique':>10} {'avg_dedup':>10} {'cycle_score':>11} {'cyc_norm':>9}")
    for experiment_dir in sorted(simulations.iterdir()):
        if not experiment_dir.is_dir():
            continue
        unique, dedup, cycle, cycle_norm = score_dir(experiment_dir)
        if not unique:
            continue
        avg_unique = sum(unique) / len(unique)
        avg_dedup = sum(dedup) / len(dedup)
        avg_cycle = sum(cycle) / len(cycle)
        avg_cycle_norm = sum(cycle_norm) / len(cycle_norm)
        print(f"{experiment_dir.name:<75} {avg_unique:>10.2f} {avg_dedup:>10.2f} {avg_cycle:>11.2f} {avg_cycle_norm:>9.2f}")

        key = combo_key(experiment_dir.name)
        combo_unique[key].extend(unique)
        combo_dedup[key].extend(dedup)
        combo_cycle[key].extend(cycle)
        combo_cycle_norm[key].extend(cycle_norm)

    print()
    print(f"{'combination':<75} {'avg_unique':>10} {'avg_dedup':>10} {'cycle_score':>11} {'cyc_norm':>9} {'n_files':>8}")
    for key in sorted(combo_unique):
        u = combo_unique[key]
        d = combo_dedup[key]
        c = combo_cycle[key]
        cn = combo_cycle_norm[key]
        print(
            f"{key:<75} {sum(u) / len(u):>10.2f} {sum(d) / len(d):>10.2f} "
            f"{sum(c) / len(c):>11.2f} {sum(cn) / len(cn):>9.2f} {len(u):>8}"
        )


if __name__ == "__main__":
    main()
