#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chain_quality import (
    cycle_score,
    cycle_score_normalized,
    dedup_consecutive_len,
    load_goal_sequence,
    unique_points,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="a single simulation folder containing goals-*.npy files")
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    files = sorted(folder.glob("goals-*.npy"))
    if not files:
        raise SystemExit(f"no goals-*.npy files found in {folder}")

    unique, dedup, cycle, cycle_norm, unique_cyc_norm = [], [], [], [], []
    print(
        f"{'file':<45} {'unique_points':>13} {'dedup_len':>10} "
        f"{'cycle_score':>11} {'cyc_norm':>9} {'uniq*cyc':>9}"
    )
    for f in files:
        seq = load_goal_sequence(f)
        u = unique_points(seq)
        d = dedup_consecutive_len(seq)
        c = cycle_score(seq)
        cn = cycle_score_normalized(seq)
        ucn = u * cn
        unique.append(u)
        dedup.append(d)
        cycle.append(c)
        cycle_norm.append(cn)
        unique_cyc_norm.append(ucn)
        print(f"{f.name:<45} {u:>13} {d:>10} {c:>11} {cn:>9.2f} {ucn:>9.2f}")

    print()
    print(f"average unique_points: {sum(unique) / len(unique):.2f}")
    print(f"average dedup_len:     {sum(dedup) / len(dedup):.2f}")
    print(f"average cycle_score:   {sum(cycle) / len(cycle):.2f}")
    print(f"average cyc_norm:      {sum(cycle_norm) / len(cycle_norm):.2f}")
    print(f"average uniq*cyc_norm: {sum(unique_cyc_norm) / len(unique_cyc_norm):.2f}")


if __name__ == "__main__":
    main()
