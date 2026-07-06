#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chain_quality import cycle_length, dedup_consecutive_len, load_goal_sequence, unique_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="a single simulation folder containing goals-*.npy files")
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    files = sorted(folder.glob("goals-*.npy"))
    if not files:
        raise SystemExit(f"no goals-*.npy files found in {folder}")

    unique, dedup, cycle = [], [], []
    print(f"{'file':<45} {'unique_points':>13} {'dedup_len':>10} {'cycle_len':>10}")
    for f in files:
        seq = load_goal_sequence(f)
        u = unique_points(seq)
        d = dedup_consecutive_len(seq)
        c = cycle_length(seq)
        unique.append(u)
        dedup.append(d)
        cycle.append(c)
        print(f"{f.name:<45} {u:>13} {d:>10} {c:>10}")

    print()
    print(f"average unique_points: {sum(unique) / len(unique):.2f}")
    print(f"average dedup_len:     {sum(dedup) / len(dedup):.2f}")
    print(f"average cycle_len:     {sum(cycle) / len(cycle):.2f}")


if __name__ == "__main__":
    main()
