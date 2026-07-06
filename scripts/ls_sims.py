#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chain_quality import cycle_score_normalized, load_goal_sequence, unique_points

SEED_RE = re.compile(r"_(\d+)$")
PARAM_RE = re.compile(r"^(decaying_speed|local_decaying_speed|match_std)\s*=\s*(\S+)")


def score_experiment_dir(experiment_dir):
    values = [
        unique_points(seq := load_goal_sequence(f)) * cycle_score_normalized(seq)
        for f in sorted(experiment_dir.glob("goals-*.npy"))
    ]
    return sum(values) / len(values) if values else None


def relevant_params(final_parameters_path):
    if not final_parameters_path.exists():
        return ""
    parts = []
    for line in final_parameters_path.read_text().splitlines():
        m = PARAM_RE.match(line.strip())
        if m and "baseline" not in m.group(1):
            parts.append(f"{m.group(1)[0]}:{m.group(2)}")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="folder containing one or more '<name>/simulations/<experiment>' trees",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    for simulations in sorted(root.glob("*/simulations")):
        for experiment_dir in sorted(simulations.iterdir()):
            if not experiment_dir.is_dir():
                continue
            avg = score_experiment_dir(experiment_dir)
            if avg is None:
                continue
            seed_match = SEED_RE.search(experiment_dir.name)
            seed = seed_match.group(1) if seed_match else experiment_dir.name
            print(f"{avg:.2f} {relevant_params(experiment_dir / 'final_parameters')} {seed}")


if __name__ == "__main__":
    main()
