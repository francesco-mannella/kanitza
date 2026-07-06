#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from slugify import slugify

SHAPES = ["triangle", "square"]
ROTATIONS = [round(i * 0.2, 1) for i in range(9)]  # 0.0 to 1.6 step 0.2
POS = [40.0, 40.0]


def goals_filename(shape, rotation):
    # Mirrors the slugified filename src/test.py writes for a given
    # shape/rotation, so completion can be checked without running it.
    base_name = f"goals_{shape}"
    parts = [f"{POS}_", f"{rotation:06.2f}_"]
    return slugify(f"{base_name}_{''.join(parts)}") + ".npy"


def pending_combinations(experiment_dir):
    # Yields only the (shape, rotation) pairs not yet tested in this dir,
    # so a partially-tested folder resumes instead of restarting from 0.
    for shape in SHAPES:
        for rotation in ROTATIONS:
            if not (experiment_dir / goals_filename(shape, rotation)).exists():
                yield shape, rotation


def run_experiment(experiment_dir, test_app, env):
    # Runs the remaining shape/rotation combinations for one experiment dir,
    # sequentially, as a single unit of work for the thread pool below.
    print(f"Testing on {experiment_dir.name} ...")
    for shape, rotation in pending_combinations(experiment_dir):
        subprocess.run(
            [sys.executable, str(test_app), "--plot", "--posrot", "40", "40", str(rotation), "--world", shape],
            cwd=experiment_dir,
            env=env,
            check=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".", help="folder containing a 'simulations' subfolder")
    parser.add_argument("--max-processes", type=int, default=1)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    simulations = root / "simulations"
    if not simulations.is_dir():
        sys.exit(f"no simulations dir found in {root}")

    test_app = Path(__file__).resolve().parent.parent / "src" / "test.py"

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    # Skip dirs that are already fully tested (no pending combinations left).
    experiment_dirs = [
        experiment_dir
        for experiment_dir in sorted(simulations.iterdir())
        if experiment_dir.is_dir() and any(pending_combinations(experiment_dir))
    ]

    # Run up to --max-processes experiment dirs concurrently. Threads (not
    # processes) are enough since each worker is blocked on subprocess.run.
    with ThreadPoolExecutor(max_workers=args.max_processes) as executor:
        futures = [
            executor.submit(run_experiment, experiment_dir, test_app, env)
            for experiment_dir in experiment_dirs
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
