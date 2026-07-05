#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

SHAPES = ["triangle", "square"]
ROTATIONS = [round(i * 0.2, 1) for i in range(9)]  # 0.0 to 1.6 step 0.2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".", help="folder containing a 'simulations' subfolder")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    simulations = root / "simulations"
    if not simulations.is_dir():
        sys.exit(f"no simulations dir found in {root}")

    test_app = Path(__file__).resolve().parent.parent / "src" / "test.py"

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    for experiment_dir in sorted(simulations.iterdir()):
        if not experiment_dir.is_dir():
            continue
        if any(experiment_dir.glob("goals*npy")):
            continue

        print(f"Testing on {experiment_dir.name} ...")
        for shape in SHAPES:
            for rotation in ROTATIONS:
                subprocess.run(
                    [sys.executable, str(test_app), "--plot", "--posrot", "40", "40", str(rotation), "--world", shape],
                    cwd=experiment_dir,
                    env=env,
                    check=True,
                )


if __name__ == "__main__":
    main()
