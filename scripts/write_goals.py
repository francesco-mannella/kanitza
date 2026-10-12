"""Write goals files into trained run folders, so results_server.py draws their scanpaths.

Usage:
    python scripts/write_goals.py (--manifest FILE | RUN_DIR ...) [--no-exclude-fixation] [--worlds W ...]

For each run, replays the test of src/test.py (object at [40, 40], both shapes, rotations 0.0 to 1.6 in steps
of 0.2 rad, seed 0, 40 decisions) with exclude_fixation on (the champion test protocol; use
--no-exclude-fixation to keep the run's own setting) and saves goals_<shape>_<pos>_<rot>.npy in the run folder
in the format of src/test.py ([{"world", "position", "angle", "saccade_id", "goal"}]). Existing files are kept.
"""
import argparse
import csv
import os
import sys
import warnings

import numpy as np
from slugify import slugify

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from exp_common import Run, set_device  # noqa: E402

POS = [40.0, 40.0]
ROTATIONS = [round(i * 0.2, 1) for i in range(9)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--manifest")
    ap.add_argument("--no-exclude-fixation", action="store_true")
    ap.add_argument("--worlds", nargs="+", default=["triangle", "square"])
    args = ap.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)
    set_device()
    runs = list(args.runs)
    if args.manifest:
        runs += [r["run"] for r in csv.DictReader(open(args.manifest))]
    for path in runs:
        run = Run(path, seed=0, **({} if args.no_exclude_fixation else {"exclude_fixation": True}))
        for world in args.worlds:
            for rot in ROTATIONS:
                name = slugify(f"goals_{world}_{POS}_{rot:06.2f}_") + ".npy"
                out = os.path.join(path, name)
                if os.path.exists(out):
                    continue
                np.save(out, [run.test(world, pos=POS, rot=rot)])
        print(os.path.basename(path), "done", flush=True)


if __name__ == "__main__":
    main()
