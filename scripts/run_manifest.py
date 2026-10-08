"""List the fully trained, current-code, non-copy runs for the Phase 1
retinotopy analysis and tag each with its settings.

Usage:
    python scripts/run_manifest.py [--root DIR] [--out manifest.csv]

Inclusion rule (fixed before running, see predictive_remapping.md 1.0): a run
is a folder with off_control_store and final_parameters that
    - lies in a sweep folder not in EXCLUDED (old, replay_old_*, test_*,
      retest_*, scanpath_*, *_verify, hparam_screen, hparam_refine, exp_*),
    - has epochs == 1000 and 1000 logged "comp:" lines in log (stored epoch
      1000),
    - has test_fovea False,
    - is not a copy: symlinks are not followed and a run whose
      off_control_store shares an inode with an earlier one is dropped.
"""
import argparse
import ast
import csv
import fnmatch
import os
import re

EXCLUDED = ["old", "replay_old_*", "test_*", "retest_*", "scanpath_*",
            "*_verify_*", "hparam_screen_*", "hparam_refine_*", "exp_*",
            "remapping_paradigm_*", "retinotopy_runs_*"]
TAGS = {
    "anchor_std": 4.0,
    "match_std_baseline": 0.5,
    "goal_inhibition": 0.0,
    "hold_fixation": False,
    "orienting_saccade": False,
    "exclude_fixation": False,
    "attention_max_variance": 6.0,
    "epochs": None,
    "test_fovea": False,
}
DEFAULT_ROOT = os.path.join(os.path.expanduser("~"), "tmp", "kanizsa_saliences")


def read_params(path):
    params = {}
    for line in open(path):
        m = re.match(r"^(\w+)\s*=\s*(.+?)\s*$", line)
        if m:
            try:
                params[m.group(1)] = ast.literal_eval(m.group(2))
            except (ValueError, SyntaxError):
                params[m.group(1)] = m.group(2)
    return params


def logged_epochs(run):
    log = os.path.join(run, "log")
    if not os.path.isfile(log):
        return 0
    return sum(1 for line in open(log) if line.startswith("comp:"))


def sweep_name(folder):
    return re.sub(r"_\d{4}-\d{2}-\d{2}$", "", folder).replace("hparam_", "")


def build(root):
    rows, seen = [], set()
    for folder in sorted(os.listdir(root)):
        if any(fnmatch.fnmatch(folder, p) for p in EXCLUDED):
            continue
        for dirpath, _, files in os.walk(os.path.join(root, folder)):
            if "off_control_store" not in files or "final_parameters" not in files:
                continue
            ino = os.stat(os.path.join(dirpath, "off_control_store")).st_ino
            params = read_params(os.path.join(dirpath, "final_parameters"))
            if params.get("epochs") != 1000 or logged_epochs(dirpath) < 1000 \
                    or params.get("test_fovea", False) or ino in seen:
                continue
            seen.add(ino)
            name = params.get("init_name", os.path.basename(dirpath))
            row = {"sweep": sweep_name(folder), "run": dirpath,
                   "name": name, "seed": name.rsplit("_", 1)[-1]}
            row.update({k: params.get(k, d) for k, d in TAGS.items() if k not in ("epochs", "test_fovea")})
            rows.append(row)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default="manifest.csv")
    args = ap.parse_args()
    rows = build(args.root)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    counts = {}
    for r in rows:
        counts[r["sweep"]] = counts.get(r["sweep"], 0) + 1
    for k, v in sorted(counts.items()):
        print(f"{k:24s} {v}")
    print(f"total {len(rows)} -> {args.out}")
