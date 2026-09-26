"""Launch src/main.py over a grid of parameters (this produced the
tests/long_search_* runs).

Usage: run from the directory that will hold the simulation folders:
    python /path/to/src/grid_search.py

Configuration is in the constants below:
    SEEDS: list of seeds, or None to draw N_SEEDS random seeds in [0, 1e5).
    WANDB: pass -w to main.py (log to wandb); if False main.py logs to
        <run>/data_sim (see local_wandb.py).
    MAX_PROCESSES: simulations run in parallel; a new one starts as soon as
        a running one ends. Each simulation uses one CPU thread
        (OMP_NUM_THREADS=1); runs slow down once they outnumber the physical
        cores (hyper-threads share a core), but total throughput still
        grows up to about the number of hardware threads minus one.
    base_name: prefix of the folder names.
    params: main.py parameters. A list value is a set of alternatives to
        grid over; a scalar is fixed. A parameter whose value is itself a
        list must therefore be wrapped twice (e.g. gabor_scales=[[1.0]]).

For each combination and seed, creates ./<base_name>_<md5(params)[:6]>_<seed:06d>
and runs, inside it,
    nohup python -u main.py -r <name> -p '<k=v;...>' -s <seed> [-w]
so stdout goes to <name>/nohup.out. To reproduce a single run, rerun that
command in an empty folder with the same -p string and seed (the exact
command is in <run>/wandb/*/files/wandb-metadata.json).

With --variants FILE.json, instead of the constants below, the sweep is a
set of one-factor variants of a base configuration:
    {"base_name": "screen", "seeds": [90902, 39973], "max_processes": 7,
     "wandb": false,
     "base": {<main.py parameters>},
     "variants": {"lr_0.03": {"maps_learning_rate": 0.03}, ...}}
It runs "base" and every variant (base updated with the variant's
overrides) for each seed, in folders ./<base_name>_<variant>_<seed:06d>.
--dry-run prints the commands without running them.
"""
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from itertools import product

import numpy as np


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

# SEEDS = [93581]
SEEDS = None
WANDB = True
# WANDB = False
N_SEEDS = 1
MAX_PROCESSES = 2
# base_name = "saliences_new_gabor_kernel"
base_name = "long_search"

params = dict(
    test_fovea=False,
    episodes=20,
    epochs=1000,
    saccade_num=10,
    saccade_time=10,
    plot_sim=False,
    plot_maps=True,
    plotting_epochs_interval=100,
    maps_output_size=100,
    action_size=2,
    attention_size=2,
    maps_learning_rate=0.1,
    saccade_threshold=12.0,
    decaying_speed=3.0,
    local_decaying_speed=[0.5, 1.0],
    learningrate_modulation=50.0,
    neighborhood_modulation=40.0,
    learningrate_modulation_baseline=0.02,
    neighborhood_modulation_baseline=0.1,
    match_std_baseline=0.5,
    match_std=[10.0],
    anchor_std=2.0,
    triangles_percent=50.0,
    agent_sampling_precision=1 - 1e-6,
    gabor_scales=[[1.0]],
    gabor_orientation_bins=5,
    gabor_frequency=0.09,
    gabor_sigma_y_multiplier=1,
    gabor_kernel_size=5,
    gabor_phase_offset=-3.141592653589793 * (0.5 - 2.8e-2),
    gabor_rgb_prop=10.0,
    gabor_bright_prop=0.0,
    attention_max_variance=6,
    attention_fixed_variance_prop=0.3,
    attention_center_distance_variance_prop=0.7,
    attention_center_distance_slope=2,
    fovea_scale=[[16, 16]],
    fovea_size=[[16, 16]],
)

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------


def get_combinations(data):
    """
    Generates all possible combinations of list elements from a dictionary.

    Lists and tuples are sets of alternatives; any other value, strings
    included, is a single fixed value. `data` is not modified.

    Args:
       data: A dictionary.

    Yields:
       A dictionary representing a single combination of elements.
    """
    alternatives = [
        v if isinstance(v, (list, tuple)) else [v] for v in data.values()
    ]

    combinations = product(*alternatives)
    for combination in combinations:
        yield dict(zip(data.keys(), combination))


def options_string(p):
    """main.py -p string for a parameter dict."""
    return ";".join(f"{k}={v}" for k, v in p.items())


def grid_jobs():
    """(folder name, params, seed) for the constants above."""
    seeds = SEEDS or np.random.randint(0, 1e5, N_SEEDS)
    for p in get_combinations(params):
        key = hashlib.md5(options_string(p).encode(encoding="utf-8")).hexdigest()[:6]
        for seed in seeds:
            yield f"{base_name}_{key}_{int(seed):06d}", p, int(seed)


def variant_jobs(config):
    """(folder name, params, seed) for a --variants configuration."""
    variants = {"base": {}, **config["variants"]}
    for name, overrides in variants.items():
        p = {**config["base"], **overrides}
        for seed in config["seeds"]:
            yield f"{config['base_name']}_{name}_{int(seed):06d}", p, int(seed)


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--variants", help="JSON file with a one-factor sweep.")
parser.add_argument("--dry-run", action="store_true",
                    help="Print the commands without running them.")
args = parser.parse_args()

if args.variants:
    with open(args.variants) as f:
        config = json.load(f)
    jobs = list(variant_jobs(config))
    max_processes = config.get("max_processes", MAX_PROCESSES)
    wandb = "-w" if config.get("wandb", False) else ""
else:
    jobs = list(grid_jobs())
    max_processes = MAX_PROCESSES
    wandb = "-w" if WANDB else ""

processes = []

orig_path = os.path.dirname(os.path.realpath(__file__))

for process_name, p, seed in jobs:
    cmd_str = (
        f"nohup python -u {orig_path}/main.py "
        f"-r {process_name} "
        f"-p {shlex.quote(options_string(p))} -s {seed} {wandb} "
    )
    print(f"Running: {cmd_str}\n\n" if not args.dry_run else cmd_str)
    if args.dry_run:
        continue

    # If max_processes are running, wait until any of them finishes.
    while len(processes) == max_processes:
        processes = [pr for pr in processes if pr.poll() is None]
        if len(processes) == max_processes:
            time.sleep(5)

    os.makedirs(process_name, exist_ok=True)
    processes.append(subprocess.Popen(cmd_str, cwd=process_name, shell=True))

# wait for all processes
exit_codes = [p.wait() for p in processes]
