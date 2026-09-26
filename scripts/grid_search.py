"""Launch src/main.py over a grid of parameters, or over one-factor variants.

Usage: run from the folder that will hold the sweep (under tests/); runs
are created in ./simulations/<variant>/:
    python /path/to/scripts/grid_search.py [--seeds S ...] [--n-seeds N]
        [--max-processes P] [--name NAME] [-w] [--variants FILE.json]
        [--dry-run]

Grid mode (default): the `params` dict below is the grid; a list value is a
set of alternatives, any other value (strings included) is fixed, and a
parameter whose value is itself a list must be wrapped twice (e.g.
gabor_scales=[[1.0]]). Each combination and seed runs in
simulations/<name>_<slugified params>_<seed:06d>.

Variants mode (--variants FILE.json): a base configuration plus named
one-factor overrides:
    {"base_name": "screen", "seeds": [90902, 39973], "max_processes": 7,
     "wandb": false,
     "base": {<main.py parameters>},
     "variants": {"lr_0.03": {"maps_learning_rate": 0.03}, ...}}
"base" and every variant (base updated with its overrides) run for each
seed in simulations/<base_name>_<variant>_<seed:06d>. Values in the file
take precedence over --seeds, --max-processes, --name and -w.

In both modes a ./loaded_params in the sweep folder, if present, is copied
into each run folder as its base configuration (main.py applies the -p
overrides on top and records the result in final_parameters). A new run
starts as soon as a running one ends. Each simulation uses one CPU thread
(OMP_NUM_THREADS=1); runs slow down once they outnumber the physical
cores, but total throughput still grows up to about the number of
hardware threads minus one. Without -w, runs log to <run>/data_sim.
--dry-run prints the commands without running them.
"""
import argparse
import json
import os
import shlex
import shutil
import subprocess
import time
from itertools import product

import numpy as np
import slugify

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

params = dict(
    decaying_speed=[3.0, 3.25, 3.5, 3.75, 4.0],
    local_decaying_speed=1.0,
    match_std=8.0,
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
    for combination in product(*alternatives):
        yield dict(zip(data.keys(), combination))


def options_string(p):
    """main.py -p string for a parameter dict."""
    return ";".join(f"{k}={v}" for k, v in p.items())


def grid_jobs(base_name, seeds):
    """(run name, params, seed) for the `params` grid."""
    for p in get_combinations(params):
        key = slugify.slugify(options_string(p))
        for seed in seeds:
            yield f"{base_name}_{key}_{int(seed):06d}", p, int(seed)


def variant_jobs(config):
    """(run name, params, seed) for a --variants configuration."""
    variants = {"base": {}, **config["variants"]}
    for name, overrides in variants.items():
        p = {**config["base"], **overrides}
        for seed in config["seeds"]:
            yield f"{config['base_name']}_{name}_{int(seed):06d}", p, int(seed)


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--seeds", type=int, nargs="+", default=None)
parser.add_argument("--n-seeds", type=int, default=1)
parser.add_argument("--max-processes", type=int, default=1)
parser.add_argument("--name", type=str, default="test")
parser.add_argument("-w", "--wandb", action="store_true")
parser.add_argument("--variants", help="JSON file with a one-factor sweep.")
parser.add_argument("--dry-run", action="store_true",
                    help="Print the commands without running them.")
args = parser.parse_args()

if args.variants:
    with open(args.variants) as f:
        config = json.load(f)
    config.setdefault("base_name", args.name)
    if args.seeds is not None:
        config.setdefault("seeds", args.seeds)
    jobs = list(variant_jobs(config))
    max_processes = config.get("max_processes", args.max_processes)
    wandb_flag = "-w" if config.get("wandb", args.wandb) else ""
else:
    seeds = args.seeds if args.seeds is not None else np.random.randint(0, 1e5, args.n_seeds)
    jobs = list(grid_jobs(args.name, seeds))
    max_processes = args.max_processes
    wandb_flag = "-w" if args.wandb else ""

processes = []

script_dir = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(os.path.dirname(script_dir), "src")
data_path = os.path.join(os.getcwd(), "simulations")
local_params = os.path.join(os.getcwd(), "loaded_params")

for variant, p, seed in jobs:
    cmd_str = (
        f"python {src_path}/main.py "
        f"--variant {variant} "
        f"--seed {seed} "
        f"--param_list {shlex.quote(options_string(p))} "
        f"{wandb_flag}"
    )
    print(f"Running: {cmd_str}")
    if args.dry_run:
        continue

    # If max_processes are running, wait until any of them finishes.
    while len(processes) == max_processes:
        processes = [pr for pr in processes if pr.poll() is None]
        if len(processes) == max_processes:
            time.sleep(5)

    run_dir = os.path.join(data_path, variant)
    os.makedirs(run_dir, exist_ok=True)
    run_params = os.path.join(run_dir, "loaded_params")
    if os.path.exists(local_params) and not os.path.exists(run_params):
        shutil.copy(local_params, run_params)

    processes.append(subprocess.Popen(cmd_str, shell=True, cwd=run_dir))

# wait for all processes
exit_codes = [p.wait() for p in processes]
