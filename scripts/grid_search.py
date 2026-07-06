import argparse
import collections.abc
import os
import shutil
import subprocess
from itertools import product

import numpy as np
import slugify

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

params = dict(
    decaying_speed=[2.0, 3.0, 5.0],
    local_decaying_speed=[1.0, 1.5],
    match_std=[2.0, 4.0],
)

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------


def get_combinations(data):
    """
    Generates all possible combinations of list elements from a dictionary.

    Args:
       data: A dictionary.

    Yields:
       A dictionary representing a single combination of elements.
    """
    for k, v in data.items():
        if not isinstance(v, collections.abc.Iterable):
            data[k] = [v]

    combinations = product(*[value for value in data.values()])
    for combination in combinations:
        yield dict(zip(data.keys(), combination))


parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="+", default=None)
parser.add_argument("--n-seeds", type=int, default=1)
parser.add_argument("--max-processes", type=int, default=1)
parser.add_argument("--name", type=str, default="test")
parser.add_argument("-w", "--wandb", action="store_true")
args = parser.parse_args()

seeds = args.seeds if args.seeds is not None else np.random.randint(0, 1e5, args.n_seeds)
wandb_flag = "-w" if args.wandb else ""
base_name = args.name

processes = []

script_dir = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(os.path.dirname(script_dir), "src")
data_path = os.path.join(os.getcwd(), "simulations")

for p in get_combinations(params):
    for seed in seeds:
        # If MAX_PROCESSES reached, wait until all of them finish.
        if len(processes) == args.max_processes:
            for process in processes:
                process.wait()
            processes = []

        param_list = ";".join(f"{k}={v}" for k, v in p.items())
        option_key = slugify.slugify(param_list)
        variant = f"{base_name}_{option_key}_{int(seed):06d}"

        run_dir = os.path.join(data_path, variant)
        os.makedirs(run_dir, exist_ok=True)

        local_params = os.path.join(os.getcwd(), "loaded_params")
        run_params = os.path.join(run_dir, "loaded_params")
        if os.path.exists(local_params) and not os.path.exists(run_params):
            shutil.copy(local_params, run_params)

        cmd_str = (
            f"python {src_path}/main.py "
            f"--variant {variant} "
            f"--name {base_name} "
            f"--seed {int(seed)} "
            f"--param_list '{param_list}' "
            f"{wandb_flag}"
        )

        print(f"Running: {cmd_str}")
        processes.append(subprocess.Popen(cmd_str, shell=True, cwd=run_dir))

# wait for all processes
exit_codes = [p.wait() for p in processes]
