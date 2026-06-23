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

SEEDS = [93581]
WANDB = False 
N_SEEDS = 1
MAX_PROCESSES = 1
base_name = "test"

params = dict(
    base_match_sigma=2,
    match_sigma=2,
    base_internal_sigma=0.1,
    cum_match_stop_th=1.0,
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


seeds = SEEDS if SEEDS is not None else np.random.randint(0, 1e5, N_SEEDS)
wandb_flag = "-w" if WANDB else ""

processes = []

orig_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(os.getcwd(), "simulations")

for p in get_combinations(params):
    for seed in seeds:
        # If MAX_PROCESSES reached, wait until all of them finish.
        if len(processes) == MAX_PROCESSES:
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
            f"python {orig_path}/main.py "
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
