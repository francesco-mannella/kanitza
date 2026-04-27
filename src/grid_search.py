import collections
import hashlib
import os
import subprocess
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


seeds = SEEDS or np.random.randint(0, 1e5, N_SEEDS)
wandb = "-w" if WANDB else ""


processes = []

orig_path = os.path.dirname(os.path.realpath(__file__))

for i, p in enumerate(get_combinations(params)):
    for seed in seeds:
        # If MAX_PROCESSES reached, wait until all of them finish.
        if len(processes) == MAX_PROCESSES:
            print("Waiting for the queue to clear...")
            for process in processes:
                process.wait()
            processes = []

        options_str = ""
        for k, v in p.items():
            options_str += f"{k}={v};"
        options_str = options_str[:-1]
        option_key = hashlib.md5(options_str.encode(encoding="utf-8")).hexdigest()[:6]
        options_str = f"-p '{options_str}'"

        process_name = f"{base_name}_{option_key}_{seed:06d}"

        base_cmd_str = (
            f"nohup python -u {orig_path}/main.py "
            f"-r {process_name} "
            f"{options_str} -s {seed} {wandb} "
        )
        cmd_str = base_cmd_str

        print(f"Running: {cmd_str}\n\n")

        os.makedirs(process_name, exist_ok=True)
        processes.append(subprocess.Popen(cmd_str, cwd=process_name, shell=True))

# wait for all processes
exit_codes = [p.wait() for p in processes]
