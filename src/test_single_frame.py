# %% IMPORTS
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gymnasium as gym
import wandb
import signal
import sys, os, re
import torch

from model.agent import Agent, gaussian_mask
from plotter import FoveaPlotter, MapsPlotter
from model.offline_controller import OfflineController
from params import Parameters


def signal_handler(signum, frame):
    signal.signal(signum, signal.SIG_IGN)  # ignore additional signals
    wandb.finish()
    sys.exit(0)


class Logger:
    def __init__(self, filename):
        self.filename = filename

    def __call__(self, number):
        with open(self.filename, 'a') as file:
            file.write(str(number) + '\n')


def collect_frames(env, agent):
    n_frames = 200

    agent.set_parameters(np.random.rand(2))
    frames = []

    for t in range(n_frames):
        if t % 100 == 0:
            env.init_world(world=env.rng.choice([0,1]))
            observation, env_info = env.reset()
        if t % 10 == 5:
            agent.set_parameters(np.random.rand(2))
        if t % 10 == 0:
            frames.append(observation['FOVEA'].copy())

        action, saliency_map, salient_point = agent.get_action(observation)
        observation, *_ = env.step(action)

    return np.stack(frames)


def test_frames(agent, off_control, frames):

    std = off_control.params.neighborhood_modulation_baseline
    frames = torch.tensor(frames.reshape(-1, 16 * 16 * 3) / 255.0)
    cond_map = off_control.visual_conditions_map
    eff_map = off_control.visual_effects_map
    att_map = off_control.attention_map

    cond_map(frames, std)
    reps = cond_map.get_representation()
    focuses = att_map.backward(reps, std)

    return reps.cpu().detach().numpy(), focuses.cpu().detach().numpy()


if __name__ == '__main__':

    # matplotlib.use("agg")

    import argparse

    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the 'seed' argument
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Set the seed for random number generation.',
    )
    parser.add_argument(
        '--decaying_speed',
        type=float,
        default=None,
        help='Speed at which decay occurs',
    )

    parser.add_argument(
        '--local_decaying_speed',
        type=float,
        default=None,
        help='Local speed at which decay occurs',
    )

    # Parse the arguments
    args = parser.parse_args()

    # Create an instance of Parameters with default or custom values
    params = Parameters()

    # Access the seed value
    seed = args.seed

    params.decaying_speed = (
        args.decaying_speed
        if args.decaying_speed is not None
        else params.decaying_speed
    )
    params.local_decaying_speed = (
        args.local_decaying_speed
        if args.local_decaying_speed is not None
        else params.local_decaying_speed
    )
    params.plot_maps = False
    params.plot_sim = False

    # Ensure values are converted to strings free of dots or special characters
    seed_str = str(seed).replace('.', '_')
    decaying_speed_str = str(params.decaying_speed).replace('.', '_')
    local_decaying_speed_str = str(params.local_decaying_speed).replace(
        '.', '_'
    )

    # Include seed, decaying_speed, and decay in init_name without dots or special characters
    params.init_name = f'test_seed_{seed_str}_decay_{decaying_speed_str}_localdecay_{local_decaying_speed_str}'

    # Configure the environment
    env = gym.make(params.env_name)
    env = env.unwrapped
    env.set_seed(seed)
    env.reset()

    # configure the agent
    agent = Agent(
        env,
        sampling_threshold=params.agent_sampling_threshold,
        seed=seed,
    )

    # configure the controller
    file_path = 'off_control_store'

    # Check if the offline control data file exists
    if os.path.exists(file_path):
        # Load the offline controller from the file if it exists
        off_control = OfflineController.load(file_path, env, params, seed)
    else:
        # Initialize a new offline controller
        off_control = OfflineController(env, params, seed)

    frames = collect_frames(env, agent)
    reps, focuses = test_frames(agent, off_control, frames)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for ax in axes:
        ax.set_axis_off()
    maps_plotter = MapsPlotter(
        env, off_control, offline=True, video_frame_duration=1000
    )
    for i, frame in enumerate(frames):
        maps_plotter.step(reps[i])
        axes[0].clear()
        axes[0].imshow(frame)
        fig.canvas.draw()
        plt.pause(5)
