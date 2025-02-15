# %% IMPORTS
import argparse
import os
import signal
import sys

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from merge_gifs import merge_gifs
from model.agent import Agent
from model.offline_controller import OfflineController
from params import Parameters
from plotter import FoveaPlotter, MapsPlotter


def signal_handler(signum, frame):
    signal.signal(signum, signal.SIG_IGN)  # ignore additional signals
    wandb.finish()
    sys.exit(0)


def init_environment(params, seed):
    env = gym.make(params.env_name)
    env = env.unwrapped
    env.set_seed(seed)
    env.rotation = 0.0
    return env


def load_offline_controller(file_path, env, params, seed):
    if os.path.exists(file_path):
        return OfflineController.load(file_path, env, params, seed)
    return OfflineController(env, params, seed)


def execute_simulation(agent, off_control, env, params, is_plotting_epoch):
    plotters = []
    for episode in range(params.episodes):
        env.init_world(world=env.rng.choice([0, 1]))
        observation, _ = env.reset()

        fovea_plotter, maps_plotter = setup_plotters(
            env, off_control, params, is_plotting_epoch
        )
        plotters.append([fovea_plotter, maps_plotter])

        run_episode(
            agent,
            env,
            off_control,
            observation,
            params,
            is_plotting_epoch,
            fovea_plotter,
            maps_plotter,
            episode,
        )

        if is_plotting_epoch:
            log_simulations(params, episode, fovea_plotter, maps_plotter)

        print(f"Episode: {episode}")

    return plotters


def setup_plotters(env, off_control, params, is_plotting_epoch):
    fovea_plotter, maps_plotter = None, None
    if is_plotting_epoch:
        if params.plot_sim:
            fovea_plotter = FoveaPlotter(env, offline=True)
        if params.plot_maps:
            maps_plotter = MapsPlotter(env, off_control, offline=True)
    return fovea_plotter, maps_plotter


def run_episode(
    agent,
    env,
    off_control,
    observation,
    params,
    is_plotting_epoch,
    fovea_plotter,
    maps_plotter,
    episode,
):
    # TODO: redundnt
    condition = observation["FOVEA"].copy()
    saccade, goal = off_control.get_action_from_condition(condition)
    agent.set_parameters(saccade)

    for time_step in range(params.saccade_time * params.saccade_num):
        condition = observation["FOVEA"].copy()
        saccade, goal = off_control.get_action_from_condition(condition)

        if time_step % 1 == 0:
            print("ts: ", time_step)
            agent.set_parameters(saccade)

        update_environment_position(env, time_step, params)

        action, saliency_map, salient_point = agent.get_action(observation)
        observation, *_ = env.step(action)

        if is_plotting_epoch:
            update_plotters(
                fovea_plotter,
                maps_plotter,
                saliency_map,
                salient_point,
                agent,
                goal,
            )


def update_environment_position(env, time_step, params):
    if time_step % 10 == 0:
        pos, rot = env.get_position_and_rotation()
        pos_trj_angle = (
            5
            * np.pi
            * (time_step / (params.saccade_time * params.saccade_num))
        )
        pos += 10 * np.array([np.cos(pos_trj_angle), np.sin(pos_trj_angle)])
        rot += pos_trj_angle
        env.update_position_and_rotation(pos, rot)


def update_plotters(
    fovea_plotter, maps_plotter, saliency_map, salient_point, agent, goal
):
    if fovea_plotter:
        fovea_plotter.step(saliency_map, salient_point, agent.attentional_mask)
    if maps_plotter:
        maps_plotter.step(goal)


def log_simulations(params, episode, fovea_plotter, maps_plotter):
    if params.plot_sim:
        gif_file = f"sim_test_{episode:04d}"
        fovea_plotter.close(gif_file)
        wandb.log(
            {"Simulation": wandb.Video(f"{gif_file}.gif", format="gif")},
            step=episode,
        )

    if params.plot_maps:
        gif_file = f"maps_test_{episode:04d}"
        maps_plotter.close(gif_file)
        wandb.log(
            {"Maps": wandb.Video(f"{gif_file}.gif", format="gif")},
            step=episode,
        )

    if params.plot_sim and params.plot_maps:
        gif_file = f"merged_test_{episode:04d}"
        merge_gifs(fovea_plotter.vm.frames, maps_plotter.vm.frames, gif_file)


def test(params, seed):
    signal.signal(signal.SIGINT, signal_handler)
    plt.ion()
    plt.close("all")
    torch.manual_seed(seed)

    env = init_environment(params, seed)
    agent = Agent(
        env, sampling_threshold=params.agent_sampling_threshold, seed=seed
    )

    file_path = "off_control_store"
    off_control = load_offline_controller(file_path, env, params, seed)

    for epoch in range(off_control.epoch, off_control.epoch + params.epochs):
        off_control.epoch = epoch
        off_control.reset_states()

        is_plotting_epoch = (epoch % params.plotting_epochs_interval == 0) or (
            epoch == params.epochs - 1
        )
        plotters = execute_simulation(
            agent, off_control, env, params, is_plotting_epoch
        )

        print(f"Epoch: {epoch}")

    return plotters


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set the seed for random number generation.",
    )
    parser.add_argument(
        "--decaying_speed",
        type=float,
        default=None,
        help="Speed at which decay occurs",
    )
    parser.add_argument(
        "--local_decaying_speed",
        type=float,
        default=None,
        help="Local speed at which decay occurs",
    )

    return parser.parse_args()


def update_params_from_args(params, args):
    params.decaying_speed = args.decaying_speed or params.decaying_speed
    params.local_decaying_speed = (
        args.local_decaying_speed or params.local_decaying_speed
    )


def format_name(param_name, value):
    return f"{param_name}_{str(value).replace('.', '_')}"


def main():
    matplotlib.use("agg")

    # Parse arguments
    args = parse_arguments()

    # Create an instance of Parameters with default or custom values
    params = Parameters()
    seed = args.seed

    # Update parameter values from args
    update_params_from_args(params, args)

    # Set additional parameters
    params.plot_maps = True
    params.plot_sim = True
    params.epochs = 1
    params.saccade_num = 4
    params.episodes = 3
    params.plotting_epochs_interval = 1

    # Generate initial name without dots or special characters
    seed_str = format_name("seed", seed)
    decay_str = format_name("decay", params.decaying_speed)
    local_decay_str = format_name("localdecay", params.local_decaying_speed)

    params.init_name = f"test_{seed_str}_{decay_str}_{local_decay_str}"

    # Initialize Weights & Biases logging
    wandb.init(
        project=params.project_name,
        entity=params.entity_name,
        name=params.init_name,
    )

    test(params, seed)

    wandb.finish()


if __name__ == "__main__":
    main()
