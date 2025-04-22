# %% IMPORTS

import argparse
import os
import signal
import sys

import EyeSim
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from slugify import slugify

from merge_gifs import merge_gifs
from model.agent import Agent
from model.offline_controller import OfflineController
from model.recurrent_generative_model import RecurrentGenerativeModel
from params import Parameters
from plotter import FoveaPlotter, MapsPlotter


_ = EyeSim  # avoid fixer erase EyeSim import


def signal_handler(signum, frame):
    signal.signal(signum, signal.SIG_IGN)  # ignore additional signals
    wandb.finish()
    sys.exit(0)


def init_environment(params, seed):
    env = gym.make(params.env_name, colors=True)
    env = env.unwrapped
    env.set_seed(seed)
    env.rotation = 0.0
    return env


def load_offline_controller(file_path, env, params, seed):
    if os.path.exists(file_path):
        return OfflineController.load(file_path, env, params, seed)
    return OfflineController(env, params, seed)


def execute_simulation(
    agent,
    off_control,
    env,
    params,
    is_plotting_epoch,
    world,
    object_params,
):
    plotters = []
    for episode in range(params.episodes):

        print(world)

        if world is None:
            world_id = env.rng.choice([0, 1])
        else:
            world_id = np.argwhere(
                [label == world for label in env.world_labels]
            )[0][0]

        env.init_world(world=world_id, object_params=object_params)
        observation, info = env.reset()
        env.info = info

        for k, v in info.items():
            print(f"{k}: {v}", end="  ")
        print()

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
            log_simulations(
                params,
                episode,
                fovea_plotter,
                maps_plotter,
                env.info,
            )

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

    gen_goal = None
    off_control.magnitude = 1

    for time_step in range(params.saccade_time * params.saccade_num):
        if time_step % 4 == 0:
            print("ts: ", time_step)

            # Get info for saccade
            condition = observation["FOVEA"].copy()
            # Compute saccade
            saccade, goal = off_control.get_action_from_condition(
                condition, gen_goal, off_control.magnitude
            )

            off_control.magnitude += params.magnitude_decay * (
                0 - off_control.magnitude
            )

            # Recurrent model step (next saccade prediction)
            gen_goal = off_control.recurrent_model.step(goal.squeeze())
            # Trigger saccade
            agent.set_parameters(saccade)

            off_control.goals["world"].append(env.info["world"])
            off_control.goals["angle"].append(env.info["angle"])
            off_control.goals["position"].append(env.info["position"])
            saccade_id = f"{episode:04d}-{time_step:04d}"
            off_control.goals["saccade_id"].append(saccade_id)
            off_control.goals["goal"].append(goal)
            off_control.goals["gen_goal"].append(gen_goal)

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
    # if time_step % 10 == 0:
    #     pos, rot = env.get_position_and_rotation()
    #     pos_trj_angle = (
    #         5
    #         * np.pi
    #         * (time_step / (params.saccade_time * params.saccade_num))
    #     )
    #     pos += 10 * np.array([np.cos(pos_trj_angle), np.sin(pos_trj_angle)])
    #     rot += pos_trj_angle
    #     env.update_position_and_rotation(pos, rot)
    pass


def update_plotters(
    fovea_plotter, maps_plotter, saliency_map, salient_point, agent, goal
):
    if fovea_plotter:
        fovea_plotter.step(saliency_map, salient_point, agent.attentional_mask)
    if maps_plotter:
        maps_plotter.step(goal)


def log_simulations(params, episode, fovea_plotter, maps_plotter, info):

    tag = slugify(f"{info['world']}_{info['angle']}_{info['position']}")

    if params.plot_sim:
        gif_file = f"sim_test_{tag}"
        fovea_plotter.close(gif_file)
        wandb.log(
            {"Simulation": wandb.Video(f"{gif_file}.gif", format="gif")},
            step=episode,
        )

    if params.plot_maps:
        gif_file = f"maps_test_{tag}"
        maps_plotter.close(gif_file)
        wandb.log(
            {"Maps": wandb.Video(f"{gif_file}.gif", format="gif")},
            step=episode,
        )

    if params.plot_sim and params.plot_maps:
        gif_file = f"merged_test_{tag}"
        merge_gifs(
            fovea_plotter.vm.frames,
            maps_plotter.vm.frames,
            gif_file,
            frame_duration=80,
        )


def test(params, seed, world=None, object_params=None):
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
    off_control.recurrent_model = RecurrentGenerativeModel()

    for epoch in range(off_control.epoch, off_control.epoch + params.epochs):
        off_control.epoch = epoch
        off_control.reset_states()
        off_control.goals = {
            "world": [],
            "position": [],
            "angle": [],
            "saccade_id": [],
            "goal": [],
            "gen_goal": [],
        }

        is_plotting_epoch = (epoch % params.plotting_epochs_interval == 0) or (
            epoch == params.epochs - 1
        )
        plotters = execute_simulation(
            agent,
            off_control,
            env,
            params,
            is_plotting_epoch,
            world,
            object_params,
        )

        np.save(
            slugify(
                f"goals_{world}_"
                f"{object_params['pos']}_"
                f"{object_params['rot']:06.2f}"
            ),
            [off_control.goals],
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
        "--world",
        type=str,
        default=None,
        help="Set the world in the test. it canbe 'square' or 'triangle'",
    )

    parser.add_argument(
        "--posrot",
        nargs=3,
        type=float,
        metavar=("x", "y", "a"),
        default=[None, None, None],
        help="Set the iposition and rotation of the object in the world",
    )

    return parser.parse_args()


def format_name(param_name, value):
    return f"{param_name}_{str(value).replace('.', '_')}"


def main():
    matplotlib.use("agg")

    # Parse arguments
    args = parse_arguments()

    # Create an instance of Parameters with default or param_list values
    params = Parameters()
    try:
        params.load("loaded_params")
    except FileNotFoundError:
        print("no local parameters")

    seed = args.seed
    world = args.world

    object_params = (
        None
        if args.posrot[0] is None
        else {"pos": args.posrot[:2], "rot": args.posrot[2]}
    )

    # Set additional parameters
    params.plot_maps = True
    params.plot_sim = True
    params.epochs = 1
    params.saccade_num = 16
    params.episodes = 1
    params.plotting_epochs_interval = 1

    # Generate initial name without dots or special characters
    seed_str = format_name("seed", seed)

    params.init_name = f"test_{seed_str}"

    # Initialize Weights & Biases logging
    wandb.init(
        project=params.project_name,
        entity=params.entity_name,
        name=params.init_name,
    )

    test(params, seed, world, object_params)

    wandb.finish()


if __name__ == "__main__":
    main()
