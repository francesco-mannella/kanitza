# %% IMPORTS

import argparse
import os
import signal
import sys

import EyeSim
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import torch
import wandb

from merge_gifs import merge_gifs
from model.offline_controller import OfflineController
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

    for time_step in range(params.saccade_time * params.saccade_num):
        condition = observation["FOVEA"].copy()
        saccade, goal = off_control.get_action_from_condition(condition)

        if time_step % 4 == 0:
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


def execute_single_frame(
    env,
    off_control,
    params,
    points,
):

    maps_plotter = MapsPlotter(env, off_control, offline=True)
    maps_plotter.step()
    maps_plotter.fig.show()

    points = torch.tensor(points).float()

    num_points = points.shape[0]
    fig, axes = plt.subplots(2, num_points)
    generated_conds = off_control.visual_conditions_map.backward(
        points, params.neighborhood_modulation_baseline
    )
    generated_effects = off_control.visual_effects_map.backward(
        points, params.neighborhood_modulation_baseline
    )
    generated_conds = (
        generated_conds.reshape(-1, 16, 16, 3)
        .cpu()
        .detach()
        .numpy()[:, ::-1]
        .transpose(0, 1, 2, 3)
        # .transpose(0, 2, 1, 3)[:, ::-1]
    )
    generated_effects = (
        generated_effects.reshape(-1, 16, 16, 3)
        .cpu()
        .detach()
        .numpy()[:, ::-1]
        .transpose(0, 1, 2, 3)
        # .transpose(0, 2, 1, 3)[:, ::-1]
    )

    for i, point in enumerate(points):
        axes[0, i].imshow(generated_conds[i])
        axes[1, i].imshow(generated_effects[i])
    input()


def test_frame_generation(params, seed, world=None, object_params=None):
    signal.signal(signal.SIGINT, signal_handler)
    plt.ion()
    plt.close("all")
    torch.manual_seed(seed)

    env = init_environment(params, seed)
    file_path = "off_control_store"
    off_control = load_offline_controller(file_path, env, params, seed)

    off_control.reset_states()

    points = [
        [4, 5],
        [9, 9],
    ]

    execute_single_frame(
        env,
        off_control,
        params,
        points,
    )


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
    matplotlib.use("QtAgg")

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
    params.saccade_num = 8
    params.episodes = 1
    params.plotting_epochs_interval = 1

    # Generate initial name without dots or special characters
    seed_str = format_name("seed", seed)

    params.init_name = f"test_{seed_str}"

    test_frame_generation(params, seed, world, object_params)


if __name__ == "__main__":
    main()
