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

from model.agent import Agent
from model.offline_controller import OfflineController
from params import Parameters
from plotter import FoveaPlotter, MapsPlotter


es = EyeSim


def signal_handler(signum, frame):
    """
    Handle incoming signals to gracefully terminate the program.

    Parameters:
    - signum (int): Signal number.
    - frame (frame object): Current stack frame.
    """
    signal.signal(signum, signal.SIG_IGN)
    wandb.finish()
    sys.exit(0)


class Logger:
    """
    Logger class to write to a file.

    Attributes:
    - filename (str): The name of the file to which the logs are written.
    """

    def __init__(self, filename):
        self.filename = filename

    def __call__(self, log):
        """
        Log to the file and stdout.

        Parameters:
        - log (any): The log.
        """
        with open(self.filename, "a") as file:
            print(log, file=file)
        print(log)


def setup_environment(seed, params):
    """
    Set up the gym environment with a specified seed.

    Parameters:
    - seed (int): Random seed to use for environment initialization.
    - params (Parameters): Parameters object containing 'env_name'.

    Returns:
    - env (gym.Env): Gym environment object.
    """
    env = gym.make(params.env_name, colors=True)
    env = env.unwrapped
    env.set_seed(seed)
    return env


def setup_agent(env, params, seed):
    """
    Set up the agent with environment and specific parameters.

    Parameters:
    - env (gym.Env): Gym environment object.
    - params (Parameters): Parameters object with 'agent_sampling_threshold'.
    - seed (int): Random seed for agent setup.

    Returns:
    - agent (Agent): Initialized Agent object.
    """
    return Agent(
        env, sampling_threshold=params.agent_sampling_threshold, seed=seed
    )


def setup_offline_controller(file_path, env, params, seed):
    """
    Set up the offline controller, loading from file if it exists.

    Parameters:
    - file_path (str): Path to check for existing controller data.
    - env (gym.Env): Gym environment object.
    - params (Parameters): Parameters for the offline controller.
    - seed (int): Random seed for controller setup.

    Returns:
    - off_controller (OfflineController): Loaded or new OfflineController
      object.

    """
    if os.path.exists(file_path):
        return OfflineController.load(file_path, env, params, seed)
    else:
        return OfflineController(env, params, seed)


def run_epoch(agent, env, off_control, params, epoch, log):
    """
    Perform multiple episodes of simulation for a single epoch.

    Parameters:
    - agent (Agent): Agent object to execute actions.
    - env (gym.Env): Gym environment object.
    - off_control (OfflineController): OfflineController for state management.
    - params (Parameters): Parameters object containing simulation settings.
    - epoch (int): Current epoch number for logging and control.
    - log (Logger): controls logging
    """
    for episode in range(params.episodes):
        info = run_episode(agent, env, off_control, params, episode, epoch)
        log(
            f"Epoch: {epoch:<5d} "
            f"Episode: {episode:<3d} "
            f"type:{info['world']:10s}"
        )


def run_episode(agent, env, off_control, params, episode, epoch):
    """
    Execute a single episode of simulation.

    Parameters:
    - agent (Agent): Agent object.
    - env (gym.Env): Gym environment object.
    - off_control (OfflineController): OfflineController object.
    - params (Parameters): Parameters object with simulation details.
    - episode (int): Current episode number.
    - epoch (int): Current epoch number.
    """
    env.init_world(world=0 if epoch % 100 < params.triangles_percent else 1)
    _, env_info = env.reset()

    plt_enabled = (
        params.plot_sim
        and episode == params.episodes - 1
        and is_plotting_epoch(epoch, params)
    )

    fovea_plotter = FoveaPlotter(env, offline=True) if plt_enabled else None

    action = np.zeros(env.action_space.shape)

    for saccade_idx in range(params.saccade_num):
        execute_saccade(
            agent,
            env,
            off_control,
            params,
            action,
            episode,
            saccade_idx,
            fovea_plotter,
        )

    if plt_enabled:
        save_simulation_gif(fovea_plotter, epoch)

    return env_info


def execute_saccade(
    agent,
    env,
    off_control,
    params,
    action,
    episode,
    saccade_idx,
    fovea_plotter,
):
    """
    Execute the attentional saccade phase of an episode.

    Parameters:
    - agent (Agent): Agent object.
    - env (gym.Env): Gym environment object.
    - off_control (OfflineController): OfflineController object.
    - params (Parameters): Parameters object.
    - action (np.ndarray): Initial action configuration.
    - episode (int): Current episode number.
    - saccade_idx (int): Current saccade index.
    - fovea_plotter (FoveaPlotter or None): Optional plotter for visual output.
    """
    observation, *_ = env.step(np.zeros(params.action_size))

    for time_step in range(params.saccade_time):
        if time_step == int(0.5 * params.saccade_time):
            saccade = self.generate_saccade(observation["FOVEA"])
            agent.set_parameters(saccade)

        observation, *_ = env.step(action)
        action, saliency_map, salient_point = agent.get_action(observation)

        if fovea_plotter:
            fovea_plotter.step(
                saliency_map, salient_point, agent.attentional_mask
            )

        state = {
            "world": env.world,
            "vision": observation["FOVEA"],
            "action": action,
            "attention": np.copy(agent.params),
        }
        off_control.record_states(episode, saccade_idx, time_step, state)


def is_plotting_epoch(epoch, params):
    """
    Determine if the current epoch is a plotting epoch.

    Parameters:
    - epoch (int): Current epoch number.
    - params (Parameters): Parameters object containing
      'plotting_epochs_interval' and 'epochs'.

    Returns:
    - is_plotting (bool): True if the current epoch is a plotting epoch;
      otherwise False.

    """
    return (
        epoch % params.plotting_epochs_interval == 0
        or epoch == params.epochs - 1
    )


def save_simulation_gif(fovea_plotter, epoch):
    """
    Save the simulation output as a GIF and log it to WandB.

    Parameters:
    - fovea_plotter (FoveaPlotter): FoveaPlotter object for visualizing data.
    - epoch (int): Current epoch number.
    """
    gif_file = f"sim_{epoch:04d}"
    fovea_plotter.close(gif_file)
    wandb.log(
        {"Simulations": wandb.Video(f"{gif_file}.gif", format="gif")},
        step=epoch,
    )


def main(params):
    """
    Main function to execute the simulation process.
    """
    main_log = Logger("log")

    signal.signal(signal.SIGINT, signal_handler)

    plt.ion()
    plt.close("all")

    torch.manual_seed(seed)

    env = setup_environment(seed, params)
    agent = setup_agent(env, params, seed)
    off_control = setup_offline_controller(
        "off_control_store", env, params, seed
    )

    if params.plot_maps:
        maps_plotter = MapsPlotter(env, off_control, offline=True)

    for epoch in range(off_control.epoch, off_control.epoch + params.epochs):
        main_log(f"epoch: {epoch}")
        off_control.epoch = epoch
        off_control.reset_states()

        run_epoch(agent, env, off_control, params, epoch, main_log)
        off_control.filter_salient_states()

        # count world types
        world_dict = {"triangle": 0, "square": 0}
        for idx in np.array(off_control.filtered_idcs).T:
            world = int(off_control.world_states[idx[0], idx[1], idx[2]][0])
            if env.world_labels[world] == "triangle":
                world_dict["triangle"] += 1
            elif env.world_labels[world] == "square":
                world_dict["square"] += 1

        main_log(
            f"triangles: {world_dict['triangle']}, "
            f"squares: {world_dict['square']}"
        )

        off_control.update()

        # Logs

        # Log to file
        main_log(f"comp: {off_control.competence.detach().cpu().numpy()}")

        # log to wandb
        wandb.log(
            dict(
                competence=off_control.competence, **off_control.weight_change
            ),
            step=epoch,
        )

        if params.plot_maps:
            maps_plotter.step()
            if is_plotting_epoch(epoch, params):
                save_maps_gif(maps_plotter, epoch)
                maps_plotter = MapsPlotter(env, off_control, offline=True)

        off_control.save("off_control_store")


def save_maps_gif(maps_plotter, epoch):
    """
    Save and log the maps as both GIF and PNG files.

    Parameters:
    - maps_plotter (MapsPlotter): MapsPlotter object for visualizing maps.
    - epoch (int): Current epoch number.
    """
    file = f"maps_{epoch:04d}"
    maps_plotter.close(file)
    wandb.log(
        {
            "history": wandb.Image(f"{file}.gif"),
            "last": wandb.Image(f"{file}.png"),
        },
        step=epoch,
    )


if __name__ == "__main__":

    matplotlib.use("agg")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set the seed for random number generation.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="base",
        help="A string describing this particular simulation",
    )
    parser.add_argument(
        "--param_list",
        type=str,
        default=None,
        help=(
            "Specify custom parameters with the format: "
            "'param1=value1;param2=value2;...'."
        ),
    )

    args = parser.parse_args()

    params = Parameters()
    seed = args.seed
    variant = args.variant
    param_list = args.param_list

    params.string_to_params(param_list)
    params.save("loaded_params")

    seed_str = str(seed).replace(".", "_")
    decaying_speed_str = str(params.decaying_speed).replace(".", "_")
    local_decaying_speed_str = str(params.local_decaying_speed).replace(
        ".", "_"
    )

    def format_scalar(x):
        return f"{x:06.3f}".replace(".", "")

    params.init_name = (
        "sim_"
        f"{variant}_"
        f"{str(hex(np.abs(hash(params))))[:6]}_"
        f"s_{seed_str}_"
        f"m_{format_scalar(params.match_std)}_"
        f"a_{format_scalar(params.anchor_std)}_"
        f"d_{format_scalar(params.decaying_speed)}_"
        f"l_{format_scalar(params.local_decaying_speed)}"
    )

    with open("NAME", "w") as fname:
        fname.write(f"{params.init_name}\n")

    wandb.init(
        project=params.project_name,
        entity=params.entity_name,
        name=params.init_name,
    )

    main(params)

    wandb.finish()
