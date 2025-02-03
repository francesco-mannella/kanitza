import os
import signal
import sys

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


def signal_handler(signum, frame):
    """
    Handle incoming signals to gracefully terminate the program.

    Parameters:
    - signum: Signal number.
    - frame: Current stack frame.
    """
    # Ignore additional signals and ensure any cleanup via wandb
    signal.signal(signum, signal.SIG_IGN)
    wandb.finish()
    sys.exit(0)


class Logger:
    """
    Logger class to write numbers to a file.

    Attributes:
    - filename: The name of the file to which the logs are written.
    """

    def __init__(self, filename):
        self.filename = filename

    def __call__(self, number):
        with open(self.filename, "a") as file:
            file.write(str(number) + "\n")


def setup_environment(seed, params):
    """
    Set up the gym environment with a specified seed.

    Parameters:
    - seed: Random seed to use for environment initialization.
    - params: Parameters object containing 'env_name'.

    Returns:
    - Gym environment object.
    """
    env = gym.make(params.env_name)
    env = env.unwrapped
    env.set_seed(seed)
    return env


def setup_agent(env, params, seed):
    """
    Set up the agent with environment and specific parameters.

    Parameters:
    - env: Gym environment object.
    - params: Parameters object with 'agent_sampling_threshold'.
    - seed: Random seed for agent setup.

    Returns:
    - Initialized Agent object.
    """
    return Agent(
        env, sampling_threshold=params.agent_sampling_threshold, seed=seed
    )


def setup_offline_controller(file_path, env, params, seed):
    """
    Set up the offline controller, loading from file if it exists.

    Parameters:
    - file_path: Path to check for existing controller data.
    - env: Gym environment object.
    - params: Parameters for the offline controller.
    - seed: Random seed for controller setup.

    Returns:
    - Loaded or new OfflineController object.
    """
    if os.path.exists(file_path):
        return OfflineController.load(file_path, env, params, seed)
    else:
        return OfflineController(env, params, seed)


def run_epoch(agent, env, off_control, params, epoch):
    """
    Perform multiple episodes of simulation for a single epoch.

    Parameters:
    - agent: Agent object to execute actions.
    - env: Gym environment object.
    - off_control: OfflineController for state management.
    - params: Parameters object containing simulation settings.
    - epoch: Current epoch number for logging and control.
    """
    for episode in range(params.episodes):
        run_episode(agent, env, off_control, params, episode, epoch)
        print(f"Episode: {episode}, Epoch: {epoch}")


def run_episode(agent, env, off_control, params, episode, epoch):
    """
    Execute a single episode of simulation.

    Parameters:
    - agent: Agent object.
    - env: Gym environment object.
    - off_control: OfflineController object.
    - params: Parameters object with simulation details.
    - episode: Current episode number.
    - epoch: Current epoch number.
    """
    env.init_world(world=env.rng.choice([0, 1]))
    _, env_info = env.reset()

    plt_enabled = (
        params.plot_sim
        and episode == params.episodes - 1
        and is_plotting_epoch(epoch, params)
    )

    fovea_plotter = FoveaPlotter(env, offline=True) if plt_enabled else None

    action = np.zeros(env.action_space.shape)
    saccades = off_control.generate_attentional_input(
        params.saccade_num
    )

    for saccade_idx, saccade in enumerate(saccades):
        execute_saccade(
            agent,
            env,
            off_control,
            params,
            action,
            saccade,
            episode,
            saccade_idx,
            fovea_plotter,
        )

    if plt_enabled:
        save_simulation_gif(fovea_plotter, epoch)


def execute_saccade(
    agent,
    env,
    off_control,
    params,
    action,
    saccade,
    episode,
    saccade_idx,
    fovea_plotter,
):
    """
    Execute the attentional saccade phase of an episode.

    Parameters:
    - agent: Agent object.
    - env: Gym environment object.
    - off_control: OfflineController object.
    - params: Parameters object.
    - action: Initial action configuration.
    - saccade: Current attentional saccade settings.
    - episode: Current episode number.
    - saccade_idx: Current saccade index.
    - fovea_plotter: Optional plotter for visual output.
    """
    for time_step in range(params.saccade_time):
        if time_step == int(0.5 * params.saccade_time):
            agent.set_parameters(saccade)

        observation, *_ = env.step(action)
        action, saliency_map, salient_point = agent.get_action(observation)

        if fovea_plotter:
            fovea_plotter.step(
                saliency_map, salient_point, agent.attentional_mask
            )

        state = {
            "vision": observation["FOVEA"],
            "action": action,
            "attention": np.copy(agent.params),
        }
        off_control.record_states(episode, saccade_idx, time_step, state)


def is_plotting_epoch(epoch, params):
    """
    Determine if the current epoch is a plotting epoch.

    Parameters:
    - epoch: Current epoch number.
    - params: Parameters object containing 'plotting_epochs_interval'
      and 'epochs'.

    Returns:
    - True if the current epoch is a plotting epoch; otherwise False.
    """
    return (
        epoch % params.plotting_epochs_interval == 0
        or epoch == params.epochs - 1
    )


def save_simulation_gif(fovea_plotter, epoch):
    """
    Save the simulation output as a GIF and log it to WandB.

    Parameters:
    - fovea_plotter: FoveaPlotter object for visualizing data.
    - epoch: Current epoch number.
    """
    gif_file = f"sim_{epoch:04d}"
    fovea_plotter.close(gif_file)
    wandb.log(
        {"Simulations": wandb.Video(f"{gif_file}.gif", format="gif")},
        step=epoch,
    )


def main():
    """
    Main function to execute the simulation process.
    """
    competence_log = Logger("comp")

    # Setup signal handling to gracefully terminate the program
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
        off_control.epoch = epoch
        off_control.reset_states()

        run_epoch(agent, env, off_control, params, epoch)

        competence_log(off_control.competence.detach().cpu().numpy())
        wandb.log({"competence": off_control.competence}, step=epoch)

        off_control.filter_salient_states()
        off_control.update_maps()

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
    - maps_plotter: MapsPlotter object for visualizing maps.
    - epoch: Current epoch number.
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

    # Use Agg backend for matplotlib
    matplotlib.use("agg")

    import argparse

    # Argument parser for command-line options
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

    args = parser.parse_args()

    params = Parameters()

    # Set up seed and decaying speed parameters from command-line arguments
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

    # Create a unique initialization name based on parameters
    seed_str = str(seed).replace(".", "_")
    decaying_speed_str = str(params.decaying_speed).replace(".", "_")
    local_decaying_speed_str = str(params.local_decaying_speed).replace(
        ".", "_"
    )

    params.init_name = (
        f"sim_seed_{seed_str}_"
        f"decay_{decaying_speed_str}_"
        f"localdecay_{local_decaying_speed_str}"
    )

    # Initialize wandb for experiment tracking
    wandb.init(
        project=params.project_name,
        entity=params.entity_name,
        name=params.init_name,
    )

    # Run the main simulation
    main()

    # Finish the wandb session
    wandb.finish()
