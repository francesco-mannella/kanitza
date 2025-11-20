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
from skimage.transform import resize

from model.agent import Agent
from model.offline_controller import OfflineController
from model.visual_processing import SaliencyMap
from params import Parameters
from plotter import FoveaPlotter, MapsPlotter


es = EyeSim


# TODO: function for debug
def ascii_imshow(matrix, nrows, ncols):
    """Prints an ASCII art representation of a numpy array."""
    levels = (
        r"$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|"
        r"()1{}[]?-_+~<>i!lI;:,\"^`'."
    )[::-1]

    matrix = resize(matrix, (nrows, ncols), anti_aliasing=True)
    min_val = matrix.min()
    max_val = matrix.max()
    value_range = max_val - min_val

    print()
    for row in matrix:
        for value in row:
            if value_range == 0:
                index = 0
            else:
                index = int(
                    (value - min_val) / value_range * (len(levels) - 1)
                )
            print(levels[index], end="")
        print()
    print()


def signal_handler(signum, frame, params):
    """
    Handle incoming signals to gracefully terminate the program.

    Parameters:
    - signum (int): Signal number.
    - frame (frame object): Current stack frame.
    """
    signal.signal(signum, signal.SIG_IGN)
    if params.use_wandb:
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


class SimulationManager:
    """Manages the simulation, including environment, agent, and offline
        control.

    Args:
        params: Configuration parameters for the simulation.
        seed: Random seed for reproducibility.
    """


def setup_agent(env, params, seed):
    """
    Set up the agent with environment and specific parameters.

    def run_epoch(self, epoch, log):
        """Runs a full epoch, iterating through all episodes.

        Args:
            epoch: Current epoch number.
            log: Logging function.
        """
        for episode in range(self.params.episodes):
            info = self.run_episode(episode, epoch)
            log(
                f"Epoch: {epoch:<5d} "
                f"Episode: {episode:<3d} "
                f"type:{info['world']:10s}"
            )

    def run_episode(self, episode, epoch):
        """Runs a single episode, handling plotting and environment reset.

        Args:
            episode: Current episode number.
            epoch: Current epoch number.

        Returns:
            dict: Information about the environment after reset.
        """
        # Select world type based on epoch and triangles_percent parameter
        self.env.init_world(
            world=0 if epoch % 100 < self.params.triangles_percent else 1,
        )
        _, env_info = self.env.reset()
        # Enable plotting only for the last episode of plotting epochs
        plt_enabled = (
            self.params.plot_sim is True
            and episode == self.params.episodes - 1
            and self.is_plotting_epoch(epoch)
        )

        if self.params.online_plot:
            plt.close("all")

        fovea_plotter = (
            FoveaPlotter(self.env, offline=True)
            if plt_enabled or self.params.online_plot
            else None
        )
        if fovea_plotter is not None:
            fovea_plotter.online = self.params.online_plot

        # Execute all saccades for this episode
        for saccade_idx in range(self.params.saccade_num):
            self.execute_saccade(episode, saccade_idx, fovea_plotter)
        if plt_enabled:
            self.save_simulation_gif(fovea_plotter, epoch)

        return env_info

    def execute_saccade(self, episode, saccade_idx, fovea_plotter):
        """Executes a single saccade, updating agent and offline controller.

        Args:
            episode: Current episode number.
            saccade_idx: Index of the current saccade.
            fovea_plotter: Optional plotter for visualization.
        """
        # Step once to get initial observation
        observation, *_ = self.env.step(np.zeros(self.params.action_size))
        # visual processing
        _, _, saliency = self.visual_map(observation["RETINA"])
        _, _, fovea = self.visual_map(observation["FOVEA"])

        competence = None
        saccade = (0.5, 0.5)
        attention = None
        salient_point, action = [0, 0], [0.0, 0.0]
        self.agent.set_parameters(saccade)
        # Iterate through all time steps of the saccade
        for time_step in range(self.params.saccade_time):
            observation, *_ = self.env.step(action)
            # visual processing
            _, _, saliency = self.visual_map(observation["RETINA"])
            _, _, fovea = self.visual_map(observation["FOVEA"])
            # At midpoint, generate new saccade and update agent parameters
            if time_step == int(0.5 * self.params.saccade_time):
                saccade, competence = self.off_control.generate_saccade(fovea)
                self.agent.set_parameters(saccade)
                attention = np.copy(saccade)
            else:
                # After midpoint, reset saccade if not at default
                if saccade is not None and not np.array_equal(
                    saccade, np.array([0.5, 0.5])
                ):
                    saccade = np.array([0.5, 0.5])
                    self.agent.set_parameters(saccade)

            action, saliency_map, salient_point = self.agent.get_action(
                saliency
            )
            # Update plotter if enabled
            if fovea_plotter:
                fovea_plotter.step(
                    fovea,
                    saliency_map,
                    salient_point,
                    self.agent.attentional_mask,
                )
                if fovea_plotter.online:
                    plt.pause(0.01)
            # Record state for offline controller
            state = {
                "world": self.env.world,
                "vision": fovea,
                "action": action,
                "attention": attention,
                "competence": competence,
            }

            row = " ".join(
                map(str, np.hstack([episode, time_step, fovea.flatten()]))
            )
            self.fovea_data.write(f"{row}\n")
            self.fovea_data.flush()

            self.off_control.record_states(
                episode, saccade_idx, time_step, state
            )

    def is_plotting_epoch(self, epoch):
        """Determines if the current epoch should trigger plotting.

        Args:
            epoch: Current epoch number.

        Returns:
            bool: True if plotting should occur, False otherwise.
        """
        return (
            epoch % self.params.plotting_epochs_interval == 0
            or epoch == self.params.epochs - 1
        )

    def save_simulation_gif(self, fovea_plotter, epoch):
        """Saves the simulation as a GIF and logs it if using wandb.

        Args:
            fovea_plotter: The plotter used for visualization.
            epoch: Current epoch number.
        """
        gif_file = f"sim_{epoch:04d}"
        fovea_plotter.close(gif_file)
        if self.params.use_wandb:
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

    sim_manager = SimulationManager(params, seed, "off_control_store")

    if params.plot_maps:
        maps_plotter = MapsPlotter(
            sim_manager.env, sim_manager.off_control, offline=True
        )

    for epoch in range(
        sim_manager.off_control.epoch,
        sim_manager.off_control.epoch + params.epochs,
    ):
        main_log(f"epoch: {epoch}")
        sim_manager.off_control.epoch = epoch
        sim_manager.off_control.reset_states()

        sim_manager.run_epoch(epoch, main_log)
        sim_manager.off_control.filter_salient_states()

        # count world types
        world_dict = {"triangle": 0, "square": 0}
        for idx in np.array(sim_manager.off_control.filtered_idcs).T:
            world = int(
                sim_manager.off_control.world_states[idx[0], idx[1], idx[2]][0]
            )
            if sim_manager.env.world_labels[world] == "triangle":
                world_dict["triangle"] += 1
            elif sim_manager.env.world_labels[world] == "square":
                world_dict["square"] += 1

        main_log(
            f"triangles: {world_dict['triangle']}, "
            f"squares: {world_dict['square']}"
        )

        sim_manager.off_control.update()

        # Logs

        # Log to file
        main_log(f"comp: {sim_manager.off_control.competence}")

        # log to wandb
        if params.use_wandb:
            wandb.log(
                dict(
                    competence=sim_manager.off_control.competence,
                    **sim_manager.off_control.weight_change,
                ),
                step=epoch,
            )

        if params.plot_maps:
            maps_plotter.step()
            if sim_manager.is_plotting_epoch(epoch):
                save_maps_gif(maps_plotter, epoch, params)
                maps_plotter = MapsPlotter(
                    sim_manager.env, sim_manager.off_control, offline=True
                )

        sim_manager.off_control.save("off_control_store")


def save_maps_gif(maps_plotter, epoch, params):
    """
    Save and log the maps as both GIF and PNG files.

    Parameters:
    - maps_plotter (MapsPlotter): MapsPlotter object for visualizing maps.
    - epoch (int): Current epoch number.
    """
    file = f"maps_{epoch:04d}"
    maps_plotter.close(file)
    if params.use_wandb:
        wandb.log(
            {
                "history": wandb.Image(f"{file}.gif"),
                "last": wandb.Image(f"{file}.png"),
            },
            step=epoch,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="Set the seed for random number generation.",
    )
    parser.add_argument(
        "-r",
        "--variant",
        type=str,
        default="base",
        help="A string describing this particular simulation",
    )
    parser.add_argument(
        "-p",
        "--param_list",
        type=str,
        default=None,
        help=(
            "Specify custom parameters with the format: "
            "'param1=value1;param2=value2;...'."
        ),
    )
    parser.add_argument(
        "-w",
        "--wandb",
        action="store_true",
        help="Enable wandb logging.",
    )
    parser.add_argument(
        "-o",
        "--online",
        action="store_true",
        help="Plot online",
    )
    return parser.parse_args()


if __name__ == "__main__":

    if torch.cuda.is_available():
        torch.set_default_device("cuda")

    args = parse_args()

    params = Parameters()
    seed = args.seed
    variant = args.variant
    params.use_wandb = args.wandb
    params.online_plot = args.online

    if not params.online_plot:
        matplotlib.use("agg")

    try:
        params.load("loaded_params")
    except FileNotFoundError:
        print("no further parameter file found.")
        if args.param_list is not None:
            print("reading parameters from the given list")
        param_list = args.param_list
        params.update(param_list)
        params.save("loaded_params")

    def format_scalar(x):
        return f"{x:06.3f}".replace(".", "")

    seed_str = str(seed).replace(".", "_")

    params.init_name = (
        f"{variant}"
    )

    with open("NAME", "w") as fname:
        fname.write(f"{params.init_name}\n")

    wandb.init(
        project=params.project_name,
        entity=params.entity_name,
        name=params.init_name,
        config=params._params_to_dict()
    )

    main(params)

    if params.use_wandb:
        wandb.finish()
