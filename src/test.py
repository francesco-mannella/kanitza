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


#####
# %% IMPORTS


_ = EyeSim  # avoid fixer erase EyeSim import


def radians_to_degrees(radians):
    return np.array(radians) * (180.0 / np.pi)

def signal_handler(signum, frame):
    """Handles signals for graceful shutdown, e.g., on Ctrl+C."""
    signal.signal(signum, signal.SIG_IGN)  # ignore additional signals
    wandb.finish()
    sys.exit(0)


class SimulationTest:
    """
    A class for running and testing the eye movement simulation.
    It initializes the environment, agent, and offline controller,
    executes the simulation, and logs the results.
    """

    def __init__(self, params, seed, world=None, object_params=None):
        """Initializes the SimulationTest object.

        Args:
            params (Parameters): The parameters for the simulation.
            seed (int): The random seed for reproducibility.
            world (str, optional): The name of the world to simulate.
                Defaults to None.
            object_params (dict, optional): Parameters for the object in the
                world. Defaults to None.
        """
        self.params = params
        self.seed = seed
        self.world = world
        self.object_params = object_params
        self.env = None
        self.agent = None
        self.off_control = None
        self.plotters = []

    def init_environment(self):
        """Initializes the Gymnasium environment.

        Returns:
            gym.Env: The initialized Gymnasium environment.
        """
        env = gym.make(self.params.env_name, colors=True)
        env = env.unwrapped
        env.set_seed(self.seed)
        env.rotation = 0.0
        self.env = env
        return env

    def load_offline_controller(self, file_path):
        """Loads the offline controller from a file, or creates a new one if
        the file doesn't exist.

        Args:
            file_path (str): The path to the offline controller file.

        Returns:
            OfflineController: The loaded or newly created offline controller.
        """
        if os.path.exists(file_path):
            return OfflineController.load(
                file_path, self.env, self.params, self.seed
            )
        return OfflineController(self.env, self.params, self.seed)

    def execute_simulation(self, is_plotting_epoch):
        """Executes the simulation for a given number of episodes.

        Args:
            is_plotting_epoch (bool): A flag indicating whether to plot the
                results of this epoch.

        Returns:
            list: A list of plotters used during the simulation.
        """
        plotters = []
        for episode in range(self.params.episodes):

            print(self.world)

            if self.world is None:
                # If no specific world is specified, randomly choose one
                world_id = self.env.rng.choice([0, 1])
            else:
                # Determine the world ID based on the provided world name
                world_id = np.argwhere(
                    [label == self.world for label in self.env.world_labels]
                )[0][0]

            self.env.init_world(
                world=world_id, object_params=self.object_params
            )
            observation, info = self.env.reset()
            self.env.info = info

            for k, v in info.items():
                print(f"{k}: {v}", end="  ")
            print()

            fovea_plotter, maps_plotter = self.setup_plotters(
                is_plotting_epoch
            )
            plotters.append([fovea_plotter, maps_plotter])

            self.run_episode(
                observation,
                is_plotting_epoch,
                fovea_plotter,
                maps_plotter,
                episode,
            )

            if is_plotting_epoch:
                self.log_simulations(
                    episode,
                    fovea_plotter,
                    maps_plotter,
                    self.env.info,
                )

            print(f"Episode: {episode}")

        return plotters

    def setup_plotters(self, is_plotting_epoch):
        """Initializes the plotters based on the is_plotting_epoch flag.

        Args:
            is_plotting_epoch (bool): A flag indicating whether to plot the
                results of this epoch.

        Returns:
            tuple: A tuple containing the fovea plotter and maps plotter.
        """
        fovea_plotter, maps_plotter = None, None
        if is_plotting_epoch:
            if self.params.plot_sim:
                fovea_plotter = FoveaPlotter(self.env, offline=True)
            if self.params.plot_maps:
                maps_plotter = MapsPlotter(
                    self.env, self.off_control, offline=True
                )
        return fovea_plotter, maps_plotter

    def run_episode(
        self,
        observation,
        is_plotting_epoch,
        fovea_plotter,
        maps_plotter,
        episode,
    ):
        """Runs a single episode of the simulation.

        Args:
            observation (numpy.ndarray): The initial observation from the
                environment.
            is_plotting_epoch (bool): A flag indicating whether to plot the
                results of this epoch.
            fovea_plotter (FoveaPlotter): The fovea plotter object.
            maps_plotter (MapsPlotter): The maps plotter object.
            episode (int): The current episode number.
        """

        saccade = None
        for time_step in range(
            self.params.saccade_time * self.params.saccade_num
        ):

            if time_step % 4 == 0:
                print(f"ts: {time_step:>3d}  ")

                # Get info for saccade
                condition = observation["FOVEA"].copy()
                # Compute saccade
                saccade, goal = self.off_control.get_action_from_condition(
                    condition
                )

                # Trigger saccade
                self.agent.set_parameters(saccade)

                self.off_control.goals["world"].append(self.env.info["world"])
                self.off_control.goals["angle"].append(self.env.info["angle"])
                self.off_control.goals["position"].append(
                    self.env.info["position"]
                )
                saccade_id = f"{episode:04d}-{time_step:04d}"
                self.off_control.goals["saccade_id"].append(saccade_id)
                self.off_control.goals["goal"].append(goal)

            elif time_step % 4 == 1:

                # Reset saccade
                if saccade is not None and not np.array_equal(
                    saccade, np.array([0.5, 0.5])
                ):
                    saccade = np.array([0.5, 0.5])
                    self.agent.set_parameters(saccade)

            self.update_environment_position(time_step)
            action, saliency_map, salient_point = self.agent.get_action(
                observation
            )
            observation, *_ = self.env.step(action)

            if is_plotting_epoch and saliency_map is not None:
                self.update_plotters(
                    fovea_plotter,
                    maps_plotter,
                    saliency_map,
                    salient_point,
                    goal,
                )

    def update_environment_position(self, time_step):
        """Placeholder for updating the environment position during the
        simulation.

        Args:
            time_step (int): The current time step in the simulation.
        """
        # if time_step % 10 == 0:
        #     pos, rot = env.get_position_and_rotation()
        #     pos_trj_angle = (
        #         5
        #         * np.pi
        #         * (time_step / (params.saccade_time * params.saccade_num))
        #     )
        #     pos += 10 * np.array([np.cos(pos_trj_angle),
        #                            np.sin(pos_trj_angle)])
        #     rot += pos_trj_angle
        #     env.update_position_and_rotation(pos, rot)
        pass

    def update_plotters(
        self, fovea_plotter, maps_plotter, saliency_map, salient_point, goal
    ):
        """Updates the plotters with the latest data from the simulation.

        Args:
            fovea_plotter (FoveaPlotter): The fovea plotter object.
            maps_plotter (MapsPlotter): The maps plotter object.
            saliency_map (numpy.ndarray): The saliency map from the agent.
            salient_point (tuple): The salient point from the agent.
            goal (numpy.ndarray): The goal for the current time step.
        """
        if fovea_plotter:
            fovea_plotter.step(
                saliency_map, salient_point, self.agent.attentional_mask
            )
        if maps_plotter:
            maps_plotter.step(goal)

    def log_simulations(self, episode, fovea_plotter, maps_plotter, info):
        """Logs the simulation results, including videos of the fovea and maps.

        Args:
            episode (int): The current episode number.
            fovea_plotter (FoveaPlotter): The fovea plotter object.
            maps_plotter (MapsPlotter): The maps plotter object.
            info (dict): The environment information.
        """

        tag = slugify(f"{info['world']}_{info['angle']}_{info['position']}")

        if self.params.plot_sim:
            gif_file = f"sim_test_{tag}"
            fovea_plotter.close(gif_file)
            if self.params.wandb:
                wandb.log(
                    {
                        "Simulation": wandb.Video(
                            f"{gif_file}.gif", format="gif"
                        )
                    },
                    step=episode,
                )

        if self.params.plot_maps:
            gif_file = f"maps_test_{tag}"
            maps_plotter.close(gif_file)
            if self.params.wandb:
                wandb.log(
                    {"Maps": wandb.Video(f"{gif_file}.gif", format="gif")},
                    step=episode,
                )

        if self.params.plot_sim and self.params.plot_maps:
            gif_file = f"merged_test_{tag}"
            merge_gifs(
                fovea_plotter.vm.frames,
                maps_plotter.vm.frames,
                gif_file,
                frame_duration=80,
            )

    def test(self):
        """Runs the main test loop for the simulation.

        Returns:
            list: A list of plotters used during the simulation.
        """
        signal.signal(signal.SIGINT, signal_handler)
        plt.ion()
        plt.close("all")
        torch.manual_seed(self.seed)

        self.env = self.init_environment()
        self.agent = Agent(
            self.env,
            sampling_threshold=self.params.agent_sampling_threshold,
            seed=self.seed,
            attention_max_variance=self.params.attention_max_variance,
            attention_fixed_variance_prop=self.params.attention_fixed_variance_prop,
            attention_center_distance_variance_prop=self.params.attention_center_distance_variance_prop,
            attention_center_distance_slope=self.params.attention_center_distance_slope,
        )

        controller_path = "off_control_store"
        rnn_path = "rnn_store.npy"
        self.off_control = self.load_offline_controller(controller_path)
        self.off_control.recurrent_model = RecurrentGenerativeModel()
        self.off_control.recurrent_model.load(rnn_path)

        for epoch in range(
            self.off_control.epoch, self.off_control.epoch + self.params.epochs
        ):
            self.off_control.epoch = epoch
            self.off_control.reset_states()
            self.off_control.goals = {
                "world": [],
                "position": [],
                "angle": [],
                "saccade_id": [],
                "offcontrol_goal": [],
                "rnn_goal": [],
                "goal": [],
            }

            is_plotting_epoch = (
                epoch % self.params.plotting_epochs_interval == 0
            ) or (epoch == self.params.epochs - 1)
            self.plotters = self.execute_simulation(is_plotting_epoch)

            base_name = f"goals_{self.world}"

            if self.object_params is not None:

                parts = []
                if self.object_params.get("pos") is not None:
                    parts.append(f"{self.object_params['pos']}_")
                if self.object_params.get("rot") is not None:
                    parts.append(f"{self.object_params['rot']:06.2f}_")

                filename = slugify(f"{base_name}_{''.join(parts)}")

            else:
                filename = slugify(f"{base_name}")

            print(filename)

            np.save(
                filename,
                [self.off_control.goals],
            )

            print(f"Epoch: {epoch}")

        return self.plotters


def parse_arguments():
    """Parses command-line arguments for the simulation.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot animations.",
    )

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
        help="Set the position and rotation of the object in the world",
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )

    return parser.parse_args()


def format_name(param_name, value):
    """Formats a parameter name and value into a string.

    Args:
        param_name (str): The name of the parameter.
        value (any): The value of the parameter.

    Returns:
        str: A formatted string representing the parameter and its value.
    """
    return f"{param_name}_{str(value).replace('.', '_')}"


def main():
    """Main function to run the simulation test."""
    matplotlib.use("agg")

    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        print("Running on CUDA")
    else:
        torch.set_default_device("cpu")
        print("Running on CPU")

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
    plot = args.plot

    object_params = (
        None
        if args.posrot[0] is None
        else {
            "pos": args.posrot[:2],
            "rot": radians_to_degrees(args.posrot[2]),
        }
    )

    # Set additional parameters
    params.plot_maps = True
    params.plot_sim = True
    params.epochs = 1
    params.saccade_num = 16
    params.episodes = 1
    params.plotting_epochs_interval = 1 if plot else 1e100

    # Generate initial name without dots or special characters
    seed_str = format_name("seed", seed)

    params.init_name = f"test_{seed_str}"
    params.wandb = args.wandb

    # Initialize Weights & Biases logging
    if args.wandb:
        wandb.init(
            project=params.project_name,
            entity=params.entity_name,
            name=params.init_name,
        )

    # Simulate
    simulation_test = SimulationTest(params, seed, world, object_params)
    simulation_test.test()

    # Close Weights & Biases logging
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
