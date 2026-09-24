"""Test the controller with goal arbitration from the recurrent model.

Usage: run inside a trained simulation folder (needs off_control_store,
rnn_store.npy and optionally loaded_params):
    python /path/to/src/test_generative.py [--plot] [--seed N]
        [--world triangle|square] [--posrot X Y ANGLE]
        [--mask_type TYPE] [--mask_posrot X Y ANGLE] [--mask_start T]
        [--arbitration] [--wandb] [--skip_existing]

Args:
    --plot, --seed, --world, --posrot, --wandb, --skip_existing: as in
        test.py.
    --mask_type: occluder body, one of white_big_mask, grey_big_mask,
        white_small_mask, grey_small_mask.
    --mask_posrot: occluder position and rotation; no occluder if omitted.
    --mask_start: timestep from which the occluder is shown.
    --arbitration: from mask_start - 4 on, bias goal selection toward the
        RNN prediction (arbitration_weight=1); otherwise weight 0.

Runs 1 episode of saccade_num=10 x saccade_time steps with a saccade every
saccade_period=4 steps; goals are computed from the agent's color-saliency
fovea, the same visual input used in training. At each saccade the unfiltered visual-conditions
goal is fed to the RNN, the RNN prediction filters the visual-conditions
map, and the resulting goal drives the attention. Exits with an error if
off_control_store is missing.

Outputs: goals_<world>_<pos>_<rot>_<w>.npy (slugified) holding
[{"world", "position", "angle", "saccade_id", "offcontrol_goal",
"rnn_goal", "goal"}]; with --plot also *_test_* gifs.
"""
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

    def __init__(
        self, params, seed, world=None, object_params=None, mask_params=None
    ):
        """Initializes the SimulationTest object.

        Args:
            params (Parameters): The parameters for the simulation.
            seed (int): The random seed for reproducibility.
            world (str, optional): The name of the world to simulate.
                Defaults to None.
            object_params (dict, optional): Parameters for the object in the
                world. Defaults to None.
            mask_params (dict, optional): Parameters for the mask object.
                Defaults to None.
        """
        self.params = params
        self.seed = seed
        self.world = world
        self.object_params = object_params
        self.mask_params = mask_params
        self.env = None
        self.agent = None
        self.off_control = None
        self.plotters = []

    def init_environment(self):
        """Initializes the Gymnasium environment.

        Returns:
            gym.Env: The initialized Gymnasium environment.
        """
        env = gym.make(self.params.env_name, params=self.params)
        env = env.unwrapped
        env.set_seed(self.seed)
        env.rotation = 0.0
        self.env = env
        return env

    def load_offline_controller(self, file_path):
        """Loads the trained offline controller from a file.

        Args:
            file_path (str): The path to the offline controller file.

        Returns:
            OfflineController: The loaded offline controller.
        """
        return OfflineController.load(
            file_path, self.env, self.params, self.seed
        )

    def update_mask(self, mask_params):
        """Updates the position and rotation of the mask in the environment.

        Args:
            mask_params (dict): A dictionary containing the position and
                rotation of the mask.
        """
        if mask_params is not None:
            self.env.update_position_and_rotation(
                position=mask_params["pos"],
                rotation=mask_params["rot"],
                obj=self.params.mask_type,
            )

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
        goal = None
        mask_updated = False
        action = (0, 0)
        saliency_map = None

        for time_step in range(
            self.params.saccade_time * self.params.saccade_num
        ):

            rnn_goal_influence = (
                0
                if time_step
                < self.params.mask_start
                - self.params.rnn_arbitration_before_mask
                else self.params.arbitration_weight
            )

            if time_step % self.params.saccade_period == 0:
                print(f"ts: {time_step:>3d}  ", end="")

                # Get info for saccade
                condition = self.agent.get_fovea(observation)

                # Goal Arbitration:
                # This section arbitrates between two potential goals:
                # 1. `offcontrol_goal`: The goal suggested by the topological
                #    alignment mechanism based on the current condition.
                # 2. `rnn_goal`: The goal predicted by the recurrent neural
                #    network (RNN), providing a sense of anticipation or
                #    prediction.

                # topological alignment goal selection, without the bias
                # applied to the map at the previous saccade
                self.off_control.visual_conditions_map.clear_filter()
                offcontrol_goal = (
                    self.off_control.get_representation_from_condition(
                        condition
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )

                # Recurrent model step
                rnn_goal = self.off_control.recurrent_model.step(
                    goal=offcontrol_goal.reshape(-1),
                    reservoir_influence=rnn_goal_influence,
                )

                # Apply bias from rnn to the visual map
                self.off_control.apply_filter_to_map(
                    map_object=self.off_control.visual_conditions_map,
                    target_goal=rnn_goal,
                    filter_strength=rnn_goal_influence,
                )

                # Compute again the goal after RNN bias
                goal = self.off_control.get_representation_from_condition(
                    condition
                )
                saccade = self.off_control.get_saccade_from_representation(
                    goal
                )

                goal = goal.cpu().detach().numpy()

                # print time_step and goals
                gs = ["SOM", "RNN", "goal"]
                vs = [
                    offcontrol_goal.flatten(),
                    rnn_goal.flatten(),
                    goal.flatten(),
                ]
                for g, (x, y) in zip(gs, vs):
                    print(f"{g:>5} {int(x):1d},{int(y):1d} ", end="")
                print()

                # Trigger saccade
                self.agent.set_parameters(saccade)

                self.off_control.goals["world"].append(self.env.info["world"])
                self.off_control.goals["angle"].append(self.env.info["angle"])
                self.off_control.goals["position"].append(
                    self.env.info["position"]
                )
                saccade_id = f"{episode:04d}-{time_step:04d}"
                self.off_control.goals["saccade_id"].append(saccade_id)
                self.off_control.goals["offcontrol_goal"].append(
                    offcontrol_goal
                )
                self.off_control.goals["rnn_goal"].append(rnn_goal)
                self.off_control.goals["goal"].append(goal)

                if time_step >= self.params.mask_start:
                    if mask_updated is False:
                        self.update_mask(self.mask_params)
                        mask_updated = True

            elif time_step % self.params.saccade_period == 1:

                # Reset saccade
                if saccade is not None and not np.array_equal(
                    saccade, np.array([0.5, 0.5])
                ):
                    saccade = np.array([0.5, 0.5])
                    self.agent.set_parameters(saccade)

            action, saliency_map, salient_point, color_saliency = (
                self.agent.get_action(observation)
            )
            observation, *_ = self.env.step(action)

            if is_plotting_epoch and saliency_map is not None:
                self.update_plotters(
                    fovea_plotter,
                    maps_plotter,
                    color_saliency,
                    saliency_map,
                    salient_point,
                    goal,
                )

    def update_plotters(
        self,
        fovea_plotter,
        maps_plotter,
        color_saliency_map,
        saliency_map,
        salient_point,
        goal,
    ):
        """Updates the plotters with the latest data from the simulation.

        Args:
            fovea_plotter (FoveaPlotter): The fovea plotter object.
            maps_plotter (MapsPlotter): The maps plotter object.
            color_saliency_map (numpy.ndarray): The fovea returned by the
                agent.
            saliency_map (numpy.ndarray): The saliency map from the agent.
            salient_point (tuple): The salient point from the agent.
            goal (numpy.ndarray): The goal for the current time step.
        """
        if fovea_plotter:
            fovea_plotter.step(
                color_saliency_map,
                saliency_map,
                salient_point,
                self.agent.attentional_mask,
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
        tag += f"_{self.params.arbitration_weight:1d}"

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

    def goals_filename(self):
        """Return the goals output path, without the .npy extension.

        Returns:
            str: slugified "goals_<world>[_<pos>_<rot>_<w>]".
        """
        base_name = f"goals_{self.world}"

        if self.object_params is not None:

            parts = []
            if self.object_params.get("pos") is not None:
                parts.append(f"{self.object_params['pos']}_")
            if self.object_params.get("rot") is not None:
                parts.append(f"{self.object_params['rot']:06.2f}_")
            parts.append(f"{self.params.arbitration_weight:1d}")

            return slugify(f"{base_name}_{''.join(parts)}")

        return slugify(f"{base_name}")

    def test(self):
        """Runs the main test loop for the simulation.

        Exits with an error if ./off_control_store is missing. With
        params.skip_existing, returns without simulating if the goals file
        already exists.

        Returns:
            list: A list of plotters used during the simulation.
        """
        controller_path = "off_control_store"
        if not os.path.exists(controller_path):
            sys.exit(f"{controller_path} not found: run inside a trained folder")
        if self.params.skip_existing and os.path.exists(
            f"{self.goals_filename()}.npy"
        ):
            print(f"{self.goals_filename()}.npy exists, skipping")
            return self.plotters

        signal.signal(signal.SIGINT, signal_handler)
        plt.ion()
        plt.close("all")
        torch.manual_seed(self.seed)

        self.env = self.init_environment()

        self.agent = Agent(
            self.env,
            seed=self.seed,
            focus_params=self.params,
        )

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

            filename = self.goals_filename()
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
        "--mask_type",
        type=str,
        default="grey_small_mask",
        help=(
            "Set the mask type. It must be one of 'white_big_mask' "
            "'grey_big_mask', 'white_small_mask', 'grey_small_mask'. "
        ),
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
        "--mask_posrot",
        nargs=3,
        type=float,
        metavar=("x", "y", "a"),
        default=[None, None, None],
        help="Set the position and rotation of the mask in the world",
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )

    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Do nothing if the goals file of this test already exists.",
    )

    parser.add_argument(
        "--mask_start",
        type=int,
        default=99999,
        help="The timestep whe the mask is shown.",
    )

    parser.add_argument(
        "--arbitration",
        action="store_true",
        help="Set the behavior of the internal model to active",
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
    mask_type = args.mask_type

    object_params = (
        None
        if args.posrot[0] is None
        else {
            "pos": args.posrot[:2],
            "rot": args.posrot[2],
        }
    )
    mask_params = (
        None
        if args.mask_posrot[0] is None
        else {"pos": args.mask_posrot[:2], "rot": args.mask_posrot[2]}
    )

    # Set additional parameters
    params.plot_maps = True
    params.plot_sim = True
    params.epochs = 1
    params.saccade_num = 10
    params.episodes = 1
    params.saccade_period = 4
    params.skip_existing = args.skip_existing
    params.plotting_epochs_interval = 1 if plot else 1e100
    params.mask_start = args.mask_start
    params.rnn_arbitration_before_mask = 4
    params.arbitration_weight = 1 if args.arbitration is True else 0
    params.mask_type = mask_type

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
    simulation_test = SimulationTest(
        params, seed, world, object_params, mask_params
    )
    simulation_test.test()

    # Close Weights & Biases logging
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
