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
from params import Parameters
from plotter import FoveaPlotter, MapsPlotter


es = EyeSim


# TODO: code for debug
def ascii_imshow(matrix, nrows, ncols):
    """Prints an ASCII art representation of a numpy array."""
    levels = (
        r"$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|" r"()1{}[]?-_+~<>i!lI;:,\"^`'."
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
                index = int((value - min_val) / value_range * (len(levels) - 1))
            print(levels[index], end="")
        print()
    print()


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


class Main:
    """Main class to initialize and run the environment, agent, and controllers."""

    def __init__(self, params):
        """Initializes the Main class with parameters, logger, signal handler,
        environment, agent, and offline controller.

        Args:
            params: Configuration parameters for the environment and agent.
        """
        self.params = params
        self.main_log = Logger("log")
        self.sh = self.SignalHandler(params.wandb)
        signal.signal(signal.SIGINT, self.sh)
        plt.ion()
        plt.close("all")
        torch.manual_seed(seed)
        self.env = self.setup_environment(seed)
        self.agent = self.setup_agent(seed)
        self.off_control = self.setup_offline_controller("off_control_store", seed)
        if params.plot_maps:
            self.maps_plotter = MapsPlotter(self.env, self.off_control, offline=True)

    class SignalHandler:
        """Handles interrupt signals to gracefully shut down the application."""

        def __init__(self, wandb=True):
            """Initializes the SignalHandler with optional Weights & Biases support.

            Args:
                wandb (bool): Whether to use Weights & Biases logging.
            """
            self._wandb = wandb

        def __call__(self, signum, frame):
            """Handles the signal by finishing Weights & Biases and exiting.

            Args:
                signum: Signal number.
                frame: Current stack frame.
            """
            signal.signal(signum, signal.SIG_IGN)
            if self._wandb:
                wandb.finish()
            sys.exit(0)

    def setup_environment(self, seed):
        """Sets up the environment with the given seed.

        Args:
            seed: Seed for random number generation.

        Returns:
            The initialized environment.
        """
        env = gym.make(self.params.env_name, colors=self.params.colors)
        env = env.unwrapped
        env.set_seed(seed)
        return env

    def setup_agent(self, seed):
        """Sets up the agent with the given seed.

        Args:
            seed: Seed for random number generation.

        Returns:
            The initialized agent.
        """
        return Agent(self.env, seed=seed, focus_params=self.params)

    def setup_offline_controller(self, file_path, seed):
        """Sets up the offline controller, loading from file if it exists.

        Args:
            file_path: Path to the offline controller file.
            seed: Seed for random number generation.

        Returns:
            The initialized offline controller.
        """
        if os.path.exists(file_path):
            return OfflineController.load(file_path, self.env, self.params, seed)
        return OfflineController(self.env, self.params, seed)

    def run_epoch(self, epoch):
        """Runs a single epoch, iterating over episodes and logging results.

        Args:
            epoch: The current epoch number.
        """
        for episode in range(self.params.episodes):
            info = self.run_episode(episode, epoch)
            self.main_log(
                f"Epoch: {epoch:<5d} "
                f"Episode: {episode:<3d} "
                f"type:{info['world']:10s}"
            )

    def run_episode(self, episode, epoch):
        """Runs a single episode, executing saccades and optionally plotting.

        Args:
            episode: The current episode number.
            epoch: The current epoch number.

        Returns:
            Information about the environment state after the episode.
        """
        self.env.init_world(
            world=0 if epoch % 100 < self.params.triangles_percent else 1,
        )
        _, env_info = self.env.reset()

        plt_enabled = (
            self.params.plot_sim
            and episode == self.params.episodes - 1
            and self.is_plotting_epoch(epoch)
        )

        fovea_plotter = FoveaPlotter(self.env, offline=True) if plt_enabled else None

        action = np.zeros(self.env.action_space.shape)

        for saccade_idx in range(self.params.saccade_num):
            self.execute_saccade(action, episode, saccade_idx, fovea_plotter)

        if plt_enabled:
            self.save_simulation_gif(fovea_plotter, epoch)

        return env_info

    def execute_saccade(self, action, episode, saccade_idx, fovea_plotter):
        """Executes a saccade, updating the agent's action and recording states.

        Args:
            action: Initial action for the saccade.
            episode: The current episode number.
            saccade_idx: Index of the current saccade.
            fovea_plotter: Optional plotter for visualizing the fovea.
        """
        observation, *_ = self.env.step(np.zeros(self.params.action_size))

        competence = None
        saccade = None
        attention = None
        salient_point, action = [0, 0], [0.0, 0.0]
        for time_step in range(self.params.saccade_time):
            observation, *_ = self.env.step(action)
            if time_step == int(0.5 * self.params.saccade_time):
                saccade, competence = self.off_control.generate_saccade(
                    observation["FOVEA"]
                )
                self.agent.set_parameters(saccade)
                attention = np.copy(saccade)
            else:
                if saccade is not None and not np.array_equal(
                    saccade, np.array([0.5, 0.5])
                ):
                    saccade = np.array([0.5, 0.5])
                    self.agent.set_parameters(saccade)
            action, saliency_map, salient_point = self.agent.get_action(observation)

            if fovea_plotter:
                fovea_plotter.step(
                    saliency_map, salient_point, self.agent.attentional_mask
                )

            state = {
                "world": self.env.world,
                "vision": observation["FOVEA"],
                "action": action,
                "attention": attention,
                "competence": competence,
            }
            self.off_control.record_states(episode, saccade_idx, time_step, state)

    def is_plotting_epoch(self, epoch):
        """Determines if the current epoch should be plotted.

        Args:
            epoch: The current epoch number.

        Returns:
            True if the epoch should be plotted, False otherwise.
        """
        return (
            epoch % self.params.plotting_epochs_interval == 0
            or epoch == self.params.epochs - 1
        )

    def save_simulation_gif(self, fovea_plotter, epoch):
        """Saves a simulation GIF using the fovea plotter.

        Args:
            fovea_plotter: The plotter used for generating the GIF.
            epoch: The current epoch number.
        """
        gif_file = f"sim_{epoch:04d}"
        fovea_plotter.close(gif_file)
        if self.params.wandb:
            wandb.log(
                {"Simulations": wandb.Video(f"{gif_file}.gif", format="gif")},
                step=epoch,
            )

    def __call__(self):
        """Runs the main loop over epochs, logging and updating the controller."""
        for epoch in range(
            self.off_control.epoch, self.off_control.epoch + self.params.epochs
        ):
            self.main_log(f"epoch: {epoch}")
            self.off_control.epoch = epoch
            self.off_control.reset_states()

            self.run_epoch(epoch)
            self.off_control.filter_salient_states()

            world_dict = {"triangle": 0, "square": 0}
            for idx in np.array(self.off_control.filtered_idcs).T:
                world = int(self.off_control.world_states[idx[0], idx[1], idx[2]][0])
                if self.env.world_labels[world] == "triangle":
                    world_dict["triangle"] += 1
                elif self.env.world_labels[world] == "square":
                    world_dict["square"] += 1

            if self.params.wandb:
                self.main_log(
                    f"triangles: {world_dict['triangle']}, "
                    f"squares: {world_dict['square']}"
                )

            self.off_control.update()

            if self.params.wandb:
                self.main_log(f"comp: {self.off_control.competence}")

                wandb.log(
                    dict(
                        competence=self.off_control.competence,
                        **self.off_control.weight_change,
                    ),
                    step=epoch,
                )

            if self.params.plot_maps:
                self.maps_plotter.step()
                if self.is_plotting_epoch(epoch):
                    self.save_maps_gif(self.maps_plotter, epoch)
                    self.maps_plotter = MapsPlotter(
                        self.env, self.off_control, offline=True
                    )

            self.off_control.save("off_control_store")

    def save_maps_gif(self, maps_plotter, epoch):
        """Saves a maps GIF using the maps plotter.

        Args:
            maps_plotter: The plotter used for generating the GIF.
            epoch: The current epoch number.
        """
        file = f"maps_{epoch:04d}"
        maps_plotter.close(file)
        if self.params.wandb:
            wandb.log(
                {
                    "history": wandb.Image(f"{file}.gif"),
                    "last": wandb.Image(f"{file}.png"),
                },
                step=epoch,
            )


if __name__ == "__main__":

    if torch.cuda.is_available():
        torch.set_default_device("cuda")

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
    parser.add_argument(
        "-w",
        "--wandb",
        action="store_true",
        help="Defines if wandb is activated",
    )

    args = parser.parse_args()

    params = Parameters()
    seed = args.seed
    variant = args.variant

    try:
        params.load("loaded_params")
    except FileNotFoundError:
        print("no local parameters")
        param_list = args.param_list
        params.update(param_list)
        params.save("loaded_params")

    params.wandb = args.wandb
    seed_str = str(seed).replace(".", "_")
    decaying_speed_str = str(params.decaying_speed).replace(".", "_")
    local_decaying_speed_str = str(params.local_decaying_speed).replace(".", "_")

    def format_scalar(x):
        return f"{x:06.3f}".replace(".", "")

    params.init_name = f"{variant}"

    with open("NAME", "w") as fname:
        fname.write(f"{params.init_name}\n")

    if args.wandb:
        wandb.init(
            project=params.project_name,
            entity=params.entity_name,
            name=params.init_name,
            config=params._params_to_dict(),
        )

    main = Main(params)

    main()

    if args.wandb:
        wandb.finish()
