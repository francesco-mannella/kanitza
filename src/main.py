# %% IMPORTS
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gymnasium as gym
import wandb
import signal
import sys,os

matplotlib.use("agg")

from model.agent import Agent, gaussian_mask
from plotter import FoveaPlotter, FakeMapsPlotter
from model.offline_controller import OfflineController


def signal_handler(signum, frame):
    signal.signal(signum, signal.SIG_IGN)  # ignore additional signals
    wandb.finish()
    sys.exit(0)


class Parameters:
    def __init__(
        self,
        project_name="eye-simulation",
        entity_name="francesco-mannella",
        init_name="offline controller tester",
        env_name="EyeSim-v0",
        episodes=2,
        epochs=600,
        focus_num=10,
        focus_time=10,
        plot_sim=False,
        plot_maps=True,
        plotting_epochs_interval=10,
        agent_sampling_threshold = 0.02,
    ):
        self.project_name = project_name
        self.entity_name = entity_name
        self.init_name = init_name
        self.env_name = env_name
        self.episodes = episodes
        self.epochs = epochs
        self.focus_num = focus_num
        self.focus_time = focus_time
        self.plot_sim = plot_sim
        self.plot_maps = plot_maps
        self.plotting_epochs_interval = plotting_epochs_interval
        self.agent_sampling_threshold = agent_sampling_threshold

# %%

# MAIN LOOP AND VISUALIZATION
if __name__ == "__main__":
    # Create an instance of Parameters with default or custom values
    params = Parameters()

    signal.signal(
        signal.SIGINT, signal_handler
    )  # register the signal with the signal handler first

    # Initialize Weights & Biases logging with project and entity names
    wandb.init(
        project=params.project_name,
        entity=params.entity_name,
        name=params.init_name,
    )

    # Enable matplotlib's interactive mode and close any existing plots
    plt.ion()
    plt.close("all")

    # Configure the environment and agent
    env = gym.make(params.env_name)
    env = env.unwrapped
    env.reset()

    agent = Agent(env, sampling_threshold=params.agent_sampling_threshold)

    off_control = OfflineController(env, params)

    for epoch in range(params.epochs):
        # Update offline controller hyperparameters based on the episode
        off_control.set_hyperparams()

        # Determine if the current epoch is a plotting epoch based on the interval
        is_plotting_epoch = epoch % params.plotting_epochs_interval == 0

        # Ensure the last epoch is always a plotting epoch
        is_plotting_epoch = is_plotting_epoch or epoch == params.epochs - 1

        # Initialize a plotting object for the current epoch
        if params.plot_maps and is_plotting_epoch:
            maps_plotter = FakeMapsPlotter(env, off_control, offline=True)

        # Execute the simulation for a specified number of episodes
        for episode in range(params.episodes):
            is_last_episode = episode == params.episodes - 1

            if params.plot_sim and is_last_episode and is_plotting_epoch:
                fovea_plotter = FoveaPlotter(env, offline=True)

            _, env_info = env.reset()

            # Precompute an action array initialized to zeros
            action = np.zeros(env.action_space.shape)

            # Create random mean values for the Gaussian masks (attention centers)
            attention_centers = off_control.generate_attentional_input(params.focus_num)

            for center in attention_centers:
                # Configure agent parameters according to the current attention center
                agent.set_parameters(center)

                # Execute the steps within the focus time
                for time_step in range(params.focus_time):
                    observation, *_ = env.step(action)
                    action, saliency_map, salient_point = agent.get_action(observation)

                    # Update the fovea_plotter with the current saliency map and salient point
                    if params.plot_sim and is_last_episode and is_plotting_epoch:
                        fovea_plotter.step(
                            saliency_map, salient_point, agent.attentional_mask
                        )

            if params.plot_sim and is_last_episode and is_plotting_epoch:
                # Save the current episode's plot as a GIF file
                gif_file = f"sim_{epoch:04d}"
                fovea_plotter.close(gif_file)

                # Log the generated GIF file to Weights & Biases
                wandb.log(
                    {
                        "Simulations": wandb.Video(f"{gif_file}.gif", format="gif"),
                    },
                    step=epoch,
                )

            print(f"Episode: {episode}, Epoch: {epoch}")

        # Update the offline controller's stored maps
        off_control.update_maps()

        if params.plot_maps and is_plotting_epoch:
            maps_plotter.step()

            # Save the current maps as a GIF file
            file = f"maps_{epoch:04d}.png"
            maps_plotter.close(file)
            if os.path.exists(file):
                # Log the generated GIF file to Weights & Biases if the file exists
                wandb.log({f"Maps": wandb.Image(f"{file}")}, step=epoch)

    # Conclude the Weights & Biases logging session
    wandb.finish()
