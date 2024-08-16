#%% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import wandb

from model.agent import Agent, gaussian_mask
from plotter import FoveaPlotter
from model.offline_controller import OfflineController 

# MAIN LOOP AND VISUALIZATION
if __name__ == '__main__':

    # Initialize Weights & Biases logging
    wandb.init(
        project='eye-simulation',
        entity='francesco-mannella',
        name='offline controller tester',
    )

    # Enable interactive mode and close any previously opened plots
    plt.ion()
    plt.close('all')

    # Set up the environment and agent
    env = gym.make('EyeSim-v0')
    env = env.unwrapped
    env.reset()
    agent = Agent(env, sampling_threshold=0.02)
    off_control = OfflineController(env)
    episodes = 5
    focus_num = 5
    focus_time = 10


    # Run the simulation for a fixed number of episodes
    for episode in range(episodes):
        _, info = env.reset()

        comp = np.exp(-episode/episodes)
        off_control.set_hyperparams(comp)

        # Precompute some constants
        action = np.zeros(env.action_space.shape)

        # Create a plotting object for the current episode
        plotter = FoveaPlotter(env, offline=True)

        # Generate random means for Gaussian masks
        attention_centers = np.random.rand(focus_num, 2)

        for center in attention_centers:
            # Set agent parameters based on the current attention center
            agent.set_parameters(center)

            # Simulate for a fixed number of time steps
            for time_step in range(10):
                observation, *_ = env.step(action)
                action, saliency_map, salient_point = agent.get_action(
                    observation
                )
                # Update the plotter with the current saliency map and salient point
                plotter.step(
                    saliency_map, salient_point, agent.attentional_mask
                )

                off_control.store_fovea_input()

        off_control.update_maps()

        # Save the plot for the current episode as a gif
        gif_file = f'episode_{episode:04d}'
        plotter.close(gif_file)

        # Log the gif file to Weights & Biases
        wandb.log(
            {'offline controller tester': wandb.Video(f'{gif_file}.gif', format='gif')}
        )

    # Close the Weights & Biases run
    wandb.finish()
