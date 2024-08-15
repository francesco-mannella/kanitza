#%% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.special import softmax
import gymnasium as gym
import EyeSim
from agent import Agent, gaussian_mask
from plotter import FoveaPlotter
import wandb


#%% MAIN LOOP AND VISUALIZATION
if __name__ == '__main__':

    # Initialize Weights & Biases logging
    wandb.init(
        project='eye-simulation',
        entity='francesco-mannella',
        name="attentional demo",
    )

    # Enable interactive mode and close any previously opened plots
    plt.ion()
    plt.close('all')

    # Set up the environment and agent
    env = gym.make('EyeSim-v0')
    penv = env.unwrapped
    env.reset()
    agent = Agent(env, sampling_threshold=0.02)

    # Precompute some constants
    action = np.zeros(env.action_space.shape)

    # Run the simulation for a fixed number of episodes
    for episode in range(5):
        _, info = env.reset()

        # Create a plotting object for the current episode
        plotter = FoveaPlotter(env, offline=True)

        # Generate random means for Gaussian masks
        attention_centers = np.random.rand(5, 2)

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

        # Save the plot for the current episode as a gif
        gif_file = f'episode_{episode:04d}'
        plotter.close(gif_file)

        # Log the gif file to Weights & Biases
        wandb.log({"attentional_demo": wandb.Video(f'{gif_file}.gif', format='gif')})

    # Close the Weights & Biases run
    wandb.finish()
