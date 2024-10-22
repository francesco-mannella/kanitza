#%% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from model.agent import Agent
from plotter import FoveaPlotter
import wandb

# This code is for running a simulation of an eye-tracking agent interacting
# with an environment and visualizing the interactions.

#%% MAIN LOOP AND VISUALIZATION
if __name__ == '__main__':

    # Initialize Weights & Biases logging
    wandb.init(
        project='eye-simulation',
        entity='francesco-mannella',
        name='random demo',
    )

    plt.ion()
    plt.close('all')

    # Prepare environment
    env = gym.make('EyeSim-v0')
    env = env.unwrapped
    env.reset()
    agent = Agent(env)

    for episode in range(1):
        _, env_info = env.reset()

        # Create plotting objects
        plotter = FoveaPlotter(env, offline=True)

        # Initialize action before starting the loop
        action = np.zeros(env.action_space.shape)

        for time_step in range(200):
            observation, *_ = env.step(action)
            action, saliency_map, salient_point = agent.get_action(observation)
            plotter.step(saliency_map, salient_point)

        # Save the plot for the current episode as a gif
        gif_file = f'episode_{episode:04d}_random'
        plotter.close(gif_file)

        # Log the gif file to Weights & Biases
        wandb.log(
            {'random_demo': wandb.Video(f'{gif_file}.gif', format='gif')}
        )

    # Close the Weights & Biases run
    wandb.finish()
