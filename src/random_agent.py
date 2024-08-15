#%% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import convolve2d
from scipy.special import softmax
import gymnasium as gym
import EyeSim
from agent import Agent
from plotter import FoveaPlotter


#%% MAIN LOOP AND VISUALIZATION
if __name__ == '__main__':

    plt.ion()
    plt.close('all')

    # Prepare environment
    env = gym.make('EyeSim-v0')
    env = env.unwrapped
    env.reset()
    agent = Agent(env)

    # Create plotting objects
    plotter = FoveaPlotter(env)

    # Initialize action before starting the loop
    action = np.zeros(env.action_space.shape)

    for time_step in range(100):
        observation, *_ = env.step(action)
        action, saliency_map, salient_point = agent.get_action(observation)
        plotter.step(saliency_map, salient_point)


