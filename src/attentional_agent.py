#%% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.special import softmax
import gymnasium as gym
import EyeSim
from agent import Agent, gaussian_mask
from plotter import FoveaPlotter



#%% MAIN LOOP AND VISUALIZATION
if __name__ == '__main__':

    plt.ion()
    plt.close('all')

    # Prepare environment
    env = gym.make('EyeSim-v0')
    env = env.unwrapped
    env.reset()
    agent = Agent(env, sampling_threshold=0.02)
    
    # Create plotting objects
    plotter = FoveaPlotter(env)

    # Initialize action before starting the loop
    action = np.zeros(env.action_space.shape)
    height, width = env.observation_space["RETINA"].shape[:-1]
    means = [
        np.array([height * 0.3, width * 0.7]),
        np.array([height * 0.5, width * 0.3]),
        np.array([height * 0.7, width * 0.7])
    ]

    means = np.array([[height, width]]) * np.random.rand(30, 2)

    angle = 0

    v1 = 18 * height * 0.5
    v2 = 18 * width * 0.5

    env.reset()
    for mean in means: 
        attentional_mask = gaussian_mask((height, width), mean, v1, v2, angle)
        for time_step in range(10):
            mask = attentional_mask 
            observation, *_ = env.step(action)
            action, saliency_map, salient_point = agent.get_action(
                    observation,
                    mask)
            plotter.render(saliency_map, salient_point, mask)

