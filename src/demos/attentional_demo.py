# %% IMPORTS

import EyeSim
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from model.agent import Agent
from params import Parameters
from plotter import FoveaPlotter


_ = EyeSim

# This code is designed for simulating and visualizing an agent’s behavior in
# an environment, specifically focusing on its attention mechanisms

# %% MAIN LOOP AND VISUALIZATION
if __name__ == "__main__":

    # Enable interactive mode and close any previously opened plots
    plt.ion()
    plt.close("all")

    # Set up the environment and agent
    env = gym.make("EyeSim/EyeSim-v0", colors=True)
    env = env.unwrapped

    params = Parameters()
    params.agent_sampling_precision = 1 - 1e-10
    params.attention_max_variance = 3
    params.attention_fixed_variance_prop = 0.3
    params.attention_center_distance_variance_prop = 0.7
    params.attention_center_distance_slope = 2

    agent = Agent(
        env,
        focus_params=params,
    )

    worlds = [
        "triangle",
        "square",
        "circle",
    ]

    data = []

    # Run the simulation for a fixed number of episodes
    for episode in range(3):
        world_id = next(
            i for i, world in enumerate(env.world_labels) if world == worlds[episode]
        )

        object_params = {"pos": [40.0, 40.0], "rot": 0.5}

        env.init_world(world=world_id, object_params=object_params)
        _, env_info = env.reset()

        # Precompute some constants
        action = [0, 0]

        # Create a plotting object for the current episode
        plotter = FoveaPlotter(env, offline=False)

        # Generate random means for Gaussian masks
        a = np.linspace(0, 2 * np.pi, 15)
        attention_centers = 0.5 + 0.3 * np.array([[np.cos(x), np.sin(x)] for x in a])

        pos = env.retina_sim_pos
        for center in attention_centers:
            # Set agent parameters based on the current attention center
            agent.set_parameters(center)

            # Simulate for a fixed number of time steps
            for time_step in range(3):
                observation, *_ = env.step(action)
                pos_prev, pos = pos, env.retina_sim_pos
                mov = np.array(pos) - pos_prev

                data.append([world_id, *list(mov)])
                print(mov)
                action, saliency_map, salient_point, color_saliency = agent.get_action(observation)
                if time_step != 0:
                    agent.set_parameters([0.5, 0.5])

                # Update the plotter with the current saliency map and salient
                # point
                plotter.step(color_saliency, saliency_map, salient_point, agent.attentional_mask)
                plt.pause(0.1)

        # Save the plot for the current episode as a gif
        gif_file = f"episode_{episode:04d}"
        plotter.close(gif_file)

        # np.save("retina_poses", data )
