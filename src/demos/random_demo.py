# %% IMPORTS

import EyeSim
import gymnasium as gym
import matplotlib.pyplot as plt
from model.agent import Agent
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
    agent = Agent(env, sampling_threshold=0.01)

    worlds = ["triangle", "square", "circle" ]

    # Run the simulation for a fixed number of episodes
    for episode in range(3):
        world_id = next(
            i
            for i, world in enumerate(env.world_labels)
            if world == worlds[episode]
        )

        object_params = {"pos": [40.0, 40.0], "rot": 0.5}

        env.init_world(world=world_id, object_params=object_params)
        _, env_info = env.reset()

        # Precompute some constants
        action = [30, 30]

        # Create a plotting object for the current episode
        plotter = FoveaPlotter(env, offline=False)

        attention_centers = [None for x in range(5)]

        for center in attention_centers:
            # Set agent parameters based on the current attention center
            agent.set_parameters(center)

            # Simulate for a fixed number of time steps
            for time_step in range(10):
                observation, *_ = env.step(action)
                action, saliency_map, salient_point = agent.get_action(
                    observation
                )
                # Update the plotter with the current saliency map and salient
                # point
                plotter.step(
                    saliency_map, salient_point, agent.attentional_mask
                )
                plt.pause(0.1)

        # Save the plot for the current episode as a gif
        gif_file = f"episode_{episode:04d}"
        plotter.close(gif_file)
