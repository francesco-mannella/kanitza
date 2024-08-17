#%% IMPORTS
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gymnasium as gym
import wandb

matplotlib.use("agg")

from model.agent import Agent, gaussian_mask
from plotter import FoveaPlotter, MapsPlotter
from model.offline_controller import OfflineController 

# MAIN LOOP AND VISUALIZATION
if __name__ == '__main__':

    # Initialize Weights & Biases logging with project and entity names
    wandb.init(
        project='eye-simulation',
        entity='francesco-mannella',
        name='offline controller tester',
    )

    # Enable matplotlib's interactive mode and close any existing plots
    plt.ion()
    plt.close('all')

    # Configure the environment and agent
    env = gym.make('EyeSim-v0')
    env = env.unwrapped
    env.reset()
    agent = Agent(env, sampling_threshold=0.02)
    off_control = OfflineController(env)
    
    # Define the number of episodes, focus points, and focus duration
    episodes = 40
    epochs = 30
    focus_num = 40
    focus_time = 10
    plot_sim = False
    plot_maps = True

    for epoch in range(epochs):
        # Update offline controller hyperparameters based on the episode
        comp = 1 - np.exp(-epoch/(0.1*epochs))
        off_control.set_hyperparams(comp)
        
        # Initialize a plotting object for the current episode
        if plot_sim : fovea_plotter = FoveaPlotter(env, offline=True)
        if plot_maps: maps_plotter = MapsPlotter(env, off_control, offline=True)
        
        # Execute the simulation for a specified number of episodes
        for episode in range(episodes):

            _, info = env.reset()

            # Precompute an action array initialized to zeros
            action = np.zeros(env.action_space.shape)


            # Create random mean values for the Gaussian masks (attention centers)
            attention_centers = np.random.rand(focus_num, 2)

            for center in attention_centers:
                # Configure agent parameters according to the current attention center
                agent.set_parameters(center)

                # Execute the steps within the focus time
                for time_step in range(focus_time):
                    observation, *_ = env.step(action)
                    action, saliency_map, salient_point = agent.get_action(observation)
                    
                    # Update the fovea_plotter with the current saliency map and salient point
                    if plot_sim: fovea_plotter.step(saliency_map, salient_point, agent.attentional_mask)

                    # Store the current inputs in the offline controller
                    off_control.store_fovea_input()
                    off_control.store_attentional_input(info["position"])
            
            print(f"Episode: {episode}, Epoch: {epoch}")
    


        # Update the offline controller's stored maps
        off_control.update_maps()
        if plot_maps: 
            maps_plotter.step()

            # Save the current maps as a GIF file
            file = f'maps_{epoch:04d}.png'
            maps_plotter.close(file)
            
            # Log the generated GIF file to Weights & Biases
            wandb.log({'Maps': wandb.Image(f'{file}')})

        if plot_sim:
            # Save the current episode's plot as a GIF file
            gif_file = f'sim_{epoch:04d}'
            fovea_plotter.close(gif_file)
            
            # Log the generated GIF file to Weights & Biases
            wandb.log({'Simulation': wandb.Video(f'{gif_file}.gif', format='gif')})

    # Conclude the Weights & Biases logging session
    wandb.finish()
