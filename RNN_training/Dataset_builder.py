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
os.environ['WANDB_MODE'] = 'disabled'

from model.offline_controller import OfflineController
from test import test
from params import Parameters
from model.agent import Agent

params = Parameters()

seed = 1

world_conditions = ['square', 'triangle']
posrot_conditions = [[45.,45.,1.57], [45.,45.,0.79], [15.,25.,1.57], [15.,25.,0.79]]

for w in world_conditions:
        
    world= w #'triangle'#"Set the world in the test. it canbe 'square' or 'triangle'"
    
    for c in posrot_conditions:
        
        posrot = c   #    [45.,45.,1.5]  #    # "Set the x postition, y position and rotation of the object "
        #rot in radianti, quindi 0.25, 0.125, 0.07  # 1.57, 0.79
        
        object_params = (
            None
            if posrot[0] is None
            else {"pos": posrot[:2], "rot": posrot[2]}
        )
        
        
        # Set additional parameters
        params.plot_maps = True
        params.plot_sim = True
        params.epochs = 1
        params.saccade_num = 4
        params.saccade_time = 10
        params.episodes = 1
        params.plotting_epochs_interval = 1
        
        #
        episode_lenght = 1000 #params.saccade_time * params.saccade_num
        num_units = params.maps_output_size
        
        storage_array_episode = np.zeros((2, episode_lenght)) 
        
        
        _ = EyeSim  # avoid fixer erase EyeSim import
        
        
        def signal_handler(signum, frame):
            signal.signal(signum, signal.SIG_IGN)  # ignore additional signals
            wandb.finish()
            sys.exit(0)
        
        
        def init_environment(params, seed):
            env = gym.make(params.env_name)
            env = env.unwrapped
            env.set_seed(seed)
            env.rotation = 0.0
            return env
        
        
        def load_offline_controller(file_path, env, params, seed):
            if os.path.exists(file_path):
                return OfflineController.load(file_path, env, params, seed)
            return OfflineController(env, params, seed)
        
        
        def run_episode(
            agent,
            env,
            off_control,
            observation,
            params,
            storage_array_episode
        ):
            # TODO: redundnt
            condition = observation["FOVEA"].copy()
            saccade, goal = off_control.get_action_from_condition(condition)
            agent.set_parameters(saccade)
        
            for time_step in range(episode_lenght):   #(params.saccade_time * params.saccade_num):
                condition = observation["FOVEA"].copy()
                saccade, goal = off_control.get_action_from_condition(condition)
        
                if time_step % 1 == 0:
                    print("ts: ", time_step)
                    agent.set_parameters(saccade)
        
                action, saliency_map, salient_point = agent.get_action(observation)
                observation, *_ = env.step(action)
        
                
                storage_array_episode[:,time_step] = goal
                
            return storage_array_episode
        
        
        
        env = init_environment(params, seed)
        agent = Agent(
            env, sampling_threshold=params.agent_sampling_threshold, seed=seed
        )
        
        file_path = "off_control_store"
        off_control = load_offline_controller(file_path, env, params, seed)
        
        if world is None:
            world_id = env.rng.choice([0, 1])
        else:
            world_id = np.argwhere(
                [label == world for label in env.world_labels]
            )[0][0]
        
        env.init_world(world=world_id, object_params=object_params)
        observation, info = env.reset()
        
        storage_array_episode = run_episode(
            agent,
            env,
            off_control,
            observation,
            params,
            storage_array_episode
        )
        
        episode_scanpath = np.zeros( (episode_lenght) ) #np.zeros( (params.saccade_time * params.saccade_num) )
        map_side = int(np.sqrt(num_units) )
        for index, i in enumerate(storage_array_episode.T):
            episode_scanpath[index] = int(i[0]) + map_side * int(i[1]) 
        
        
        plt.figure()
        plt.scatter(range(len(episode_scanpath)), episode_scanpath)
        
        scanp = np.zeros( (len(episode_scanpath), 2) )
        for index, winning_unit in enumerate(episode_scanpath):
            scanp[index, 0] = winning_unit%10
            scanp[index, 1] = winning_unit//10
            
        plt.figure()   
        plt.plot(scanp[-500::,0],scanp[-500::,1], alpha = 0.2)
        plt.scatter(scanp[-500:,0],scanp[-500::,1], s=10)
        plt.xlim(-0.5,10.5)
        plt.ylim(-0.5,10.5)
        
        #'''
        
        Saving_array = np.zeros(len(episode_scanpath)+5)
        Saving_array[0] = seed
        if world == 'triangle':
            Saving_array[1] = 0
        else:
            Saving_array[1] = 1
        Saving_array[2] = posrot[0]
        Saving_array[3] = posrot[1]
        Saving_array[4] = posrot[2]
        Saving_array[5::] = episode_scanpath
        
        text = 'Scanpath_seed' + str(seed) + '_' + world +'_pos' + str(posrot[0]) + '_' + str(posrot[1]) + '_rot' + str(posrot[2]) + '.csv'
        np.savetxt(text, Saving_array, delimiter=',')
        
        #'''
        
        
        
        
        
        
        
        
