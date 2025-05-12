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

#plt.figure()   

fig, axes = plt.subplots(4, 2, figsize=(12, 12))
fig2,ax=plt.subplots()

for w in world_conditions:
        
    world= w #'triangle'#"Set the world in the test. it canbe 'square' or 'triangle'"
    
    for num, c in enumerate(posrot_conditions):
        
        posrot = c   #    [45.,45.,1.5]  #    # "Set the x postition, y position and rotation of the object "
        #rot in radianti, quindi 0.25, 0.125, 0.07  # 1.57, 0.79
        
        text = 'Scanpath_seed' + str(seed) + '_' + world +'_pos' + str(posrot[0]) + '_' + str(posrot[1]) + '_rot' + str(posrot[2]) + '.csv'
        data = np.loadtxt(text, delimiter=',')
        scanpath = data[5::]
        
        
        if world == 'square':
            cmap = 'r'# plt.cm.autumn
            column = 0
        else:
            cmap = 'b'# plt.cm.winter
            column = 1
        col = cmap#(num/4)
        
        
        
        scanp = np.zeros( (len(scanpath), 2) )
        for index, winning_unit in enumerate(scanpath):
            scanp[index, 0] = winning_unit%10
            scanp[index, 1] = winning_unit//10
        '''
        plt.figure()  
        plt.plot(scanp[-500::,0],scanp[-500::,1], alpha = 0.2, color = col)
        plt.scatter(scanp[-500:,0],scanp[-500::,1], s=10, color = col)
        
        
        
        #plt.xlim(-0.5,10.5)
        #plt.ylim(-0.5,10.5)
        
        '''
        
        axes[num,column].plot(scanp[-500::,0],scanp[-500::,1], alpha = 0.2, color = col)
        axes[num,column].scatter(scanp[-500:,0],scanp[-500::,1], s=10, color = col)
        axes[num,column].set_xlim(-0.5,10.5)
        axes[num,column].set_ylim(-0.5,10.5)
        ax.plot(scanp[-500::,0],scanp[-500::,1], alpha = 0.2, color = col)
        ax.scatter(scanp[-500:,0],scanp[-500::,1], s=10, color = col)
        ax.set_xlim(-0.5,10.5)
        ax.set_ylim(-0.5,10.5)
        
        
        
        
