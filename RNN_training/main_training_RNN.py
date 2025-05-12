import matplotlib.pyplot as plt
import numpy as np

from recurrent_generative_model import RecurrentGenerativeModelUpdater, RecurrentGenerativeModel
from params_recurrent_generative_model import ParamsFORCE

paramsFORCE = ParamsFORCE()

'''
Parameters definition
'''

seed = 1


'''
Dataset definition
'''

objects = ['square', 'triangle']
rot_conditions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

dataset = 's_1_m_08000_a_02000_d_01500_l_04000'


'''
Create network
'''
np.random.seed(seed)
RNN = RecurrentGenerativeModel()
updater = RecurrentGenerativeModelUpdater(RNN)


'''
Training
'''

fig, axs = plt.subplots(len(rot_conditions)*len(objects), 3)
axs_num=0
for obj in objects:
          
    for num, c in enumerate(rot_conditions):
        
        posrot = c
        
        text =  dataset +'_Object_' + obj + '_Angle_' + str(c) + '_.csv' 
        data = np.loadtxt(text, delimiter=',')
        
        scanpath = np.zeros( (len(data),2)  )
        for index, value in enumerate(data):
            scanpath[index, 0] = value//RNN.output_side
            scanpath[index, 1] = value%RNN.output_side
        
        training_time = len(scanpath) 
        test_time = 1000
        simtime = training_time + test_time
        
        
        '''
        Create saving arrays
        '''
        predicted_goal_history = np.zeros(simtime)
        
        
        '''
        Run simulation
        '''
        for t in range(simtime):
        
            if t < training_time - 1:
                predicted_goal = RNN.step(goal = scanpath[t], RNN_updater = updater, mode = "training")
            else:
                predicted_goal = RNN.step(mode = "default")
                
            
            #reservoir_history[:, t] = network.reservoir_activity[:, 0]
            #predicted_goal_history[t,:] = predicted_goal
            predicted_goal_history[t] = int(predicted_goal[0] * 10 + predicted_goal[1])
            print('Object: ', obj, ' Angle:', posrot, ' Timestep:',  t)
        
        axs[axs_num,0].scatter(range(len(data)), data, s=3) 
        axs[axs_num,0].set_ylim(-0.5,101)
        axs[axs_num,1].scatter(range(len(predicted_goal_history)), predicted_goal_history, s=3) 
        axs[axs_num,1].set_ylim(-0.5,101)
        axs_num += 1
        
        
        #reservoir_text = 'Scanpath_seed' + str(seed) + '_' + world +'_pos' + str(posrot[0]) + '_' + str(posrot[1]) + '_rot' + str(posrot[2]) + 'reservoir.csv'
        readout_text = dataset +'_Object_' + obj + '_Angle_' + str(c) + 'readout_training.csv' 
        #scanpath_2D_text = 'Scanpath_seed' + str(seed) + '_' + world +'_pos' + str(posrot[0]) + '_' + str(posrot[1]) + '_rot' + str(posrot[2]) + 'scanpath_2D.csv'
        
        #np.savetxt(reservoir_text, reservoir_history[:,training_time::], delimiter=',')
        #np.savetxt(readout_text, readout_history, delimiter=',')
        #np.savetxt(scanpath_2D_text, scanpath_2Dcoord, delimiter=',')
        np.savetxt(readout_text, predicted_goal_history, delimiter=',')
        
        
        RNN.reset()
        
        #'''
RNN_save_text = 'Trained_RNN' + dataset
RNN.save(RNN_save_text)






'''
Testing
'''

test_time = 500
axs_num=0

RNN.reset()

for obj in objects:
          
    for num, c in enumerate(rot_conditions):
        
        posrot = c
        
        text = dataset +'_Object_' + obj + '_Angle_' + str(c) + '_.csv'
        data = np.loadtxt(text, delimiter=',')
        
        scanpath = np.zeros( (len(data),2)  )
        for index, value in enumerate(data):
            scanpath[index, 0] = value//RNN.output_side
            scanpath[index, 1] = value%RNN.output_side
        
        '''
        Create saving arrays
        '''
        predicted_goal_history = np.zeros(test_time)
          
        for k in range(test_time):
            if k <7:#<3:
                predicted_goal = RNN.step(goal = scanpath[k], mode = "input")
            else:
                predicted_goal = RNN.step(mode = "default")
            predicted_goal_history[k] = int(predicted_goal[0] * 10 + predicted_goal[1])
        
        
        axs[axs_num,2].scatter(range(len(predicted_goal_history)), predicted_goal_history, s=3) 
        axs[axs_num,2].set_ylim(-0.5,101)
        axs_num += 1
        
        readout_text = dataset +'_Object_' + obj + '_Angle_' + str(c) + 'readout_test.csv'
        np.savetxt(readout_text, predicted_goal_history, delimiter=',')
        
        RNN.reset()
        





'''
Plotting

'''
objects = ['square', 'triangle']
rot_conditions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

fig, axs = plt.subplots(len(rot_conditions)*len(objects), 3)
axs_num=0

for obj in objects:
          
    for num, c in enumerate(rot_conditions):
        
        posrot = c
        
        text =  dataset +'_Object_' + obj + '_Angle_' + str(c) + '_.csv' 
        data = np.loadtxt(text, delimiter=',')
        axs[axs_num,0].scatter(range(len(data)), data, s=3) 
        axs[axs_num,0].set_ylim(-0.5,101)
        
        text =  dataset +'_Object_' + obj + '_Angle_' + str(c) + 'readout_training.csv'
        predicted_training = np.loadtxt(text, delimiter=',')
        axs[axs_num,1].scatter(range(len(predicted_training)), predicted_training, s=3) 
        axs[axs_num,1].set_ylim(-0.5,101)
        
        text =  dataset +'_Object_' + obj + '_Angle_' + str(c) + 'readout_test.csv'
        predicted_test = np.loadtxt(text, delimiter=',')
        axs[axs_num,2].scatter(range(len(predicted_test)), predicted_test, s=3) 
        axs[axs_num,2].set_ylim(-0.5,101)
        
        axs_num += 1













