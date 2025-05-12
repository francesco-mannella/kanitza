import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('paths.csv')

'''
#dataset visualization
df.columns

print(df.info())

print(df)

print(df.to_string()) 


#pick objects
obj = df[['Object', 'sim']]
'''
'''
#visualize scanpaths
plt.figure()

for world in ['square', 'triangle']:

    for angle in [0., 0.2, 0.4, 0.6, 0.8, 1.]:
    
        query_text = "Object == '" + world + "' "+ \
                     "and angle ==" + str(angle) +\
                     "and sim == " +\
                     "'s_1_m_08000_a_02000_d_01500_l_04000'"
                     
        d = df.query(query_text)
        
        string_goals = d[['goal']]
        
        string_goals = string_goals[5::].to_numpy()
        
        scanpath = np.zeros((string_goals.shape[0], 2))
        
        for index, item in enumerate(string_goals):
            scanpath[index,0] = np.array(item[0][2],dtype=int)
            scanpath[index,1] = np.array(item[0][5],dtype=int)
        
        
        print(scanpath)
        
        if world == 'square':
            red = 0.2
            green = 0.2
            blue = angle/4 + 0.4
        if world == 'triangle':
            red = angle/4 + 0.4
            green = 0.2
            blue = 0.2
        
        plt.scatter(scanpath[5:,0],scanpath[5:,1], color = (red, green, blue))
        plt.plot(scanpath[5:,0],scanpath[5:,1], color = (red, green, blue), linewidth = angle*4+0.2)
        
'''


#prepare scanpaths for FORCE

plt.figure()        
for world in ['square', 'triangle']:

    for angle in [0.2,0.4]:#[0., 0.2, 0.4, 0.6, 0.8, 1.]:
        
        file = 's_1_m_08000_a_02000_d_01500_l_04000' #  's_1_m_08000_a_02000_d_01500_l_02500'  #
        filestring = "'" + file + "'"
    
        query_text = "Object == '" + world + "' "+ \
                     "and angle ==" + str(angle) +\
                     "and sim == " +\
                     filestring#"''"#"'s_1_m_08000_a_02000_d_02500_l_01000'"#"'s_1_m_08000_a_02000_d_02500_l_00500'"  # 
                     
        d = df.query(query_text)
        
        #scanpath = d[['pos.x','pos.y']]
        
        #scanpath = scanpath[5::].to_numpy()     
        
        string_goals = d[['goal']]
        
        string_goals = string_goals[5::].to_numpy()
        
        scanpath = np.zeros((string_goals.shape[0], 2))
        
        for index, item in enumerate(string_goals):
            scanpath[index,0] = np.array(item[0][2],dtype=int)
            scanpath[index,1] = np.array(item[0][5],dtype=int)
        
        linearized_scanpath = np.zeros(len(scanpath))
        
        for index, values in enumerate(scanpath):
            linearized_scanpath[index] = values[0]*10 + values[1]
        
        sequence = []
        for i in linearized_scanpath:
            if i in sequence:
                break
            sequence.append(i)
            
        sequence = np.array( sequence*2000 )
            
        if world == 'square':
            red = 0.2
            green = 0.2
            blue = angle/4 + 0.4
        if world == 'triangle':
            red = angle/4 + 0.4
            green = 0.2
            blue = 0.2
        
        plt.scatter(scanpath[5:,0],scanpath[5:,1], color = (red, green, blue))
        plt.plot(scanpath[5:,0],scanpath[5:,1], color = (red, green, blue), linewidth = angle*4+0.2)
        
        print(sequence[-15::])

        #savetext = file + '_Object_' + world + '_Angle_' + str(angle) + '_.csv'
        #np.savetxt(savetext, sequence, delimiter=',')





